"""Clean evaluation of the three validation-selected EEGNet checkpoints."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


class CheckpointAuditError(ValueError):
    """A saved classifier cannot be faithfully reconstructed or audited."""


_ARMS = ("eegnet_reference__full", "masked_eegnet__full", "masked_eegnet__mixed")
_METRIC_KEYS = ("balanced_accuracy", "accuracy", "f1", "log_loss")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_digest(state: dict[str, Any]) -> str:
    import torch
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _metrics(labels, probabilities):
    import numpy as np
    from agfl.metrics import classification_metrics
    from scipy.special import xlogy

    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    result = classification_metrics(labels, probabilities)
    result["balanced_accuracy"] = float(np.mean([row["recall"] for row in result["per_class"]]))
    result["log_loss"] = float(-xlogy(np.eye(4)[labels], np.clip(probabilities, 1e-12, 1.0)).sum(axis=1).mean())
    return result


def _predict(model, x, batch_size, device):
    import numpy as np
    import torch

    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            logits = model(x[start:start + batch_size].to(device))
            if logits.shape != (min(batch_size, len(x) - start), 4) or not bool(torch.isfinite(logits).all()):
                raise CheckpointAuditError("Checkpoint emitted malformed or nonfinite four-class logits")
            pieces.append(logits.softmax(-1).cpu().numpy())
    probabilities = np.concatenate(pieces, axis=0)
    if not np.isfinite(probabilities).all():
        raise CheckpointAuditError("Checkpoint emitted nonfinite probabilities")
    return probabilities


def _history_metrics(row, partition):
    values = row.get(f"clean_{partition}_metrics")
    if not isinstance(values, dict):
        raise CheckpointAuditError(f"Selected history row lacks clean {partition} metrics")
    return values


def _compare_history(recorded, replayed, tolerance, label):
    import numpy as np

    errors = {}
    for key in _METRIC_KEYS:
        if key not in recorded:
            raise CheckpointAuditError(f"Selected history row lacks {label} {key}")
        expected, actual = float(recorded[key]), float(replayed[key])
        errors[key] = abs(expected - actual)
        if not np.isfinite(expected) or not np.isclose(expected, actual,
                rtol=tolerance["rtol"], atol=tolerance["atol"]):
            raise CheckpointAuditError(
                f"Selected-epoch {label} {key} differs: recorded={expected:g}, replayed={actual:g}")
    if recorded.get("confusion_matrix") != replayed["confusion_matrix"]:
        raise CheckpointAuditError(f"Selected history {label} confusion matrix differs from replay")
    return errors


def audit_checkpoints(sources: Any, aligned: dict, cfg: dict, device: str = "cpu") -> dict:
    """Replay selected checkpoints on clean train/validation inputs only.

    Returns ``arms`` keyed by the three stable arm IDs. Each has train and
    validation metrics, train-minus-validation metric gaps, selected-epoch
    history comparison, and separately labeled optimization-time metrics.
    """
    import copy
    import numpy as np
    import torch
    from inm.baselines.eegnet import EEGNetClassifier
    from . import alignment as alignment_module
    from .artifacts import LABEL_NAMES, LEGACY_CHANNELS

    baseline_alignment = aligned.get("within_study", {}).get("baseline", {})
    if baseline_alignment.get("status") != "verified":
        raise CheckpointAuditError("Session 03 did not verify baseline alignment/normalization")
    if any(status != "match" for status in sources.baseline_metadata.replay_source_status.values()):
        bad = [path for path, status in sources.baseline_metadata.replay_source_status.items()
               if status != "match"]
        raise CheckpointAuditError("Historical baseline replay source mismatch: " + ", ".join(bad))
    if tuple(cfg.get("partitions", ())) != ("train", "validation"):
        raise CheckpointAuditError("Checkpoint replay is restricted to train and validation")

    bundle = alignment_module._load_source_bundle(cfg, sources.subject)
    source_ids = tuple(bundle.sample_ids)
    if len(set(source_ids)) != len(source_ids):
        raise CheckpointAuditError("Canonical source contains duplicate sample IDs")
    index_by_id = {sample_id: index for index, sample_id in enumerate(source_ids)}
    if tuple(bundle.metadata.get("channel_names", ())) != LEGACY_CHANNELS:
        raise CheckpointAuditError("Raw source channel order differs from the saved EEGNet protocol")
    if tuple(bundle.metadata.get("label_names", ())) != LABEL_NAMES:
        raise CheckpointAuditError("Raw source class order differs from the saved EEGNet protocol")
    mean = np.asarray(sources.baseline.normalization_mean, dtype=np.float64).reshape(-1)
    std = np.asarray(sources.baseline.normalization_std, dtype=np.float64).reshape(-1)
    if mean.shape != (22,) or std.shape != (22,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise CheckpointAuditError("Saved baseline normalization statistics are invalid")
    raw = np.asarray(bundle.x)
    source_labels = np.asarray(bundle.y)
    if raw.ndim != 3 or raw.shape[1:] != (22, 1000):
        raise CheckpointAuditError("Canonical EEG source has invalid shape")

    partitions = {}
    for part in ("train", "validation"):
        view = getattr(sources.baseline, part)
        if not view.sample_ids or len(view.sample_ids) != len(view.labels):
            raise CheckpointAuditError(f"Baseline {part} identities/labels are empty or mismatched")
        if len(set(view.sample_ids)) != len(view.sample_ids) or any(sid not in index_by_id for sid in view.sample_ids):
            raise CheckpointAuditError(f"Baseline {part} sample IDs are duplicate or absent from source")
        indices = [index_by_id[sid] for sid in view.sample_ids]
        labels = np.asarray(view.labels, dtype=np.int64)
        if not np.array_equal(np.asarray(source_labels[indices], dtype=np.int64), labels):
            raise CheckpointAuditError(f"Baseline {part} labels do not match stable source identities")
        counts = np.bincount(labels, minlength=4)
        if labels.ndim != 1 or len(labels) != len(indices) or (counts == 0).any():
            raise CheckpointAuditError(f"Baseline {part} must contain all four classes")
        selected_raw = raw[indices].astype(np.float64)
        if not np.isfinite(selected_raw).all():
            raise CheckpointAuditError(f"Baseline {part} source contains nonfinite values")
        normalized = ((selected_raw - mean[None, :, None]) / std[None, :, None]).astype(np.float32)
        partitions[part] = (torch.from_numpy(normalized.reshape(-1, 22, 4, 250)), labels)

    results = {}
    for arm in _ARMS:
        if arm not in sources.baseline.checkpoints:
            raise CheckpointAuditError(f"Missing required saved neural arm {arm}")
        checkpoint = sources.baseline.checkpoints[arm]
        if checkpoint.arm != arm or checkpoint.selected_epoch < 1 or len(checkpoint.history_selection) != 1:
            raise CheckpointAuditError(f"Wrong selected epoch/history identity for {arm}")
        history = checkpoint.history_selection[0]
        if history.get("epoch") != checkpoint.selected_epoch:
            raise CheckpointAuditError(f"Wrong selected epoch in history for {arm}")
        metadata = checkpoint.metadata
        if (metadata.get("selected_epoch") != checkpoint.selected_epoch or metadata.get("subject") != sources.subject
                or metadata.get("seed") != sources.seed or metadata.get("arm") != arm):
            raise CheckpointAuditError(f"Checkpoint metadata identity/epoch mismatch for {arm}")
        normalization = metadata.get("normalization_stats")
        if not isinstance(normalization, dict) or not {"mean", "std"} <= set(normalization):
            raise CheckpointAuditError(f"Checkpoint lacks saved normalization metadata for {arm}")
        if (not np.allclose(np.asarray(normalization["mean"], dtype=np.float64), mean, rtol=0, atol=0)
                or not np.allclose(np.asarray(normalization["std"], dtype=np.float64), std, rtol=0, atol=0)):
            raise CheckpointAuditError(f"Checkpoint normalization differs from aligned baseline statistics for {arm}")
        if metadata.get("class_order") != list(LABEL_NAMES) or metadata.get("channel_order") != list(LEGACY_CHANNELS):
            raise CheckpointAuditError(f"Wrong saved class or channel order for {arm}")
        if checkpoint.model_type != "inm.baselines.eegnet.EEGNetClassifier":
            raise CheckpointAuditError(f"Unsupported historical classifier model type {checkpoint.model_type!r}")
        settings = dict(checkpoint.constructor)
        if settings.get("channels") != 22 or settings.get("windows") != 4 or settings.get("window_samples") != 250 or settings.get("num_classes") != 4:
            raise CheckpointAuditError(f"Wrong EEGNet constructor dimensions for {arm}")
        expected_mask_conditioned = arm != "eegnet_reference__full"
        if settings.get("mask_conditioned") is not expected_mask_conditioned:
            raise CheckpointAuditError(f"Wrong model constructor/arm identity for {arm}")
        try:
            model = EEGNetClassifier(**settings)
            model.load_state_dict(checkpoint.state_dict, strict=True)
        except Exception as error:
            raise CheckpointAuditError(f"Unsupported checkpoint state/constructor for {arm}: {error}") from error
        model.to(device).eval()
        bad_modes = [name for name, module in model.named_modules()
                     if isinstance(module, (torch.nn.modules.batchnorm._BatchNorm,
                                            torch.nn.modules.dropout._DropoutNd)) and module.training]
        if model.training or bad_modes:
            raise CheckpointAuditError(f"Evaluation mode did not disable dropout/BatchNorm for {arm}")
        state_before = _state_digest(model.state_dict())
        file_hash_before = {}
        for relative, expected in sources.baseline_metadata.artifact_sha256.items():
            if relative.startswith(f"{arm}/"):
                path = Path(sources.baseline_metadata.task_path) / "ARMS" / relative
                if not path.is_file() or _sha256(path) != expected:
                    raise CheckpointAuditError(f"Checkpoint/history input changed before replay: {path}")
                file_hash_before[relative] = expected
        if {f"{arm}/checkpoint.pt", f"{arm}/history.json"} - set(file_hash_before):
            raise CheckpointAuditError(f"Verified checkpoint/history file hashes are missing for {arm}")
        probability_sets, metrics = {}, {}
        for part in ("train", "validation"):
            x, labels = partitions[part]
            probabilities = _predict(model, x, int(cfg.get("checkpoint_batch_size", 64)), device)
            probability_sets[part] = probabilities
            metrics[part] = _metrics(labels, probabilities)
        state_after = _state_digest(model.state_dict())
        if state_after != state_before:
            raise CheckpointAuditError(f"Evaluation modified model parameters or buffers for {arm}")
        tolerance = cfg["tolerances"]["probabilities"]
        history_errors = {}
        for part in ("train", "validation"):
            history_errors[part] = _compare_history(_history_metrics(history, part), metrics[part],
                                                     tolerance, f"{arm} {part}")
        # Independently reconstruct from the saved constructor/state and prove
        # evaluation probabilities do not depend on the first model instance.
        reloaded = EEGNetClassifier(**settings)
        reloaded.load_state_dict(copy.deepcopy(checkpoint.state_dict), strict=True)
        reloaded.to(device).eval()
        reload_errors = {}
        for part in ("train", "validation"):
            x, _ = partitions[part]
            second = _predict(reloaded, x, int(cfg.get("checkpoint_batch_size", 64)), device)
            reload_errors[part] = float(np.max(np.abs(second - probability_sets[part])))
            if not np.allclose(second, probability_sets[part], rtol=tolerance["rtol"], atol=tolerance["atol"]):
                raise CheckpointAuditError(f"Reload predictions differ for {arm} {part}")
        file_hash_after = {relative: _sha256(Path(sources.baseline_metadata.task_path) / "ARMS" / relative)
                           for relative in file_hash_before}
        if file_hash_after != file_hash_before:
            raise CheckpointAuditError(f"Input checkpoint/history bytes changed during replay for {arm}")
        gaps = {key: metrics["train"][key] - metrics["validation"][key] for key in _METRIC_KEYS}
        results[arm] = {"selected_epoch": checkpoint.selected_epoch,
            "class_order": list(LABEL_NAMES), "channel_order": list(LEGACY_CHANNELS),
            "metrics": metrics, "train_minus_validation_gaps": gaps,
            "selected_history_max_abs_error": history_errors,
            "reload_probability_max_abs_error": reload_errors,
            "model_state_unchanged": True, "input_artifact_sha256_unchanged": file_hash_after,
            "optimization_time_metrics": {key: history.get(key) for key in
                ("masked_train_loss", "masked_train_accuracy") if key in history},
            "optimization_time_label": "training_mode_with_epoch_masks_and_dropout"}

    legacy = sources.legacy.encoder_selection
    return {"status": "complete", "partitions": ["train", "validation"], "arms": results,
        "tolerances": cfg["tolerances"]["probabilities"],
        "legacy_encoder_selection": {"historical_metadata": legacy,
            "clean_classifier_replay": "unavailable_pretraining_mha_head_not_persisted",
            "probes_are_separate_diagnostic_models": True},
        "no_test_rows_used": True}
