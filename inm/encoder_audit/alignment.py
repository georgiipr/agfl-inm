"""Trial identity, training-only normalization, and frozen feature replay."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifacts import LABEL_NAMES, LEGACY_CHANNELS


class AlignmentError(ValueError):
    """A source trial or within-study replay invariant failed."""


def _load_source_bundle(cfg: dict, subject: int):
    # Keep this identical to the historical data loader settings. It only reads
    # recordings; it does not call get_split or create a .splits directory.
    from inm.data import load_subject
    prep = cfg["preprocessing"]
    return load_subject({"data_dir": cfg["data_dir"], "subjects": [subject], "sessions": ["T"],
        "artifact_policy": "exclude", "filter_scope": "window", "window": prep["trial_samples"],
        "offset_seconds": 0.0, "filter_window_samples": prep["window_samples"],
        "lowcut": prep["filter_low_hz"], "highcut": prep["filter_high_hz"],
        "normalization": "none", "labels_dir": None})


def _max_error(actual, expected) -> float:
    import numpy as np
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape:
        raise AlignmentError(f"Shape mismatch: replay {a.shape}, saved {b.shape}")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AlignmentError("Nonfinite values encountered during alignment/replay")
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.0


def _require_close(actual, expected, tolerance, name):
    import numpy as np
    error = _max_error(actual, expected)
    if not np.allclose(actual, expected, rtol=tolerance["rtol"], atol=tolerance["atol"]):
        raise AlignmentError(f"{name} differs (maximum absolute error {error:g})")
    return error


def _split(task_path: str):
    path = Path(task_path) / "split.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise AlignmentError(f"Cannot read persisted split {path}: {error}") from error


def _dataset_metadata(task_path: str):
    path = Path(task_path) / "dataset.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise AlignmentError(f"Cannot read persisted dataset metadata {path}: {error}") from error


def _indices(split: dict, part: str, ids: tuple[str, ...], all_ids: tuple[str, ...]):
    indices = split.get(part)
    stored_ids = split.get("sample_ids", {}).get(part)
    if (not isinstance(indices, list) or not isinstance(stored_ids, list) or
            len(indices) != len(ids) or len(stored_ids) != len(ids) or
            any(type(i) is not int or i < 0 or i >= len(all_ids) for i in indices) or
            len(set(indices)) != len(indices) or len(set(stored_ids)) != len(stored_ids)):
        raise AlignmentError(f"Malformed persisted {part} membership")
    if [all_ids[i] for i in indices] != stored_ids or tuple(stored_ids) != ids:
        raise AlignmentError(f"Persisted {part} identities are reordered or differ from verified view")
    return indices


def _normalization(raw, train_indices, saved_mean, saved_std, floor, tolerance, label):
    import numpy as np
    training = np.asarray(raw[train_indices], dtype=np.float64)
    if training.ndim != 3 or training.shape[1] != 22 or not np.isfinite(training).all():
        raise AlignmentError(f"{label} training raw signals are malformed or nonfinite")
    mean = training.mean(axis=(0, 2))
    std = np.maximum(training.std(axis=(0, 2), ddof=0), floor)
    saved_mean = np.asarray(saved_mean, dtype=np.float64).reshape(-1)
    saved_std = np.asarray(saved_std, dtype=np.float64).reshape(-1)
    if saved_mean.shape != (22,) or saved_std.shape != (22,) or (saved_std <= 0).any():
        raise AlignmentError(f"{label} saved normalization has invalid shape or values")
    return {"mean_max_abs_error": _require_close(mean, saved_mean, tolerance, f"{label} raw mean"),
            "std_max_abs_error": _require_close(std, saved_std, tolerance, f"{label} raw std"),
            "mean": mean, "std": std}


def _legacy_replay(raw, indices, sources, cfg):
    import numpy as np
    import torch
    from inm.model import EEGWindowEncoder

    legacy = sources.legacy
    channels, windows, size = 22, 4, 250
    model = EEGWindowEncoder(channels=channels, windows=windows, window_samples=size)
    try:
        model.load_state_dict(legacy.encoder_state, strict=True)
    except Exception as error:
        raise AlignmentError(f"Legacy encoder state cannot be reconstructed: {error}") from error
    model.freeze()
    bad_modes = [name for name, module in model.named_modules()
        if isinstance(module, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.modules.dropout._DropoutNd)) and module.training]
    if model.training or bad_modes or any(parameter.requires_grad for parameter in model.parameters()):
        raise AlignmentError("Encoder BatchNorm/dropout or gradients are not frozen")
    raw_mean = np.asarray(legacy.raw_mean, dtype=np.float64).reshape(1, 22, 1)
    raw_std = np.asarray(legacy.raw_std, dtype=np.float64).reshape(1, 22, 1)
    if raw_mean.shape != (1, 22, 1) or raw_std.shape != (1, 22, 1) or not np.isfinite(raw_mean).all() or not np.isfinite(raw_std).all() or (raw_std <= 0).any():
        raise AlignmentError("Legacy raw normalization arrays are invalid")
    normalized = {part: ((np.asarray(raw[part_indices], dtype=np.float64) - raw_mean) / raw_std).astype(np.float32)
                  for part, part_indices in indices.items()}
    all_results = {}
    batch_error = 0.0
    feature_mean_error = feature_std_error = None
    for part in ("train", "validation"):
        part_indices = indices[part]
        outputs = []
        with torch.inference_mode():
            full_batch = model(torch.from_numpy(normalized[part])).cpu().numpy()
            for start in range(0, len(part_indices), 2):
                outputs.append(model(torch.from_numpy(normalized[part][start:start + 2])).cpu().numpy())
        separately_batched = np.concatenate(outputs, axis=0)
        batch_error = max(batch_error, _require_close(full_batch, separately_batched,
            cfg["tolerances"]["features"], f"{part} batch-size invariance"))
        # _calibrate standardizes features with training feature moments only.
        feature_mean = np.asarray(legacy.feature_mean, dtype=np.float32)
        feature_std = np.asarray(legacy.feature_std, dtype=np.float32)
        if feature_mean.shape != (1, 22, 1, 32) or feature_std.shape != feature_mean.shape or (feature_std <= 0).any():
            raise AlignmentError("Saved training feature mean/std have invalid shapes or values")
        if part == "train":
            train_raw_features = full_batch.astype(np.float64)
            expected_mean = train_raw_features.mean(axis=(0, 2), keepdims=True).astype(np.float32)
            expected_std = np.maximum(train_raw_features.std(axis=(0, 2), keepdims=True), 1e-6).astype(np.float32)
            feature_mean_error = _require_close(feature_mean, expected_mean,
                cfg["tolerances"]["features"], "saved training feature mean")
            feature_std_error = _require_close(feature_std, expected_std,
                cfg["tolerances"]["features"], "saved training feature std")
        replay = (full_batch - feature_mean) / feature_std
        cache = legacy.train_features if part == "train" else legacy.validation_features
        all_results[part] = _require_close(replay, cache, cfg["tolerances"]["features"], f"{part} cached features")
    # Locality: perturb a different electrode/window and compare the selected
    # output. Hidden NaNs are selected away before any arithmetic.
    probe = torch.from_numpy(normalized["validation"][:1])
    if not len(probe):
        raise AlignmentError("No validation sample available for encoder invariants")
    baseline = model(probe)
    changed = probe.clone(); changed[0, 1, 250:500] += 3.0
    local_error = float((model(changed)[0, 0, 0] - baseline[0, 0, 0]).abs().max().item())
    if local_error != 0.0:
        raise AlignmentError("Perturbing another electrode/window changed a local feature")
    nan_input = probe.clone(); mask = torch.ones((1, 22, 4), dtype=torch.bool); mask[:, 0, 0] = False
    nan_input.reshape(1, 22, 4, 250)[:, 0, 0] = float("nan")
    masked = model(nan_input, mask)
    zeroed = probe.clone(); zeroed.reshape(1, 22, 4, 250)[:, 0, 0] = 0.
    mask_error = float((masked - model(zeroed, mask)).abs().max().item())
    if not np.isfinite(masked.numpy()).all() or mask_error != 0.0:
        raise AlignmentError("Hidden NaN leaked through encoder masking")
    return {"train_cache_max_abs_error": all_results["train"],
        "validation_cache_max_abs_error": all_results["validation"],
        "batch_size_max_abs_error": batch_error, "locality_max_abs_error": local_error,
        "hidden_nan_invariance_max_abs_error": mask_error,
        "feature_mean_max_abs_error": feature_mean_error,
        "feature_std_max_abs_error": feature_std_error,
        "encoder_mode": "eval_frozen", "batchnorm_frozen": True, "dropout_frozen": True,
        "training_feature_statistics_source": "saved_training_only_feature_mean_std"}


def audit_alignment(sources: Any, cfg: dict) -> dict:
    """Audit original trial IDs, persisted partitions, normalization, and features.

    The result distinguishes a failed within-study replay from an unavailable
    cross-study pairing. No split writer or test labels are accessed.
    """
    import numpy as np
    subject = sources.subject
    bundle = _load_source_bundle(cfg, subject)
    loaded_ids = tuple(bundle.sample_ids)
    if any(not sid.startswith(f"A{subject:02d}T:cue:") or ":sample:" not in sid for sid in loaded_ids):
        raise AlignmentError("Source trial IDs do not encode the requested T-session cue identities")
    if tuple(bundle.metadata.get("channel_names", ())) != LEGACY_CHANNELS or tuple(bundle.metadata.get("label_names", ())) != LABEL_NAMES:
        raise AlignmentError("Source channel/class order differs from the fixed protocol")
    prep = bundle.metadata.get("preprocessing", {})
    if (prep.get("offset_seconds") != 0.0 or prep.get("filter_scope") != "window" or
            prep.get("filter_window_samples") != 250 or prep.get("window") != 1000 or
            prep.get("lowcut") != 2.0 or prep.get("highcut") != 30.0 or prep.get("artifact_policy") != "exclude"):
        raise AlignmentError("Cue offset, trial exclusion, or window-local filtering differs from protocol")
    if bundle.x.shape[1:] != (22, 1000) or float(bundle.metadata.get("sampling_rate")) != 250.0:
        raise AlignmentError("Source sample count, channel count, or sample rate differs from protocol")
    if bundle.metadata.get("skipped", {}).get("artifact", 0) < 0:
        raise AlignmentError("Invalid source exclusion counts")
    labels = np.asarray(bundle.y, dtype=np.int64)
    if not np.isfinite(bundle.x).all():
        raise AlignmentError("Source signals contain nonfinite values")
    if set(labels.tolist()) != set(range(4)):
        raise AlignmentError("Cue-derived source lacks one or more classes")
    views = {"baseline": sources.baseline, "legacy": sources.legacy}
    raw = np.asarray(bundle.x)
    study = {}
    partition_indices = {}
    for name, view in views.items():
        metadata = getattr(sources, f"{name}_metadata")
        split = _split(metadata.task_path)
        persisted_dataset = _dataset_metadata(metadata.task_path)
        persisted_exclusions = persisted_dataset.get("source_exclusions", persisted_dataset.get("skipped"))
        if persisted_exclusions is not None and persisted_exclusions != bundle.metadata.get("skipped", {}):
            raise AlignmentError(f"{name} source trial exclusions differ from rebuilt source exclusions")
        indices = {part: _indices(split, part, getattr(view, part).sample_ids, loaded_ids)
                   for part in ("train", "validation")}
        partition_indices[name] = indices
        if any(int(labels[loaded_ids.index(sid)]) != int(label) for part in indices for sid, label in
               zip(getattr(view, part).sample_ids, getattr(view, part).labels)):
            raise AlignmentError(f"{name} cue-derived labels are shifted relative to persisted trial IDs")
        if name == "baseline":
            saved_mean, saved_std = sources.baseline.normalization_mean, sources.baseline.normalization_std
            floor = 1e-8
        else:
            saved_mean, saved_std = sources.legacy.raw_mean, sources.legacy.raw_std
            floor = 1e-12
        normalization = _normalization(raw, indices["train"], saved_mean, saved_std, floor,
                                       cfg["tolerances"]["normalization"], name)
        normalized_view = {part: ((raw[indices[part]].astype(np.float64) - np.asarray(saved_mean).reshape(1, 22, 1)) /
                           np.asarray(saved_std).reshape(1, 22, 1)).astype(np.float32)
                           for part in ("train", "validation")}
        if not all(np.isfinite(values).all() for values in normalized_view.values()):
            raise AlignmentError(f"{name} saved normalization produces nonfinite train/validation values")
        # The baseline stores channel vectors; legacy stores [1,C,1]. Both are
        # checked with their own protocol floor, then validation uses those exact values.
        study[name] = {"status": "verified", "split_id": metadata.split_id,
            "train_count": len(indices["train"]), "validation_count": len(indices["validation"]),
            "normalization_mean_max_abs_error": normalization["mean_max_abs_error"],
            "normalization_std_max_abs_error": normalization["std_max_abs_error"],
            "epsilon_floor": floor}
    cross_match = all(getattr(sources.baseline, p).sample_ids == getattr(sources.legacy, p).sample_ids and
        np.array_equal(getattr(sources.baseline, p).labels, getattr(sources.legacy, p).labels)
        for p in ("train", "validation"))
    source_integrity = all(all(value == "match" for value in md.replay_source_status.values())
        for md in (sources.baseline_metadata, sources.legacy_metadata))
    mismatches = {name: [path for path, status in md.replay_source_status.items() if status != "match"]
        for name, md in (("baseline", sources.baseline_metadata), ("legacy", sources.legacy_metadata))}
    mismatches = {name: paths for name, paths in mismatches.items() if paths}
    replay = (_legacy_replay(raw, partition_indices["legacy"], sources, cfg) if source_integrity else
        {"status": "blocked_historical_source_mismatch", "mismatches": mismatches})
    if not source_integrity:
        study["source_integrity"] = {"status": "blocked", "mismatches": {
            **mismatches}}
    return {"status": "complete" if source_integrity else "blocked",
        "within_study": study, "legacy_feature_replay": replay,
        "cross_study_pairing": {"status": "matched" if cross_match else "unavailable_unmatched_inputs",
            "train_ids_match": sources.baseline.train.sample_ids == sources.legacy.train.sample_ids,
            "validation_ids_match": sources.baseline.validation.sample_ids == sources.legacy.validation.sample_ids,
            "paired_descriptive_contrast_available": bool(cross_match and source_integrity)},
        "source_trial_count": len(loaded_ids), "source_exclusions": dict(bundle.metadata.get("skipped", {})),
        "cue_alignment": "verified_by_stable_ids_and_labels", "partitions_exposed": ["train", "validation"],
        "tolerances": cfg["tolerances"], "invariant_violations": [] if source_integrity else ["historical_source_mismatch"]}
