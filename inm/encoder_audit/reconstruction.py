"""Paired validation reconstruction and fixed-probe diagnostics.

Completion is label-blind. Labels are read only after every mask bank has been
completed, for classification metrics and descriptive class stratification.
"""
from __future__ import annotations


class ReconstructionError(ValueError):
    """Invalid reconstruction input, protocol, or completion result."""


_CONDITIONS = (
    ("full_22", "full", 22, 1),
    ("random_static_16", "random_static", 16, 5),
    ("dynamic_random_16", "dynamic_random", 16, 5),
    ("random_static_6", "random_static", 6, 5),
    ("dynamic_random_6", "dynamic_random", 6, 5),
)
_CLASSES = (0, 1, 2, 3)


def _conditions(cfg):
    declared = cfg.get("conditions")
    expected = [{"name": n, "pattern": p, "retained": k, "repeats": r}
                for n, p, k, r in _CONDITIONS]
    if declared != expected:
        raise ReconstructionError("Validation conditions must be exactly full and the four declared random 16/6 conditions (five repeats)")
    return _CONDITIONS


def _factors(state):
    """Load the immutable saved factor buffers; never fit factors here."""
    import torch
    from inm.tensor_attention import Tucker2
    try:
        u_value, v_value = state["U"], state["V"]
        u = u_value.detach().cpu() if hasattr(u_value, "detach") else torch.as_tensor(u_value)
        v = v_value.detach().cpu() if hasattr(v_value, "detach") else torch.as_tensor(v_value)
        if u.ndim != 2 or u.shape[0] != 22 or v.ndim != 2 or v.shape[0] != 32:
            raise ValueError("unsupported factor shapes")
        solver = Tucker2(22, 32, rank_channels=u.shape[1], rank_features=v.shape[1], ridge=.001)
        tensor_state = {key: value.detach().cpu().clone() if hasattr(value, "detach") else torch.as_tensor(value).clone()
                        for key, value in state.items()}
        solver.load_state_dict(tensor_state, strict=True)
        if not bool(solver.fitted.item()):
            raise ValueError("factors are not marked fitted")
        solver.eval()
        return solver
    except Exception as error:
        raise ReconstructionError(f"Cannot load frozen Tucker factors: {error}") from error


def _complete(solver, values, available):
    """Complete a batch safely, preserving observed values exactly."""
    import numpy as np
    import torch
    x = np.asarray(values)
    mask = np.asarray(available)
    if x.ndim != 4 or x.shape[1:] != (22, 4, 32) or mask.shape != x.shape[:3] or mask.dtype != np.bool_:
        raise ReconstructionError("Features/mask must have shapes [N,22,4,32] and Boolean [N,22,4]")
    if not len(x) or not np.isfinite(np.where(mask[..., None], x, 0)).all():
        raise ReconstructionError("Observed features must be nonempty and finite")
    tx = torch.as_tensor(x, dtype=torch.float32)
    tm = torch.as_tensor(mask, dtype=torch.bool)
    with torch.inference_mode():
        completed = solver.complete(tx, tm).cpu().numpy()
    if not np.isfinite(completed).all():
        raise ReconstructionError("Tucker completion produced nonfinite values")
    if not np.array_equal(completed[mask[..., None].repeat(x.shape[-1], axis=-1)],
                          x[mask[..., None].repeat(x.shape[-1], axis=-1)]):
        raise ReconstructionError("Tucker completion changed observed entries")
    return completed


def _metrics(labels, probabilities):
    import numpy as np
    from sklearn.metrics import balanced_accuracy_score, log_loss, confusion_matrix
    predicted = np.asarray(_CLASSES)[probabilities.argmax(axis=1)]
    cm = confusion_matrix(labels, predicted, labels=_CLASSES)
    counts = cm.sum(axis=1)
    recalls = [float(cm[i, i] / counts[i]) if counts[i] else None for i in range(4)]
    # sklearn's balanced accuracy averages recall over labels present in truth.
    return {"balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
            "log_loss": float(log_loss(labels, probabilities, labels=_CLASSES)),
            "per_class_recall": recalls, "class_counts": counts.tolist(),
            "confusion_matrix": cm.tolist()}


def _probability_log_loss(labels, probability):
    import numpy as np
    return (-np.log(np.clip(probability[np.arange(len(labels)), labels], 1e-15, 1.0))).tolist()


def _hidden_error_sums(truth, estimate, mask):
    """Return the legacy pooled hidden SSE and target energy in float64."""
    import numpy as np
    truth64 = np.asarray(truth, dtype=np.float64)
    estimate64 = np.asarray(estimate, dtype=np.float64)
    hidden = ~np.asarray(mask, dtype=bool)[..., None]
    if truth64.shape != estimate64.shape or truth64.ndim != 4 or hidden.shape != (*truth64.shape[:3], 1):
        raise ReconstructionError("Hidden scoring expects matching [N,C,P,F] tensors and [N,C,P] mask")
    return (float(np.where(hidden, (estimate64 - truth64) ** 2, 0.0).sum()),
            float(np.where(hidden, truth64 ** 2, 0.0).sum()), int(hidden.sum() * truth64.shape[-1]))


def audit_reconstruction(sources, ordered_probe, cfg):
    """Compare paired full, zero-fill, and Tucker-completed validation features.

    Returns per-condition mask digests, aggregate hidden-entry sums/NRMSE,
    fixed-probe metrics and per-trial diagnostics. Repeats remain separate;
    downstream aggregation is repeats, then seeds, then participants.
    """
    import numpy as np
    from inm.availability import make_mask_bank, mask_bank_digest
    conditions = _conditions(cfg)
    if getattr(ordered_probe, "name", None) != "features_ordered":
        raise ReconstructionError("A fitted features_ordered probe is required")
    metadata = getattr(ordered_probe, "metadata", {})
    if metadata.get("subject") != int(sources.subject) or metadata.get("seed") != int(sources.seed):
        raise ReconstructionError("Probe task identity does not match source task")
    expected_studies = {"baseline": sources.baseline_metadata.study_id,
                        "legacy": sources.legacy_metadata.study_id}
    if metadata.get("input_study_ids") != expected_studies:
        raise ReconstructionError("Probe input study identities do not match source task")
    if metadata.get("sample_ids", {}).get("validation") != list(sources.legacy.validation.sample_ids):
        raise ReconstructionError("Probe validation sample identities/order do not match sources")

    features = np.asarray(sources.legacy.validation_features)
    ids = tuple(sources.legacy.validation.sample_ids)
    if features.ndim != 4 or features.shape[1:] != (22, 4, 32) or not len(features):
        raise ReconstructionError("Validation features must have shape [N,22,4,32]")
    if len(ids) != len(features) or len(set(ids)) != len(ids):
        raise ReconstructionError("Validation sample IDs are missing, duplicated, or misaligned")
    if not np.isfinite(features).all():
        raise ReconstructionError("Validation reference features must be finite for scoring")

    solver = _factors(sources.legacy.factor_state)
    # Build and complete every bank before reading labels. Mask identities are
    # independent of labels and keyed by stable trial IDs.
    banks = []
    for name, pattern, retained, repeats in conditions:
        for repeat in range(repeats):
            mask = make_mask_bank(len(features), 4, retained, pattern,
                seed=int(sources.seed), partition="validation", subject=f"A{int(sources.subject):02d}",
                repeat=repeat, sample_ids=ids)
            mask_digest = mask_bank_digest(mask)
            if retained == 22:
                completed = features.copy()
            else:
                completed = _complete(solver, features, mask)
            zero_filled = np.where(mask[..., None], features, 0.0)
            if not np.array_equal(zero_filled[mask[..., None].repeat(32, axis=-1)],
                                  features[mask[..., None].repeat(32, axis=-1)]):
                raise ReconstructionError("Zero-fill changed observed entries")
            banks.append({"condition": name, "pattern": pattern, "retained": retained,
                          "repeat": repeat, "mask": mask, "mask_sha256": mask_digest,
                          "completed": completed, "zero_filled": zero_filled})

    labels = np.asarray(sources.legacy.validation.labels)
    if labels.shape != (len(features),) or not np.issubdtype(labels.dtype, np.integer) or not np.isin(labels, _CLASSES).all():
        raise ReconstructionError("Validation labels must be integer class IDs 0..3")

    output = []
    truth64 = features.astype(np.float64, copy=False)
    for bank in banks:
        mask = bank["mask"]
        hidden = ~mask[..., None]
        hidden_count = int(hidden.sum() * 32)
        if hidden_count:
            errors = {}
            for key, estimate in (("zero_fill", bank["zero_filled"]), ("tucker_completion", bank["completed"])):
                errors[key], energy, _ = _hidden_error_sums(features, estimate, mask)
            nrmse = {key: (float(np.sqrt(value / energy)) if energy > 0 else None)
                     for key, value in errors.items()}
        else:
            energy, errors, nrmse = 0.0, {"zero_fill": 0.0, "tucker_completion": 0.0}, {
                "zero_fill": None, "tucker_completion": None}

        full_p = ordered_probe.predict_proba(features)
        zero_p = ordered_probe.predict_proba(bank["zero_filled"])
        completed_p = ordered_probe.predict_proba(bank["completed"])
        metric_map = {"full_features": _metrics(labels, full_p), "zero_fill": _metrics(labels, zero_p),
                      "tucker_completion": _metrics(labels, completed_p)}
        paired = {}
        for key in ("balanced_accuracy", "log_loss"):
            paired[key] = float(metric_map["tucker_completion"][key] - metric_map["zero_fill"][key])
        paired["per_class_recall"] = [
            (float(a - b) if a is not None and b is not None else None)
            for a, b in zip(metric_map["tucker_completion"]["per_class_recall"],
                            metric_map["zero_fill"]["per_class_recall"])]
        per_trial = []
        for i, sample_id in enumerate(ids):
            trial_hidden = hidden[i]
            trial_energy = float(np.where(trial_hidden, truth64[i] ** 2, 0.0).sum()) if hidden_count else 0.0
            trial_error = {key: float(np.where(trial_hidden, (arr[i].astype(np.float64)-truth64[i])**2, 0.0).sum())
                           for key, arr in (("zero_fill", bank["zero_filled"]), ("tucker_completion", bank["completed"]))}
            per_trial.append({"sample_id": sample_id, "label": int(labels[i]),
                "hidden_squared_error_sum": trial_error, "hidden_target_energy_sum": trial_energy,
                "hidden_nrmse": {key: (float(np.sqrt(value/trial_energy)) if trial_energy > 0 else None)
                                 for key, value in trial_error.items()},
                "log_loss": {"full_features": _probability_log_loss(labels[i:i+1], full_p[i:i+1])[0],
                    "zero_fill": _probability_log_loss(labels[i:i+1], zero_p[i:i+1])[0],
                    "tucker_completion": _probability_log_loss(labels[i:i+1], completed_p[i:i+1])[0]}})
        class_reconstruction = {}
        for cls in _CLASSES:
            members = [row for row in per_trial if row["label"] == cls]
            # Explicit null/zero count for absent groups; no fabricated mean.
            class_sse = {key: (float(sum(row["hidden_squared_error_sum"][key] for row in members)) if members else None)
                         for key in ("zero_fill", "tucker_completion")}
            class_energy = float(sum(row["hidden_target_energy_sum"] for row in members)) if members else None
            class_reconstruction[str(cls)] = {"count": len(members),
                "hidden_squared_error_sum": class_sse,
                "hidden_target_energy_sum": class_energy,
                "hidden_nrmse": {key: (float(np.sqrt(value / class_energy))
                    if value is not None and class_energy is not None and class_energy > 0 else None)
                    for key, value in class_sse.items()},
                "mean_log_loss": {key: (float(np.mean([row["log_loss"][key] for row in members])) if members else None)
                    for key in ("full_features", "zero_fill", "tucker_completion")}}
        output.append({"condition": bank["condition"], "pattern": bank["pattern"], "retained": bank["retained"],
            "repeat": bank["repeat"], "mask_sha256": bank["mask_sha256"], "mask_shape": list(mask.shape),
            "mask_identities": {key: bank["mask_sha256"] for key in ("full_features", "zero_fill", "tucker_completion")},
            "hidden_entries": hidden_count, "hidden_target_energy_sum": energy,
            "hidden_squared_error_sum": errors, "hidden_nrmse": nrmse,
            "metrics": metric_map, "paired_tucker_minus_zero_fill": paired,
            "per_trial": per_trial, "class_stratified": class_reconstruction})
    return {"status": "complete", "synthetic": False, "partitions": ["validation"],
        "sample_ids": list(ids), "conditions": output,
        "aggregation": "Keep mask repeats separate; average repeats within seed, then seeds within participant, then participants equally.",
        "nrmse_definition": "sqrt(sum(hidden squared error) / sum(hidden target energy)); full input is null."}
