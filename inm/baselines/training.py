"""Validation-selected, reloadable neural classifiers for baseline studies."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
from torch import nn

from agfl.optimization import ClassificationLoss, gradients_are_finite
from agfl.reproducibility import seed_everything
from inm.availability import make_mask_bank, training_mask_bank, mask_bank_digest
from inm.training import metrics as _legacy_metrics


_ROBUST_BANKS = ((22, "full"), (16, "random_static"),
                 (16, "dynamic_random"), (6, "random_static"),
                 (6, "dynamic_random"))


def _validate_partition(name: str, labels: np.ndarray) -> None:
    counts = np.bincount(labels.astype(np.int64, copy=False), minlength=4)
    if len(counts) != 4 or np.any(counts == 0):
        raise ValueError(f"{name} partition lacks one or more of four classes: {counts.tolist()}")


def _eval(model, x, labels, mask, batch_size, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = x[start:start + batch_size].to(device)
            mb = mask[start:start + batch_size].to(device)
            logits = model(xb, mb)
            if logits.shape != (len(xb), 4) or not bool(torch.isfinite(logits).all()):
                raise FloatingPointError("Invalid or nonfinite four-class logits")
            rows.append(logits.softmax(-1).cpu().numpy())
    probabilities = np.concatenate(rows)
    result = _legacy_metrics(labels, probabilities)
    result["log_loss"] = float(-np.log(np.clip(
        probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)).mean())
    return result


def _selection_evidence(model, x, labels, cfg, sample_ids, seed, subject, batch_size, device):
    policy = cfg["selection"]["policy"]
    specs = ((22, "full"),) if policy == "full" else _ROBUST_BANKS
    evidence = {}
    for retained, pattern in specs:
        name = "full" if pattern == "full" else f"{pattern}_{retained}"
        mask = make_mask_bank(len(x), cfg["preprocessing"]["windows"], retained, pattern,
                              seed=seed, partition="validation", subject=f"A{subject:02d}",
                              sample_ids=sample_ids)
        evidence[name] = _eval(model, x, labels, torch.as_tensor(mask), batch_size, device)
        evidence[name]["mask_sha256"] = mask_bank_digest(mask)
    if policy == "full":
        selected = evidence["full"]
        score, loss = selected["balanced_accuracy"], selected["log_loss"]
    elif policy == "robust":
        names = ["full" if pattern == "full" else f"{pattern}_{retained}"
                 for retained, pattern in _ROBUST_BANKS]
        score = float(np.mean([evidence[name]["balanced_accuracy"] for name in names]))
        loss = float(np.mean([evidence[name]["log_loss"] for name in names]))
    else:
        raise ValueError("selection.policy must be 'full' or 'robust'")
    return {"policy": policy, "score": score, "tie_break_log_loss": loss,
            "validation_banks": evidence}


def _atomic_torch_save(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, temp_name)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _atomic_json(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def checkpoint_payload(model: nn.Module, metadata: dict) -> dict:
    """Build a weights-only safe checkpoint; constructor settings are primitives."""
    if not hasattr(model, "constructor_settings"):
        raise TypeError("Reloadable classifier must implement constructor_settings()")
    settings = model.constructor_settings()
    return {"format": "agfl-baseline-classifier-v1",
            "model_type": f"{type(model).__module__}.{type(model).__qualname__}",
            "constructor": settings,
            "state_dict": {key: value.detach().cpu().clone()
                           for key, value in model.state_dict().items()},
            "metadata": copy.deepcopy(metadata)}


def load_classifier(checkpoint_path, *, model_factory=None, device="cpu"):
    """Restore an EEGNet or caller-supplied compatible model factory checkpoint."""
    from .eegnet import EEGNetClassifier
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload.get("format") != "agfl-baseline-classifier-v1":
        raise ValueError("Unsupported classifier checkpoint format")
    factory = model_factory or EEGNetClassifier
    model = factory(**payload["constructor"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def fit_classifier(model, prepared, cfg, arm, directory, device="cpu") -> dict:
    """Fit and reload a neural classifier using training and validation only.

    Returns selected model, selected epoch/evidence, paths, timing, and history.
    ``prepared`` follows the PreparedData API. ``arm`` may be an arm dict or its
    ``model__regime`` name. Test arrays/labels are deliberately never indexed.
    """
    device = torch.device(device)
    seed = int(prepared.metadata.get("seed", 0))
    subject = int(prepared.metadata.get("subject", 1))
    seed_everything(seed)
    settings = cfg["training"]
    epochs, batch_size = settings["epochs"], settings["batch_size"]
    if epochs < 1 or batch_size < 1 or settings["minimum_epochs"] < 1:
        raise ValueError("training epochs, batch_size, and minimum_epochs must be positive")
    if settings["minimum_epochs"] > epochs:
        raise ValueError("minimum_epochs cannot exceed epochs")
    regime = arm.get("regime") if isinstance(arm, dict) else str(arm).rsplit("__", 1)[-1]
    if regime not in ("full", "mixed"):
        raise ValueError("arm regime must be 'full' or 'mixed'")
    train_idx = np.asarray(prepared.split_indices["train"], dtype=np.int64)
    val_idx = np.asarray(prepared.split_indices["validation"], dtype=np.int64)
    train_labels = np.asarray(prepared.labels[train_idx], dtype=np.int64)
    val_labels = np.asarray(prepared.labels[val_idx], dtype=np.int64)
    _validate_partition("training", train_labels)
    _validate_partition("validation", val_labels)
    prep = cfg["preprocessing"]
    shape = (prep["channels"], prep["windows"], prep["window_samples"])
    signals = np.asarray(prepared.signals, dtype=np.float32)
    if signals.ndim != 3 or signals.shape[1:] != (shape[0], shape[1] * shape[2]):
        raise ValueError("Prepared signals do not match configured channels/windows/samples")
    x = torch.from_numpy(signals.reshape(len(signals), *shape))
    train_x, val_x = x[train_idx], x[val_idx]
    train_ids = [prepared.sample_ids[i] for i in train_idx]
    val_ids = [prepared.sample_ids[i] for i in val_idx]
    train_labels_t = torch.as_tensor(train_labels, dtype=torch.long)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"])
    warmup = settings["warmup_epochs"]

    def schedule(epoch):
        if warmup and epoch < warmup:
            return 0.1 + 0.9 * epoch / warmup
        return (1 + np.cos(np.pi * min(1., (epoch - warmup) / max(1, epochs - warmup)))) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    criterion = ClassificationLoss()
    output = Path(directory)
    history_path, checkpoint_path = output / "history.json", output / "checkpoint.pt"
    history, best_key, best_state, best_evidence, best_epoch = [], None, None, None, 0
    started = time.monotonic()
    for epoch in range(epochs):
        model.train()
        masks = training_mask_bank(len(train_x), prep["windows"], regime=regime,
                                   seed=seed, epoch=epoch, subject=f"A{subject:02d}",
                                   sample_ids=train_ids)
        order = np.random.default_rng(seed + 100003 * epoch).permutation(len(train_x))
        loss_total, correct = 0.0, 0
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            xb, yb = train_x[indices].to(device), train_labels_t[indices].to(device)
            mask = torch.as_tensor(masks[indices], dtype=torch.bool, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, mask)
            loss = criterion(logits, yb)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite training loss at epoch {epoch + 1}")
            loss.backward()
            if not gradients_are_finite(model.parameters()):
                raise FloatingPointError(f"Nonfinite gradients at epoch {epoch + 1}")
            if settings["gradient_clip"] is not None:
                nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            if hasattr(model, "clip_weights"):
                model.clip_weights()
            loss_total += float(loss.detach()) * len(indices)
            correct += int((logits.argmax(-1) == yb).sum())
        evidence = _selection_evidence(model, val_x, val_labels, cfg, val_ids, seed,
                                       subject, batch_size, device)
        key = (evidence["score"], -evidence["tie_break_log_loss"])
        # Strict lexicographic improvement preserves the earliest exact tie.
        if best_key is None or key > best_key:
            best_key, best_epoch = key, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_evidence = copy.deepcopy(evidence)
        # Clean full-input evaluation metrics are separate from masked train loss.
        full_mask = torch.ones((len(val_x), shape[0], shape[1]), dtype=torch.bool)
        clean = evidence["validation_banks"]["full"] if "full" in evidence["validation_banks"] else \
            _eval(model, val_x, val_labels, full_mask, batch_size, device)
        history.append({"epoch": epoch + 1, "masked_train_loss": loss_total / len(train_x),
                        "masked_train_accuracy": correct / len(train_x),
                        "clean_train_metrics": _eval(model, train_x, train_labels, torch.ones(
                            (len(train_x), shape[0], shape[1]), dtype=torch.bool), batch_size, device),
                        "clean_validation_metrics": clean, "selection": evidence,
                        "learning_rate": optimizer.param_groups[0]["lr"]})
        _atomic_json(history, history_path)
        scheduler.step()
        if (settings["patience"] and epoch + 1 >= settings["minimum_epochs"]
                and epoch + 1 - best_epoch >= settings["patience"]):
            break
    if best_state is None:
        raise RuntimeError("Training produced no validation-selected checkpoint")
    metadata = {"selected_epoch": best_epoch, "selection": best_evidence,
                "normalization_stats": copy.deepcopy(prepared.normalization_stats),
                "class_order": list(prepared.metadata.get("label_names",
                                      ["left_hand", "right_hand", "feet", "tongue"])),
                "channel_order": list(prepared.channel_names),
                "protocol": cfg["protocol"], "protocol_description": cfg["protocol_description"],
                "subject": subject, "seed": seed, "arm": arm if isinstance(arm, str) else arm.get("name"),
                "data_fingerprint": prepared.data_fingerprint,
                "split_id": prepared.metadata.get("split_id"),
                "source_sha256": {str(source.name): hashlib.sha256(source.read_bytes()).hexdigest()
                                  for source in (Path(__file__), Path(__file__).with_name("eegnet.py"))},
                "config_sha256": hashlib.sha256(Path(cfg["config_path"]).read_bytes()).hexdigest()
                    if cfg.get("config_path") and Path(cfg["config_path"]).is_file() else None,
                "synthetic": bool(prepared.metadata.get("synthetic", False))}
    model.load_state_dict(best_state)
    model.eval()
    _atomic_torch_save(checkpoint_payload(model, metadata), checkpoint_path)
    restored, payload = load_classifier(checkpoint_path, model_factory=type(model), device=device)
    elapsed = time.monotonic() - started
    return {"model": restored, "selected_epoch": best_epoch, "epochs_trained": len(history),
            "selection": best_evidence, "history": history, "history_path": str(history_path),
            "checkpoint_path": str(checkpoint_path), "checkpoint": payload,
            "elapsed_seconds": elapsed}
