"""Validation-selected training and reloadable checkpoints for candidate arms."""
from __future__ import annotations

import copy
from contextlib import contextmanager
import hashlib
import json
import math
import os
import platform
import random
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
from torch import nn

from .models import restore_model


_CLASSES = ("left_hand", "right_hand", "feet", "tongue")
_CHANNELS = 22
_WINDOWS = 4
_SAMPLES = 250
_RNG_POLICY_NAME = "agfl-candidate-fit-rng-v1"
_FIT_STREAM_OFFSET = 1_900_001
_DATA_ORDER_STREAM_OFFSET = 700_001
_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def _derive_seed(task_seed: int, stream_offset: int) -> int:
    """Stable arithmetic seed derivation; independent of Python hashing/state."""
    if type(task_seed) is not int or type(stream_offset) is not int:
        raise TypeError("Task seeds and stream offsets must be integers")
    return (task_seed + stream_offset) % (2**63 - 1)


def _rng_metadata(task_seed: int, device: torch.device) -> dict:
    fit_seed = _derive_seed(task_seed, _FIT_STREAM_OFFSET)
    return {
        "name": _RNG_POLICY_NAME,
        "version": 1,
        "task_seed": task_seed,
        "fit_stream_offset": _FIT_STREAM_OFFSET,
        "fit_seed": fit_seed,
        "data_order_stream_offset": _DATA_ORDER_STREAM_OFFSET,
        "data_order_seed_formula": "(task_seed + 100003 * one_based_epoch + 700001) mod (2^63-1)",
        "device": str(device),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": str(torch.__version__),
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cublas_workspace_config": _CUBLAS_WORKSPACE_CONFIG if device.type == "cuda" else None,
    }


@contextmanager
def _scoped_fit_rng(task_seed: int, device: torch.device):
    """Seed fit randomness and restore caller RNG/backend state on every exit."""
    metadata = _rng_metadata(task_seed, device)
    cuda_devices = []
    previous_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if device.type == "cuda":
        already_initialized = torch.cuda.is_initialized()
        if already_initialized:
            if previous_cublas != _CUBLAS_WORKSPACE_CONFIG:
                raise RuntimeError("CUDA was initialized before deterministic CUBLAS_WORKSPACE_CONFIG=:4096:8")
        else:
            # CUBLAS reads this on first context creation. Establish it before
            # querying availability or selecting a device.
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS_WORKSPACE_CONFIG
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA training requested, but CUDA is unavailable")
            index = torch.cuda.current_device() if device.index is None else device.index
            if index < 0 or index >= torch.cuda.device_count():
                raise RuntimeError(f"CUDA device index {index} is unavailable")
        except Exception:
            if not already_initialized:
                if previous_cublas is None:
                    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
                else:
                    os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous_cublas
            raise
        cuda_devices = [index]
        metadata["device"] = f"cuda:{index}"

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_benchmark = torch.backends.cudnn.benchmark
    cudnn_deterministic = torch.backends.cudnn.deterministic
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            seed = metadata["fit_seed"]
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.random.default_generator.manual_seed(seed)
            if cuda_devices:
                torch.cuda.default_generators[cuda_devices[0]].manual_seed(seed)
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            yield metadata
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic
        if device.type == "cuda":
            if previous_cublas is None:
                os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            else:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous_cublas


def learning_rate_for_epoch(epoch: int, settings: dict) -> float:
    """Return the protocol LR for a one-based epoch, including exact endpoints."""
    maximum = int(settings["maximum_epochs"])
    warmup = int(settings["warmup_epochs"])
    base = float(settings["learning_rate"])
    if type(epoch) is not int or not 1 <= epoch <= maximum:
        raise ValueError("epoch must be an integer in 1..maximum_epochs")
    if warmup and epoch <= warmup:
        return base * epoch / warmup
    span = max(1, maximum - warmup)
    progress = min(1.0, max(0.0, (epoch - warmup) / span))
    return base * 0.5 * (1.0 + math.cos(math.pi * progress))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _atomic_torch_save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, name)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (len(labels), 4) or not np.isfinite(probabilities).all():
        raise FloatingPointError("Evaluation probabilities must be finite [N,4]")
    counts = np.bincount(labels, minlength=4)
    if len(labels) == 0 or np.any(counts == 0):
        raise ValueError("Each evaluated partition must contain all four classes")
    predicted = probabilities.argmax(axis=1)
    confusion = np.zeros((4, 4), dtype=np.int64)
    np.add.at(confusion, (labels, predicted), 1)
    recalls = np.diag(confusion) / confusion.sum(axis=1)
    f1s = []
    for cls in range(4):
        tp = confusion[cls, cls]
        denom = 2 * tp + confusion[:, cls].sum() - tp + confusion[cls, :].sum() - tp
        f1s.append(0.0 if denom == 0 else float(2 * tp / denom))
    clipped = np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)
    return {"balanced_accuracy": float(recalls.mean()),
            "accuracy": float((predicted == labels).mean()),
            "macro_f1": float(np.mean(f1s)),
            "log_loss": float(-np.log(clipped).mean()),
            "per_class_recall": recalls.tolist(), "confusion_matrix": confusion.tolist()}


@torch.no_grad()
def _evaluate(model, x: torch.Tensor, labels: np.ndarray, batch_size: int,
              device: torch.device) -> tuple[dict, np.ndarray]:
    model.eval()
    outputs = []
    mask = torch.ones((min(batch_size, len(x)), _CHANNELS, _WINDOWS), dtype=torch.bool)
    for start in range(0, len(x), batch_size):
        xb = x[start:start + batch_size].to(device)
        mb = mask[:len(xb)].to(device)
        logits = model(xb, mb)
        if tuple(logits.shape) != (len(xb), 4) or not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("Model produced invalid or nonfinite four-class logits")
        outputs.append(logits.softmax(dim=-1).cpu().numpy())
    probabilities = np.concatenate(outputs)
    return _metrics(labels, probabilities), probabilities


def _state_copy(model) -> dict:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _compatibility(prepared, cfg, arm_id, model) -> dict:
    cfg_hash = None
    if cfg.get("config_path") and Path(cfg["config_path"]).is_file():
        cfg_hash = _sha256(Path(cfg["config_path"]))
    provenance = prepared.provenance if isinstance(prepared.provenance, dict) else {}
    base_study_id = provenance.get("study_id")
    if cfg.get("config_path") and Path(cfg["config_path"]).is_file():
        # The report evidence schema has one study_id shared across all 27
        # tasks. Per-task verified data identity is recorded separately below
        # through split_id/data_id and in the task provenance manifest.
        from .protocol import study_identity
        base_study_id = study_identity(cfg)["study_id"]
    return {"subject": int(prepared.subject), "seed": int(prepared.seed), "arm": arm_id,
            "synthetic": bool(cfg.get("synthetic", False)),
            "study_id": base_study_id, "config_sha256": cfg_hash,
            "split_id": str(prepared.split_id), "data_id": str(prepared.data_id),
            "recording_sha256": copy.deepcopy(provenance.get("recording_sha256", {})),
            "original_metadata_sha256": copy.deepcopy(provenance.get("original_metadata_sha256", {})),
            "constructor": model.constructor_settings()}


def _reuse_existing(directory: Path, identity: dict, arm_id: str, device):
    paths = [directory / name for name in ("result.json", "history.json", "checkpoint.pt")]
    present = [path.exists() for path in paths]
    if not any(present):
        return None
    if not all(present) or any(not path.is_file() for path in paths):
        raise FileExistsError(f"Incomplete candidate fit artifacts at {directory}; refusing reuse/overwrite")
    try:
        result = json.loads(paths[0].read_text(encoding="utf-8"))
        hashes = result["artifact_sha256"]
        if result.get("compatibility") != identity or any(
                hashes.get(name) != _sha256(directory / name)
                for name in ("history.json", "checkpoint.pt")):
            raise ValueError("artifact identity or checksum mismatch")
        payload = torch.load(paths[2], map_location="cpu", weights_only=True)
        if (payload.get("format") != "agfl-encoder-candidate-classifier-v1" or
                payload.get("metadata") != identity or
                payload.get("rng_policy") != identity.get("rng_policy") or
                payload.get("execution_device") != identity.get("execution_device")):
            raise ValueError("checkpoint metadata mismatch")
        if (result.get("status") != "complete" or
                result.get("selected_epoch") != payload.get("selected_epoch")):
            raise ValueError("result/checkpoint selected state mismatch")
        model = restore_model(arm_id, payload["constructor"], payload["state_dict"])
        model.to(device).eval()
        history = json.loads(paths[1].read_text(encoding="utf-8"))
        if (not isinstance(history, list) or not history or
                result.get("epochs_trained") != len(history) or
                any(row.get("epoch") != index for index, row in enumerate(history, 1))):
            raise ValueError("history structure or epoch sequence is invalid")
        return {**result, "model": model, "history": history,
                "checkpoint_path": str(paths[2]), "history_path": str(paths[1]),
                "result_path": str(paths[0]), "reused": True}
    except Exception as error:
        raise ValueError(f"Existing candidate fit at {directory} failed verification: {error}") from error


def fit_classifier(model, prepared, cfg, arm_id, directory, device="cpu") -> dict:
    """Fit inside an isolated, repeatable RNG and deterministic-kernel scope."""
    requested_device = torch.device(device)
    if requested_device.type not in ("cpu", "cuda"):
        raise ValueError("Candidate fitting supports CPU or CUDA devices")
    with _scoped_fit_rng(int(prepared.seed), requested_device) as rng_policy:
        return _fit_classifier(model, prepared, cfg, arm_id, directory,
                               requested_device, rng_policy)


def _fit_classifier(model, prepared, cfg, arm_id, directory, device,
                    rng_policy) -> dict:
    """Train on full-input training trials, select on full-input validation, save and reload.

    The public PreparedData contains only train and validation arrays. Data-order
    permutations use a private NumPy generator, separate from model/dropout RNG.
    """
    settings = cfg["training"]
    if arm_id not in {arm["id"] for arm in cfg.get("arms", [])} and cfg.get("arms"):
        raise ValueError(f"Arm {arm_id!r} is not declared in this configuration")
    maximum = int(settings["maximum_epochs"])
    minimum = int(settings["minimum_epochs"])
    patience = int(settings["patience"])
    batch_size = int(settings["batch_size"])
    synthetic = bool(cfg.get("synthetic", False))
    if not synthetic and (maximum, minimum, patience, batch_size, int(settings["warmup_epochs"])) != (250, 75, 50, 32, 10):
        raise ValueError("Real candidate fits must use the fixed full training budget")
    if synthetic and (maximum > 250 or minimum > 75 or patience > 50 or
                      batch_size > 32 or int(settings["warmup_epochs"]) > 10 or
                      int(settings["warmup_epochs"]) >= maximum):
        raise ValueError("Synthetic candidate budgets may only shrink fixed training settings")
    if min(maximum, minimum, batch_size) < 1 or minimum > maximum or patience < 0:
        raise ValueError("Invalid training budget, minimum epoch, patience or batch size")
    if cfg.get("training_regime", "full") != "full":
        raise ValueError("Encoder candidate classifiers train on full-input data only")
    train = prepared.train
    validation = prepared.validation
    train_x_np = np.asarray(train.raw)
    validation_x_np = np.asarray(validation.raw)
    train_y = np.asarray(train.labels, dtype=np.int64)
    validation_y = np.asarray(validation.labels, dtype=np.int64)
    expected = (_CHANNELS, _WINDOWS, _SAMPLES)
    if train_x_np.ndim != 4 or tuple(train_x_np.shape[1:]) != expected:
        raise ValueError("Prepared training data must be [N,22,4,250]")
    if validation_x_np.ndim != 4 or tuple(validation_x_np.shape[1:]) != expected:
        raise ValueError("Prepared validation data must be [N,22,4,250]")
    if len(train_x_np) != len(train_y) or len(validation_x_np) != len(validation_y):
        raise ValueError("Prepared signal and label counts differ")
    if (not np.isfinite(train_x_np).all() or not np.isfinite(validation_x_np).all()):
        raise ValueError("Full-input training and validation signals must be finite")
    if (not np.isin(train_y, np.arange(4)).all() or
            not np.isin(validation_y, np.arange(4)).all()):
        raise ValueError("Labels must be integer class IDs 0..3")
    for name, ids in (("train", train.sample_ids), ("validation", validation.sample_ids)):
        if len(ids) != len(set(ids)):
            raise ValueError(f"Prepared {name} sample IDs must be unique")
    if set(train.sample_ids) & set(validation.sample_ids):
        raise ValueError("Training and validation stable sample IDs overlap")
    for name, labels in (("training", train_y), ("validation", validation_y)):
        if np.any(np.bincount(labels, minlength=4) == 0):
            raise ValueError(f"{name} partition must contain all four classes")

    model.to(device)
    output = Path(directory)
    identity = _compatibility(prepared, cfg, arm_id, model)
    identity["rng_policy"] = copy.deepcopy(rng_policy)
    identity["execution_device"] = "cpu" if device.type == "cpu" else "cuda"
    reused = _reuse_existing(output, identity, arm_id, device)
    if reused is not None:
        return reused
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite unrecognized files in {output}")
    train_x = torch.as_tensor(train_x_np, dtype=torch.float32)
    validation_x = torch.as_tensor(validation_x_np, dtype=torch.float32)
    train_y_t = torch.as_tensor(train_y, dtype=torch.long)
    full_mask = torch.ones((_CHANNELS, _WINDOWS), dtype=torch.bool)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]),
                                  weight_decay=float(settings["weight_decay"]))
    criterion = nn.CrossEntropyLoss()
    best_key = None
    best_state = None
    best_epoch = None
    best_validation = None
    best_train = None
    history = []
    stale = 0
    started = time.monotonic()

    for epoch in range(1, maximum + 1):
        lr = learning_rate_for_epoch(epoch, settings)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        order_seed = (int(prepared.seed) + 100003 * epoch +
                      _DATA_ORDER_STREAM_OFFSET) % (2**63 - 1)
        order_rng = np.random.default_rng(order_seed)
        order = order_rng.permutation(len(train_x))
        epoch_loss = 0.0
        epoch_correct = 0
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            xb = train_x[indices].to(device)
            yb = train_y_t[indices].to(device)
            mask = full_mask.unsqueeze(0).expand(len(indices), -1, -1).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, mask)
            loss = criterion(logits, yb)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite cross-entropy at epoch {epoch}")
            loss.backward()
            parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
            if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
                   for parameter in parameters):
                raise FloatingPointError(f"Nonfinite gradients at epoch {epoch}")
            torch.nn.utils.clip_grad_norm_(parameters, float(settings["gradient_clip_norm"]),
                                           error_if_nonfinite=True)
            optimizer.step()
            if hasattr(model, "clip_weights"):
                model.clip_weights()
            epoch_loss += float(loss.detach()) * len(indices)
            epoch_correct += int((logits.detach().argmax(-1) == yb).sum())

        validation_metrics, _ = _evaluate(model, validation_x, validation_y,
                                            batch_size, device)
        train_metrics, _ = _evaluate(model, train_x, train_y, batch_size, device)
        key = (validation_metrics["balanced_accuracy"], -validation_metrics["log_loss"])
        improved = best_key is None or key > best_key
        if improved:
            best_key = key
            best_epoch = epoch
            best_state = _state_copy(model)
            best_validation = copy.deepcopy(validation_metrics)
            best_train = copy.deepcopy(train_metrics)
            stale = 0
        else:
            stale += 1
        history.append({"epoch": epoch, "learning_rate": lr,
                        "optimization_loss": epoch_loss / len(train_x),
                        "optimization_accuracy": epoch_correct / len(train_x),
                        "clean_train_metrics": train_metrics,
                        "clean_validation_metrics": validation_metrics,
                        "selected": bool(improved)})
        if epoch >= minimum and patience == 0 and not improved:
            break
        if epoch >= minimum and patience > 0 and stale >= patience:
            break

    if best_state is None:
        raise RuntimeError("Training ended without a validation-selected checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    selected_train, train_probabilities = _evaluate(model, train_x, train_y, batch_size, device)
    selected_validation, validation_probabilities = _evaluate(
        model, validation_x, validation_y, batch_size, device)
    if selected_train != best_train or selected_validation != best_validation:
        raise RuntimeError("Selected checkpoint metrics changed after restoring selected weights")
    elapsed = time.monotonic() - started
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.pt"
    history_path = output / "history.json"
    result_path = output / "result.json"
    constructor = model.constructor_settings()
    checkpoint = {"format": "agfl-encoder-candidate-classifier-v1",
                  "constructor": constructor,
                  "state_dict": _state_copy(model),
                  "metadata": identity,
                  "rng_policy": copy.deepcopy(rng_policy),
                  "execution_device": identity["execution_device"],
                  "class_order": list(_CLASSES),
                  "channel_order": list(prepared.provenance.get(
                      "channel_names", prepared.provenance.get("channel_order", []))),
                  "normalization": copy.deepcopy(prepared.normalization),
                  "selected_epoch": int(best_epoch)}
    for row in history:
        row["selected"] = row["epoch"] == best_epoch
    _atomic_torch_save(checkpoint_path, checkpoint)
    _atomic_json(history_path, history)
    reloaded_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    reloaded = restore_model(arm_id, reloaded_payload["constructor"],
                             reloaded_payload["state_dict"]).to(device).eval()
    _, reloaded_train_prob = _evaluate(reloaded, train_x, train_y, batch_size, device)
    _, reloaded_val_prob = _evaluate(reloaded, validation_x, validation_y, batch_size, device)
    if (not np.allclose(train_probabilities, reloaded_train_prob, rtol=1e-4, atol=1e-5) or
            not np.allclose(validation_probabilities, reloaded_val_prob,
                            rtol=1e-4, atol=1e-5)):
        raise RuntimeError("Checkpoint reload changed selected probabilities")
    artifact_hashes = {"checkpoint.pt": _sha256(checkpoint_path),
                       "history.json": _sha256(history_path)}
    result = {"status": "complete", "synthetic": synthetic,
              "partitions": ["train", "validation"], "training_regime": "full",
              **identity, "compatibility": identity, "selected_epoch": int(best_epoch),
              "epochs_trained": len(history), "parameter_count": sum(
                  parameter.numel() for parameter in model.parameters()),
              "clean_train": selected_train, "clean_validation": selected_validation,
              "optimization": [{"epoch": row["epoch"],
                                  "loss": row["optimization_loss"],
                                  "accuracy": row["optimization_accuracy"],
                                  "learning_rate": row["learning_rate"]}
                                 for row in history],
              "elapsed_seconds": elapsed, "artifact_sha256": artifact_hashes}
    _atomic_json(result_path, result)
    return {**result, "model": reloaded, "history": history,
            "checkpoint_path": str(checkpoint_path), "history_path": str(history_path),
            "result_path": str(result_path), "reused": False}
