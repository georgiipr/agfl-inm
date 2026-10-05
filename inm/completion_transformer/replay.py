"""Strict, read-only replay of historical encoder-candidate fits.

Only the two fixed spatial backbones are restored.  Historical study files are
inputs; ``prepare_task`` writes the paired prepared-data provenance exclusively
under the caller's new output staging directory.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

from .protocol import ARM_IDS, ROOT, digest, file_sha256

EVIDENCE_RELATIVE = Path("report/evidence.json")
ORIGINAL_CONFIG_RELATIVE = Path("configs/encoder-candidates-reproducible.json")
CLASS_ORDER = ["left_hand", "right_hand", "feet", "tongue"]
REUSED_SOURCE_KEYS = {
    "agfl/datasets/base.py", "agfl/datasets/eeg.py", "agfl/datasets/splits.py",
    "inm/data.py", "inm/baselines/eegnet.py", "inm/encoder_candidates/data.py",
    "inm/encoder_candidates/models.py", "inm/encoder_candidates/spatial.py",
    "inm/encoder_candidates/transformer.py", "inm/encoder_candidates/training.py",
}


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read historical JSON {path}: {exc}") from exc


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _same(actual: Any, expected: Any, name: str) -> None:
    _require(actual == expected, f"historical {name} mismatch")


def _sha(path: Path) -> str:
    _require(path.is_file(), f"historical artifact is missing: {path}")
    return file_sha256(path)


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def _task_name(subject: int, seed: int) -> str:
    return f"A{subject:02d}_seed_{seed}"


def _source_map(root: Path, saved: dict[str, str]) -> None:
    _require(isinstance(saved, dict) and REUSED_SOURCE_KEYS.issubset(saved),
             "historical source map omits a reused data/model/training dependency")
    for relative, expected in saved.items():
        rel = PurePosixPath(relative)
        _require(not rel.is_absolute() and ".." not in rel.parts and bool(rel.parts),
                 f"unsafe historical source path: {relative}")
        path = (root / Path(*rel.parts)).resolve()
        _require(path.is_relative_to(root.resolve()), f"historical source path escapes repository: {relative}")
        _same(_sha(path), expected, f"source checksum for {relative}")


def _expected_source_packages(evidence: dict, task: dict) -> None:
    source_map = evidence.get("source_files_sha256")
    _require(isinstance(source_map, dict), "historical report source map is absent")
    _same(task.get("source_files_sha256"), source_map, "task source map")
    _source_map(ROOT, source_map)
    packages = evidence.get("packages")
    _require(isinstance(packages, dict), "historical package identity is absent")
    for package, expected in packages.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        _same(actual, expected, f"installed package {package}")


def _manifest_task(evidence: dict, subject: int, seed: int) -> dict:
    expected = {f"tasks/A{sub:02d}_seed_{task_seed}/task.json"
                for sub in range(1, 10) for task_seed in range(3)}
    rows = evidence.get("tasks")
    _require(isinstance(rows, list) and len(rows) == 27,
             "historical manifest must inventory all 27 task metadata files")
    paths = [row.get("path") for row in rows if isinstance(row, dict)]
    _require(len(paths) == 27 and len(set(paths)) == 27 and set(paths) == expected,
             "historical manifest task inventory is incomplete or duplicated")
    relative = f"tasks/{_task_name(subject, seed)}/task.json"
    records = [item for item in evidence.get("tasks", [])
               if item.get("path") == relative]
    _require(len(records) == 1, f"historical manifest has no unique task record for {relative}")
    record = records[0]
    rel = PurePosixPath(relative)
    _require(not rel.is_absolute() and ".." not in rel.parts,
             "unsafe task path in historical manifest")
    _same(record.get("subject"), subject, "manifest task subject")
    _same(record.get("seed"), seed, "manifest task seed")
    return record


def verify_historical_task(cfg: dict, subject: int, seed: int) -> dict:
    """Verify all identity metadata and byte hashes before loading tensors."""
    _require(subject in cfg.get("subjects", []), f"subject {subject} is not declared in config")
    _require(seed in cfg.get("seeds", []), f"seed {seed} is not declared in config")
    candidate = Path(cfg["candidate_source_dir"]).resolve()
    split_root = Path(cfg["split_source_dir"]).resolve()
    evidence_path = candidate / EVIDENCE_RELATIVE
    evidence = _json(evidence_path)
    _same(evidence.get("schema_name"), "agfl-encoder-candidates-report-v1", "manifest schema")
    _same(evidence.get("status"), "complete", "manifest completion status")
    _same(evidence.get("synthetic"), False, "manifest synthetic flag")
    _same(evidence.get("subjects"), list(range(1, 10)), "manifest cohort")
    _same(evidence.get("seeds"), [0, 1, 2], "manifest seeds")
    _same(evidence.get("partitions"), ["train", "validation"], "manifest partitions")
    _require(set(ARM_IDS).issubset(evidence.get("arms", [])), "manifest lacks a required backbone")

    original_config = ROOT / ORIGINAL_CONFIG_RELATIVE
    config_sha = _sha(original_config)
    _same(evidence.get("config_sha256"), config_sha, "original candidate config checksum")
    task_record = _manifest_task(evidence, subject, seed)
    task_path = candidate / task_record["path"]
    _require(task_path.resolve().is_relative_to(candidate), "historical task path escapes candidate directory")
    _same(_sha(task_path), task_record.get("sha256"), "task manifest checksum")
    task = _json(task_path)
    task_dir = task_path.parent
    _same(task.get("schema_name"), "agfl-encoder-candidates-task-v1", "task schema")
    _same(task.get("status"), "complete", "task completion status")
    _same(task.get("synthetic"), False, "task synthetic flag")
    _same(task.get("subject"), subject, "task subject")
    _same(task.get("seed"), seed, "task seed")
    _same(task.get("study_id"), evidence.get("study_id"), "task study identity")
    _same(task.get("declared_study_id"), evidence.get("study_id"), "declared study identity")
    _same(task.get("config_sha256"), config_sha, "task original config checksum")
    _same(task.get("packages"), evidence.get("packages"), "task package identities")
    _same(task.get("partitions"), ["train", "validation"], "task partitions")
    _same(task.get("training_regime"), "full", "task training regime")
    _same(task.get("channel_order"), ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz"], "task channel order")
    checks = task.get("checks")
    _require(isinstance(checks, dict) and checks and all(value is True for value in checks.values()),
             "historical task acceptance checks are not all true")
    _expected_source_packages(evidence, task)

    provenance_path = task_dir / "data_provenance.json"
    provenance = _json(provenance_path)
    _same(provenance.get("subject"), subject, "data provenance subject")
    _same(provenance.get("seed"), seed, "data provenance seed")
    _same(provenance.get("data_id"), task.get("data_id"), "data identity")
    _same(provenance.get("split_id"), task.get("split_id"), "split identity")
    _same(provenance.get("normalization"), task.get("normalization"), "training normalization")
    _same(provenance.get("partition_sample_ids"), task.get("partition_sample_ids"), "ordered partition IDs")
    _same(provenance.get("channel_names"), task.get("channel_order"), "data channel order")
    _same(provenance.get("recording_sha256"), task.get("recording_sha256"), "recording identity")
    _same(provenance.get("original_metadata_sha256"), task.get("original_metadata_sha256"), "original metadata identity")
    preprocessing = provenance.get("preprocessing")
    original = _json(original_config)
    _same(preprocessing, original.get("preprocessing"), "original preprocessing config")
    _require(preprocessing.get("sessions") == ["T"] and
             preprocessing.get("filter_scope") == "window" and
             preprocessing.get("normalization") == "training_channel_mean_population_std_floor_1e-8",
             "historical preprocessing differs from declared replay protocol")

    # The baseline source supplies split/dataset metadata only.  Its manifest is
    # consulted solely for the saved checksum; no baseline model artifacts load.
    original_hashes = task.get("original_metadata_sha256")
    _require(isinstance(original_hashes, dict), "task original metadata checksums are absent")
    split_dir = split_root / "artifacts" / _task_name(subject, seed)
    for name in ("split.json", "dataset.json"):
        _same(_sha(split_dir / name), original_hashes.get(name), f"baseline {name} checksum")
    _same(_sha(split_root / "study.json"), original_hashes.get("study.json"), "baseline study metadata checksum")
    split = _json(split_dir / "split.json")
    dataset = _json(split_dir / "dataset.json")
    historical_loader = dataset.get("provenance", {}).get("loader_config", {})
    _same(Path(historical_loader.get("data_dir", "")).resolve(),
          Path(cfg["data_dir"]).expanduser().resolve(), "configured historical data directory")
    baseline_manifest = _json(split_root / "study.json")
    _same(baseline_manifest.get("format"), "agfl-baseline-study-v1", "baseline manifest schema")
    _same(baseline_manifest.get("synthetic"), False, "baseline manifest synthetic flag")
    _same(baseline_manifest.get("identity", {}).get("config", {}).get("protocol"),
          "within_session", "baseline split protocol")
    _same(split.get("split_id"), task.get("split_id"), "baseline split ID")
    _same(split.get("seed"), seed, "baseline split seed")
    _same(split.get("protocol"), "stratified", "baseline split kind")
    _same(split.get("subject_independent"), False, "baseline split subject isolation")
    fingerprint = split.get("fingerprint")
    expected_split_identity = {"version": "agfl-splits-v2", "fingerprint": fingerprint,
                               "seed": seed, "config": {"protocol": "stratified", "train": 0.6,
                               "validation": 0.2, "test": 0.2}}
    _same(split.get("identity"), expected_split_identity, "baseline split identity fields")
    _same(_canonical_hash(expected_split_identity), split.get("split_id"), "baseline split identity checksum")
    _same(dataset.get("dataset_fingerprint"), fingerprint, "baseline dataset fingerprint")
    _same(dataset.get("provenance", {}).get("dataset_fingerprint"), fingerprint,
          "baseline dataset provenance fingerprint")
    _same(dataset.get("provenance", {}).get("split_id"), split.get("split_id"),
          "baseline dataset provenance split ID")
    _same(dataset.get("provenance", {}).get("subject"), subject, "baseline dataset subject")
    _same(dataset.get("provenance", {}).get("seed"), seed, "baseline dataset seed")
    normalization_stats = dataset.get("normalization_stats")
    _require(isinstance(normalization_stats, dict), "baseline training normalization stats are absent")
    fingerprint_contents = {"provenance": dataset.get("provenance"), "normalization": normalization_stats,
                            "sample_ids": dataset.get("sample_ids")}
    _same(_canonical_hash(fingerprint_contents), dataset.get("data_fingerprint"),
          "baseline dataset metadata identity")
    _same(dataset.get("normalization_stats", {}).get("mean"), task.get("normalization", {}).get("mean"), "baseline normalization mean")
    _same(dataset.get("normalization_stats", {}).get("std"), task.get("normalization", {}).get("std"), "baseline normalization std")
    _same(dataset.get("channel_names"), task.get("channel_order"), "baseline channel order")
    _require(isinstance(dataset.get("sample_ids"), list) and len(set(dataset["sample_ids"])) == len(dataset["sample_ids"]),
             "baseline dataset sample IDs are malformed")
    historical_partition_ids = split.get("sample_ids")
    _require(isinstance(historical_partition_ids, dict), "baseline split partition IDs are absent")
    _same(task.get("partition_sample_ids"),
          {name: historical_partition_ids.get(name) for name in ("train", "validation")},
          "baseline ordered train/validation IDs")
    indices = provenance.get("partition_indices")
    _require(isinstance(indices, dict), "candidate data partition indices are absent")
    for name in ("train", "validation"):
        members = split.get(name)
        _require(isinstance(members, list) and all(type(index) is int for index in members),
                 f"baseline {name} indices are malformed")
        _same(indices.get(name), members, f"candidate {name} membership")
        _same(historical_partition_ids.get(name), [dataset["sample_ids"][index] for index in members],
              f"baseline {name} ordered IDs")

    fit_rows = {row.get("arm"): row for row in task.get("fits", []) if isinstance(row, dict)}
    fits = {}
    for arm in ARM_IDS:
        _require(arm in fit_rows, f"task manifest has no fit record for {arm}")
        result_path = task_dir / "ARMS" / arm / "result.json"
        row = fit_rows[arm]
        _same(row.get("path"), f"ARMS/{arm}/result.json", f"{arm} fit path")
        _same(_sha(result_path), row.get("sha256"), f"{arm} fit manifest checksum")
        result = _json(result_path)
        _same(result.get("arm"), arm, f"{arm} arm")
        for name, expected in (("status", "complete"), ("synthetic", False), ("subject", subject), ("seed", seed),
                               ("study_id", evidence.get("study_id")), ("config_sha256", config_sha),
                               ("data_id", task.get("data_id")), ("split_id", task.get("split_id")),
                               ("training_regime", "full"), ("partitions", ["train", "validation"])):
            _same(result.get(name), expected, f"{arm} result {name}")
        compatibility = result.get("compatibility", {})
        for name in ("arm", "config_sha256", "data_id", "original_metadata_sha256", "recording_sha256",
                     "seed", "split_id", "study_id", "subject", "synthetic"):
            expected = {"arm": arm, "config_sha256": config_sha, "data_id": task.get("data_id"),
                        "original_metadata_sha256": original_hashes, "recording_sha256": task.get("recording_sha256"),
                        "seed": seed, "split_id": task.get("split_id"), "study_id": evidence.get("study_id"),
                        "subject": subject, "synthetic": False}[name]
            _same(compatibility.get(name), expected, f"{arm} compatibility {name}")
        _same(compatibility.get("constructor"), result.get("constructor"), f"{arm} constructor")
        selected_epoch = result.get("selected_epoch")
        _require(type(selected_epoch) is int and selected_epoch > 0,
                 f"{arm} selected epoch is invalid")
        history_path = result_path.parent / "history.json"
        history = _json(history_path)
        _require(isinstance(history, list) and len(history) >= selected_epoch,
                 f"{arm} history is incomplete before selected epoch")
        _same(result.get("epochs_trained"), len(history), f"{arm} recorded history length")
        _require(all(isinstance(item, dict) and item.get("epoch") == index
                     for index, item in enumerate(history, 1)),
                 f"{arm} history epoch sequence is malformed")
        selected = [row for row in history if row.get("selected") is True]
        _require(len(selected) == 1 and selected[0].get("epoch") == selected_epoch,
                 f"{arm} selected epoch does not match history")
        _same(result.get("clean_validation", {}).get("balanced_accuracy"),
              selected[0].get("clean_validation_metrics", {}).get("balanced_accuracy"),
              f"{arm} selected validation metric")
        artifact_hashes = result.get("artifact_sha256")
        _require(isinstance(artifact_hashes, dict), f"{arm} artifact checksums are absent")
        for artifact in ("checkpoint.pt", "history.json", "predictions.npz"):
            _same(_sha(result_path.parent / artifact), artifact_hashes.get(artifact), f"{arm} {artifact} checksum")
        import numpy as np
        with np.load(result_path.parent / "predictions.npz", allow_pickle=False) as predictions:
            _require(set(predictions.files) == {"validation_sample_ids", "validation_labels",
                                                "validation_probabilities"},
                     f"{arm} prediction file has unknown or missing arrays")
            prediction_ids = predictions["validation_sample_ids"].astype(str).tolist()
            prediction_labels = predictions["validation_labels"]
            prediction_probs = predictions["validation_probabilities"]
        _same(prediction_ids, task.get("partition_sample_ids", {}).get("validation"),
              f"{arm} ordered prediction IDs")
        _require(prediction_labels.shape == (len(prediction_ids),) and
                 prediction_probs.shape == (len(prediction_ids), 4) and
                 np.isfinite(prediction_probs).all() and
                 np.all(prediction_probs >= 0) and np.all(prediction_probs <= 1) and
                 np.allclose(prediction_probs.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6),
                 f"{arm} saved validation prediction arrays are malformed")
        _same(row.get("path"), f"ARMS/{arm}/result.json", f"{arm} fit path")
        fits[arm] = {"result": result, "directory": result_path.parent}

    return {"subject": subject, "seed": seed, "candidate_dir": str(candidate),
            "task_dir": str(task_dir), "evidence": evidence, "task": task,
            "data_provenance": provenance, "fits": fits,
            "source_files_sha256": evidence["source_files_sha256"],
            "packages": evidence["packages"], "original_config_sha256": config_sha}


def prepare_task(cfg: dict, subject: int, seed: int, staging_dir: str | Path,
                 *, loader=None):
    """Verify historical IDs then prepare train/validation data in new staging."""
    record = verify_historical_task(cfg, subject, seed)
    output_root = Path(cfg["output_dir"]).expanduser().resolve()
    stage = Path(staging_dir).expanduser().resolve()
    _require(stage != output_root and stage.is_relative_to(output_root),
             "task staging directory must be strictly inside the new output directory")
    for key in ("candidate_source_dir", "split_source_dir", "data_dir"):
        protected = Path(cfg[key]).expanduser().resolve()
        _require(stage != protected and not stage.is_relative_to(protected) and
                 not protected.is_relative_to(stage),
                 f"task staging directory overlaps protected historical input {key}")
    if stage.exists():
        _require(stage.is_dir() and not any(stage.iterdir()),
                 "task staging directory already contains files; refusing to overwrite")
    # Imported only when a task is prepared, preserving dependency-light planning.
    from inm.encoder_candidates.data import prepare_subject

    old_preprocessing = record["data_provenance"]["preprocessing"]
    data_cfg = {"subjects": [subject], "seeds": [seed], "data_dir": cfg["data_dir"],
                "split_source_dir": cfg["split_source_dir"], "preprocessing": old_preprocessing}
    prepared = prepare_subject(data_cfg, subject, seed, staging_dir, loader=loader)
    task = record["task"]
    _same(prepared.data_id, task["data_id"], "prepared data ID")
    _same(prepared.split_id, task["split_id"], "prepared split ID")
    _same(prepared.normalization, task["normalization"], "prepared training normalization")
    _same(list(prepared.train.sample_ids), task["partition_sample_ids"]["train"], "prepared training IDs")
    _same(list(prepared.validation.sample_ids), task["partition_sample_ids"]["validation"], "prepared validation IDs")
    return prepared, record


def restore_backbones(record: dict) -> dict[str, Any]:
    """Restore both candidate models from weights-only checkpoints."""
    import torch
    from inm.encoder_candidates.models import restore_model

    restored = {}
    for arm in ARM_IDS:
        fit = record["fits"][arm]
        checkpoint_path = fit["directory"] / "checkpoint.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        _require(isinstance(checkpoint, dict), f"{arm} checkpoint must be a primitive dictionary")
        _same(checkpoint.get("format"), "agfl-encoder-candidate-classifier-v1", f"{arm} checkpoint format")
        result = fit["result"]
        task = record["task"]
        _same(checkpoint.get("constructor"), result.get("constructor"), f"{arm} checkpoint constructor")
        _same(checkpoint.get("selected_epoch"), result.get("selected_epoch"), f"{arm} selected epoch")
        _same(checkpoint.get("normalization"), task.get("normalization"), f"{arm} checkpoint normalization")
        _same(checkpoint.get("channel_order"), task.get("channel_order"), f"{arm} checkpoint channel order")
        _same(checkpoint.get("class_order"), CLASS_ORDER, f"{arm} checkpoint class order")
        metadata = checkpoint.get("metadata", {})
        _same(metadata, fit["result"].get("compatibility"), f"{arm} checkpoint fit metadata")
        model = restore_model(arm, checkpoint["constructor"], checkpoint["state_dict"])
        model.eval()
        model.requires_grad_(False)
        restored[arm] = model
    return restored


def replay_full_input(record: dict, prepared, *, batch_size: int = 64) -> dict[str, dict]:
    """Compare restored models' validation probabilities to persisted outputs."""
    import numpy as np
    import torch

    _require(type(batch_size) is int and batch_size > 0, "batch_size must be positive")
    models = restore_backbones(record)
    x = torch.from_numpy(np.array(prepared.validation.raw, dtype=np.float32, copy=True))
    mask = torch.ones((len(x), 22, 4), dtype=torch.bool)
    reports = {}
    for arm, model in models.items():
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(x), batch_size):
                logits = model(x[start:start + batch_size], mask[start:start + batch_size])
                chunks.append(torch.softmax(logits, dim=-1).cpu().numpy())
        actual = np.concatenate(chunks).astype(np.float32, copy=False)
        prediction_path = record["fits"][arm]["directory"] / "predictions.npz"
        with np.load(prediction_path, allow_pickle=False) as saved:
            _require(set(saved.files) == {"validation_sample_ids", "validation_labels", "validation_probabilities"},
                     f"{arm} saved predictions contain unknown or missing arrays")
            ids = saved["validation_sample_ids"].astype(str).tolist()
            labels = saved["validation_labels"]
            expected = saved["validation_probabilities"]
        _same(ids, list(prepared.validation.sample_ids), f"{arm} prediction IDs")
        _same(labels.tolist(), prepared.validation.labels.tolist(), f"{arm} prediction labels")
        _require(expected.shape == actual.shape and np.isfinite(expected).all() and np.isfinite(actual).all(),
                 f"{arm} replay prediction shape/finiteness mismatch")
        # Saved probabilities are float32; tolerate only ordinary deterministic
        # CPU kernel rounding, never a semantic or checkpoint mismatch.
        _require(np.allclose(actual, expected, rtol=1e-5, atol=1e-7),
                 f"{arm} full-input validation probabilities do not replay")
        reports[arm] = {"n_validation": len(ids), "max_abs_probability_error": float(np.max(np.abs(actual - expected))),
                        "prediction_sha256": file_sha256(prediction_path)}
    return reports


__all__ = ["verify_historical_task", "prepare_task", "restore_backbones", "replay_full_input"]
