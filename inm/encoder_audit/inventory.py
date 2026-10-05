"""Read-only JSON/file inventory. Never import numeric code or load tensors.

Existence and byte integrity establish availability, not successful replay.
Metric records and histories may contain test scores: no scores are extracted.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys

from .protocol import ROOT, audit_identity, digest, file_sha256, tasks

BASELINE_ARMS = ("eegnet_reference__full", "masked_eegnet__full",
                 "masked_eegnet__mixed", "covariance__full")
LEGACY_ARMS = tuple(f"mha__{view}__{regime}" for view in
                    ("baseline", "tensor_completion", "tensor_core")
                    for regime in ("full", "mixed"))
# Known replay dependencies, rather than a digest of the expanded checkout.
# Later replay must also check any additional files it actually imports.
SHARED_REPLAY = ("inm/data.py", "inm/availability.py", "inm/training.py",
    "agfl/datasets/__init__.py", "agfl/datasets/base.py", "agfl/datasets/eeg.py",
    "agfl/datasets/splits.py", "agfl/reproducibility.py", "agfl/optimization.py")
REPLAY_PATHS = {
    "baseline": SHARED_REPLAY + ("inm/baselines/__init__.py", "inm/baselines/data.py",
        "inm/baselines/eegnet.py", "inm/baselines/training.py"),
    "legacy": SHARED_REPLAY + ("inm/model.py", "inm/tensor_attention.py",
        "agfl/attention/mha/layer.py", "agfl/attention/performer/layer.py",
        "agfl/attention/projections.py"),
}


def _block(blockers, code, path, message):
    blockers.append({"code": code, "path": str(path), "message": message})


def _read(path, blockers, *, kind=dict):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, kind):
            raise ValueError(f"expected {kind.__name__}")
        return value
    except (OSError, UnicodeError, ValueError) as error:
        _block(blockers, "missing_file" if not path.exists() else "invalid_json", path, str(error))
        return None


def _valid_hash(value):
    return (isinstance(value, str) and len(value) == 64 and
            all(c in "0123456789abcdef" for c in value))


def _file(path, blockers, expected=None, *, require_hash=False):
    record = {"path": str(path), "exists": path.is_file()}
    if expected is not None or require_hash:
        record["recorded_sha256"] = expected
    if not path.is_file():
        _block(blockers, "missing_file", path, "Required artifact is missing")
        return record
    if expected is not None or require_hash:
        try:
            actual = file_sha256(path)
        except OSError as error:
            _block(blockers, "unreadable_file", path, str(error))
            return record
        record.update(sha256=actual, checksum_verified=_valid_hash(expected) and expected == actual)
        if not _valid_hash(expected):
            _block(blockers, "missing_checksum", path, "A valid recorded SHA-256 is required")
        elif actual != expected:
            _block(blockers, "checksum_mismatch", path, "File bytes differ from the recorded SHA-256")
    return record


def _source_map(study, mapping, blockers):
    if not isinstance(mapping, dict) or not mapping:
        _block(blockers, "missing_source_map", study, "Historical source map is missing")
        mapping = {}
    records = []
    for name, expected in sorted(mapping.items()):
        path = (ROOT / name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(ROOT):
            _block(blockers, "invalid_source_path", name, "Historical source must be relative to repository root")
            continue
        actual = file_sha256(path) if path.is_file() else None
        status = "match" if _valid_hash(expected) and actual == expected else "mismatch"
        records.append({"path": name, "recorded_sha256": expected, "sha256": actual,
                        "status": status, "replay_dependency": name in REPLAY_PATHS[study]})
        if not _valid_hash(expected):
            _block(blockers, "invalid_source_checksum", name, "Invalid historical SHA-256")
        elif status != "match" and name in REPLAY_PATHS[study]:
            _block(blockers, "replay_source_mismatch", name, f"{study}: faithful replay requires historical bytes")
    for name in REPLAY_PATHS[study]:
        if name not in mapping:
            _block(blockers, "missing_replay_source", name, f"{study}: replay file absent from historical source map")
    return records


def _input_protocol(study, config):
    """Recognize the supplied historical designs; never initialize their runners."""
    split = config.get("split", {})
    if study == "baseline":
        prep = config.get("preprocessing", {})
        return (isinstance(prep, dict) and config.get("schema_name") == "agfl-baselines-v1" and
            config.get("protocol") == "within_session" and
            split == {"strategy": "stratified", "train": .6, "validation": .2, "test": .2} and
            all(prep.get(k) == v for k, v in {"sampling_rate": 250, "channels": 22,
                "windows": 4, "window_samples": 250, "trial_samples": 1000,
                "filter_low_hz": 2., "filter_high_hz": 30., "filter_scope": "window",
                "normalization": "train_channel"}.items()) and
            "head_ablation" not in config and "execution_arms" not in config and
            config.get("external_labels_dir") is None)
    prep = config.get("data", {})
    enc = config.get("encoder", {})
    tensor = config.get("tensor", {})
    followup = config.get("tensor_followup", {})
    return (all(isinstance(v, dict) for v in (prep, enc, tensor, followup)) and
        config.get("schema_version") == 2 and
        followup.get("schema_name") == "agfl-legacy-tensor-followup-v1" and
        split == {"protocol": "stratified", "train": .6, "validation": .2, "test": .2} and
        all(prep.get(k) == v for k, v in {"sessions": ["T"], "offset_seconds": 0.,
            "window": 1000, "filter_window_samples": 250, "filter_scope": "window",
            "lowcut": 2., "highcut": 30., "normalization": "train_channel",
            "artifact_policy": "exclude", "labels_dir": None}.items()) and
        all(enc.get(k) == v for k, v in {"channels": 22, "windows": 4,
                                       "window_samples": 250, "f2": 32}.items()) and
        all(tensor.get(k) == v for k, v in {"rank_channels": 4, "rank_features": 4,
                                          "ridge": .001}.items()))


def _split_metadata(split, dataset, path, seed, blockers):
    """Validate saved metadata without returning samples, labels or test counts."""
    if split is None:
        return None
    if split.get("protocol") != "stratified" or split.get("seed") != seed:
        _block(blockers, "split_identity_mismatch", path, "Wrong saved split protocol or seed")
    if any(not _valid_hash(split.get(key)) for key in ("split_id", "fingerprint")):
        _block(blockers, "missing_split_identity", path, "Saved split and dataset fingerprints must be SHA-256")
    indices = {key: split.get(key) for key in ("train", "validation", "test")}
    valid = all(isinstance(x, list) and x and all(type(i) is int and i >= 0 for i in x)
                and len(set(x)) == len(x) for x in indices.values())
    if not valid:
        _block(blockers, "invalid_split", path, "Saved partitions require nonempty unique nonnegative indices")
    else:
        sets = {k: set(v) for k, v in indices.items()}
        if any(sets[a] & sets[b] for a, b in
               (("train", "validation"), ("train", "test"), ("validation", "test"))):
            _block(blockers, "split_leakage", path, "Saved partitions overlap")
        ids = split.get("sample_ids", {})
        for partition in ("train", "validation"):
            rows = ids.get(partition) if isinstance(ids, dict) else None
            if (not isinstance(rows, list) or not all(isinstance(r, str) for r in rows) or
                    len(rows) != len(indices[partition]) or len(set(rows)) != len(rows)):
                _block(blockers, "invalid_sample_ids", path, f"{partition}: missing or duplicate saved IDs")
            elif dataset and isinstance(dataset.get("sample_ids"), list):
                saved = dataset["sample_ids"]
                if any(i >= len(saved) for i in indices[partition]) or rows != [saved[i] for i in indices[partition]]:
                    _block(blockers, "sample_order_mismatch", path, f"{partition}: IDs disagree with saved trial order")
    return {"split_id": split.get("split_id"), "fingerprint": split.get("fingerprint"),
            "rows": {k: len(indices[k]) if isinstance(indices[k], list) else None
                     for k in ("train", "validation")}}


def _identity_check(actual, expected, path, blockers):
    if not isinstance(actual, dict):
        _block(blockers, "identity_mismatch", path, "Missing artifact identity")
        return
    wrong = [key for key, value in expected.items() if actual.get(key) != value]
    if wrong:
        _block(blockers, "identity_mismatch", path, "Mismatched identity fields: " + ", ".join(wrong))


def _finite_stats(value):
    if isinstance(value, list):
        return bool(value) and all(_finite_stats(v) for v in value)
    return type(value) in (int, float) and math.isfinite(value)


def _inspect_study(study, cfg, blockers):
    root = Path(cfg[f"{study}_dir"])
    path = root / "study.json"
    manifest = _read(path, blockers)
    result = {"path": str(root), "study_id": None, "tasks": [], "source_files": [],
        "coverage": {"tasks_expected": len(tasks(cfg)), "tasks_available": 0,
            "fits_expected": len(tasks(cfg)) * (4 if study == "baseline" else 6), "fits_available": 0},
        "capabilities": ({"saved_neural_classifiers": "requires_verified_files_and_replay",
            "covariance": "saved_context_only_no_refit"} if study == "baseline" else
            {"checkpoint_kind": "encoder_only_with_factors_and_feature_cache",
             "encoder_reconstruction": "requires_verified_cache_and_replay",
             "pretraining_classifier_reloadable": False, "downstream_classifiers_reloadable": False,
             "reason": "Whole pretraining MHA head and selected downstream heads were not persisted"})}
    if manifest is None:
        return result
    before = len(blockers)
    result["manifest_sha256"] = file_sha256(path)
    identity = manifest.get("identity", {}) if study == "baseline" else {}
    if not isinstance(identity, dict):
        _block(blockers, "invalid_identity", path, "Manifest identity must be an object")
        return result
    config = identity.get("config", {}) if study == "baseline" else manifest.get("config", {})
    if not isinstance(config, dict) or not _input_protocol(study, config):
        _block(blockers, "input_protocol_mismatch", path, f"Expected the supplied {study} within-T-session protocol")
        return result
    expected_arms = BASELINE_ARMS if study == "baseline" else LEGACY_ARMS
    declared_arms = manifest.get("arms", [])
    names = ([a.get("name") if isinstance(a, dict) else a for a in declared_arms]
             if isinstance(declared_arms, list) else [])
    if names != list(expected_arms):
        _block(blockers, "arm_coverage_mismatch", path, "Historical arms differ from the supplied study")
    declared = manifest.get("tasks", [])
    subjects, seeds = config.get("subjects"), config.get("seeds")
    valid_tasks = (isinstance(subjects, list) and isinstance(seeds, list) and subjects and seeds and
                   all(type(s) is int and 1 <= s <= 9 for s in subjects) and
                   all(type(s) is int and 0 <= s <= 2 for s in seeds) and
                   len(set(subjects)) == len(subjects) and len(set(seeds)) == len(seeds))
    configured = [[s, seed] for s in subjects for seed in seeds] if valid_tasks else []
    if not isinstance(declared, list) or declared != configured or any([s, seed] not in declared for s, seed in tasks(cfg)):
        _block(blockers, "task_coverage_mismatch", path, "Manifest task list does not cover configured tasks")
    result["declared_task_count"] = len(declared) if isinstance(declared, list) else None
    if study == "baseline":
        result["study_id"] = digest(identity)
        result["study_id_kind"] = "derived_sha256_of_recorded_identity"
        source = identity.get("source_sha256")
        result["recorded_config_sha256"] = identity.get("config_sha256")
        config_path = config.get("config_path")
        if isinstance(config_path, str):
            result["original_config"] = _file(Path(config_path), blockers,
                identity.get("config_sha256"), require_hash=True)
        else:
            _block(blockers, "missing_config_path", path, "Historical config path is missing")
        result["recorded_packages"] = manifest.get("package_versions", {})
        inputs = manifest.get("input_sha256")
        result["recorded_input_sha256"] = inputs
        if not isinstance(inputs, dict) or not inputs or any(not _valid_hash(h) for h in inputs.values()):
            _block(blockers, "invalid_input_checksums", path, "Historical recording checksum map is missing or invalid")
        if manifest.get("synthetic") is not False:
            _block(blockers, "synthetic_input", path, "Primary audit input must be real")
    else:
        result["study_id"] = manifest.get("study_id")
        result["study_id_kind"] = "stored_legacy_study_id"
        original = manifest.get("source", {})
        if not isinstance(original, dict):
            _block(blockers, "invalid_source", path, "Manifest source must be an object")
            return result
        source = original.get("files_sha256")
        result["recorded_source_sha256"] = original.get("source_sha256")
        result["recorded_packages"] = original.get("packages", {})
        if result["study_id"] != digest({"config": config, "source": original}):
            _block(blockers, "study_identity_mismatch", path, "Stored legacy study ID does not match config/source")
        if source and original.get("source_sha256") != digest(source):
            _block(blockers, "source_identity_mismatch", path, "Historical source digest does not match its map")
    result["source_files"] = _source_map(study, source, blockers)
    result["source_map_sha256"] = digest(source) if isinstance(source, dict) else None
    manifest_valid = len(blockers) == before
    for subject, seed in tasks(cfg):
        start = len(blockers)
        directory = root / "artifacts" / f"A{subject:02d}_seed_{seed}"
        dataset = _read(directory / "dataset.json", blockers)
        split = _read(directory / "split.json", blockers)
        split_info = _split_metadata(split, dataset, directory / "split.json", seed, blockers)
        entry = {"subject": subject, "seed": seed, "path": str(directory), "split": split_info,
                 "arms": [], "files": []}
        entry["dataset_metadata"] = {
            "channel_names": dataset.get("channel_names") if dataset else None,
            "normalization_convention": "baseline_raw_train_channel" if study == "baseline" else
                                        "legacy_raw_and_learned_features_train_only",
            "normalization_replay_verified": False,
        }
        expected = {"subject": subject, "seed": seed,
                    "split_id": split_info["split_id"] if split_info else None}
        if study == "legacy":
            expected.update(study_id=result["study_id"],
                            dataset_fingerprint=split.get("fingerprint") if split else None)
            calibration = _read(directory / "calibration.json", blockers)
            _identity_check(calibration.get("identity") if calibration else None, expected,
                            directory / "calibration.json", blockers)
            checksum = calibration.get("cache_sha256") if calibration else None
            entry["calibration_metadata"] = {
                "feature_count": calibration.get("feature_count") if calibration else None,
                "encoder_selection_summary_available": isinstance(calibration.get("encoder_selection"), dict)
                                                       if calibration else False,
                "saved_feature_cache": "all-partition container; extract train/validation immediately in later reader",
                "frozen_encoder_and_factors": "writer-declared; tensor contents not inspected",
            }
            if calibration:
                if calibration.get("feature_count") != 32:
                    _block(blockers, "invalid_feature_count", directory / "calibration.json", "Expected 32 learned features")
                for key in ("raw_mean", "raw_std", "feature_mean", "feature_std"):
                    if not _finite_stats(calibration.get(key)):
                        _block(blockers, "invalid_normalization_metadata", directory / "calibration.json",
                               f"{key}: missing or nonfinite saved statistics")
            entry["files"].append(_file(directory / "calibration.pt", blockers, checksum, require_hash=True))
            expected["calibration_sha256"] = checksum
            for name in ("encoder_history.json", "factor_history.json"):
                _read(directory / name, blockers, kind=list)
                entry["files"].append({"path": str(directory / name), "exists": (directory / name).is_file()})
        else:
            expected.update(config_sha256=identity.get("config_sha256"), source_sha256=source,
                synthetic=False, protocol="within_session",
                data_fingerprint=dataset.get("data_fingerprint") if dataset else None)
            stats = dataset.get("normalization_stats", {}) if dataset else {}
            if not isinstance(stats, dict) or any(not _finite_stats(stats.get(k)) for k in ("mean", "std")):
                _block(blockers, "invalid_normalization_metadata", directory / "dataset.json",
                       "Missing or nonfinite saved baseline channel statistics")
        provenance = dataset.get("provenance", {}) if dataset else {}
        if not isinstance(provenance, dict):
            _block(blockers, "invalid_provenance", directory / "dataset.json", "Expected provenance object")
            provenance = {}
        sources = (provenance.get("sources", []) if study == "baseline" else
                   dataset.get("sources", []) if dataset else [])
        entry["recorded_data_sources"] = ([{k: s.get(k) for k in ("name", "sha256", "bytes")}
                                         for s in sources if isinstance(s, dict)]
                                         if isinstance(sources, list) else [])
        if not isinstance(sources, list) or not sources or any(not isinstance(s, dict) or s.get("name") != f"A{subject:02d}T.gdf"
                              or not _valid_hash(s.get("sha256")) for s in sources):
            _block(blockers, "invalid_data_sources", directory / "dataset.json", "Expected checksummed T-session source")
        elif study == "baseline" and isinstance(inputs, dict):
            recorded = inputs.get(str(Path(config.get("data_dir", "")) / f"A{subject:02d}T.gdf"))
            if any(s["sha256"] != recorded for s in sources):
                _block(blockers, "data_checksum_identity_mismatch", directory / "dataset.json",
                       "Task recording checksum differs from manifest")
        for arm in expected_arms:
            arm_dir = directory / "ARMS" / arm
            arm_path = arm_dir / "result.json"
            saved = _read(arm_path, blockers)
            arm_start = len(blockers)
            item = {"name": arm, "result_path": str(arm_path), "status": "unavailable"}
            if saved is not None:
                _identity_check(saved.get("identity"), expected, arm_path, blockers)
                actual_arm = saved.get("arm")
                actual_name = actual_arm.get("name") if isinstance(actual_arm, dict) else actual_arm
                if (saved.get("status") != "complete" or actual_name != arm or
                        saved.get("subject") != subject or saved.get("seed") != seed):
                    _block(blockers, "incomplete_or_wrong_arm", arm_path, "Result status or task/arm identity differs")
                if study == "baseline":
                    neural = arm != "covariance__full"
                    item["checkpoint_kind"] = "selected_neural_classifier" if neural else "covariance_context"
                    filename, hash_key = (("checkpoint.pt", "checkpoint_sha256") if neural else
                                          ("model.npz", "model_sha256"))
                    item["checkpoint"] = _file(arm_dir / filename, blockers, saved.get(hash_key), require_hash=True)
                    if neural:
                        item["history"] = _file(arm_dir / "history.json", blockers,
                                                saved.get("history_sha256"), require_hash=True)
                        _read(arm_dir / "history.json", blockers, kind=list)
                    item["classifier_checkpoint_available"] = item["checkpoint"].get("checksum_verified", False)
                    item["classifier_replay_verified"] = False
                else:
                    _read(arm_dir / "history.json", blockers, kind=list)
                    item["classifier_reloadable"] = False
                if len(blockers) == arm_start:
                    item["status"] = "metadata_and_files_available"
                    result["coverage"]["fits_available"] += 1
            entry["arms"].append(item)
        entry["status"] = "available" if len(blockers) == start else "blocked"
        if entry["status"] == "available":
            result["coverage"]["tasks_available"] += 1
        result["tasks"].append(entry)
    result["manifest_and_source_ready"] = manifest_valid
    return result


def inspect_inputs(cfg):
    """Return stdlib inventory and structured blockers; write nothing anywhere."""
    blockers = []
    identity = audit_identity(cfg)
    dependencies = {}
    for name, module in (("torch", "torch"), ("numpy", "numpy"), ("scipy", "scipy"),
                         ("scikit-learn", "sklearn"), ("mne", "mne"), ("tqdm", "tqdm")):
        try:
            available = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            available = False
        dependencies[name] = {"discoverable": available, "version": identity["packages"][name]}
        if not available:
            _block(blockers, "missing_dependency", name, "Required for later real audit; not imported by inventory")
    studies = {key: _inspect_study(key, cfg, blockers) for key in ("baseline", "legacy")}
    recordings = []
    for subject in cfg["subjects"]:
        path = Path(cfg["data_dir"]) / f"A{subject:02d}T.gdf"
        record = _file(path, blockers)
        # Presence is inventoried; no GDF parsing or trial loading in session 01.
        record["checksum_verification"] = "deferred_to_artifact_reader"
        recordings.append(record)
    return {"schema_name": "agfl-encoder-audit-inventory-v1", "status": "blocked" if blockers else "ready",
        "inventory_only": True, "measurements_performed": False, "probe_fits_performed": 0,
        "partitions": ["train", "validation"], "audit_identity": identity,
        "input_study_ids": {key: value["study_id"] for key, value in studies.items()},
        "studies": studies, "dependencies": dependencies, "python_executable": sys.executable,
        "data": {"path": cfg["data_dir"], "recordings": recordings}, "blockers": blockers,
        "limitations": ["No checkpoint was deserialized or replayed; file availability is not replay evidence",
            "Historical validation maxima are selection-biased and are not extracted",
            "Cross-study trial pairing and normalization replay require subsequent sessions"]}
