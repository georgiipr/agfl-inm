"""Stdlib-only readiness and saved-result inspection for the baseline study."""
from __future__ import annotations

import importlib.util
import json
import platform
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
LEGACY_CONFIG = ROOT / "configs" / "study.json"
PACKAGES = {
    "torch": "torch",
    "numpy": "numpy",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "mne": "mne",
    "tqdm": "tqdm",
}


def _read_json(path: Path) -> tuple[object | None, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except FileNotFoundError:
        return None, "missing"
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return None, f"corrupt: {type(error).__name__}: {error}"


def _resolve_legacy_paths(config: dict, config_path: Path) -> tuple[Path, Path]:
    # The legacy loader resolves these paths from the process working directory.
    data_dir = Path(config["data"]["data_dir"]).expanduser().resolve()
    output_dir = Path(config["output_dir"]).expanduser().resolve()
    return data_dir, output_dir


def _resolve_baseline_paths(config: dict, config_path: Path) -> tuple[Path, Path]:
    base = config_path.parent
    data = Path(config["data_dir"]).expanduser()
    output = Path(config["output_dir"]).expanduser()
    return ((base / data if not data.is_absolute() else data).resolve(),
            (base / output if not output.is_absolute() else output).resolve())


def inspect_environment(config_path=None) -> dict:
    """Report package discoverability and expected paths without importing them."""
    selected = Path(config_path).expanduser().resolve() if config_path else LEGACY_CONFIG
    cfg, error = _read_json(selected)
    if error or not isinstance(cfg, dict):
        return {
            "python": {"version": platform.python_version(), "executable": sys.executable},
            "packages": {name: {"discoverable": importlib.util.find_spec(module) is not None}
                         for name, module in PACKAGES.items()},
            "config_path": str(selected), "config_status": error or "invalid: expected JSON object",
            "input_dir": None, "output_dir": None, "expected_recordings": [],
            "readiness": "unavailable",
        }
    try:
        is_baseline = cfg.get("schema_name") == "agfl-baselines-v1"
        input_dir, output_dir = (_resolve_baseline_paths(cfg, selected) if is_baseline
                                 else _resolve_legacy_paths(cfg, selected))
    except (KeyError, TypeError) as exception:
        return {
            "python": {"version": platform.python_version(), "executable": sys.executable},
            "packages": {name: {"discoverable": importlib.util.find_spec(module) is not None}
                         for name, module in PACKAGES.items()},
            "config_path": str(selected), "config_status": f"invalid: {exception}",
            "input_dir": None, "output_dir": None, "expected_recordings": [],
            "readiness": "unavailable",
        }
    recordings = []
    for subject in cfg.get("subjects", range(1, 10)):
        sessions = ("T", "E") if is_baseline and cfg.get("protocol") == "cross_session" else ("T",)
        for session in sessions:
            recording = input_dir / f"A{subject:02d}{session}.gdf"
            recordings.append({"path": str(recording), "exists": recording.is_file()})
    packages = {name: {"discoverable": importlib.util.find_spec(module) is not None}
                for name, module in PACKAGES.items()}
    required_discoverable = all(packages[name]["discoverable"]
                                for name in ("torch", "numpy", "scipy", "scikit-learn", "mne"))
    recordings_exist = all(item["exists"] for item in recordings)
    labels_dir = None
    labels_exist = True
    if is_baseline and cfg.get("protocol") == "cross_session":
        labels_value = cfg.get("external_labels_dir")
        if labels_value:
            labels_dir = Path(labels_value).expanduser()
            if not labels_dir.is_absolute():
                labels_dir = (selected.parent / labels_dir).resolve()
            labels_exist = labels_dir.is_dir()
        else:
            labels_exist = False
    return {
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "packages": packages,
        "config_path": str(selected), "config_status": "available",
        "input_dir": str(input_dir), "output_dir": str(output_dir),
        "expected_recordings": recordings,
        "external_labels_dir": str(labels_dir) if labels_dir else None,
        "external_labels_required": bool(is_baseline and cfg.get("protocol") == "cross_session"),
        "readiness": "ready" if recordings_exist and required_discoverable and labels_exist else "blocked",
        "blockers": ([f"Missing recording: {item['path']}" for item in recordings if not item["exists"]]
                     + [f"Missing required package: {name}" for name in
                        ("torch", "numpy", "scipy", "scikit-learn", "mne")
                        if not packages[name]["discoverable"]]
                     + (["Cross-session preflight requires an explicit external_labels_dir containing E-session labels."]
                        if not labels_exist else [])),
    }


def _identity(value: object) -> dict | None:
    if not isinstance(value, dict):
        return None
    identity = value.get("identity")
    if not isinstance(identity, dict):
        identity = value
    keys = ("study_id", "subject", "seed", "dataset_fingerprint", "split_id",
            "calibration_sha256", "source_sha256")
    result = {key: identity[key] for key in keys if key in identity}
    # Legacy calibration.json stores the frozen-cache digest at the top level;
    # downstream result identities call the same digest calibration_sha256.
    if "cache_sha256" in value:
        if ("calibration_sha256" in result and
                result["calibration_sha256"] != value["cache_sha256"]):
            return None
        result["calibration_sha256"] = value["cache_sha256"]
    return result or None


def _metric_rows(value: object) -> list[dict]:
    if not isinstance(value, dict) or not isinstance(value.get("metrics"), list):
        return []
    extracted = []
    for row in value["metrics"]:
        partition = row.get("partition", "").lower() if isinstance(row, dict) and isinstance(
            row.get("partition", ""), str) else ""
        if (isinstance(row, dict) and row.get("scenario") == "full_22"
                and partition in ("validation", "test")):
            extracted.append({
                "partition": partition.upper(), "scenario": "full_22",
                **{name: row.get(name) for name in
                   ("accuracy", "balanced_accuracy", "f1_macro", "mask_sha256") if name in row},
            })
    return extracted


def _stage_metrics(value: object) -> dict | None:
    """Copy the scalar full-input metrics used for a stage comparison."""
    if not isinstance(value, dict):
        return None
    return {name: value.get(name) for name in
            ("accuracy", "balanced_accuracy", "f1_macro", "log_loss")
            if name in value}


def _history_score(path: Path, epoch: object) -> tuple[dict | None, str]:
    if not isinstance(epoch, int) or isinstance(epoch, bool):
        return None, "selected epoch is missing or invalid"
    history, error = _read_json(path)
    if error:
        return None, f"history {error}"
    if not isinstance(history, list):
        return None, "history is not a list"
    for row in history:
        if isinstance(row, dict) and row.get("epoch") == epoch:
            score = row.get("validation")
            if isinstance(score, dict):
                return _stage_metrics(score), "available"
            return None, "selected history row has no validation metrics"
    return None, f"selected epoch {epoch} not found in history"


def _identity_mismatches(candidate: object, calibration: object,
                         split: object, manifest: object, subject: int, seed: int) -> list[str]:
    expected = _identity(calibration)
    actual = _identity(candidate)
    if not expected or not actual:
        return ["identity is missing"]
    identity_keys = ("study_id", "subject", "seed", "dataset_fingerprint", "split_id",
                     "calibration_sha256")
    mismatch = [f"missing {key}" for key in identity_keys
                if key not in expected or key not in actual]
    mismatch.extend(key for key in identity_keys
                    if key in expected and key in actual and expected[key] != actual[key])
    for key in ("study_id",):
        if isinstance(manifest, dict) and manifest.get(key) is not None and \
                expected.get(key) != manifest.get(key):
            mismatch.append(f"calibration.{key}")
    if expected.get("subject") != subject:
        mismatch.append("calibration.subject")
    if expected.get("seed") != seed:
        mismatch.append("calibration.seed")
    if isinstance(split, dict):
        if split.get("split_id") is not None and expected.get("split_id") != split.get("split_id"):
            mismatch.append("calibration.split_id")
        fingerprint = split.get("fingerprint")
        if fingerprint is not None and expected.get("dataset_fingerprint") != fingerprint:
            mismatch.append("calibration.dataset_fingerprint")
    return sorted(set(mismatch))


def _stage_comparison(output: Path, tasks: list[dict], manifest: object) -> dict:
    """Build validation-only stage comparisons and separate test evidence."""
    comparisons = []
    for task in tasks:
        subject, seed = task["subject"], task["seed"]
        task_dir = output / "artifacts" / f"A{subject:02d}_seed_{seed}"
        calibration_path = task_dir / "calibration.json"
        calibration, calibration_error = _read_json(calibration_path)
        split, _ = _read_json(task_dir / "split.json")
        if calibration_error or not isinstance(calibration, dict):
            comparisons.append({"subject": subject, "seed": seed, "status": "unavailable",
                                "validation_stages": [], "test_metrics": [],
                                "missing_sources": [f"calibration.json: {calibration_error or 'invalid object'}"],
                                "comparability": "No stage comparison without a valid calibration record."})
            continue
        calibration_identity = _identity(calibration)
        calibration_problems = _identity_mismatches(calibration, calibration, split, manifest,
                                                     subject, seed)
        selection = calibration.get("encoder_selection")
        encoder_epoch = selection.get("best_epoch") if isinstance(selection, dict) else None
        encoder_history, encoder_history_status = _history_score(task_dir / "encoder_history.json", encoder_epoch)
        encoder_score = _stage_metrics(selection.get("selected_validation")) if isinstance(selection, dict) else None
        missing = []
        if calibration_problems:
            missing.append("calibration identity mismatch: " + ", ".join(calibration_problems))
            encoder_score = None
        if encoder_score is None:
            missing.append("original shared pretraining score unavailable in calibration.json")
        if encoder_history is None:
            missing.append(f"original shared pretraining history unavailable: {encoder_history_status}")
        stages = [{"stage": "shared_pretraining", "representation": "shared_encoder",
                   "selected_epoch": encoder_epoch, "validation_score": encoder_score,
                   "history_validation_score": encoder_history,
                   "history_status": encoder_history_status,
                   "score_source": "calibration.json encoder_selection.selected_validation"}]
        tests = []
        head_entries = []
        for arm in task.get("arms", []):
            arm_dir = task_dir / "ARMS" / str(arm.get("arm", ""))
            result_path = arm_dir / "result.json"
            result, result_error = _read_json(result_path)
            if result_error or not isinstance(result, dict):
                missing.append(f"{result_path.relative_to(output)}: {result_error or 'invalid object'}")
                continue
            mismatches = _identity_mismatches(result, calibration, split, manifest, subject, seed)
            if mismatches:
                missing.append(f"{result_path.relative_to(output)} identity mismatch: {', '.join(mismatches)}")
                continue
            arm_info = result.get("arm") if isinstance(result.get("arm"), dict) else {}
            representation = arm_info.get("representation")
            # Only legacy feature-classifier representations have compatible stages.
            if representation not in ("baseline", "tensor_core"):
                continue
            selection_info = result.get("selection") if isinstance(result.get("selection"), dict) else {}
            selected_epoch = selection_info.get("best_epoch")
            history_score, history_status = _history_score(arm_dir / "history.json", selected_epoch)
            rows = _metric_rows(result)
            validation_row = next((row for row in rows if row["partition"] == "VALIDATION"), None)
            test_row = next((row for row in rows if row["partition"] == "TEST"), None)
            item = {"stage": "retrained_baseline_head" if representation == "baseline" else "tucker_head",
                    "arm": arm_info.get("name", arm.get("arm")),
                    "representation": representation, "attention": arm_info.get("attention"),
                    "regime": arm_info.get("regime"), "selected_epoch": selected_epoch,
                    "validation_score": _stage_metrics(validation_row),
                    "history_validation_score": history_score, "history_status": history_status,
                    "score_source": "result.json full_22 VALIDATION row"}
            if validation_row is None:
                missing.append(f"{result_path.relative_to(output)} has no full-input VALIDATION metrics")
            if test_row is None:
                missing.append(f"{result_path.relative_to(output)} has no full-input TEST metrics")
            if history_score is None:
                missing.append(f"{arm_dir.relative_to(output) / 'history.json'}: {history_status}")
            stages.append(item)
            head_entries.append(item)
            if test_row is not None:
                tests.append({"arm": item["arm"], "representation": representation,
                              "attention": item["attention"], "regime": item["regime"],
                              "selected_epoch": selected_epoch, "test_score": _stage_metrics(test_row),
                              "score_source": "result.json full_22 TEST row",
                              "excluded_from_architecture_recommendations": True})
        pairs = []
        for baseline in head_entries:
            if baseline["representation"] != "baseline":
                continue
            for core in head_entries:
                if core["representation"] != "tensor_core":
                    continue
                if (baseline["attention"], baseline["regime"]) == (core["attention"], core["regime"]):
                    pairs.append({"baseline_arm": baseline["arm"], "tucker_arm": core["arm"],
                                  "matched_attention": baseline["attention"],
                                  "matched_regime": baseline["regime"],
                                  "comparable_features": True})
        comparisons.append({"subject": subject, "seed": seed,
                            "status": "available" if not calibration_problems else "identity_mismatch",
                            "calibration_identity": calibration_identity,
                            "validation_stages": stages, "matched_head_pairs": pairs,
                            "test_metrics": tests, "missing_sources": missing,
                            "common_heldout_raw_predictions": "unavailable unless explicitly saved; summary metrics cannot reconstruct paired predictions",
                            "comparability": "Selected validation maxima from independently trained stages are diagnostic and selection-biased, not independent test evidence."})
    return {"validation_only_for_architecture_recommendations": True,
            "test_metrics_separate_and_excluded_from_recommendations": True,
            "tasks": comparisons}


def inspect_legacy(output_dir) -> dict:
    """Summarize saved legacy task records without loading scientific artifacts."""
    output = Path(output_dir).expanduser().resolve()
    manifest_path = output / "study.json"
    manifest, manifest_error = _read_json(manifest_path)
    if manifest_error == "missing":
        return {
            "output_dir": str(output), "status": "unavailable",
            "completion": {"state": "unavailable", "tasks_complete": 0, "tasks_expected": None},
            "study_identity": None, "tasks": [],
            "stage_comparison": {"validation_only_for_architecture_recommendations": True,
                                 "test_metrics_separate_and_excluded_from_recommendations": True,
                                 "tasks": []},
            "findings": [{"status": "unavailable", "path": str(manifest_path),
                          "message": "Legacy study manifest is missing."}],
        }
    findings = []
    if manifest_error:
        findings.append({"status": "corrupt", "path": str(manifest_path), "message": manifest_error})
    if not isinstance(manifest, dict):
        manifest = {}
        findings.append({"status": "corrupt", "path": str(manifest_path),
                         "message": "Expected a JSON object study manifest."})
    declared = manifest.get("tasks")
    expected_pairs = []
    if isinstance(declared, list):
        for task in declared:
            if isinstance(task, (list, tuple)) and len(task) == 2:
                expected_pairs.append((task[0], task[1]))
    if not expected_pairs:
        cfg, _ = _read_json(LEGACY_CONFIG)
        if isinstance(cfg, dict):
            expected_pairs = [(subject, seed) for subject in cfg.get("subjects", [])
                              for seed in cfg.get("seeds", [])]

    task_dirs = {}
    artifacts = output / "artifacts"
    if artifacts.is_dir():
        for path in sorted(artifacts.glob("A*_seed_*")):
            try:
                subject_text, seed_text = path.name.split("_seed_")
                pair = (int(subject_text[1:]), int(seed_text))
                task_dirs[pair] = path
            except (ValueError, IndexError):
                findings.append({"status": "issue", "path": str(path),
                                 "message": "Unrecognized task directory name."})

    tasks_out = []
    for subject, seed in expected_pairs or sorted(task_dirs):
        directory = task_dirs.get((subject, seed), artifacts / f"A{subject:02d}_seed_{seed}")
        calibration_path = directory / "calibration.json"
        calibration, calibration_error = _read_json(calibration_path)
        if calibration_error and calibration_error != "missing":
            findings.append({"status": "corrupt", "path": str(calibration_path),
                             "message": calibration_error})
        selection = calibration.get("encoder_selection") if isinstance(calibration, dict) else None
        encoder = None
        if isinstance(selection, dict):
            encoder = {
                "selection_rule": selection.get("selection"),
                "selected_epoch": selection.get("best_epoch"),
                "validation_metrics": selection.get("selected_validation"),
                "partition": "VALIDATION",
            }
        elif calibration_error == "missing":
            findings.append({"status": "unavailable", "path": str(calibration_path),
                             "message": "Encoder selection record is unavailable."})
        split_path = directory / "split.json"
        split, split_error = _read_json(split_path)
        if split_error and split_error != "missing":
            findings.append({"status": "corrupt", "path": str(split_path), "message": split_error})
        split_identity = ({key: split[key] for key in ("split_id", "fingerprint", "protocol") if key in split}
                          if isinstance(split, dict) else None)
        arms = []
        arms_dir = directory / "ARMS"
        if arms_dir.is_dir():
            for result_path in sorted(arms_dir.glob("*/result.json")):
                value, result_error = _read_json(result_path)
                if result_error:
                    findings.append({"status": "corrupt" if result_error != "missing" else "unavailable",
                                     "path": str(result_path), "message": result_error})
                    arms.append({"arm": result_path.parent.name,
                                 "status": "corrupt" if result_error != "missing" else "unavailable",
                                 "identity": None, "full_input_scores": []})
                    continue
                if not isinstance(value, dict):
                    findings.append({"status": "corrupt", "path": str(result_path),
                                     "message": "Expected a JSON object result."})
                    arms.append({"arm": result_path.parent.name, "status": "corrupt",
                                 "identity": None, "full_input_scores": []})
                    continue
                arm_info = value.get("arm") if isinstance(value.get("arm"), dict) else {}
                if arm_info.get("representation") not in ("baseline", "tensor_core"):
                    continue
                scores = _metric_rows(value)
                result_status = value.get("status", "incomplete")
                has_both_partitions = {row["partition"] for row in scores} >= {"VALIDATION", "TEST"}
                arm_status = ("complete" if result_status == "complete" and has_both_partitions
                              else "incomplete")
                arms.append({"arm": arm_info.get("name", result_path.parent.name),
                             "representation": arm_info.get("representation"),
                             "attention": arm_info.get("attention"), "regime": arm_info.get("regime"),
                             "status": arm_status,
                             "identity": _identity(value), "full_input_scores": scores})
        expected_arms = manifest.get("arms", []) if isinstance(manifest.get("arms"), list) else []
        expected_names = {item.get("name") for item in expected_arms
                          if isinstance(item, dict) and item.get("representation") in ("baseline", "tensor_core")}
        found_complete = {item["arm"] for item in arms
                          if item.get("status") == "complete" and item.get("full_input_scores")}
        task_complete = bool(expected_names) and expected_names <= found_complete
        tasks_out.append({"subject": subject, "seed": seed,
                          "status": "complete" if task_complete else "partial",
                          "encoder_selection": encoder,
                          "split_identity": split_identity,
                          "calibration_identity": _identity(calibration) if calibration else None,
                          "arms": arms})
        if not task_complete:
            findings.append({"status": "unavailable", "path": str(directory),
                             "message": "Task lacks one or more complete full-input baseline/core results."})

    expected_count = len(expected_pairs) or len(tasks_out)
    completed_count = sum(task["status"] == "complete" for task in tasks_out)
    state = "complete" if expected_count and completed_count == expected_count else (
        "unavailable" if not tasks_out else "partial")
    status = "corrupt" if any(item["status"] == "corrupt" for item in findings) else state
    return {
        "output_dir": str(output), "status": status,
        "completion": {"state": state, "tasks_complete": completed_count,
                       "tasks_expected": expected_count or None},
        "study_identity": {key: manifest[key] for key in ("study_id", "source", "config") if key in manifest},
        "tasks": tasks_out, "stage_comparison": _stage_comparison(output, tasks_out, manifest),
        "findings": findings,
    }
