"""Task orchestration and atomic audit artifacts."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

from .protocol import ROOT, audit_identity, digest, file_sha256, tasks

CHECK_NAMES = ("source_integrity", "trial_alignment", "split_isolation",
               "normalization_replay", "checkpoint_replay", "probe_controls", "mask_pairing")


def _plain(value):
    """Convert analysis outputs to strict JSON primitives, excluding model objects."""
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, dict, list, tuple)):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items() if k not in ("probe", "ordered_probe")}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return _plain(value.item())
    raise TypeError(f"Cannot serialize audit value {type(value).__name__}")


def _atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(_plain(value), sort_keys=True, indent=2, allow_nan=False) + "\n"
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _atomic_text(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="ascii") as stream:
            stream.write(value + "\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def _task_dir(cfg, subject, seed):
    return Path(cfg["output_dir"]) / "tasks" / f"A{subject:02d}_seed_{seed}"


def _ids(sources):
    return {"baseline": sources.baseline_metadata.study_id,
            "legacy": sources.legacy_metadata.study_id}


def _checks(alignment, checkpoints, probes, reconstruction):
    within = alignment.get("within_study", {})
    replay = alignment.get("legacy_feature_replay", {})
    # Alignment records executed status. Source maps are also persisted by the artifact reader.
    source = alignment.get("source_integrity", {}).get("status") == "verified" or (
        alignment.get("status") == "complete" and all(
            item.get("status") == "verified" for item in within.values() if isinstance(item, dict))
    )
    normalization = all(isinstance(within.get(k), dict) and
        "normalization_mean_max_abs_error" in within[k] and
        "normalization_std_max_abs_error" in within[k] for k in ("baseline", "legacy"))
    trial = (all(within.get(k, {}).get("status") == "verified" for k in ("baseline", "legacy"))
             and alignment.get("cross_study_pairing", {}).get("status") in
             ("matched", "unavailable_unmatched_inputs"))
    return {"source_integrity": bool(source), "trial_alignment": bool(trial),
        "split_isolation": alignment.get("partitions_exposed") == ["train", "validation"] or
            (alignment.get("partitions") == ["train", "validation"] and alignment.get("test_rows_exposed") is False),
        "normalization_replay": bool(normalization),
        "checkpoint_replay": checkpoints.get("status") == "complete" and
            checkpoints.get("partitions") == ["train", "validation"],
        "probe_controls": probes.get("status") == "complete" and
            len(probes.get("probes", {})) == 4 and all(
                x.get("converged") is True for x in probes.get("probes", {}).values()),
        "mask_pairing": reconstruction.get("status") == "complete" and
            bool(reconstruction.get("conditions")) and all(
                len(set(row.get("mask_identities", {}).values())) == 1
                for row in reconstruction.get("conditions", []))}


def _run(cfg, subject, seed, device, synthetic=False, sources=None):
    from .alignment import audit_alignment
    from .artifacts import load_task_sources
    from .checkpoints import audit_checkpoints
    from .probes import fit_probes
    from .reconstruction import audit_reconstruction

    out = _task_dir(cfg, subject, seed)
    audit_path = out / "audit.json"
    sidecar_path = out / "audit.json.sha256"
    identity = audit_identity(cfg)
    if audit_path.exists():
        old = json.loads(audit_path.read_text(encoding="utf-8"))
        if not sidecar_path.is_file() or sidecar_path.read_text(encoding="ascii").strip() != file_sha256(audit_path):
            raise ValueError(f"Task audit checksum mismatch: {audit_path}")
        if old.get("synthetic") is not synthetic or old.get("audit_id") != identity["audit_id"]:
            raise ValueError(f"Refusing incompatible task reuse: {audit_path}")
        if old.get("status") != "complete" or not old.get("artifact_sha256"):
            raise ValueError(f"Refusing incomplete/corrupt task reuse: {audit_path}")
        for rel, expected in old["artifact_sha256"].items():
            artifact = out / rel
            if not artifact.is_file() or file_sha256(artifact) != expected:
                raise ValueError(f"Task artifact checksum mismatch: {artifact}")
        return old
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Refusing to overwrite task directory without a valid audit record: {out}")
    started = time.time()
    alignment, checkpoint, probe_result, reconstruction = {}, {}, {}, {}
    error_rows = []
    try:
        sources = sources or load_task_sources(cfg, subject, seed)
        alignment = audit_alignment(sources, cfg)
        checkpoint = audit_checkpoints(sources, alignment, cfg, device=device)
        probe_result = fit_probes(sources, cfg, out / "probes")
        reconstruction = audit_reconstruction(sources, probe_result["ordered_probe"], cfg)
    except Exception as error:
        error_rows.append({"type": type(error).__name__, "message": str(error)})
    # A failed later stage does not erase checks already executed by an earlier
    # stage; stages that did not return their validation remain false.
    checks = _checks(alignment, checkpoint, probe_result, reconstruction)
    data = {"schema_name": "agfl-encoder-audit-task-v1", "subject": subject, "seed": seed,
        "status": "complete" if all(checks.values()) and not error_rows else "failed", "synthetic": synthetic,
        "partitions": ["train", "validation"], "input_study_ids": _ids(sources) if sources else
            {"baseline": None, "legacy": None},
        "audit_id": identity["audit_id"], "config_sha256": identity["config_sha256"],
        "source_sha256": identity["source_sha256"], "source_files_sha256": identity["source_files_sha256"],
        "packages": identity["packages"], "input_checksums": ({
            "baseline": sources.baseline_metadata.artifact_sha256,
            "legacy": sources.legacy_metadata.artifact_sha256} if sources else {}),
        "input_manifest_sha256": ({"baseline": sources.baseline_metadata.manifest_sha256,
            "legacy": sources.legacy_metadata.manifest_sha256} if sources else {}),
        "split_ids": ({"baseline": sources.baseline_metadata.split_id,
            "legacy": sources.legacy_metadata.split_id} if sources else {}),
        "alignment": alignment or {"status": "not_run"}, "clean_checkpoints": checkpoint or {"status": "not_run"},
        "probes": probe_result or {"status": "not_run"},
        "reconstruction": reconstruction or {"status": "not_run"}, "checks": checks,
        "errors": error_rows, "timings": {"elapsed_seconds": time.time() - started},
        "tolerances": cfg["tolerances"]}
    # Numeric probe NPZ files are separately checksummed. JSON contains metadata and metrics only.
    data = _plain(data)
    artifacts = {}
    for path in sorted(out.rglob("*")):
        if path.is_file() and path != audit_path:
            artifacts[path.relative_to(out).as_posix()] = file_sha256(path)
    data["artifact_sha256"] = artifacts
    _atomic_json(audit_path, data)
    _atomic_text(sidecar_path, file_sha256(audit_path))
    return data


def run_task(cfg, task_index, device="cpu"):
    pairs = tasks(cfg)
    if type(task_index) is not int or not 0 <= task_index < len(pairs):
        raise ValueError(f"task_index must be in 0..{len(pairs)-1}")
    return _run(cfg, *pairs[task_index], device=device)


def run_smoke(cfg, output_dir):
    """Create an independent deterministic synthetic receipt without real inputs."""
    from .reporting import _write_task_fixture
    target = Path(output_dir).expanduser().resolve()
    for key in ("data_dir", "baseline_dir", "legacy_dir", "output_dir"):
        source = Path(cfg[key]).resolve()
        if target == source or target.is_relative_to(source) or source.is_relative_to(target):
            raise ValueError(f"Synthetic smoke output overlaps {key}")
    if target.exists() and any(target.iterdir()):
        raise ValueError(f"Synthetic smoke requires a fresh output directory: {target}")
    target.mkdir(parents=True, exist_ok=True)
    identity = audit_identity(cfg)
    # Smoke deliberately creates a small synthetic task; no original reader, split,
    # checkpoint, or study writer is invoked.
    task = _write_task_fixture(target, identity, subject=1, seed=0, synthetic=True)
    report_dir = target / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {"schema_name": "agfl-encoder-audit-report-v1", "status": "partial",
        "synthetic": True, "partitions": ["train", "validation"], "subjects": [1], "seeds": [0],
        "input_study_ids": task["input_study_ids"], "source_files_sha256": identity["source_files_sha256"],
        "tasks": [{"subject": 1, "seed": 0, "path": "tasks/A01_seed_0/audit.json",
                   "sha256": file_sha256(target / "tasks/A01_seed_0/audit.json")}],
        "audit_id": identity["audit_id"], "config_sha256": identity["config_sha256"],
        "aggregation": [], "unavailable_comparisons": []}
    _atomic_json(report_dir / "evidence.json", report)
    (report_dir / "summary.md").write_text(
        "# Synthetic encoder audit smoke\n\nThis fixture is software acceptance only; it contains no study measurements.\n",
        encoding="utf-8")
    return {"status": "complete", "synthetic": True, "output_dir": str(target),
            "task": task, "report": str(report_dir / "evidence.json"), "audit_id": identity["audit_id"]}
