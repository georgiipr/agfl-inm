"""Tiny stdlib fixtures; no scientific imports, recordings or research fits."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from inm.encoder_audit.inventory import (BASELINE_ARMS, LEGACY_ARMS, REPLAY_PATHS,
                                         inspect_inputs)
from inm.encoder_audit.protocol import ROOT, audit_identity, digest, file_sha256, load_config, tasks


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


class FixtureCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "config/audit.json"
        self.raw = json.loads((ROOT / "configs/encoder-audit.json").read_text())
        self.raw.update(data_dir="../data", baseline_dir="../baseline", legacy_dir="../legacy",
                        output_dir="../audit")

    def config(self, raw=None):
        write(self.path, self.raw if raw is None else raw)
        return load_config(self.path)


class ProtocolTests(FixtureCase):
    def test_default_plan_and_task_order(self):
        cfg = load_config(ROOT / "configs/encoder-audit.json")
        self.assertEqual(len(tasks(cfg)), 27)
        self.assertEqual(tasks(cfg)[:4], [(1, 0), (1, 1), (1, 2), (2, 0)])
        self.assertEqual(tasks(cfg)[-1], (9, 2))
        self.assertEqual(cfg["budgets"], {"cpu_probe_fits_per_task": 4, "neural_refits": 0})
        self.assertEqual(cfg["data_dir"], str(ROOT.parent / "ml"))

    def test_paths_are_config_relative_not_cwd(self):
        cfg = self.config()
        self.assertEqual(cfg["output_dir"], str(self.root / "audit"))
        self.assertFalse(Path(cfg["output_dir"]).exists())
        self.assertEqual(self.path.read_bytes(), (json.dumps(self.raw) + "\n").encode())

    def test_invalid_fixed_parameters_and_unknown_search(self):
        changes = [lambda c: c.update(partitions=["train", "validation", "test"]),
            lambda c: c.update(protocol="cross_session"),
            lambda c: c.update(search={"ranks": [3, 4]}),
            lambda c: c.update(selection={"policy": "best_test"}),
            lambda c: c["probes"].update(C=2),
            lambda c: c["probes"].update(multi_class="multinomial"),
            lambda c: c["preprocessing"].update(sessions=["T", "E"]),
            lambda c: c["preprocessing"].update(offset_seconds=2),
            lambda c: c["tolerances"]["features"].update(atol=.01),
            lambda c: c["conditions"][1].update(retained=0),
            lambda c: c["conditions"][1].update(retained=11),
            lambda c: c["conditions"][1].update(pattern="spatial_static"),
            lambda c: c["conditions"][1].update(repeats=4),
            lambda c: c["budgets"].update(neural_refits=True),
            lambda c: c["preprocessing"].update(windows=4.0),
            lambda c: c.update(subjects=[1, 1]), lambda c: c.update(subjects=[True]),
            lambda c: c.update(seeds=[0, 0]), lambda c: c.update(seeds=[3]),
            lambda c: c.update(name=""), lambda c: c.update(data_dir=None)]
        for change in changes:
            raw = copy.deepcopy(self.raw)
            change(raw)
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                self.config(raw)

    def test_nonfinite_and_duplicate_json_keys(self):
        raw = copy.deepcopy(self.raw)
        raw["probes"]["C"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            self.config(raw)
        for value in (float("inf"), -float("inf")):
            raw["probes"]["C"] = value
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                self.config(raw)
        self.config()
        self.path.write_text(self.path.read_text().replace('"seeds": [0, 1, 2]', '"seeds": [0], "seeds": [1]'))
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            load_config(self.path)

    def test_overlap_in_both_directions_and_symlinks(self):
        for key in ("data_dir", "baseline_dir", "legacy_dir"):
            for output in (self.raw[key], self.raw[key] + "/child", ".."):
                raw = copy.deepcopy(self.raw)
                raw["output_dir"] = output
                with self.subTest(key=key, output=output), self.assertRaisesRegex(ValueError, "overlap"):
                    self.config(raw)
            source = self.root / key
            source.mkdir()
            link = self.root / (key + "-link")
            link.symlink_to(source, target_is_directory=True)
            raw = copy.deepcopy(self.raw)
            raw[key] = str(source)
            raw["output_dir"] = str(link / "future")
            with self.assertRaisesRegex(ValueError, "overlap"):
                self.config(raw)
            raw[key] = str(link / "future")
            raw["output_dir"] = str(source)
            with self.assertRaisesRegex(ValueError, "overlap"):
                self.config(raw)
        raw["output_dir"] = str(ROOT)
        with self.assertRaisesRegex(ValueError, "repository root"):
            self.config(raw)

    def test_missing_config_and_same_input(self):
        with self.assertRaises(FileNotFoundError):
            load_config(self.path)
        self.raw["baseline_dir"] = self.raw["legacy_dir"]
        with self.assertRaisesRegex(ValueError, "distinct input"):
            self.config()

    def test_identity_hashes_config_and_source_bytes(self):
        cfg = self.config()
        identity = audit_identity(cfg)
        self.assertEqual(identity["config_sha256"], file_sha256(self.path))
        self.assertEqual(identity["source_sha256"], digest(identity["source_files_sha256"]))
        files = [*ROOT.glob("*.py"), *(ROOT / "agfl").rglob("*.py"), *(ROOT / "inm").rglob("*.py")]
        self.assertEqual(set(identity["source_files_sha256"]), {p.relative_to(ROOT).as_posix() for p in files})
        self.assertEqual(identity, audit_identity(cfg))
        self.path.write_text(self.path.read_text() + "\n")
        self.assertNotEqual(identity["audit_id"], audit_identity(load_config(self.path))["audit_id"])

    def test_cli_lazy_plan_and_integrated_actions(self):
        # A fresh interpreter cannot accidentally use already-loaded numeric modules.
        script = '''
import importlib.abc, runpy, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy','torch','scipy','sklearn','mne'}:
            raise AssertionError('numeric import attempted: ' + fullname)
sys.meta_path.insert(0, Block())
sys.argv = ['encoder-audit', '--config', sys.argv[1], '--plan']
runpy.run_module('inm.encoder_audit', run_name='__main__')
'''
        cfg = self.config()
        result = subprocess.run([sys.executable, "-c", script, str(self.path)], cwd=ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Tasks: 27; CPU probe fits: 108; neural refits: 0", result.stdout)
        self.assertIn("train/validation only", result.stdout)
        smoke_dir = Path(cfg["config_path"]).parent / "synthetic-smoke"
        for action in (["--task-index", "0"], ["--task-index", "27"], ["--plan", "--inventory"]):
            result = subprocess.run([sys.executable, "-m", "inm.encoder_audit", "--config", cfg["config_path"], *action],
                                    cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
        result = subprocess.run([sys.executable, "-m", "inm.encoder_audit", "--config", cfg["config_path"],
                                 "--smoke", "--output-dir", str(smoke_dir)], cwd=ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"synthetic": true', result.stdout)
        self.assertTrue((smoke_dir / "report/evidence.json").is_file())
        result = subprocess.run([sys.executable, "-m", "inm.encoder_audit", "--config", cfg["config_path"],
                                 "--summarize-only"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class InventoryTests(FixtureCase):
    def fixture(self):
        self.raw.update(subjects=[1], seeds=[0])
        cfg = self.config()
        self.cfg = cfg
        self.manifests = {}
        for study, arms in (("baseline", BASELINE_ARMS), ("legacy", LEGACY_ARMS)):
            root = Path(cfg[f"{study}_dir"])
            source = {p: file_sha256(ROOT / p) for p in REPLAY_PATHS[study]}
            if study == "baseline":
                config = json.loads((ROOT / "configs/baselines-reproducible.json").read_text())
                config.update(subjects=[1], seeds=[0], data_dir=cfg["data_dir"],
                              config_path=str(self.root / "baseline-config.json"))
                write(Path(config["config_path"]), config)
                identity = {"config": config, "config_sha256": file_sha256(config["config_path"]),
                            "source_sha256": source}
                manifest = {"format": "agfl-baseline-study-v1", "identity": identity, "synthetic": False,
                    "tasks": [[1, 0]], "arms": list(arms), "package_versions": {},
                    "input_sha256": {str(Path(cfg["data_dir"]) / "A01T.gdf"): "a" * 64}}
            else:
                config = json.loads((ROOT / "configs/study.json").read_text())
                config.update(subjects=[1], seeds=[0], tensor_followup={"schema_name": "agfl-legacy-tensor-followup-v1"})
                config["data"]["data_dir"] = cfg["data_dir"]
                original = {"files_sha256": source, "source_sha256": digest(source), "packages": {}}
                manifest = {"config": config, "source": original,
                    "study_id": digest({"config": config, "source": original}), "tasks": [[1, 0]],
                    "arms": [{"name": arm} for arm in arms]}
            write(root / "study.json", manifest)
            self.manifests[study] = manifest
            directory = root / "artifacts/A01_seed_0"
            ids = [f"synthetic:cue:{i}" for i in range(6)]
            split = {"protocol": "stratified", "seed": 0, "split_id": "b" * 64,
                "fingerprint": "c" * 64, "train": [0, 1], "validation": [2, 3], "test": [4, 5],
                "sample_ids": {"train": ids[:2], "validation": ids[2:4], "test": ids[4:]}}
            sources = [{"name": "A01T.gdf", "sha256": "a" * 64, "bytes": 1}]
            dataset = ({"sample_ids": ids, "data_fingerprint": "d" * 64, "provenance": {"sources": sources},
                        "normalization_stats": {"mean": [0.] * 22, "std": [1.] * 22}}
                       if study == "baseline" else {"sources": sources})
            write(directory / "split.json", split)
            write(directory / "dataset.json", dataset)
            expected = {"subject": 1, "seed": 0, "split_id": "b" * 64}
            if study == "legacy":
                expected.update(study_id=manifest["study_id"], dataset_fingerprint="c" * 64)
                # Deliberately not a tensor. Deserialization would fail this fixture.
                (directory / "calibration.pt").write_bytes(b"encoder-only fixture; no MHA or downstream head")
                checksum = file_sha256(directory / "calibration.pt")
                write(directory / "calibration.json", {"identity": expected, "cache_sha256": checksum,
                    "feature_count": 32, "raw_mean": [0.] * 22, "raw_std": [1.] * 22,
                    "feature_mean": [[0.] * 32] * 22, "feature_std": [[1.] * 32] * 22,
                    "encoder_selection": {"best_epoch": 1}})
                expected = {**expected, "calibration_sha256": checksum}
                for name in ("encoder_history.json", "factor_history.json"):
                    write(directory / name, [])
            else:
                expected.update(config_sha256=identity["config_sha256"], source_sha256=source,
                    synthetic=False, protocol="within_session", data_fingerprint="d" * 64)
            for arm in arms:
                arm_dir = directory / "ARMS" / arm
                saved = {"identity": expected, "subject": 1, "seed": 0, "status": "complete",
                    "arm": arm if study == "baseline" else {"name": arm},
                    "metrics": [{"partition": "test", "balanced_accuracy": .99, "labels": [99]}]}
                write(arm_dir / "history.json", [])
                if study == "baseline":
                    neural = arm != "covariance__full"
                    filename, key = ("checkpoint.pt", "checkpoint_sha256") if neural else ("model.npz", "model_sha256")
                    (arm_dir / filename).write_bytes(b"fake selected classifier; never loaded")
                    saved[key] = file_sha256(arm_dir / filename)
                    if neural:
                        saved["history_sha256"] = file_sha256(arm_dir / "history.json")
                write(arm_dir / "result.json", saved)
        return cfg

    def mutate(self, study, name, update):
        path = Path(self.cfg[f"{study}_dir"]) / name
        value = json.loads(path.read_text())
        update(value)
        write(path, value)

    def test_inventory_capabilities_and_input_bytes_unchanged(self):
        cfg = self.fixture()
        original = {str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        result = inspect_inputs(cfg)
        self.assertEqual(original, {str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()})
        for study, count in (("baseline", 4), ("legacy", 6)):
            self.assertEqual(result["studies"][study]["coverage"]["fits_available"], count)
            self.assertEqual(result["studies"][study]["coverage"]["tasks_available"], 1)
        caps = result["studies"]["legacy"]["capabilities"]
        self.assertFalse(caps["pretraining_classifier_reloadable"])
        self.assertFalse(caps["downstream_classifiers_reloadable"])
        self.assertIn("encoder_only", caps["checkpoint_kind"])
        self.assertEqual(result["input_study_ids"]["baseline"], digest(self.manifests["baseline"]["identity"]))
        self.assertFalse(result["measurements_performed"])
        self.assertEqual(result["probe_fits_performed"], 0)
        self.assertFalse(Path(cfg["output_dir"]).exists())
        self.assertEqual(result["status"], "blocked")  # No recordings created by fixtures.

    def test_test_metrics_labels_and_features_never_exposed(self):
        cfg = self.fixture()
        before = inspect_inputs(cfg)
        for study, arms in (("baseline", BASELINE_ARMS), ("legacy", LEGACY_ARMS)):
            for arm in arms:
                self.mutate(study, f"artifacts/A01_seed_0/ARMS/{arm}/result.json",
                    lambda v: v.update(metrics=[{"partition": "test", "accuracy": float("nan"),
                        "labels": ["POISON"], "features": [float("inf")]}]))
        self.assertEqual(before, inspect_inputs(cfg))
        serialized = json.dumps(before, allow_nan=False)
        self.assertNotIn("balanced_accuracy", serialized)
        self.assertNotIn("POISON", serialized)

    def test_missing_files_and_checkpoint_checksum_failure(self):
        cfg = self.fixture()
        path = Path(cfg["legacy_dir"]) / "artifacts/A01_seed_0/calibration.pt"
        path.write_bytes(path.read_bytes() + b"corrupt")
        codes = {b["code"] for b in inspect_inputs(cfg)["blockers"]}
        self.assertIn("checksum_mismatch", codes)
        path.unlink()
        self.assertIn("missing_file", {b["code"] for b in inspect_inputs(cfg)["blockers"]})
        Path(cfg["baseline_dir"], "study.json").unlink()
        result = inspect_inputs(cfg)
        self.assertIsNone(result["input_study_ids"]["baseline"])
        self.assertEqual(result["studies"]["baseline"]["coverage"]["fits_available"], 0)

    def test_wrong_protocol_and_missing_task_coverage(self):
        cfg = self.fixture()
        self.mutate("baseline", "study.json", lambda v: v["identity"]["config"].update(protocol="cross_session"))
        self.assertIn("input_protocol_mismatch", {b["code"] for b in inspect_inputs(cfg)["blockers"]})
        self.mutate("legacy", "study.json", lambda v: v.update(tasks=[]))
        self.assertIn("task_coverage_mismatch", {b["code"] for b in inspect_inputs(cfg)["blockers"]})

    def test_identity_mismatch_leakage_and_reordered_samples(self):
        cfg = self.fixture()
        self.mutate("legacy", "artifacts/A01_seed_0/calibration.json", lambda v: v["identity"].update(seed=2))
        self.mutate("baseline", "artifacts/A01_seed_0/split.json", lambda v: v.update(validation=[1, 3]))
        result = inspect_inputs(cfg)
        codes = {b["code"] for b in result["blockers"]}
        self.assertIn("identity_mismatch", codes)
        self.assertIn("split_leakage", codes)
        self.assertIn("sample_order_mismatch", codes)
        self.mutate("baseline", "artifacts/A01_seed_0/split.json", lambda v: v.update(validation=[2, 3]))
        self.mutate("baseline", "artifacts/A01_seed_0/dataset.json", lambda v: v["sample_ids"].reverse())
        self.assertIn("sample_order_mismatch", {b["code"] for b in inspect_inputs(cfg)["blockers"]})

    def test_exact_replay_source_mismatch_reported(self):
        cfg = self.fixture()
        self.mutate("baseline", "study.json", lambda v: v["identity"]["source_sha256"].update({"inm/baselines/eegnet.py": "0" * 64}))
        result = inspect_inputs(cfg)
        self.assertTrue(any(b["code"] == "replay_source_mismatch" and b["path"] == "inm/baselines/eegnet.py"
                            for b in result["blockers"]))

    def test_nonfinite_normalization_metadata_blocks(self):
        cfg = self.fixture()
        self.mutate("legacy", "artifacts/A01_seed_0/calibration.json", lambda v: v.update(raw_mean=[float("nan")]))
        self.mutate("baseline", "artifacts/A01_seed_0/dataset.json", lambda v: v["normalization_stats"].update(std=[float("inf")]))
        problems = [b for b in inspect_inputs(cfg)["blockers"] if b["code"] == "invalid_normalization_metadata"]
        self.assertEqual(len(problems), 2)
        # Nonfinite values are reported as failures without being emitted in JSON.
        json.dumps(inspect_inputs(cfg), allow_nan=False)

    def test_corrupt_json_and_missing_recorded_checksum(self):
        cfg = self.fixture()
        path = Path(cfg["legacy_dir"]) / "artifacts/A01_seed_0/calibration.json"
        path.write_text("{invalid")
        self.mutate("baseline", "artifacts/A01_seed_0/ARMS/masked_eegnet__full/result.json",
                    lambda v: v.pop("checkpoint_sha256"))
        result = inspect_inputs(cfg)
        codes = {b["code"] for b in result["blockers"]}
        self.assertIn("invalid_json", codes)
        self.assertIn("missing_checksum", codes)
        self.assertLess(result["studies"]["baseline"]["coverage"]["fits_available"], 4)

    def test_inventory_cli_blockers_are_json_and_nonzero(self):
        cfg = self.config()
        result = subprocess.run([sys.executable, "-m", "inm.encoder_audit", "--config", cfg["config_path"], "--inventory"],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1, result.stderr)
        output = json.loads(result.stdout)
        self.assertEqual(output["status"], "blocked")
        self.assertTrue(output["blockers"])

    def test_inventory_never_imports_numeric_modules(self):
        cfg = self.fixture()
        script = '''
import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'numpy','torch','scipy','sklearn','mne'}:
        raise AssertionError('numeric import attempted: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = ['encoder-audit', '--config', sys.argv[1], '--inventory']
runpy.run_module('inm.encoder_audit', run_name='__main__')
'''
        result = subprocess.run([sys.executable, "-c", script, cfg["config_path"]], cwd=ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 1, result.stderr)  # Missing fixture recordings.
        self.assertEqual(json.loads(result.stdout)["studies"]["legacy"]["coverage"]["fits_available"], 6)


if __name__ == "__main__":
    unittest.main()
