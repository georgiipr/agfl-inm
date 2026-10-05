import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from inm.encoder_audit import reporting, study
from inm.encoder_audit.protocol import audit_identity, load_config


ROOT = Path(__file__).resolve().parents[2]


def config_at(root, subjects=(1, 2), seeds=(0, 1)):
    cfg = load_config(ROOT / "configs/encoder-audit.json")
    cfg.update(subjects=list(subjects), seeds=list(seeds), output_dir=str(root / "out"),
               baseline_dir=str(root / "baseline"), legacy_dir=str(root / "legacy"),
               data_dir=str(root / "data"))
    return cfg


def synthetic_sources():
    md = lambda name: SimpleNamespace(study_id=("a" if name == "baseline" else "b") * 64,
        artifact_sha256={}, manifest_sha256="c" * 64, split_id="d" * 64,
        source_sha256="e" * 64)
    return SimpleNamespace(subject=1, seed=0, baseline_metadata=md("baseline"), legacy_metadata=md("legacy"))


class IntegrationTests(unittest.TestCase):
    def test_end_to_end_synthetic_task_and_resume_checksum(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = config_at(Path(temp))
            aligned = {"status": "complete", "within_study": {
                name: {"status": "verified", "normalization_mean_max_abs_error": 0.,
                       "normalization_std_max_abs_error": 0.} for name in ("baseline", "legacy")},
                "cross_study_pairing": {"status": "matched"}, "partitions_exposed": ["train", "validation"]}
            checkpoints = {"status": "complete", "partitions": ["train", "validation"], "arms": {"fixture": {}}}
            probes = {"status": "complete", "probes": {name: {"converged": True, "validation": {"balanced_accuracy": .5}}
                for name in ("features_ordered", "features_window_mean", "core_ordered", "features_ordered_shuffled")},
                "ordered_probe": object()}
            reconstruction = {"status": "complete", "conditions": [{"condition": "full_22",
                "mask_identities": {"full_features": "x", "zero_fill": "x", "tucker_completion": "x"}}]}
            with patch("inm.encoder_audit.artifacts.load_task_sources", return_value=synthetic_sources()), \
                 patch("inm.encoder_audit.alignment.audit_alignment", return_value=aligned), \
                 patch("inm.encoder_audit.checkpoints.audit_checkpoints", return_value=checkpoints), \
                 patch("inm.encoder_audit.probes.fit_probes", side_effect=lambda sources, cfg, directory:
                       (Path(directory).mkdir(parents=True, exist_ok=True),
                        (Path(directory) / "fixture.npz").write_bytes(b"numeric fixture"), probes)[-1]), \
                 patch("inm.encoder_audit.reconstruction.audit_reconstruction", return_value=reconstruction):
                first = study._run(cfg, 1, 0, "cpu", synthetic=True)
                second = study._run(cfg, 1, 0, "cpu", synthetic=True)
            self.assertEqual(first, second)
            self.assertTrue(all(first["checks"].values()))
            audit = Path(cfg["output_dir"]) / "tasks/A01_seed_0/audit.json"
            self.assertEqual(json.loads(audit.read_text())["partitions"], ["train", "validation"])
            artifact = Path(cfg["output_dir"]) / "tasks/A01_seed_0/probes/fake.npz"
            artifact.write_bytes(b"fixture")
            # The task has an empty artifact map in this mocked run; add one and prove tamper rejection.
            record = json.loads(audit.read_text()); record["artifact_sha256"] = {"probes/fake.npz": "0" * 64}
            audit.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                study._run(cfg, 1, 0, "cpu", synthetic=True)

    def _real_fixture(self, root, cfg, identity, subject, seed, value, trials):
        task = reporting._write_task_fixture(root, identity, subject, seed, synthetic=False)
        path = root / "tasks" / f"A{subject:02d}_seed_{seed}" / "audit.json"
        task["input_study_ids"] = {"baseline": "a" * 64, "legacy": "b" * 64}
        task["alignment"] = {"cross_study_pairing": {"status": "unavailable_unmatched_inputs"}}
        task["reconstruction"] = {"conditions": [{"condition": "random_static_6", "pattern": "random_static",
            "retained": 6, "repeat": 0, "metrics": {
                "full_features": {"balanced_accuracy": value}, "zero_fill": {"balanced_accuracy": value},
                "tucker_completion": {"balanced_accuracy": value}},
            "per_trial": [{"sample_id": f"id-{i}", "label": 1, "test_metric": 999} for i in range(trials)]}]}
        # Recompute only allowed serialized fixture checksums.
        task["artifact_sha256"] = {"probes/fixture.npz": study.file_sha256(path.parent / "probes/fixture.npz")}
        path.write_text(json.dumps(task))

    def test_partial_hierarchical_mean_and_no_test_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = config_at(Path(temp), subjects=(1, 2), seeds=(0,))
            identity = audit_identity(cfg); root = Path(cfg["output_dir"])
            self._real_fixture(root, cfg, identity, 1, 0, .2, 1)
            self._real_fixture(root, cfg, identity, 2, 0, .8, 100)
            result = reporting.summarize(cfg)
            self.assertEqual(result["status"], "complete")
            report = json.loads(Path(result["paths"]["evidence"]).read_text())
            row = report["aggregation"][0]
            self.assertAlmostEqual(row["views"]["full_features"]["balanced_accuracy"], .5)
            self.assertEqual(len(report["unavailable_comparisons"]), 2)
            self.assertNotIn("test_metric", json.dumps(report))
            # One missing task suppresses all cohort means.
            cfg["seeds"] = [0, 1]
            # A changed configured task matrix has a new identity. Existing tasks
            # are intentionally rejected; build a fresh partial output instead.
            cfg["output_dir"] = str(Path(temp) / "partial-out")
            identity = audit_identity(cfg)
            self._real_fixture(Path(cfg["output_dir"]), cfg, identity, 1, 0, .2, 1)
            result = reporting.summarize(cfg)
            report = json.loads(Path(result["paths"]["evidence"]).read_text())
            self.assertEqual(result["status"], "partial")
            self.assertIsNone(report["aggregation"][0]["views"]["full_features"]["balanced_accuracy"])

    def test_smoke_is_synthetic_and_separate_from_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = config_at(Path(temp))
            for key in ("data_dir", "baseline_dir", "legacy_dir"):
                path = Path(cfg[key]); path.mkdir(); (path / "preserve.txt").write_text("original")
            smoke = study.run_smoke(cfg, Path(temp) / "smoke")
            self.assertTrue(smoke["synthetic"])
            self.assertEqual(smoke["task"]["synthetic"], True)
            self.assertTrue(Path(smoke["report"]).is_file())
            for key in ("data_dir", "baseline_dir", "legacy_dir"):
                self.assertEqual((Path(cfg[key]) / "preserve.txt").read_text(), "original")
            with self.assertRaisesRegex(ValueError, "overlaps"):
                study.run_smoke(cfg, cfg["data_dir"])

    def test_source_config_identity_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = config_at(Path(temp))
            identity = audit_identity(cfg)
            self._real_fixture(Path(cfg["output_dir"]), cfg, identity, 1, 0, .5, 1)
            changed = dict(cfg, name="changed_protocol_name")
            with self.assertRaisesRegex(ValueError, "identity"):
                reporting.summarize(changed)
            task_path = Path(cfg["output_dir"]) / "tasks/A01_seed_0/audit.json"
            task = json.loads(task_path.read_text()); task["source_files_sha256"] = {}
            task_path.write_text(json.dumps(task))
            with self.assertRaisesRegex(ValueError, "Source files changed"):
                reporting.summarize(cfg)
            duplicate = dict(cfg, subjects=[1, 1])
            with self.assertRaisesRegex(ValueError, "Duplicate configured task"):
                reporting._task_records(duplicate, audit_identity(duplicate))


if __name__ == "__main__":
    unittest.main()
