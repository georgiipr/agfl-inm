from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from inm.completion_transformer import protocol


ROOT = Path(__file__).resolve().parents[2]
DECLARATION = ROOT / "configs" / "completion-transformer.json"


class ProtocolTests(unittest.TestCase):
    def test_protocol_import_does_not_load_scientific_dependencies(self):
        check = (
            "import sys; import inm.completion_transformer.protocol; "
            "roots=('torch','numpy','scipy'); "
            "loaded=[name for name in sys.modules if any(name == root or name.startswith(root + '.') "
            "for root in roots)]; "
            "assert not loaded, loaded"
        )
        subprocess.run([sys.executable, "-c", check], check=True, cwd=ROOT,
                       capture_output=True, text=True)

    def config_copy(self, directory: Path, **changes):
        value = json.loads(DECLARATION.read_text(encoding="utf-8"))
        value.update(changes)
        path = directory / "study.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_fixed_declaration_plan_and_config_relative_paths(self):
        cfg = protocol.load_config(DECLARATION)
        self.assertEqual(len(protocol.tasks(cfg)), 27)
        self.assertEqual(protocol.tasks(cfg)[0], {"subject": 1, "seed": 0})
        self.assertEqual(protocol.tasks(cfg)[-1], {"subject": 9, "seed": 2})
        self.assertEqual(cfg["data_dir"], str((ROOT / "configs" / "../../ml").resolve()))
        self.assertEqual(cfg["output_dir"], str((ROOT / "configs" / "../results/completion-transformer-v1").resolve()))
        self.assertEqual(protocol.ARM_IDS, ("spatial_eegnet", "spatial_transformer"))
        self.assertEqual(protocol.STRATEGY_IDS, ("zero", "covariance", "tucker"))

    def test_synthetic_subset_and_output_occupancy_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            cfg_path = self.config_copy(directory, synthetic=True, subjects=[1], seeds=[0],
                data_dir="data", candidate_source_dir="transformer",
                split_source_dir="eegnet", output_dir="output")
            cfg = protocol.load_config(cfg_path)
            self.assertEqual(protocol.tasks(cfg), [{"subject": 1, "seed": 0}])
            output = Path(cfg["output_dir"])
            output.mkdir()
            (output / "old.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "already contains artifacts"):
                protocol.load_config(cfg_path)
            self.assertEqual(protocol.load_config(cfg_path, allow_existing_output=True)["output_dir"], str(output))

    def test_rejects_protocol_drift_duplicate_keys_and_bad_cohort(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            drift_path = self.config_copy(directory, protocol={"unexpected": True})
            with self.assertRaisesRegex(ValueError, "unknown settings forbidden"):
                protocol.load_config(drift_path)
            cohort_path = self.config_copy(directory, subjects=[1])
            with self.assertRaisesRegex(ValueError, "fixed cohort"):
                protocol.load_config(cohort_path)
            duplicate_path = directory / "duplicate.json"
            duplicate_path.write_text('{"schema_name":"a","schema_name":"b"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
                protocol.load_config(duplicate_path)

    def test_rejects_path_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            overlapping = self.config_copy(directory, synthetic=True, subjects=[1], seeds=[0],
                data_dir="same", candidate_source_dir="transformer",
                split_source_dir="eegnet", output_dir="same/output")
            with self.assertRaisesRegex(ValueError, "overlaps data_dir"):
                protocol.load_config(overlapping)

    def test_identity_and_inventory_are_deterministic_and_metadata_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            transformer = directory / "transformer"
            eegnet = directory / "eegnet"
            for source in (transformer, eegnet):
                source.mkdir()
                (source / "study.json").write_text('{"study_id":"fixture"}', encoding="utf-8")
            (transformer / "report").mkdir()
            (transformer / "report" / "evidence.json").write_text('{"status":"fixture"}', encoding="utf-8")
            cfg_path = self.config_copy(directory, synthetic=True, subjects=[1], seeds=[0],
                data_dir="data", candidate_source_dir="transformer",
                split_source_dir="eegnet", output_dir="output")
            cfg = protocol.load_config(cfg_path)
            identity_a = protocol.study_identity(cfg)
            identity_b = protocol.study_identity(cfg)
            self.assertEqual(identity_a["study_id"], identity_b["study_id"])
            self.assertEqual(len(identity_a["source_sha256"]), 64)
            self.assertIn("historical_manifest_sha256", identity_a)

            # Arbitrary bytes prove inventory hashes artifact files without
            # trying to deserialize them as checkpoints or prediction arrays.
            task = transformer / "tasks" / "A01_seed_0"
            model = task / "ARMS" / "spatial_transformer"
            model.mkdir(parents=True)
            eegnet_model = task / "ARMS" / "spatial_eegnet"
            eegnet_model.mkdir(parents=True)
            (task / "task.json").write_text("{}", encoding="utf-8")
            (task / "data_provenance.json").write_text("{}", encoding="utf-8")
            for arm_model in (model, eegnet_model):
                for name in ("checkpoint.pt", "result.json", "predictions.npz", "history.json"):
                    (arm_model / name).write_bytes(b"opaque fixture bytes")
            base_task = eegnet / "artifacts" / "A01_seed_0"
            base_task.mkdir(parents=True)
            for name in ("split.json", "dataset.json"):
                (base_task / name).write_text("{}", encoding="utf-8")
            inventory = protocol.inspect_inputs(cfg)
            self.assertEqual(inventory["status"], "unavailable")
            self.assertEqual(inventory["candidate_tasks"][0]["files"]["spatial_transformer_checkpoint"]["sha256"],
                             protocol.file_sha256(model / "checkpoint.pt"))
            self.assertEqual(inventory["candidate_tasks"][0]["files"]["spatial_eegnet_checkpoint"]["sha256"],
                             protocol.file_sha256(eegnet_model / "checkpoint.pt"))
            self.assertEqual(len(inventory["missing_recordings"]), 9)
            self.assertIn("were not loaded", " ".join(inventory["limitations"]))


if __name__ == "__main__":
    unittest.main()
