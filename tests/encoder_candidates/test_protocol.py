import builtins
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from inm.encoder_candidates import protocol
from inm.encoder_candidates.__main__ import main


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "encoder-candidates.json"


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.cfg = protocol.load_config(CONFIG)
        self.raw = json.loads(CONFIG.read_text(encoding="utf-8"))

    def write_config(self, value, directory):
        path = Path(directory) / "config.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_fixed_fit_matrix_and_arm_order(self):
        self.assertEqual(len(protocol.tasks(self.cfg)), 27)
        self.assertEqual(protocol.tasks(self.cfg)[:4], [
            {"subject": 1, "seed": 0}, {"subject": 1, "seed": 1},
            {"subject": 1, "seed": 2}, {"subject": 2, "seed": 0}])
        self.assertEqual(protocol.arms(self.cfg), list(protocol.ARM_IDS))
        self.assertEqual(len(protocol.tasks(self.cfg)) * len(protocol.arms(self.cfg)), 135)

    def test_plan_runs_without_numerical_imports(self):
        original_import = builtins.__import__
        numerical = ("numpy", "scipy", "torch", "sklearn", "mne")

        def guarded(name, *args, **kwargs):
            if name.split(".", 1)[0] in numerical:
                raise AssertionError(f"numerical import attempted: {name}")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=guarded), \
             mock.patch("sys.stdout", new_callable=__import__("io").StringIO) as output:
            self.assertEqual(main(["--config", str(CONFIG), "--plan"]), 0)
        for phrase in ("27", "135", "108", "Tucker fits: 0", "No fits or measurements"):
            self.assertIn(phrase, output.getvalue())

    def test_unknown_keys_settings_arms_regime_and_test_are_rejected(self):
        cases = []
        changed = copy.deepcopy(self.raw); changed["unreviewed"] = 1; cases.append(changed)
        changed = copy.deepcopy(self.raw); changed["training"]["future_budget"] = 4; cases.append(changed)
        changed = copy.deepcopy(self.raw); changed["arms"][0]["id"] = "mystery"; cases.append(changed)
        changed = copy.deepcopy(self.raw); changed["arms"][0]["encoder_settings"]["f1"] = 8; cases.append(changed)
        changed = copy.deepcopy(self.raw); changed["partitions"] = ["train", "validation", "test"]; cases.append(changed)
        changed = copy.deepcopy(self.raw); changed["training_regime"] = "mixed"; cases.append(changed)
        changed = copy.deepcopy(self.raw); changed["training"]["maximum_epochs"] = 251; cases.append(changed)
        with tempfile.TemporaryDirectory() as temporary:
            for index, value in enumerate(cases):
                with self.subTest(index=index):
                    subdir = Path(temporary) / str(index); subdir.mkdir()
                    path = self.write_config(value, subdir)
                    with self.assertRaises(ValueError):
                        protocol.load_config(path)

    def test_duplicate_ids_and_duplicate_json_keys_are_rejected(self):
        for field in ("subjects", "seeds"):
            changed = copy.deepcopy(self.raw); changed[field][-1] = changed[field][0]
            with tempfile.TemporaryDirectory() as temporary:
                with self.assertRaises(ValueError):
                    protocol.load_config(self.write_config(changed, temporary))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "duplicate.json"
            path.write_text('{"schema_name":"wrong","schema_name":"wrong"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
                protocol.load_config(path)

    def test_paths_reject_input_overlap_and_symlink_alias(self):
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            changed = copy.deepcopy(self.raw)
            changed["data_dir"] = str(temp / "input")
            changed["split_source_dir"] = str(temp / "splits")
            changed["output_dir"] = str(temp / "input" / "nested")
            with self.assertRaisesRegex(ValueError, "overlaps data_dir"):
                protocol.load_config(self.write_config(changed, temp))
            (temp / "input").mkdir()
            (temp / "splits").mkdir()
            alias = temp / "alias"
            alias.symlink_to(temp / "input", target_is_directory=True)
            changed["output_dir"] = str(alias)
            with self.assertRaisesRegex(ValueError, "overlaps data_dir"):
                protocol.load_config(self.write_config(changed, temp))

    def test_invalid_and_nonempty_output_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            changed = copy.deepcopy(self.raw)
            changed["output_dir"] = " "
            with self.assertRaisesRegex(ValueError, "output_dir"):
                protocol.load_config(self.write_config(changed, temp))
            occupied = temp / "occupied"; occupied.mkdir(); (occupied / "old").touch()
            changed["output_dir"] = str(occupied)
            with self.assertRaisesRegex(ValueError, "already contains"):
                protocol.load_config(self.write_config(changed, temp))

    def test_only_fixed_real_cohort_is_accepted(self):
        changed = copy.deepcopy(self.raw); changed["subjects"] = [1]
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "only synthetic fixtures"):
                protocol.load_config(self.write_config(changed, temporary))

    def test_explicit_synthetic_fixture_may_shrink_cohort_and_budget(self):
        changed = copy.deepcopy(self.raw)
        changed["synthetic"] = True
        changed["subjects"] = [1]
        changed["seeds"] = [0]
        changed["training"].update({"maximum_epochs": 2, "minimum_epochs": 1,
                                    "patience": 0, "batch_size": 2, "warmup_epochs": 0})
        with tempfile.TemporaryDirectory() as temporary:
            cfg = protocol.load_config(self.write_config(changed, temporary))
        self.assertTrue(cfg["synthetic"])
        self.assertEqual(len(protocol.tasks(cfg)), 1)

    def test_identity_tracks_config_bytes_and_source_map(self):
        first = protocol.study_identity(self.cfg)
        with tempfile.TemporaryDirectory() as temporary:
            cfg_path = Path(temporary) / "same.json"
            cfg_path.write_text(CONFIG.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            raw_cfg = protocol.load_config(cfg_path)
            raw_cfg["config_path"] = str(cfg_path)
            # Keep resolved values from the original config: only raw bytes change.
            raw_cfg.update({key: self.cfg[key] for key in protocol.PATH_KEYS})
            second = protocol.study_identity(raw_cfg)
        self.assertNotEqual(first["config_sha256"], second["config_sha256"])
        self.assertNotEqual(first["study_id"], second["study_id"])
        with mock.patch.object(protocol, "_source_hashes", return_value={"changed.py": "0" * 64}):
            third = protocol.study_identity(self.cfg)
        self.assertNotEqual(first["source_sha256"], third["source_sha256"])
        self.assertNotEqual(first["study_id"], third["study_id"])

    def test_preflight_is_inventory_only_and_reports_missing_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            changed = copy.deepcopy(self.raw)
            changed["data_dir"] = str(temp / "missing_data")
            changed["split_source_dir"] = str(temp / "missing_splits")
            changed["output_dir"] = str(temp / "output")
            cfg = protocol.load_config(self.write_config(changed, temp))
            result = protocol.inspect_inputs(cfg)
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(len(result["recordings"]), 9)
        self.assertEqual(len(result["splits"]), 27)
        self.assertEqual(len(result["missing_recordings"]), 9)
        self.assertIn("not an input", result["historical_checkpoint_source_mismatch"])
        self.assertIn("were not loaded", result["limits"][0])

    def test_smoke_is_implemented_and_summarize_reports_partial_without_evidence(self):
        with mock.patch("sys.stdout", new_callable=__import__("io").StringIO) as output:
            self.assertEqual(main(["--config", str(CONFIG), "--smoke"]), 0)
        self.assertIn('"synthetic": true', output.getvalue())
        self.assertIn('"fits": 5', output.getvalue())
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            raw = copy.deepcopy(self.raw)
            raw.update({"data_dir": str(temp / "data"),
                        "split_source_dir": str(temp / "splits"),
                        "output_dir": str(temp / "output")})
            config_path = temp / "study.json"
            config_path.write_text(json.dumps(raw), encoding="utf-8")
            with mock.patch("sys.stdout", new_callable=__import__("io").StringIO) as output:
                self.assertEqual(main(["--config", str(config_path), "--summarize-only"]), 1)
            self.assertIn('"status": "partial"', output.getvalue())


if __name__ == "__main__":
    unittest.main()
