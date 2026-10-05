from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from inm.task_driven_completion import protocol

ROOT = Path(__file__).resolve().parents[2]
DECLARATION = ROOT / "configs/task-driven-completion.json"


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.base = json.loads(DECLARATION.read_text())
        self.base.update(synthetic=True, subjects=[1], seeds=[0], data_dir="data",
                         candidate_source_dir="candidate", split_source_dir="splits",
                         output_dir="output")

    def config(self, **changes):
        value = copy.deepcopy(self.base)
        value.update(changes)
        path = self.directory / "study.json"
        path.write_text(json.dumps(value))
        return path

    def test_fixed_cohort_counts_and_scientific_contract(self):
        cfg = protocol.load_config(DECLARATION)
        planned = protocol.plan(cfg)
        self.assertEqual(planned["tasks"], 27)
        self.assertEqual(planned["supervised_completion_fits"], 54)
        self.assertEqual(planned["backbone_fits"], 0)
        self.assertEqual(planned["strategies_per_task"], 5)
        self.assertEqual(planned["evaluation_rows_per_strategy"], 21)
        self.assertEqual(protocol.tasks(cfg)[0], {"subject": 1, "seed": 0})
        self.assertEqual(protocol.tasks(cfg)[-1], {"subject": 9, "seed": 2})
        self.assertEqual(protocol.ARM_IDS, ("spatial_transformer",))
        self.assertEqual(protocol.HISTORICAL_ARM_IDS, ("spatial_eegnet", "spatial_transformer"))
        self.assertEqual(protocol.STRATEGY_IDS,
                         ("zero", "covariance_frozen", "covariance_learned", "tucker_frozen", "tucker_learned"))
        science = cfg["protocol"]
        self.assertEqual(science["execution"]["device"], "cpu")
        self.assertEqual(science["execution"]["threads"], 1)
        self.assertEqual(science["tucker"]["rank_channels"], 4)
        self.assertEqual(science["tucker"]["rank_features"], 16)
        self.assertEqual(science["tucker"]["ridge_penalty"], "n_observed*250*0.001")
        self.assertEqual(science["training"]["reconstruction_weight"], 0.1)
        self.assertEqual(science["training"]["anchor_weight"], 1e-4)
        self.assertEqual(science["selection"]["metric"], "mean_degraded_balanced_accuracy")
        self.assertTrue(science["selection"]["epoch_zero_eligible"])
        self.assertEqual(science["selection"]["minimum_training_epochs"], 10)
        self.assertEqual(science["selection"]["conditions"], [row["name"] for row in protocol.CONDITIONS[1:]])
        self.assertFalse(science["selection"]["test_data"])
        self.assertFalse(planned["real_fitting_ready"])

    def test_config_paths_are_relative_to_config_not_cwd(self):
        cfg = protocol.load_config(self.config())
        for key, relative in (("data_dir", "data"), ("candidate_source_dir", "candidate"),
                              ("split_source_dir", "splits"), ("output_dir", "output")):
            self.assertEqual(cfg[key], str(self.directory / relative))
        result = subprocess.run([sys.executable, "-m", "inm.task_driven_completion", "--config",
                                 str(DECLARATION), "--plan"], cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["supervised_completion_fits"], 54)
        self.assertFalse((self.directory / "output").exists())

    def test_every_scientific_leaf_is_fixed_including_types(self):
        def leaves(value, prefix=()):
            if isinstance(value, dict):
                for key, item in value.items():
                    yield from leaves(item, (*prefix, key))
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    yield from leaves(item, (*prefix, index))
            else:
                yield prefix, value
        for keys, value in leaves(self.base["protocol"]):
            with self.subTest(field=keys):
                changed = copy.deepcopy(self.base["protocol"])
                parent = changed
                for key in keys[:-1]:
                    parent = parent[key]
                parent[keys[-1]] = "protocol-drift" if not isinstance(value, str) else value + "_drift"
                with self.assertRaisesRegex(ValueError, "fixed protocol"):
                    protocol.load_config(self.config(protocol=changed))
        changed = copy.deepcopy(self.base["protocol"])
        changed["execution"]["threads"] = True
        with self.assertRaisesRegex(ValueError, "fixed protocol"):
            protocol.load_config(self.config(protocol=changed))
        changed["execution"]["threads"] = 1.0
        with self.assertRaisesRegex(ValueError, "fixed protocol"):
            protocol.load_config(self.config(protocol=changed))

    def test_unknown_missing_and_duplicate_settings_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "unknown settings"):
            protocol.load_config(self.config(extra="forbidden"))
        for malformed in ({}, {**self.base["protocol"], "extra": False}):
            with self.assertRaisesRegex(ValueError, "unknown settings"):
                protocol.load_config(self.config(protocol=malformed))
        path = self.config()
        path.write_text('{"schema_name":"one","schema_name":"two"}')
        with self.assertRaisesRegex(ValueError, "Duplicate JSON"):
            protocol.load_config(path)
        for constant in ("NaN", "Infinity", "-Infinity"):
            path.write_text('{"value":' + constant + '}')
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                protocol.load_config(path)
        path.write_text('[]')
        with self.assertRaisesRegex(ValueError, "unknown settings"):
            protocol.load_config(path)

    def test_cohort_and_synthetic_types_are_strict(self):
        for changes in ({"synthetic": 1}, {"subjects": [True]}, {"seeds": [0.0]},
                        {"subjects": [1, 1]}, {"seeds": [3]}, {"subjects": []},
                        {"subjects": [10]}, {"seeds": [-1]}, {"synthetic": False}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                protocol.load_config(self.config(**changes))
        cfg = protocol.load_config(self.config(subjects=[1, 9], seeds=[0, 2]))
        self.assertEqual(len(protocol.tasks(cfg)), 4)
        self.assertTrue(cfg["synthetic"])

    def test_output_overlap_both_directions_and_source_protection(self):
        for key in protocol.PATH_KEYS[:-1]:
            for candidate in (self.base[key], self.base[key] + "/child"):
                with self.subTest(path=candidate), self.assertRaisesRegex(ValueError, "overlaps"):
                    protocol.load_config(self.config(output_dir=candidate))
        with self.assertRaisesRegex(ValueError, "overlaps"):
            protocol.load_config(self.config(data_dir="output/child"))
        for output in (ROOT, ROOT.parent, ROOT / "inm/new-output", ROOT / "configs/new-output",
                       ROOT / "docs/new-output", ROOT / ".session-runs/new-output",
                       ROOT / "results", ROOT / "results/supervised-tucker-v1/nested"):
            with self.subTest(path=output), self.assertRaisesRegex(ValueError, "output_dir"):
                protocol.load_config(self.config(output_dir=str(output)))
        with self.assertRaisesRegex(ValueError, "config_path"):
            protocol.load_config(self.config(data_dir="/tmp/other-data", candidate_source_dir="/tmp/other-candidates",
                                             split_source_dir="/tmp/other-splits", output_dir="."))

    def test_output_symlinks_cannot_escape_path_protection(self):
        (self.directory / "source-link").symlink_to(ROOT / "inm", target_is_directory=True)
        (self.directory / "data-link").symlink_to(self.directory / "data", target_is_directory=True)
        for output in ("source-link/output", "data-link/output"):
            with self.subTest(path=output), self.assertRaisesRegex(ValueError, "overlaps"):
                protocol.load_config(self.config(output_dir=output))
        (self.directory / "historical-link").symlink_to(ROOT / "results", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "historical results"):
            protocol.load_config(self.config(output_dir="historical-link/completion-transformer-v1/new"))

    def test_output_occupancy_is_not_resume_authorization(self):
        config = self.config()
        output = self.directory / "output"
        output.mkdir()
        protocol.load_config(config)
        (output / "foreign.json").write_text('{}')
        with self.assertRaisesRegex(ValueError, "fresh output identity"):
            protocol.load_config(config)
        cfg = protocol.load_config(config, allow_existing_output=True)
        self.assertEqual(cfg["output_dir"], str(output))
        (self.directory / "file-output").write_text("not a directory")
        with self.assertRaisesRegex(ValueError, "fresh output identity"):
            protocol.load_config(self.config(output_dir="file-output"), allow_existing_output=True)

    def test_existing_output_symlink_descendants_rejected_even_for_inspection(self):
        output = self.directory / "output"
        output.mkdir()
        (output / "task").symlink_to(ROOT / "results/completion-transformer-v1", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink descendant"):
            protocol.load_config(self.config(), allow_existing_output=True)

    def test_missing_empty_or_aliased_paths_rejected(self):
        for value in ("", "  ", None, 42, "bad\x00path"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                protocol.load_config(self.config(data_dir=value))
        with self.assertRaisesRegex(ValueError, "distinct"):
            protocol.load_config(self.config(data_dir="candidate"))
        with self.assertRaisesRegex(ValueError, "cannot read"):
            protocol.load_config(self.directory / "absent.json")

    def test_identity_tracks_config_source_packages_and_historical_metadata(self):
        path = self.config()
        cfg = protocol.load_config(path)
        identity = protocol.study_identity(cfg)
        self.assertEqual(identity, protocol.study_identity(cfg))
        source = identity["source_files_sha256"]
        self.assertIn("inm/task_driven_completion/protocol.py", source)
        self.assertIn("inm/completion_transformer/replay.py", source)
        self.assertIn("inm/encoder_candidates/models.py", source)
        self.assertIn("agfl/datasets/eeg.py", source)
        path.write_text(path.read_text() + "\n")
        self.assertNotEqual(identity["study_id"], protocol.study_identity(cfg)["study_id"])
        with patch.object(protocol, "_source_hashes", return_value={**source, "changed.py": "0" * 64}):
            changed = protocol.study_identity(cfg)
        self.assertNotEqual(changed["source_sha256"], identity["source_sha256"])
        with patch.object(protocol, "_package_versions", return_value={"torch": "changed"}):
            changed_packages = protocol.study_identity(cfg)
        self.assertNotEqual(changed["study_id"], changed_packages["study_id"])
        manifest = self.directory / "candidate/report/evidence.json"
        manifest.parent.mkdir(parents=True)
        before = protocol.study_identity(cfg)
        manifest.write_text('{"not_a_verified_manifest":true}')
        after = protocol.study_identity(cfg)
        self.assertNotEqual(before["study_id"], after["study_id"])
        self.assertEqual(after["historical_manifest_sha256"][str(manifest)], protocol.file_sha256(manifest))

    def test_preflight_hashes_opaque_inputs_without_claiming_replay(self):
        cfg = protocol.load_config(self.config())
        files = [self.directory / "candidate/report/evidence.json", self.directory / "splits/study.json"]
        task = self.directory / "candidate/tasks/A01_seed_0"
        files += [task / item for item in ("task.json", "data_provenance.json")]
        files += [task / "ARMS" / arm / item for arm in protocol.HISTORICAL_ARM_IDS
                  for item in ("checkpoint.pt", "result.json", "predictions.npz", "history.json")]
        files += [self.directory / "splits/artifacts/A01_seed_0" / item for item in ("split.json", "dataset.json")]
        files += [self.directory / "data" / f"A{i:02d}T.gdf" for i in range(1, 10)]
        for file in files:
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(b"opaque non-numeric fixture")
        with patch("inm.completion_transformer.protocol._package_versions",
                   return_value={name: "fixture" for name in protocol.PACKAGE_NAMES}):
            inventory = protocol.inspect_inputs(cfg)
        self.assertEqual(inventory["status"], "inventory_complete")
        self.assertFalse(inventory["real_fitting_ready"])
        self.assertFalse(inventory["cuda_verified"])
        self.assertEqual(inventory["historical_verification"], "not_run")
        self.assertEqual(inventory["original_full_input_replay"], "not_run")
        for arm in protocol.HISTORICAL_ARM_IDS:
            checkpoint = task / "ARMS" / arm / "checkpoint.pt"
            self.assertEqual(inventory["candidate_tasks"][0]["files"][arm + "_checkpoint"]["sha256"],
                             protocol.file_sha256(checkpoint))
        self.assertFalse((self.directory / "output").exists())

    def test_import_plan_and_preflight_are_dependency_light(self):
        config_path = self.config()
        code = """
import importlib.abc, sys
class NoScientificImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'torch', 'scipy', 'sklearn', 'mne'}:
            raise AssertionError('scientific import: ' + fullname)
sys.meta_path.insert(0, NoScientificImports())
from inm.task_driven_completion.__main__ import main
raise SystemExit(main(sys.argv[1:]))
"""
        for mode, expected_exit in (("--plan", 0), ("--preflight", 2)):
            result = subprocess.run([sys.executable, "-c", code, "--config", str(config_path), mode],
                                    cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, expected_exit, result.stderr)
            parsed = json.loads(result.stdout)
            self.assertFalse(parsed["real_fitting_ready"])
        self.assertFalse((self.directory / "output").exists())

    def test_cli_cannot_fit_or_offer_cuda_and_invalid_config_fails(self):
        for arguments in (("--task-index", "0"), ("--plan", "--device", "cuda"),
                          ("--plan", "--preflight")):
            result = subprocess.run([sys.executable, "-m", "inm.task_driven_completion", "--config",
                                     str(self.config()), *arguments], cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(result.returncode, 2)
        result = subprocess.run([sys.executable, "-m", "inm.task_driven_completion", "--config",
                                 str(self.config(extra=0)), "--plan"], cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("unknown settings", result.stderr)


if __name__ == "__main__":
    unittest.main()
