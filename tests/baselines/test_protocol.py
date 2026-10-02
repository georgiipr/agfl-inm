"""Configuration and dependency-light planning checks for baseline studies."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from inm.baselines.protocol import arms, load_config, tasks


ROOT = Path(__file__).resolve().parents[2]
PYTHON = os.environ.get("AGFL_PYTHON", sys.executable)


class BaselineProtocolTests(unittest.TestCase):
    def test_both_declared_configs_validate_and_resolve_from_config_directory(self):
        for config_name, output in (("baselines.json", "baselines-v1"),
                                    ("baselines-cross-session.json", "baselines-cross-session-v1")):
            with self.subTest(config=config_name):
                cfg = load_config(ROOT / "configs" / config_name)
                self.assertEqual(cfg["data_dir"], str((ROOT.parent / "ml").resolve()))
                self.assertEqual(cfg["output_dir"], str((ROOT / "results" / output).resolve()))
                self.assertEqual(len(tasks(cfg)), 27)
                self.assertEqual(cfg["protocol"], "cross_session" if "cross" in config_name else "within_session")

    def test_default_matrix_and_full_only_classical_coverage(self):
        cfg = load_config(ROOT / "configs" / "baselines.json")
        matrix = arms(cfg)
        self.assertEqual([arm["name"] for arm in matrix], [
            "eegnet_reference__full", "masked_eegnet__full", "masked_eegnet__mixed",
            "covariance__full",
        ])
        self.assertEqual(len(matrix) * len(tasks(cfg)), 108)
        self.assertEqual(matrix[-1]["coverage"], ["full"])
        for arm in matrix[:3]:
            self.assertEqual(len(arm["coverage"]), 13)
            self.assertEqual(set(arm["coverage"]), {
                "full", "random_static_16", "spatial_static_16", "dynamic_random_16",
                "dynamic_spatial_16", "random_static_11", "spatial_static_11",
                "dynamic_random_11", "dynamic_spatial_11", "random_static_6",
                "spatial_static_6", "dynamic_random_6", "dynamic_spatial_6",
            })

    def test_rejects_unknown_options_invalid_dimensions_and_ambiguous_selection(self):
        original = json.loads((ROOT / "configs" / "baselines.json").read_text())
        malformed = []
        value = json.loads(json.dumps(original)); value["unexpected"] = True; malformed.append(value)
        value = json.loads(json.dumps(original)); value["preprocessing"]["trial_samples"] = 999; malformed.append(value)
        value = json.loads(json.dumps(original)); value["model"]["spatial_kernel"] = [21, 1]; malformed.append(value)
        value = json.loads(json.dumps(original)); value["selection"] = {"policy": "robust"}; malformed.append(value)
        value = json.loads(json.dumps(original)); value["arms"][0]["regime"] = "mixed"; malformed.append(value)
        for candidate in malformed:
            with self.subTest(candidate=candidate):
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "config.json"
                    path.write_text(json.dumps(candidate))
                    with self.assertRaises(ValueError):
                        load_config(path)

    def test_plan_runs_from_other_cwd_without_torch(self):
        code = (
            f"import builtins, runpy, sys; sys.path.insert(0, {str(ROOT)!r}); "
            "original=builtins.__import__; "
            "builtins.__import__=lambda name,*a,**k: (_ for _ in ()).throw("
            "AssertionError('torch imported')) if name == 'torch' or name.startswith('torch.') "
            "else original(name,*a,**k); "
            f"sys.argv=['inm.baselines','--config','{ROOT / 'configs/baselines.json'}','--plan']; "
            "runpy.run_module('inm.baselines', run_name='__main__')"
        )
        with tempfile.TemporaryDirectory() as other_cwd:
            result = subprocess.run([PYTHON, "-c", code], cwd=other_cwd, text=True,
                                    capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Tasks: 27; fits: 108", result.stdout)
        self.assertIn("Arm covariance__full: coverage=full", result.stdout)
        self.assertIn(str((ROOT.parent / "ml").resolve()), result.stdout)


if __name__ == "__main__":
    unittest.main()
