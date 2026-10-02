"""Independent command-line integration checks for the baseline package."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
PYTHON = os.environ.get("AGFL_PYTHON", sys.executable)


class BaselineIntegrationTests(unittest.TestCase):
    def _cli(self, *args, cwd=ROOT):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        return subprocess.run([PYTHON, "-m", "inm.baselines", *map(str, args)],
                              cwd=cwd, env=env, text=True, capture_output=True,
                              timeout=120)

    def test_cli_plan_smoke_summary_and_checkpoint_reload(self):
        plan = self._cli("--config", ROOT / "configs/baselines.json", "--plan")
        self.assertEqual(plan.returncode, 0, plan.stderr)
        self.assertIn("Tasks: 27; fits: 108", plan.stdout)
        self.assertIn("coverage=full,random_static_16", plan.stdout)

        with tempfile.TemporaryDirectory(prefix="agfl-baseline-integration-") as temporary:
            temporary = Path(temporary)
            cfg = json.loads((ROOT / "configs/baselines.json").read_text())
            cfg["output_dir"] = str(temporary / "real-study")
            config_path = temporary / "baselines.json"
            config_path.write_text(json.dumps(cfg), encoding="utf-8")

            smoke = self._cli("--config", config_path, "--smoke")
            self.assertEqual(smoke.returncode, 0, smoke.stderr)
            smoke_summary = json.loads(smoke.stdout)
            self.assertEqual(smoke_summary["status"], "complete")
            self.assertIs(smoke_summary["synthetic"], True)
            smoke_dir = Path(smoke_summary["output_dir"])
            self.assertNotEqual(smoke_dir, Path(cfg["output_dir"]).resolve())
            result_path = smoke_dir / "artifacts/synthetic_A01_seed_0/ARMS/eegnet_reference__full/result.json"
            record = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertIs(record["synthetic"], True)
            self.assertEqual(record["status"], "complete")
            self.assertEqual(set(record["coverage"]), {
                "full", "random_static_16", "spatial_static_16", "dynamic_random_16",
                "dynamic_spatial_16", "random_static_11", "spatial_static_11",
                "dynamic_random_11", "dynamic_spatial_11", "random_static_6",
                "spatial_static_6", "dynamic_random_6", "dynamic_spatial_6"})

            checkpoint = result_path.parent / "checkpoint.pt"
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
            code = (
                "import torch; from inm.baselines.training import load_classifier; "
                f"a, pa = load_classifier({str(checkpoint)!r}); "
                f"b, pb = load_classifier({str(checkpoint)!r}); "
                "x=torch.zeros(2,22,4,250); m=torch.ones(2,22,4,dtype=torch.bool); "
                "assert pa['metadata']['synthetic'] is True; "
                "assert pa['metadata']['selected_epoch'] >= 1; "
                "assert torch.equal(a(x,m), b(x,m)); print('reload-ok')"
            )
            reload = subprocess.run([PYTHON, "-c", code], cwd=ROOT, env=env,
                                    text=True, capture_output=True, timeout=30)
            self.assertEqual(reload.returncode, 0, reload.stderr)
            self.assertIn("reload-ok", reload.stdout)

            summary = self._cli("--config", config_path, "--summarize-only")
            self.assertEqual(summary.returncode, 1, summary.stderr)
            progress = json.loads(summary.stdout)
            self.assertFalse(progress["complete"])
            self.assertEqual(progress["completed_fits"], 0)

    def test_missing_real_recording_fails_without_synthetic_or_complete_result(self):
        with tempfile.TemporaryDirectory(prefix="agfl-baseline-missing-data-") as temporary:
            temporary = Path(temporary)
            cfg = json.loads((ROOT / "configs/baselines.json").read_text())
            cfg.update({"subjects": [1], "seeds": [0],
                        "data_dir": str(temporary / "missing-data"),
                        "output_dir": str(temporary / "study"),
                        "execution_arms": ["eegnet_reference__full"]})
            config_path = temporary / "baselines.json"
            config_path.write_text(json.dumps(cfg), encoding="utf-8")

            preflight = self._cli("--config", config_path, "--preflight")
            self.assertEqual(preflight.returncode, 1)
            readiness = json.loads(preflight.stdout)
            self.assertEqual(readiness["readiness"], "blocked")
            self.assertTrue(any("Missing recording" in item for item in readiness["blockers"]))

            task = self._cli("--config", config_path, "--task-index", "0", "--device", "cpu")
            self.assertEqual(task.returncode, 2)
            self.assertIn("Task failed", task.stderr)
            output = Path(cfg["output_dir"])
            manifest = json.loads((output / "study.json").read_text())
            self.assertIs(manifest["synthetic"], False)
            task_dir = output / "artifacts/A01_seed_0"
            failure = json.loads((task_dir / "failure.json").read_text())
            self.assertEqual(failure["status"], "failed")
            self.assertIs(failure["synthetic"], False)
            self.assertFalse(list(output.rglob("result.json")))

    def test_cuda_request_does_not_fall_back_to_cpu(self):
        import torch
        from inm.baselines.protocol import load_config
        from inm.baselines.study import run_task

        with tempfile.TemporaryDirectory(prefix="agfl-baseline-cuda-guard-") as temporary:
            config = json.loads((ROOT / "configs/baselines.json").read_text())
            config.update({"subjects": [1], "seeds": [0],
                           "output_dir": str(Path(temporary) / "study"),
                           "execution_arms": ["eegnet_reference__full"]})
            config_path = Path(temporary) / "baselines.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            cfg = load_config(config_path)
            with mock.patch.object(torch.cuda, "is_available", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "CUDA was requested but is unavailable"):
                    run_task(cfg, 0, device="cuda")
            self.assertFalse(Path(cfg["output_dir"]).exists())


if __name__ == "__main__":
    unittest.main()
