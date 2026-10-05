from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/completion-transformer.json"
PYTHON = ROOT / ".venv/bin/python"


class PublicCliTests(unittest.TestCase):
    def _run(self, *args):
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        return subprocess.run([str(PYTHON), "-m", "inm.completion_transformer", *map(str, args)],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)

    def _run_main_and_check_lazy_imports(self, *args):
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        script = ("import sys; from inm.completion_transformer.__main__ import main; "
                  "code = main(sys.argv[1:]); "
                  "assert not any(name in sys.modules for name in ('torch', 'numpy', 'scipy')); "
                  "raise SystemExit(code)")
        return subprocess.run([str(PYTHON), "-c", script, *map(str, args)], cwd=ROOT,
            env=env, capture_output=True, text=True, timeout=180)

    def test_plan_and_preflight_are_dependency_light(self):
        plan = self._run_main_and_check_lazy_imports("--config", CONFIG, "--plan")
        self.assertEqual(plan.returncode, 0, plan.stderr)
        data = json.loads(plan.stdout)
        self.assertEqual(data["tasks"], 27)
        self.assertEqual(data["cells_per_task"], 6)
        self.assertEqual(data["neural_fits"], 0)
        self.assertEqual(data["tucker_calibrations"], 27)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = json.loads(CONFIG.read_text())
            cfg.update(data_dir=str(root / "missing-recordings"),
                candidate_source_dir=str(root / "missing-candidates"),
                split_source_dir=str(root / "missing-splits"), output_dir=str(root / "new-output"))
            config_path = root / "missing-input-config.json"
            config_path.write_text(json.dumps(cfg, indent=2) + "\n")
            preflight = self._run_main_and_check_lazy_imports("--config", config_path, "--preflight")
            self.assertEqual(preflight.returncode, 2, preflight.stderr)
            inventory = json.loads(preflight.stdout)
            self.assertEqual(inventory["status"], "unavailable")
            self.assertEqual(len(inventory["missing_recordings"]), 9)
            self.assertTrue(inventory["missing_candidate_artifacts"])

    def test_synthetic_smoke_summary_resume_corruption_and_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = json.loads(CONFIG.read_text())
            cfg.update(subjects=[1], seeds=[0], synthetic=True,
                data_dir=str((ROOT / "ml").resolve()),
                candidate_source_dir=str((ROOT / "results/encoder-candidates-reproducible-v1").resolve()),
                split_source_dir=str((ROOT / "results/baselines-reproducible-v1").resolve()),
                output_dir=str(root / "smoke-output"))
            config_path = root / "synthetic-config.json"
            config_path.write_text(json.dumps(cfg, indent=2) + "\n")
            first = self._run("--config", config_path, "--smoke", "--output-dir", cfg["output_dir"])
            self.assertEqual(first.returncode, 0, first.stderr)
            result = json.loads(first.stdout)
            self.assertEqual(result["cells"], 126)
            self.assertEqual(result["report_status"], "partial")
            task_dir = Path(cfg["output_dir"]) / "tasks/A01_seed_0"
            task = json.loads((task_dir / "task.json").read_text())
            self.assertTrue(task["synthetic"])
            self.assertEqual(task["classifier_fits"], 0)
            self.assertEqual(task["synthetic_factor_epoch_override"], 1)

            summary = self._run("--config", config_path, "--summarize-only")
            self.assertEqual(summary.returncode, 0, summary.stderr)
            self.assertEqual(json.loads(summary.stdout)["valid_tasks"], 0)
            evidence = json.loads((Path(cfg["output_dir"]) / "report/evidence.json").read_text())
            self.assertEqual(len(evidence["synthetic_tasks_excluded"]), 1)
            self.assertEqual(evidence["status"], "partial")

            resumed = self._run("--config", config_path, "--smoke", "--resume-smoke", "--output-dir", cfg["output_dir"])
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            artifact = task_dir / "cells/spatial_transformer__zero__random_static_16__r0.npz"
            original = artifact.read_bytes()
            artifact.write_bytes(original + b"corrupt")
            corrupted = self._run("--config", config_path, "--smoke", "--resume-smoke", "--output-dir", cfg["output_dir"])
            self.assertNotEqual(corrupted.returncode, 0)
            self.assertIn("checksum mismatch", corrupted.stderr)
            artifact.write_bytes(original)
            task_path = task_dir / "task.json"
            task_original = task_path.read_bytes()
            failed_task = json.loads(task_original)
            failed_task["status"] = "failed"
            task_path.write_text(json.dumps(failed_task) + "\n")
            partial = self._run("--config", config_path, "--smoke", "--resume-smoke", "--output-dir", cfg["output_dir"])
            self.assertNotEqual(partial.returncode, 0)
            self.assertIn("refusing retry", partial.stderr)
            task_path.write_bytes(task_original)
            marker = Path(cfg["output_dir"]) / "study-identity.json"
            identity = json.loads(marker.read_text())
            identity["study_id"] = "0" * 64
            marker.write_text(json.dumps(identity) + "\n")
            changed = self._run("--config", config_path, "--smoke", "--resume-smoke", "--output-dir", cfg["output_dir"])
            self.assertNotEqual(changed.returncode, 0)
            self.assertIn("study identity differs", changed.stderr)


if __name__ == "__main__":
    unittest.main()
