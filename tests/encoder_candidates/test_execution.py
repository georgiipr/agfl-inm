"""Public subprocess checks for safe candidate CLI resume and identity gates."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = ROOT / "configs/encoder-candidates-reproducible.json"


class CandidateCliExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "out"
        self.config_path = self.root / "candidate.json"
        cfg = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
        cfg.update({"synthetic": True, "subjects": [1, 2], "seeds": [0],
                    "data_dir": str(self.root / "data"),
                    "split_source_dir": str(self.root / "splits"),
                    "output_dir": str(self.output)})
        cfg["training"].update({"maximum_epochs": 2, "minimum_epochs": 1,
                                "patience": 0, "batch_size": 2,
                                "warmup_epochs": 0})
        self.config = cfg
        self.write_config()

    def write_config(self, cfg=None):
        self.config_path.write_text(json.dumps(cfg or self.config, indent=2) + "\n",
                                    encoding="utf-8")

    def cli(self, *args, expected=0):
        env = os.environ.copy()
        env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1"})
        result = subprocess.run(
            [sys.executable, "-m", "inm.encoder_candidates", "--config",
             str(self.config_path), *args], cwd=ROOT, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=240)
        self.assertEqual(result.returncode, expected,
                         f"CLI {args} returned {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
        return result

    def test_two_tasks_resume_summary_and_synthetic_exclusion(self):
        first = self.cli("--task-index", "0")
        self.assertIn('"status": "complete"', first.stdout)
        self.assertIn('"synthetic": true', json.dumps(
            json.loads((self.output / "tasks/A01_seed_0/task.json").read_text())))
        second = self.cli("--task-index", "1")
        self.assertIn('"subject": 2', second.stdout)
        resumed = self.cli("--task-index", "0")
        self.assertIn('"status": "complete"', resumed.stdout)
        summary = self.cli("--summarize-only", expected=1)
        self.assertIn('"status": "partial"', summary.stdout)
        self.assertIn("synthetic", summary.stdout.lower())

    def test_config_source_package_corruption_and_partial_output_fail_closed(self):
        self.cli("--task-index", "0")
        task_path = self.output / "tasks/A01_seed_0/task.json"
        original = json.loads(task_path.read_text(encoding="utf-8"))

        changed_config = copy.deepcopy(self.config)
        changed_config["name"] += "_changed"
        self.write_config(changed_config)
        self.cli("--task-index", "0", expected=2)

        self.write_config()
        for field, replacement in (("source_files_sha256", {"altered.py": "0" * 64}),
                                   ("packages", {"torch": "mismatch"})):
            manifest = json.loads(task_path.read_text(encoding="utf-8"))
            manifest[field] = replacement
            task_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.cli("--task-index", "0", expected=2)
        task_path.write_text(json.dumps(original), encoding="utf-8")

        fit_path = self.output / "tasks/A01_seed_0/ARMS/local_control/result.json"
        fit_path.write_text(fit_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
        self.cli("--task-index", "0", expected=2)

        partial = self.output / "tasks/A02_seed_0"
        partial.mkdir(parents=True)
        (partial / "orphan.bin").write_bytes(b"partial")
        self.cli("--task-index", "1", expected=2)


if __name__ == "__main__":
    unittest.main()
