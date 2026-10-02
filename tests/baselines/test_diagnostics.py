import json
from pathlib import Path
import tempfile
import unittest

from inm.baselines.diagnostics import inspect_environment, inspect_legacy


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def small_study(root: Path):
    output = root / "study"
    output.mkdir()
    write_json(output / "study.json", {
        "study_id": "study-id", "source": {"source_sha256": "source-id"},
        "tasks": [[1, 0]],
        "arms": [
            {"name": "mha__baseline__full", "representation": "baseline"},
            {"name": "mha__tensor_core__full", "representation": "tensor_core"},
        ],
    })
    task = output / "artifacts" / "A01_seed_0"
    write_json(task / "split.json", {"split_id": "split-id", "fingerprint": "data-id", "protocol": "stratified"})
    write_json(task / "calibration.json", {
        "identity": {"study_id": "study-id", "subject": 1, "seed": 0,
                     "dataset_fingerprint": "data-id", "split_id": "split-id",
                     "calibration_sha256": "calibration-id"},
        "encoder_selection": {"selection": "validation_balanced_accuracy_then_log_loss",
                              "best_epoch": 7,
                              "selected_validation": {"balanced_accuracy": 0.5, "log_loss": 1.0}},
    })
    return output, task


def result(arm, status="complete"):
    return {"status": status,
            "identity": {"study_id": "study-id", "subject": 1, "seed": 0,
                         "dataset_fingerprint": "data-id", "split_id": "split-id",
                         "calibration_sha256": "calibration-id"},
            "arm": {"name": arm, "attention": "mha",
                    "representation": "baseline" if "baseline" in arm else "tensor_core",
                    "regime": "full"},
            "metrics": [
                {"partition": "validation", "scenario": "full_22", "accuracy": 0.7,
                 "balanced_accuracy": 0.6, "f1_macro": 0.5},
                {"partition": "test", "scenario": "full_22", "accuracy": 0.4,
                 "balanced_accuracy": 0.3, "f1_macro": 0.2},
            ]}


class DiagnosticsTests(unittest.TestCase):
    def test_missing_results_are_unavailable(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = inspect_legacy(Path(temporary) / "absent")
        self.assertEqual(report["status"], "unavailable")
        self.assertEqual(report["completion"]["state"], "unavailable")
        self.assertEqual(report["tasks"], [])
        self.assertEqual(report["findings"][0]["status"], "unavailable")

    def test_corrupt_record_is_visible_and_not_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, task = small_study(Path(temporary))
            result_path = task / "ARMS" / "mha__baseline__full" / "result.json"
            result_path.parent.mkdir(parents=True)
            result_path.write_text("{broken", encoding="utf-8")
            report = inspect_legacy(output)
        self.assertEqual(report["status"], "corrupt")
        self.assertTrue(any(item["status"] == "corrupt" for item in report["findings"]))
        self.assertNotEqual(report["completion"]["state"], "complete")

    def test_partial_results_are_not_completed(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, task = small_study(Path(temporary))
            write_json(task / "ARMS" / "mha__baseline__full" / "result.json",
                       result("mha__baseline__full"))
            report = inspect_legacy(output)
        self.assertEqual(report["completion"], {"state": "partial", "tasks_complete": 0,
                                                 "tasks_expected": 1})
        self.assertEqual(report["tasks"][0]["status"], "partial")

    def test_encoder_validation_is_distinct_from_downstream_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, task = small_study(Path(temporary))
            for name in ("mha__baseline__full", "mha__tensor_core__full"):
                write_json(task / "ARMS" / name / "result.json", result(name))
            report = inspect_legacy(output)
        item = report["tasks"][0]
        self.assertEqual(report["completion"]["state"], "complete")
        self.assertEqual(item["encoder_selection"]["partition"], "VALIDATION")
        self.assertEqual(item["encoder_selection"]["selected_epoch"], 7)
        scores = item["arms"][0]["full_input_scores"]
        self.assertEqual([row["partition"] for row in scores], ["VALIDATION", "TEST"])
        self.assertEqual(item["arms"][0]["identity"]["split_id"], "split-id")

    def test_environment_reports_paths_without_loading_numerical_packages(self):
        report = inspect_environment()
        self.assertIn("torch", report["packages"])
        self.assertIn("mne", report["packages"])
        self.assertTrue(Path(report["config_path"]).is_file())
        self.assertEqual(len(report["expected_recordings"]), 9)
        self.assertTrue(all("exists" in item for item in report["expected_recordings"]))

    def test_environment_resolves_baseline_paths_from_config_directory(self):
        report = inspect_environment(Path(__file__).resolve().parents[2] / "configs" / "baselines-cross-session.json")
        root = Path(__file__).resolve().parents[2]
        self.assertEqual(report["input_dir"], str((root.parent / "ml").resolve()))
        self.assertEqual(report["output_dir"], str((root / "results" / "baselines-cross-session-v1").resolve()))
        self.assertEqual(len(report["expected_recordings"]), 18)
        self.assertTrue(report["external_labels_required"])
        self.assertIn("external_labels_dir", report["blockers"][-1])


if __name__ == "__main__":
    unittest.main()
