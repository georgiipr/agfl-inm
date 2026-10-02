import json
from pathlib import Path
import tempfile
import unittest

from inm.baselines.diagnostics import inspect_legacy


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(root: Path, *, calibration=True):
    output = root / "study"
    task = output / "artifacts" / "A01_seed_0"
    write_json(output / "study.json", {
        "study_id": "study-id", "tasks": [[1, 0]],
        "arms": [{"name": "mha__baseline__full", "representation": "baseline"},
                 {"name": "mha__tensor_core__full", "representation": "tensor_core"}],
    })
    write_json(task / "split.json", {"split_id": "split-id", "fingerprint": "data-id"})
    if calibration:
        write_json(task / "calibration.json", {
            "identity": {"study_id": "study-id", "subject": 1, "seed": 0,
                         "dataset_fingerprint": "data-id", "split_id": "split-id"},
            "cache_sha256": "cal-id",
            "encoder_selection": {"best_epoch": 2, "selected_validation": {
                "accuracy": 0.6, "balanced_accuracy": 0.55, "log_loss": 1.1}},
        })
        write_json(task / "encoder_history.json", [
            {"epoch": 1, "validation": {"balanced_accuracy": 0.5}},
            {"epoch": 2, "validation": {"accuracy": 0.6, "balanced_accuracy": 0.55, "log_loss": 1.1}},
            {"epoch": 3, "validation": {"balanced_accuracy": 0.9}},
        ])
    for name, representation, epoch, score in (
            ("mha__baseline__full", "baseline", 3, 0.7),
            ("mha__tensor_core__full", "tensor_core", 1, 0.58)):
        arm = task / "ARMS" / name
        write_json(arm / "history.json", [
            {"epoch": i, "validation": {"balanced_accuracy": score if i == epoch else 0.4}}
            for i in range(1, 4)])
        write_json(arm / "result.json", {
            "status": "complete",
            "identity": {"study_id": "study-id", "subject": 1, "seed": 0,
                         "dataset_fingerprint": "data-id", "split_id": "split-id",
                         "calibration_sha256": "cal-id"},
            "arm": {"name": name, "representation": representation,
                    "attention": "mha", "regime": "full"},
            "selection": {"best_epoch": epoch},
            "metrics": [
                {"partition": "validation", "scenario": "full_22", "accuracy": score,
                 "balanced_accuracy": score, "f1_macro": score},
                {"partition": "test", "scenario": "full_22", "accuracy": 0.2,
                 "balanced_accuracy": 0.1, "f1_macro": 0.15},
            ],
        })
    return output


class LegacyDiagnosisTests(unittest.TestCase):
    def test_nested_calibration_hash_remains_supported(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = fixture(Path(temporary))
            path = output / 'artifacts/A01_seed_0/calibration.json'
            value = json.loads(path.read_text())
            value['identity']['calibration_sha256'] = value.pop('cache_sha256')
            write_json(path, value)
            diagnosis = inspect_legacy(output)['stage_comparison']['tasks'][0]
        self.assertEqual(diagnosis['status'], 'available')
        self.assertEqual(len(diagnosis['matched_head_pairs']), 1)

    def test_conflicting_calibration_hash_fields_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = fixture(Path(temporary))
            path = output / 'artifacts/A01_seed_0/calibration.json'
            value = json.loads(path.read_text())
            value['identity']['calibration_sha256'] = 'different-cache'
            write_json(path, value)
            diagnosis = inspect_legacy(output)['stage_comparison']['tasks'][0]
        self.assertEqual(diagnosis['status'], 'identity_mismatch')
        self.assertEqual(diagnosis['matched_head_pairs'], [])

    def test_missing_calibration_hash_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = fixture(Path(temporary))
            path = output / 'artifacts/A01_seed_0/calibration.json'
            value = json.loads(path.read_text())
            del value['cache_sha256']
            write_json(path, value)
            diagnosis = inspect_legacy(output)['stage_comparison']['tasks'][0]
        self.assertEqual(diagnosis['status'], 'identity_mismatch')
        self.assertEqual(diagnosis['matched_head_pairs'], [])

    def test_selected_epochs_and_test_metrics_are_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = inspect_legacy(fixture(Path(temporary)))
        diagnosis = report["stage_comparison"]["tasks"][0]
        self.assertEqual(diagnosis['status'], 'available')
        self.assertEqual(diagnosis['missing_sources'], [])
        stages = diagnosis["validation_stages"]
        self.assertEqual([stage["selected_epoch"] for stage in stages], [2, 3, 1])
        self.assertEqual(stages[1]["history_validation_score"]["balanced_accuracy"], 0.7)
        self.assertEqual(len(diagnosis["matched_head_pairs"]), 1)
        self.assertEqual(len(diagnosis["test_metrics"]), 2)
        self.assertTrue(all(row["excluded_from_architecture_recommendations"]
                            for row in diagnosis["test_metrics"]))
        self.assertIn("selection-biased", diagnosis["comparability"])
        self.assertIn("cannot reconstruct paired predictions",
                      diagnosis["common_heldout_raw_predictions"])

    def test_mismatched_calibration_identity_blocks_stage_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = fixture(Path(temporary))
            result_path = output / "artifacts/A01_seed_0/ARMS/mha__baseline__full/result.json"
            value = json.loads(result_path.read_text())
            value["identity"]["calibration_sha256"] = "wrong-cal-id"
            write_json(result_path, value)
            diagnosis = inspect_legacy(output)["stage_comparison"]["tasks"][0]
        self.assertTrue(any("identity mismatch" in item for item in diagnosis["missing_sources"]))
        self.assertFalse(any(row["stage"] == "retrained_baseline_head"
                             for row in diagnosis["validation_stages"]))

    def test_corrupt_calibration_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = fixture(Path(temporary))
            path = output / "artifacts/A01_seed_0/calibration.json"
            path.write_text("{bad", encoding="utf-8")
            diagnosis = inspect_legacy(output)["stage_comparison"]["tasks"][0]
        self.assertEqual(diagnosis["status"], "unavailable")
        self.assertIn("calibration.json: corrupt", diagnosis["missing_sources"][0])

    def test_absent_calibration_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            diagnosis = inspect_legacy(fixture(Path(temporary), calibration=False))[
                "stage_comparison"]["tasks"][0]
        self.assertEqual(diagnosis["status"], "unavailable")
        self.assertEqual(diagnosis["test_metrics"], [])
        self.assertIn("calibration.json: missing", diagnosis["missing_sources"][0])


if __name__ == "__main__":
    unittest.main()
