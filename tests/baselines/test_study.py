import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from inm.baselines.data import make_synthetic_signal_dataset, prepare_subject
from inm.baselines.protocol import arms, load_config
from inm.baselines.study import (_execute_prepared, _manifest, _result_complete,
                                _evaluation_banks, run_smoke, run_task)
from inm.baselines.covariance import CovarianceClassifier


class TinyClassifier(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.linear = nn.Linear(1, 4)
        self.mask_conditioned = bool(kwargs.get("mask_conditioned", False))

    def constructor_settings(self):
        return {"mask_conditioned": self.mask_conditioned}

    def forward(self, raw, mask=None):
        observed = torch.where(mask[..., None], raw, torch.zeros_like(raw))
        return self.linear(observed.mean((1, 2, 3)).unsqueeze(1))


def fixture_config(output_dir):
    cfg = load_config("configs/baselines.json")
    cfg["subjects"] = [1]
    cfg["seeds"] = [0]
    cfg["output_dir"] = str(output_dir)
    cfg["execution_arms"] = ["eegnet_reference__full"]
    cfg["training"].update({"epochs": 2, "minimum_epochs": 1, "warmup_epochs": 0,
                            "patience": 0, "batch_size": 16})
    cfg["mask_repeats"] = 1
    return cfg


class BaselineStudyTests(unittest.TestCase):
    def test_same_task_seed_reproduces_weights_despite_ambient_rng(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = fixture_config(Path(temp) / 'real')
            states = []
            for ambient_seed in (11, 99):
                torch.manual_seed(ambient_seed)
                output = Path(temp) / f'smoke-{ambient_seed}'
                with patch('inm.baselines.eegnet.EEGNetClassifier', TinyClassifier):
                    run_smoke(cfg, output)
                checkpoint = output / 'artifacts/synthetic_A01_seed_0/ARMS/eegnet_reference__full/checkpoint.pt'
                states.append(torch.load(checkpoint, map_location='cpu', weights_only=True)['state_dict'])
            self.assertEqual(set(states[0]), set(states[1]))
            for name in states[0]:
                self.assertTrue(torch.equal(states[0][name], states[1][name]), name)

    def test_synthetic_smoke_fits_reloads_evaluates_and_records_complete_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = fixture_config(Path(temp) / "real")
            with patch("inm.baselines.eegnet.EEGNetClassifier", TinyClassifier):
                result = run_smoke(cfg, Path(temp) / "smoke")
            self.assertTrue(result["synthetic"])
            self.assertEqual(result["format"], "agfl-baseline-task-v1")
            arm_result = result["arms"][0]
            self.assertEqual(arm_result["format"], "agfl-baseline-task-result-v1")
            self.assertTrue(arm_result["synthetic"])
            self.assertEqual(len(arm_result["coverage"]), 13)
            self.assertEqual(len(arm_result["metrics"]), 26)
            self.assertTrue(all(len(row["per_class_recall"]) == 4 for row in arm_result["metrics"]))
            self.assertEqual(arm_result["selected_epoch"] in (1, 2), True)
            self.assertTrue((Path(temp) / "smoke/artifacts/synthetic_A01_seed_0/ARMS/eegnet_reference__full/checkpoint.pt").is_file())

    def test_checkpoint_corruption_and_missing_checkpoint_reject_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = fixture_config(Path(temp) / "real")
            output = Path(temp) / "smoke"
            with patch("inm.baselines.eegnet.EEGNetClassifier", TinyClassifier):
                result = run_smoke(cfg, output)
            arm_result = result["arms"][0]
            rows = arm_result["metrics"]
            banks = [{"partition": row["partition"], "scenario": row["scenario"],
                      "mask_repeat": row["mask_repeat"], "mask_sha256": row["mask_sha256"]}
                     for row in rows]
            result_path = output / "artifacts/synthetic_A01_seed_0/ARMS/eegnet_reference__full/result.json"
            identity = arm_result["identity"]
            self.assertIsNotNone(_result_complete(result_path, identity, "eegnet_reference__full", banks))
            checkpoint = result_path.parent / "checkpoint.pt"
            original = checkpoint.read_bytes()
            checkpoint.write_bytes(b"corrupt")
            with self.assertRaisesRegex(RuntimeError, "corruption"):
                _result_complete(result_path, identity, "eegnet_reference__full", banks)
            checkpoint.unlink()
            with self.assertRaisesRegex(RuntimeError, "Missing checkpoint"):
                _result_complete(result_path, identity, "eegnet_reference__full", banks)
            checkpoint.write_bytes(original)

    def test_changed_manifest_identity_and_synthetic_real_collision_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = fixture_config(Path(temp) / "real")
            output = Path(temp) / "smoke"
            _manifest(output, cfg, [1], synthetic=True)
            changed = copy.deepcopy(cfg)
            changed["name"] += "-changed"
            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                _manifest(output, changed, [1], synthetic=True)
            with patch("inm.baselines.study._source_hashes", return_value={"study.py": "changed"}):
                with self.assertRaisesRegex(RuntimeError, "identity changed"):
                    _manifest(output, cfg, [1], synthetic=True)
            with self.assertRaisesRegex(ValueError, "separate"):
                run_smoke(cfg, cfg["output_dir"])

    def test_invalid_task_fails_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = fixture_config(Path(temp) / "real")
            with self.assertRaisesRegex(ValueError, "task-index"):
                run_task(cfg, 9, device="cpu")
            self.assertFalse(Path(cfg["output_dir"]).exists())

    def test_classical_execution_records_full_only_coverage_and_roundtrips(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = fixture_config(Path(temp) / "real")
            cfg["execution_arms"] = ["covariance__full"]
            task_dir = Path(temp) / "artifacts/A01_seed_0"
            output = Path(temp) / "study"
            fixture = make_synthetic_signal_dataset(seed=27, samples_per_class=15)
            prepared = prepare_subject(cfg, 1, 0, task_dir, loader=lambda _: fixture)
            prepared.metadata["synthetic"] = True
            result = _execute_prepared(cfg, prepared, task_dir, output, 1, 0,
                                       arms(cfg), device="cpu", synthetic=True)
            arm_result = result["arms"][0]
            self.assertEqual(arm_result["coverage"], ["full"])
            self.assertEqual(arm_result["coverage_scope"], "full_input_only")
            self.assertEqual(len(arm_result["metrics"]), 2)
            self.assertEqual({row["partition"] for row in arm_result["metrics"]},
                             {"validation", "test"})
            self.assertTrue(all(row["scenario"] == "full" for row in arm_result["metrics"]))
            self.assertEqual(arm_result["class_order"], [0, 1, 2, 3])
            arm_dir = task_dir / "ARMS/covariance__full"
            restored = CovarianceClassifier.load(arm_dir / "model.npz")
            test_indices = prepared.split_indices["test"]
            self.assertEqual(restored.predict_proba(prepared.filtered_signals[test_indices]).shape,
                             (len(test_indices), 4))
            banks = [bank for bank in _evaluation_banks(prepared, cfg, 1, 0, 1)
                     if bank["scenario"] == "full"]
            self.assertIsNotNone(_result_complete(arm_dir / "result.json", arm_result["identity"],
                                                  "covariance__full", banks, family="classical"))

    def test_incomplete_arm_artifacts_are_not_reusable(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = fixture_config(Path(temp) / "real")
            smoke_cfg = copy.deepcopy(cfg)
            task_dir = Path(temp) / "smoke/artifacts/synthetic_A01_seed_0"
            output = task_dir.parents[1]
            fixture = make_synthetic_signal_dataset(seed=17, samples_per_class=15)
            prepared = prepare_subject(smoke_cfg, 1, 0, task_dir, loader=lambda _: fixture)
            prepared.metadata["synthetic"] = True
            arm_dir = task_dir / "ARMS/eegnet_reference__full"
            arm_dir.mkdir(parents=True)
            (arm_dir / "history.json").write_text("[]", encoding="utf-8")
            with patch("inm.baselines.eegnet.EEGNetClassifier", TinyClassifier):
                with self.assertRaisesRegex(RuntimeError, "Incomplete arm artifacts"):
                    _execute_prepared(smoke_cfg, prepared, task_dir, output, 1, 0,
                                      arms(smoke_cfg)[:1], device="cpu", synthetic=True)
            self.assertEqual((arm_dir / "history.json").read_text(encoding="utf-8"), "[]")


if __name__ == "__main__":
    unittest.main()
