"""Training RNG and deterministic execution acceptance checks."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from inm.encoder_candidates import training
from inm.encoder_candidates.models import build_model
from tests.encoder_candidates.test_training import _cfg, _prepared


ARMS = ("local_control", "local_power", "spatial_eegnet",
        "spatial_filterbank", "spatial_transformer")


def _fit(arm, seed, directory, ambient_seed, prepared=None):
    random.seed(ambient_seed)
    np.random.seed(ambient_seed)
    torch.manual_seed(ambient_seed)
    model = build_model(arm, seed=seed)
    result = training.fit_classifier(model, prepared or _prepared(),
        _cfg(epochs=1, minimum=1, patience=0, batch=8), arm, directory)
    return result


def _state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _probabilities(model, prepared):
    model.eval()
    raw = torch.as_tensor(prepared.validation.raw)
    mask = torch.ones((len(raw), 22, 4), dtype=torch.bool)
    with torch.no_grad():
        return model(raw, mask).softmax(-1).cpu()


def _same_history(left, right):
    return left == right


def _assert_rng_state(test, before, after):
    test.assertEqual(before[0], after[0])
    test.assertEqual(before[1][0], after[1][0])
    np.testing.assert_array_equal(before[1][1], after[1][1])
    test.assertEqual(before[1][2:], after[1][2:])
    torch.testing.assert_close(before[2], after[2], rtol=0, atol=0)
    test.assertEqual(before[3:], after[3:])


class ReproducibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_original_cpu_dropout_rng_failure_is_reproduced_and_fixed(self):
        with tempfile.TemporaryDirectory() as temp:
            first = _fit("spatial_eegnet", 0, Path(temp) / "first", 123)
            second = _fit("spatial_eegnet", 0, Path(temp) / "second", 456)
        self.assertTrue(_same_history(first["history"], second["history"]))
        for key, value in _state(first["model"]).items():
            torch.testing.assert_close(value, _state(second["model"])[key], rtol=0, atol=0)

    def test_every_arm_repeats_actual_short_fits_across_ambient_rng_histories(self):
        data = _prepared()
        with tempfile.TemporaryDirectory() as temp:
            for arm in ARMS:
                with self.subTest(arm=arm):
                    first = _fit(arm, 17, Path(temp) / f"{arm}-a", 123, data)
                    # Perturb all ambient generators between fresh-directory fits.
                    random.random()
                    np.random.random(17)
                    torch.rand(41)
                    second = _fit(arm, 17, Path(temp) / f"{arm}-b", 456, data)
                    self.assertEqual(first["history"], second["history"])
                    for key, value in _state(first["model"]).items():
                        torch.testing.assert_close(value, _state(second["model"])[key],
                                                   rtol=0, atol=0)
                    torch.testing.assert_close(_probabilities(first["model"], data),
                                               _probabilities(second["model"], data),
                                               rtol=0, atol=0)
                    self.assertEqual(first["execution_device"], "cpu")
                    self.assertEqual(first["rng_policy"], second["rng_policy"])

    def test_fit_is_independent_of_arm_order_and_other_prior_fits(self):
        with tempfile.TemporaryDirectory() as temp:
            alone = _fit("spatial_eegnet", 21, Path(temp) / "alone", 51)
            _fit("local_power", 909, Path(temp) / "other", 52)
            after = _fit("spatial_eegnet", 21, Path(temp) / "after", 53)
        self.assertEqual(alone["history"], after["history"])
        for key, value in _state(alone["model"]).items():
            torch.testing.assert_close(value, _state(after["model"])[key], rtol=0, atol=0)

    def test_different_task_seeds_change_training_randomness(self):
        with tempfile.TemporaryDirectory() as temp:
            data_a, data_b = _prepared(), _prepared()
            data_a.seed, data_b.seed = 3, 4
            first_model, second_model = build_model("spatial_eegnet", seed=33), build_model(
                "spatial_eegnet", seed=33)
            first_model.load_state_dict(second_model.state_dict())
            cfg = _cfg(epochs=1, minimum=1, patience=0, batch=8)
            first = training.fit_classifier(first_model, data_a, cfg, "spatial_eegnet",
                                             Path(temp) / "seed3")
            second = training.fit_classifier(second_model, data_b, cfg, "spatial_eegnet",
                                              Path(temp) / "seed4")
        self.assertNotEqual(first["optimization"], second["optimization"])

    def test_caller_rng_and_deterministic_settings_restore_after_fit_and_exception(self):
        def caller_state():
            return (random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone(),
                    torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.benchmark,
                    torch.backends.cudnn.deterministic)
        with tempfile.TemporaryDirectory() as temp:
            random.seed(81); np.random.seed(81); torch.manual_seed(81)
            before = caller_state()
            result = training.fit_classifier(build_model("spatial_eegnet", seed=7),
                _prepared(), _cfg(epochs=1, minimum=1, patience=0, batch=8),
                "spatial_eegnet", Path(temp) / "success")
            after = caller_state()
            _assert_rng_state(self, before, after)
            before = caller_state()
            original_evaluate = training._evaluate
            with unittest.mock.patch.object(training, "_evaluate",
                    side_effect=RuntimeError("synthetic fit failure")):
                with self.assertRaisesRegex(RuntimeError, "synthetic fit failure"):
                    training.fit_classifier(build_model("spatial_eegnet", seed=7),
                        _prepared(), _cfg(epochs=1, minimum=1, patience=0, batch=8),
                        "spatial_eegnet", Path(temp) / "failure")
            after = caller_state()
            _assert_rng_state(self, before, after)

    def test_rng_provenance_roundtrips_and_policy_mismatch_refuses_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            fit = _fit("spatial_eegnet", 5, Path(temp) / "fit", 10)
            payload = torch.load(fit["checkpoint_path"], map_location="cpu", weights_only=True)
            self.assertEqual(payload["rng_policy"], fit["rng_policy"])
            self.assertEqual(payload["metadata"]["rng_policy"], fit["rng_policy"])
            # The recorded policy is part of fit compatibility, so changing it
            # must be rejected before any existing artifact can be reused.
            payload["rng_policy"] = {"name": "incompatible"}
            torch.save(payload, fit["checkpoint_path"])
            result_path = Path(fit["result_path"])
            result = json.loads(result_path.read_text())
            result["artifact_sha256"]["checkpoint.pt"] = training._sha256(
                Path(fit["checkpoint_path"]))
            result_path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "failed verification"):
                training.fit_classifier(build_model("spatial_eegnet", seed=5),
                    _prepared(), _cfg(epochs=1, minimum=1, patience=0, batch=8),
                    "spatial_eegnet", Path(temp) / "fit")

    def test_separate_process_short_fit_matches_selected_state_and_history(self):
        script = r'''import json, random, sys, torch, numpy as np
from pathlib import Path
from inm.encoder_candidates.models import build_model
from inm.encoder_candidates.training import fit_classifier
from tests.encoder_candidates.test_training import _cfg, _prepared
random.seed(int(sys.argv[2])); np.random.seed(int(sys.argv[2])); torch.manual_seed(int(sys.argv[2]))
torch.set_num_threads(1)
fit = fit_classifier(build_model("spatial_eegnet", seed=42), _prepared(),
    _cfg(epochs=1, minimum=1, patience=0, batch=8), "spatial_eegnet", Path(sys.argv[1]))
print(json.dumps({"history": fit["history"], "state": {k: v.tolist() for k,v in fit["model"].state_dict().items()}}, sort_keys=True))
'''
        with tempfile.TemporaryDirectory() as temp:
            outputs = []
            for name, ambient in (("one", 111), ("two", 987)):
                env = dict(os.environ, PYTHONHASHSEED=str(ambient))
                completed = subprocess.run([sys.executable, "-c", script,
                    str(Path(temp) / name), str(ambient)], check=True, capture_output=True, text=True,
                    env=env, cwd=Path(__file__).resolve().parents[2])
                outputs.append(json.loads(completed.stdout))
        self.assertEqual(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main()
