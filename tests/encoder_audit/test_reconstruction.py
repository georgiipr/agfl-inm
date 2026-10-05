"""Synthetic paired reconstruction checks; no recordings or real fits."""
from dataclasses import dataclass
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from inm.encoder_audit.probes import Probe
from inm.encoder_audit.reconstruction import (ReconstructionError, _complete,
    _hidden_error_sums, audit_reconstruction)
from inm.tensor_attention import Tucker2


@dataclass
class Part:
    sample_ids: tuple
    labels: np.ndarray


def _config():
    return {"conditions": [{"name": n, "pattern": p, "retained": k, "repeats": r}
            for n, p, k, r in (("full_22", "full", 22, 1),
                ("random_static_16", "random_static", 16, 5),
                ("dynamic_random_16", "dynamic_random", 16, 5),
                ("random_static_6", "random_static", 6, 5),
                ("dynamic_random_6", "dynamic_random", 6, 5))]}


class ReconstructionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(18)
        # Unequal per-trial energies make pooled NRMSE differ from averaging
        # individual ratios. The probe is fixed and intentionally simple.
        x = rng.normal(size=(8, 22, 4, 32)).astype(np.float32)
        x[0] *= .05
        x[-1] *= 7
        y = np.tile(np.arange(4, dtype=np.int64), 2)
        ids = tuple(f"v{i}" for i in range(len(x)))
        model = Tucker2(22, 32, rank_channels=4, rank_features=4, ridge=.001)
        with torch.no_grad():
            model.U.copy_(torch.tensor(rng.normal(size=(22, 4)), dtype=torch.float32))
            model.V.copy_(torch.tensor(rng.normal(size=(32, 4)), dtype=torch.float32))
            model.fitted.fill_(True)
            model.fit_steps.fill_(1)
        self.state = {k: v.clone() for k, v in model.state_dict().items()}
        self.features = x
        self.labels = y
        self.sources = SimpleNamespace(subject=1, seed=1,
            baseline_metadata=SimpleNamespace(study_id="baseline-id"),
            legacy_metadata=SimpleNamespace(study_id="legacy-id"), legacy=SimpleNamespace(
            validation=Part(ids, y.copy()), validation_features=x, factor_state=self.state))
        # A numeric fixed-state probe with all four classes. No fitting occurs.
        width = 22 * 4 * 32
        self.probe = Probe("features_ordered", np.zeros(width), np.ones(width),
            np.zeros((4, width)), np.zeros(4), np.arange(4),
            {"subject": 1, "seed": 1, "input_study_ids": {"baseline": "baseline-id", "legacy": "legacy-id"},
             "sample_ids": {"validation": list(ids)}})

    def test_condition_nrmse_is_pooled_and_zero_fill_is_one(self):
        # Hand calculation with unequal target energy across trials. The pooled
        # result is sqrt((4+16)/(4+64)); averaging trial ratios is different.
        truth = np.zeros((2, 22, 4, 32), dtype=np.float64)
        estimate = truth.copy()
        mask = np.ones((2, 22, 4), dtype=bool)
        mask[:, 1, 0] = False
        truth[0, 1, 0, 0] = 2
        truth[1, 1, 0, 0] = 8
        estimate[1, 1, 0, 0] = 4
        sse, energy, count = _hidden_error_sums(truth, estimate, mask)
        self.assertEqual((sse, energy, count), (20.0, 68.0, 64))
        pooled = np.sqrt(20 / 68)
        self.assertAlmostEqual(pooled, np.sqrt(sse / energy))
        self.assertNotEqual((np.sqrt(4 / 4) + np.sqrt(16 / 64)) / 2, pooled)

        result = audit_reconstruction(self.sources, self.probe, _config())
        by_condition = result["conditions"]
        full = by_condition[0]
        self.assertIsNone(full["hidden_nrmse"]["zero_fill"])
        self.assertIsNone(full["hidden_nrmse"]["tucker_completion"])
        degraded = by_condition[1]
        self.assertGreater(degraded["hidden_target_energy_sum"], 0)
        self.assertAlmostEqual(degraded["hidden_nrmse"]["zero_fill"], 1.0, places=12)
        truth = self.features.astype(np.float64)
        mask = __import__("inm.availability", fromlist=["make_mask_bank"]).make_mask_bank(
            len(truth), 4, 16, "random_static", seed=1, partition="validation", subject="A01",
            repeat=0, sample_ids=self.sources.legacy.validation.sample_ids)
        hidden = ~mask[..., None]
        energy = float(np.where(hidden, truth**2, 0).sum())
        completed = by_condition[1]
        estimate_numerator = float(completed["hidden_squared_error_sum"]["tucker_completion"])
        manual = np.sqrt(estimate_numerator / energy)
        self.assertAlmostEqual(completed["hidden_nrmse"]["tucker_completion"], manual, places=12)
        trial_ratios = [row["hidden_nrmse"]["tucker_completion"] for row in completed["per_trial"]
                        if row["hidden_nrmse"]["tucker_completion"] is not None]
        self.assertGreater(abs(manual - float(np.mean(trial_ratios))), 1e-5)

    def test_paired_mask_and_observed_values_and_hidden_nan_invariance(self):
        result = audit_reconstruction(self.sources, self.probe, _config())
        row = result["conditions"][1]
        self.assertEqual(len(row["mask_sha256"]), 64)
        self.assertEqual(set(row["mask_identities"].values()), {row["mask_sha256"]})
        rng = np.random.default_rng(5)
        solver = Tucker2(22, 32, rank_channels=4, rank_features=4, ridge=.001)
        solver.load_state_dict(self.state)
        mask = np.ones((2, 22, 4), dtype=bool)
        mask[:, 4:, :] = False
        x = rng.normal(size=(2, 22, 4, 32)).astype(np.float32)
        expected = _complete(solver, x, mask)
        poisoned = x.copy()
        poisoned[~mask] = np.nan
        actual = _complete(solver, poisoned, mask)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual[mask[..., None].repeat(32, axis=-1)],
            x[mask[..., None].repeat(32, axis=-1)])
        # Each method is evaluated against the same single bank identity.
        self.assertEqual(row["metrics"].keys(), {"full_features", "zero_fill", "tucker_completion"})

    def test_completion_does_not_depend_on_labels_and_forbidden_conditions_fail(self):
        first = audit_reconstruction(self.sources, self.probe, _config())
        self.sources.legacy.validation.labels[:] = np.roll(self.labels, 1)
        second = audit_reconstruction(self.sources, self.probe, _config())
        for a, b in zip(first["conditions"], second["conditions"]):
            self.assertEqual(a["mask_sha256"], b["mask_sha256"])
            self.assertEqual(a["hidden_squared_error_sum"], b["hidden_squared_error_sum"])
        bad = _config()
        bad["conditions"].append({"name": "spatial_11", "pattern": "spatial_static", "retained": 11, "repeats": 5})
        with self.assertRaisesRegex(ReconstructionError, "exactly"):
            audit_reconstruction(self.sources, self.probe, bad)

    def test_probe_identity_order_and_nonfinite_reference_rejected(self):
        bad_study_probe = Probe("features_ordered", self.probe.mean, self.probe.scale,
            self.probe.coef, self.probe.intercept, self.probe.classes,
            {**self.probe.metadata, "input_study_ids": {"baseline": "other", "legacy": "legacy-id"}})
        with self.assertRaisesRegex(ReconstructionError, "study identities"):
            audit_reconstruction(self.sources, bad_study_probe, _config())
        with self.assertRaisesRegex(ReconstructionError, "identities/order"):
            audit_reconstruction(self.sources, Probe("features_ordered", self.probe.mean, self.probe.scale,
                self.probe.coef, self.probe.intercept, self.probe.classes,
                {**self.probe.metadata, "sample_ids": {"validation": list(reversed(self.sources.legacy.validation.sample_ids))}}), _config())
        self.sources.legacy.validation_features[0, 0, 0, 0] = np.inf
        with self.assertRaisesRegex(ReconstructionError, "finite"):
            audit_reconstruction(self.sources, self.probe, _config())

    def test_missing_class_is_explicit(self):
        self.sources.legacy.validation.labels[:] = np.array([0, 1, 2, 0, 1, 2, 0, 1])
        metrics = audit_reconstruction(self.sources, self.probe, _config())["conditions"][1]["metrics"]["zero_fill"]
        self.assertIsNone(metrics["per_class_recall"][3])
        self.assertEqual(metrics["class_counts"][3], 0)


if __name__ == "__main__":
    unittest.main()
