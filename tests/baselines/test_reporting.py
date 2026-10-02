import csv
import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stdout
import io
from unittest.mock import patch
from pathlib import Path

from inm.baselines.protocol import load_config
from inm.baselines.reporting import METRICS, summarize


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


class BaselineReportingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.output = root / "study"
        cfg = load_config("configs/baselines.json")
        cfg["subjects"] = [1, 2]
        cfg["seeds"] = [0, 1]
        cfg["mask_repeats"] = 2
        cfg["output_dir"] = str(self.output)
        cfg["execution_arms"] = ["eegnet_reference__full", "masked_eegnet__full", "covariance__full"]
        self.cfg = cfg
        self.arms = [
            ("eegnet_reference__full", "neural", "full_and_degraded"),
            ("masked_eegnet__full", "neural", "full_and_degraded"),
            ("covariance__full", "classical", "full_input_only"),
        ]
        for subject in cfg["subjects"]:
            for seed in cfg["seeds"]:
                self._write_dataset(subject, seed)
                for arm, family, scope in self.arms:
                    self._write(subject, seed, arm, family, scope)

    def tearDown(self):
        self.temp.cleanup()

    def _write_dataset(self, subject, seed, raw_suffix=""):
        provenance = {"subject": subject, "seed": seed,
            "dataset_fingerprint": digest(f"raw-{subject}{raw_suffix}"),
            "split_id": digest(f"split-{subject}-{seed}"),
            "config": {"config_sha256": hashlib.sha256(Path(self.cfg["config_path"]).read_bytes()).hexdigest()}}
        stats = {"mean": [float(seed)] * 22, "std": [1.0] * 22}
        ids = [f"A{subject:02d}:trial:{i}" for i in range(20)]
        prepared_hash = hashlib.sha256(json.dumps({"provenance": provenance,
            "normalization": stats, "sample_ids": ids}, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        dataset = {"dataset_fingerprint": provenance["dataset_fingerprint"],
            "data_fingerprint": prepared_hash, "provenance": provenance,
            "normalization_stats": stats, "sample_ids": ids}
        path = self.output / 'artifacts' / f'A{subject:02d}_seed_{seed}' / 'dataset.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dataset))
        return path

    def _write(self, subject, seed, arm, family, scope, *, synthetic=False, mask_suffix="", data_suffix=""):
        neural = family == "neural"
        scenarios = ["full"] + [f"{pattern}_{n}" for n in (16, 11, 6)
            for pattern in ("random_static", "spatial_static", "dynamic_random", "dynamic_spatial")] if neural else ["full"]
        # Distinct seed and participant scores; repeat effects average analytically to subject + seed.
        base = 0.4 + 0.1 * (subject - 1) + 0.05 * seed
        rows = []
        for partition in ("validation", "test"):
            for scenario in scenarios:
                repeats = range(1 if scenario == "full" else self.cfg["mask_repeats"])
                for repeat in repeats:
                    degraded = scenario != "full"
                    adjustment = (-0.04 if degraded else 0.0) + (0.02 * repeat if degraded else 0)
                    rows.append({"partition": partition, "scenario": scenario, "mask_repeat": repeat,
                        "mask_sha256": digest(f"{subject}-{seed}-{partition}-{scenario}-{repeat}{mask_suffix}"),
                        "sample_count": 10 + subject * 20, "balanced_accuracy": base + adjustment,
                        "accuracy": base + adjustment, "f1_macro": base + adjustment,
                        "per_class_recall": [base + adjustment] * 4})
        dataset = json.loads((self.output / 'artifacts' / f'A{subject:02d}_seed_{seed}' / 'dataset.json').read_text())
        prepared_hash = dataset['data_fingerprint'] if not data_suffix else digest(data_suffix)
        identity = {"synthetic": synthetic, "config_sha256": hashlib.sha256(Path(self.cfg["config_path"]).read_bytes()).hexdigest(),
                    "source_sha256": {"src": digest("source")}, "data_fingerprint": prepared_hash,
                    "split_id": digest(f"split-{subject}-{seed}"), "subject": subject, "seed": seed,
                    "protocol": "within_session"}
        value = {"format": "agfl-baseline-task-result-v1", "status": "complete", "synthetic": synthetic,
                 "subject": subject, "seed": seed, "protocol": "within_session", "arm": arm,
                 "family": family, "regime": "full", "coverage_scope": scope, "selection_policy": "full",
                 "identity": identity, "data_fingerprint": identity["data_fingerprint"],
                 "split_id": identity["split_id"], "metrics": rows}
        path = self.output / "artifacts" / f"A{subject:02d}_seed_{seed}" / "ARMS" / arm / "result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def _read_csv(self, name):
        with (self.output / "report" / name).open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    def test_repeat_seed_subject_means_are_equal_weight_and_classical_is_full_only(self):
        datasets = [json.loads((self.output / 'artifacts' / f'A01_seed_{seed}' / 'dataset.json').read_text())
                    for seed in self.cfg['seeds']]
        self.assertEqual(datasets[0]['dataset_fingerprint'], datasets[1]['dataset_fingerprint'])
        self.assertNotEqual(datasets[0]['data_fingerprint'], datasets[1]['data_fingerprint'])
        progress = summarize(self.cfg)
        self.assertTrue(progress["complete"])
        rows = self._read_csv("accuracy_table.csv")
        selected = next(row for row in rows if row["arm"] == "eegnet_reference__full" and
                        row["partition"] == "test" and row["scenario"] == "full")
        # Participant means are 0.425 and 0.525, independent of their 30 vs 50 trials.
        self.assertAlmostEqual(float(selected["balanced_accuracy"]), 0.475)
        classical = [row for row in rows if row["arm"] == "covariance__full"]
        self.assertEqual({row["scenario"] for row in classical}, {"full"})
        self.assertEqual(len([row for row in self._read_csv("accuracy_by_run.csv")
                              if row["arm"] == "covariance__full"]), 8)
        self.assertTrue((self.output / "report" / "summary.md").is_file())
        self.assertTrue((self.output / "report" / "full_input_degradation.csv").is_file())
        degradation = self._read_csv("full_input_degradation.csv")
        static = next(row for row in degradation if row["arm"] == "eegnet_reference__full" and
                      row["partition"] == "test" and row["scenario"] == "random_static_16")
        self.assertAlmostEqual(float(static["balanced_accuracy_degradation"]), 0.03)
        paired = self._read_csv("paired_neural_differences.csv")
        aggregate = next(row for row in paired if row.get("scope") == "cohort" and row["partition"] == "test" and
                         row["scenario"] == "full")
        self.assertEqual(aggregate["complete"], "True")

    def test_missing_seed_blanks_cohort_and_mask_mismatch_is_audited(self):
        path = self._write(1, 0, "masked_eegnet__full", "neural", "full_and_degraded", mask_suffix="different")
        progress = summarize(self.cfg)
        self.assertTrue(progress["complete"])
        issue = self._read_csv("pairing_issues.csv")
        self.assertTrue(any(row["subject"] == "1" and "mask hash mismatch" in row["reason"] for row in issue))
        path.unlink()
        progress = summarize(self.cfg)
        self.assertFalse(progress["complete"])
        table = self._read_csv("accuracy_table.csv")
        row = next(row for row in table if row["arm"] == "masked_eegnet__full" and
                   row["partition"] == "test" and row["scenario"] == "full")
        self.assertEqual(row["balanced_accuracy"], "")

    def test_synthetic_and_corrupt_records_are_visible_and_not_counted(self):
        path = self._write(1, 0, "eegnet_reference__full", "neural", "full_and_degraded", synthetic=True)
        progress = summarize(self.cfg)
        self.assertFalse(progress["complete"])
        self.assertTrue(any("synthetic" in row["reason"] for row in progress["invalid"]))
        path.write_text("{broken", encoding="utf-8")
        progress = summarize(self.cfg)
        self.assertTrue(any("Expecting" in row["reason"] for row in progress["invalid"]))
        self.assertTrue((self.output / "report" / "invalid_results.csv").is_file())

    def test_prepared_identity_mismatch_is_invalid_and_excluded_from_pairs(self):
        for seed in self.cfg["seeds"]:
            self._write(1, seed, "masked_eegnet__full", "neural", "full_and_degraded", data_suffix="-alternate")
        progress = summarize(self.cfg)
        self.assertFalse(progress["complete"])
        self.assertTrue(any("data_fingerprint" in row["reason"] for row in progress['invalid']))
        pairs = self._read_csv('paired_neural_differences.csv')
        self.assertFalse(any(row['subject'] == '1' for row in pairs))

    def test_different_underlying_dataset_across_seeds_is_still_rejected(self):
        self._write_dataset(1, 1, raw_suffix='changed-recording')
        for arm, family, scope in self.arms:
            self._write(1, 1, arm, family, scope)
        progress = summarize(self.cfg)
        self.assertFalse(progress['complete'])
        self.assertTrue(any('underlying dataset fingerprint differs' in r['reason']
                            for r in progress['invalid']))

    def test_missing_or_tampered_dataset_metadata_cannot_validate_a_result(self):
        path = self.output / 'artifacts/A01_seed_0/dataset.json'
        original = path.read_text()
        path.unlink()
        self.assertFalse(summarize(self.cfg)['complete'])
        dataset = json.loads(original)
        dataset['normalization_stats']['mean'][0] += 1.0
        path.write_text(json.dumps(dataset))
        progress = summarize(self.cfg)
        self.assertTrue(any('dataset.json contents' in r['reason'] for r in progress['invalid']))

    def test_actual_preparation_of_two_seeds_reports_as_one_dataset(self):
        from inm.baselines.data import make_synthetic_signal_dataset, prepare_subject
        self.cfg['subjects'] = [1]
        bundle = make_synthetic_signal_dataset(15, samples_per_class=10)
        fingerprints = []
        for seed in self.cfg['seeds']:
            prepared = prepare_subject(self.cfg, 1, seed,
                self.output / 'artifacts' / f'A01_seed_{seed}', loader=lambda _: bundle)
            fingerprints.append(prepared.data_fingerprint)
            dataset = json.loads((self.output / 'artifacts' / f'A01_seed_{seed}' / 'dataset.json').read_text())
            for arm, family, scope in self.arms:
                path = self._write(1, seed, arm, family, scope)
                result = json.loads(path.read_text())
                result['split_id'] = result['identity']['split_id'] = dataset['provenance']['split_id']
                path.write_text(json.dumps(result))
        self.assertNotEqual(*fingerprints)
        self.assertTrue(summarize(self.cfg)['complete'])

    def test_summarize_only_dispatches_without_study_import(self):
        from inm.baselines import __main__ as cli
        with patch.object(cli, "load_config", return_value=self.cfg), redirect_stdout(io.StringIO()):
            result = cli.main(["--config", "unused.json", "--summarize-only"])
        self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
