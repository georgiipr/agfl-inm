import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
from scipy.io import savemat
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from published_covariance import data, masks


class NativeDataTests(unittest.TestCase):
    def fixture(self, path):
        dtype = [(name, "O") for name in ("X", "trial", "y", "fs", "classes", "artifacts")]
        runs = np.empty((1, 2), dtype=object)
        for i in range(2):
            rec = np.empty((1, 1), dtype=dtype)
            starts = np.array([101, 2101, 4101, 6101]) if i else np.array([], dtype=int)
            x = np.arange(8000)[:, None] + np.arange(25)[None, :] * 100000.
            fields = {
                "X": x, "trial": starts[:, None],
                "y": np.array([4, 1, 3, 2] if i else [], dtype=int)[:, None],
                "fs": np.array([[250]]), "classes": np.array(["left", "right", "feet", "tongue"], dtype=object),
                "artifacts": np.array([1, 0, 1, 0] if i else [], dtype=int)[:, None],
            }
            for key, value in fields.items():
                rec[key][0, 0] = value
            runs[0, i] = rec
        savemat(path, {"data": runs})
        return x, starts

    def test_native_trial_slice_order_mapping_and_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "A01T.mat"
            x, starts = self.fixture(path)
            recording = data.load_native_recording(path, "A01", "T")
        np.testing.assert_array_equal(recording.values, np.stack([x[s + 375:s + 1500, :22].T for s in starts]))
        np.testing.assert_array_equal(recording.labels, [3, 0, 2, 1])
        np.testing.assert_array_equal(recording.artifacts, [True, False, True, False])
        np.testing.assert_array_equal(recording.run_ids, [2, 2, 2, 2])
        self.assertEqual(len(set(recording.trial_ids)), 4)
        self.assertEqual(recording.metadata["slice_from_stored_start"], [375, 1500])
        self.assertEqual(recording.metadata["units"], "microvolts (native source values; no conversion)")

    def test_native_input_rejects_ambiguous_session_and_label_only_mat(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "A01T.mat"
            savemat(bad, {"labels": np.arange(4)})
            with self.assertRaises(ValueError):
                data.load_native_recording(bad, "A01", "T")
            with self.assertRaises(ValueError):
                data.load_native_recording(bad, "A01", "E")

    def test_native_metadata_integrality_and_outer_window_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "A01T.mat"
            for field, value in (("y", 1.5), ("trial", 101.5), ("fs", 250.5), ("artifacts", 0.5)):
                dtype = [(name, "O") for name in ("X", "trial", "y", "fs", "classes", "artifacts")]
                record = np.empty((1, 1), dtype=dtype)
                values = {"X": np.zeros((2100, 25)), "trial": [[100]], "y": [[1]],
                          "fs": [[250]], "classes": np.array(["left", "right", "feet", "tongue"], dtype=object),
                          "artifacts": [[0]]}
                values[field] = [[value]]
                for key, item in values.items():
                    record[key][0, 0] = item
                runs = np.empty((1, 1), dtype=object)
                runs[0, 0] = record
                savemat(path, {"data": runs})
                with self.subTest(field=field), self.assertRaises(ValueError):
                    data.load_native_recording(path, "A01", "T")
            dtype = [(name, "O") for name in ("X", "trial", "y", "fs", "classes", "artifacts")]
            record = np.empty((1, 1), dtype=dtype)
            values = {"X": np.zeros((1700, 25)), "trial": [[100]], "y": [[1]],
                      "fs": [[250]], "classes": np.array(["left", "right", "feet", "tongue"], dtype=object),
                      "artifacts": [[0]]}
            for key, value in values.items():
                record[key][0, 0] = value
            runs = np.empty((1, 1), dtype=object)
            runs[0, 0] = record
            savemat(path, {"data": runs})
            with self.assertRaisesRegex(ValueError, "1750-sample"):
                data.load_native_recording(path, "A01", "T")

    def test_normalization_matches_author_and_sanitizes_hidden(self):
        x = np.random.default_rng(71).normal(size=(12, 22, 1125))
        mean, scale = data.fit_native_normalization(x)
        order = shuffle(np.arange(12), random_state=42)
        expected = np.empty_like(x)
        for c in range(22):
            scaler = StandardScaler().fit(x[order, c, :])
            np.testing.assert_array_equal(mean[c], scaler.mean_)
            np.testing.assert_array_equal(scale[c], scaler.scale_)
            expected[:, c, :] = scaler.transform(x[:, c, :])
        full = np.ones((12, 22), dtype=bool)
        np.testing.assert_array_equal(data.normalize(x, mean, scale, full), expected)
        mask = full.copy()
        mask[:, [0, 4, 9]] = False
        clean = data.normalize(x, mean, scale, mask)
        for bad in (np.nan, np.inf, -np.inf, 1e250):
            corrupt = x.copy()
            corrupt[~mask] = bad
            np.testing.assert_array_equal(data.normalize(corrupt, mean, scale, mask), clean)
        np.testing.assert_array_equal(clean[mask], expected[mask])
        self.assertTrue((clean[~mask] == 0).all())
        with self.assertRaises(ValueError):
            data.normalize(x, mean, scale, mask.astype(np.int8))
        with self.assertRaises(ValueError):
            data.normalize(x, mean, scale, np.zeros_like(mask))
        corrupt = x.copy()
        corrupt[0, 1, 0] = np.nan
        with self.assertRaises(ValueError):
            data.normalize(corrupt, mean, scale, mask)

    def test_normalizer_roundtrip_binds_subject_and_source(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=(5, 22, 1125))
        mean, scale = data.fit_native_normalization(x)
        ids = [f"A01:T:r04:t{i:03}" for i in range(5)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "normalizer.npz"
            data.save_normalization(path, mean, scale, subject="A01", source_sha256="a" * 64, trial_ids=ids)
            mu2, s2, metadata = data.load_normalization(path, subject="A01", source_sha256="a" * 64)
            np.testing.assert_array_equal(mu2, mean)
            np.testing.assert_array_equal(s2, scale)
            self.assertEqual(metadata["fit_session"], "T")
            self.assertEqual(metadata["fit_trial_ids"], ids)
            with self.assertRaises(ValueError):
                data.load_normalization(path, subject="A02")
            with self.assertRaises(ValueError):
                data.load_normalization(path, source_sha256="b" * 64)
            with self.assertRaises(FileExistsError):
                data.save_normalization(path, mean + 1, scale, subject="A01",
                                        source_sha256="a" * 64, trial_ids=ids)
            for invalid_ids in (("A01:E:r04:t001",), ("A02:T:r04:t001",)):
                with self.assertRaises(ValueError):
                    data.save_normalization(Path(tmp) / "invalid.npz", mean, scale,
                                            subject="A01", source_sha256="a" * 64,
                                            trial_ids=invalid_ids)
            with self.assertRaises(ValueError):
                data.save_normalization(Path(tmp) / "bad-hash.npz", mean, scale,
                                        subject="A01", source_sha256="not-a-hash", trial_ids=ids)

    def test_normalizer_load_rejects_tampered_provenance_and_arrays(self):
        mean, scale = np.zeros((22, 1125)), np.ones((22, 1125))
        ids = ["A01:T:r04:t001"]
        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "good.npz"
            data.save_normalization(good, mean, scale, subject="A01",
                                    source_sha256="a" * 64, trial_ids=ids)
            with np.load(good, allow_pickle=False) as archive:
                baseline = {key: archive[key].copy() for key in archive.files}
            for name, edit in (
                ("id-hash", lambda m: m.update(fit_trial_ids_sha256="0" * 64)),
                ("subject", lambda m: m.update(subject="A02")),
                ("schema", lambda m: m.update(schema_version=2)),
            ):
                with self.subTest(name=name):
                    arrays = {key: value.copy() for key, value in baseline.items()}
                    meta = json.loads(str(arrays["metadata"].item()))
                    edit(meta)
                    arrays["metadata"] = np.asarray(json.dumps(meta))
                    target = Path(tmp) / f"{name}.npz"
                    np.savez_compressed(target, **arrays)
                    with self.assertRaises(ValueError):
                        data.load_normalization(target)
            for name, key, value in (("nonfinite-mean", "mean", np.nan),
                                     ("zero-scale", "scale", 0.0)):
                arrays = {field: item.copy() for field, item in baseline.items()}
                arrays[key].flat[0] = value
                target = Path(tmp) / f"{name}.npz"
                np.savez_compressed(target, **arrays)
                with self.assertRaises(ValueError):
                    data.load_normalization(target)

    def test_exact_stratified_membership_and_invalid_classes(self):
        labels = np.array([3, 1, 0, 2] * 11 + [0, 0, 1, 2, 3])
        for seed in (0, 1, 2):
            rng = np.random.Generator(np.random.PCG64(seed))
            validation = []
            for klass in range(4):
                indices = np.flatnonzero(labels == klass)
                validation.extend(rng.permutation(indices)[:max(1, int(np.floor(.2 * len(indices))))])
            train, val = data.stratified_split(labels, seed)
            np.testing.assert_array_equal(val, sorted(validation))
            np.testing.assert_array_equal(train, sorted(set(range(len(labels))) - set(validation)))
            self.assertEqual(set(labels[train]), {0, 1, 2, 3})
        with self.assertRaises(ValueError):
            data.stratified_split(np.arange(4), 0)


class StaticMaskTests(unittest.TestCase):
    def test_static_masks_are_keyed_fixed_count_and_partition_disjoint(self):
        ids = [f"A01:T:{i:03}" for i in range(100)]
        for retained in (6, 16):
            banks = []
            for partition in ("train", "validation", "E"):
                bank = masks.static_masks("A01", 0, ids, partition, retained, repeat=2, epoch=3)
                self.assertEqual(bank.shape, (100, 22))
                self.assertEqual(bank.dtype, np.bool_)
                np.testing.assert_array_equal(bank.sum(1), np.full(100, retained))
                np.testing.assert_array_equal(bank, masks.static_masks("A01", 0, ids, partition, retained, repeat=2, epoch=3))
                banks.append({tuple(row) for row in bank})
            for i in range(3):
                for j in range(i):
                    self.assertFalse(banks[i] & banks[j])
        self.assertTrue(masks.static_masks("A01", 0, ids, "train", 22).all())

    def test_static_mask_keys_and_rejects_bad_protocol_inputs(self):
        ids = ["A01:T:r04:t001"]
        baseline = masks.static_masks("A01", 0, ids, "train", 16)
        self.assertFalse(np.array_equal(baseline, masks.static_masks("A02", 0, ids, "train", 16)))
        self.assertFalse(np.array_equal(baseline, masks.static_masks("A01", 1, ids, "train", 16)))
        self.assertFalse(np.array_equal(baseline, masks.static_masks("A01", 0, ids, "train", 16, epoch=1)))
        for retained in (0, 11, 23):
            with self.assertRaises(ValueError):
                masks.static_masks("A01", 0, ids, "train", retained)
        with self.assertRaises(ValueError):
            masks.static_masks("A01", 0, ids, "test", 16)
        with self.assertRaises(ValueError):
            masks.static_masks("A01", 0, ids + ids, "train", 16)


if __name__ == "__main__":
    unittest.main(verbosity=2)
