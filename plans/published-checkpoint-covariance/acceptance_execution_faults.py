"""Adversarial persistence checks independent of stored self-reported hashes."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

class PersistenceFaults(unittest.TestCase):
    def call(self, mode, output, ok=True):
        p = subprocess.run([sys.executable, '-m', 'published_covariance', mode,
                            '--synthetic', '--device', 'cpu', '--output', str(output)],
                           cwd=ROOT, capture_output=True, text=True, timeout=180)
        if ok: self.assertEqual(p.returncode, 0, p.stdout+p.stderr)
        else: self.assertNotEqual(p.returncode, 0, p.stdout+p.stderr)
        return p

    def test_read_only_modes_never_create_study(self):
        with tempfile.TemporaryDirectory() as tmp:
            for mode in ('audit', 'summarize', 'review-pilot', 'evaluate-cohort'):
                out = Path(tmp)/mode
                self.call(mode, out, ok=False)
                self.assertFalse(out.exists(), mode+' initialized a nonexistent study')

    def test_completed_fit_reuse_replays_not_only_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'fixture'
            self.call('pilot', out)
            state = out/'fits/A01-s0.npz'
            with np.load(state, allow_pickle=False) as z:
                arrays = {k:z[k].copy() for k in z.files}
            arrays['selected_free'] += .25
            with state.open('wb') as stream: np.savez_compressed(stream, **arrays)
            meta = state.with_suffix('.json')
            record = json.loads(meta.read_text())
            record['npz_sha256'] = hashlib.sha256(state.read_bytes()).hexdigest()
            meta.write_text(json.dumps(record))
            self.call('pilot', out, ok=False)

    def test_forged_pilot_review_cannot_authorize_cohort(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'fixture'
            self.call('pilot', out)
            (out/'pilot-review.json').write_text('{"accepted": true}')
            self.call('train-cohort', out, ok=False)
            self.assertFalse((out/'fits/A02-s0.npz').exists())

if __name__ == '__main__': unittest.main(verbosity=2)
