"""Independent source-review regressions for native input/persistence boundaries."""
from pathlib import Path
import sys
import tempfile
import unittest
import json
import numpy as np
import scipy.io as sio
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from published_covariance import data

def fixture(path, *, n_samples=2100, label=1, start=100, fs=250, bad_hidden=False):
    dtype=[(n,'O') for n in ['X','trial','y','fs','classes','artifacts']]
    r=np.empty((1,1),dtype=dtype)
    x=np.arange(n_samples*25,dtype=float).reshape(n_samples,25)
    if bad_hidden:x[:,0]=np.nan
    for k,v in dict(X=x,trial=np.array([[start]]),y=np.array([[label]]),fs=np.array([[fs]]),classes=np.array(['left','right','feet','tongue'],dtype=object),artifacts=np.array([[0]])).items():r[k][0,0]=v
    runs=np.empty((1,1),dtype=object);runs[0,0]=r
    sio.savemat(path,{'data':runs})

class BoundaryAcceptance(unittest.TestCase):
    def test_no_fractional_metadata_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'A01T.mat'
            for kw in [{'label':1.5},{'start':100.5},{'fs':250.5}]:
                fixture(p,**kw)
                with self.subTest(kw=kw),self.assertRaises(ValueError):data.load_native_recording(p,'A01','T')

    def test_native_outer_window_must_exist(self):
        # Author first materializes 1750 samples; a 1500-sample short trial
        # must not be admitted merely because the final crop fits.
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'A01T.mat';fixture(p,n_samples=1700)
            with self.assertRaises(ValueError):data.load_native_recording(p,'A01','T')

    def test_hidden_raw_values_through_complete_preprocessing(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'A01T.mat';fixture(p)
            clean=data.load_native_recording(p,'A01','T')
            fixture(p,bad_hidden=True)
            # Extraction has no arithmetic; observed-only finite validation
            # belongs at the explicit masking/normalization boundary.
            bad=data.load_native_recording(p,'A01','T')
            mask=np.ones((1,22),bool);mask[:,0]=False
            mean=np.zeros((22,1125));scale=np.ones_like(mean)
            np.testing.assert_array_equal(data.normalize(clean.values,mean,scale,mask),data.normalize(bad.values,mean,scale,mask))
            with self.assertRaises(ValueError):data.normalize(bad.values,mean,scale,np.ones_like(mask))

    def test_normalization_no_overwrite_or_false_T_provenance(self):
        mean=np.zeros((22,1125));scale=np.ones_like(mean)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'norm.npz'
            for ids in [('A01:E:r01:t001',),('A02:T:r01:t001',)]:
                with self.subTest(ids=ids),self.assertRaises(ValueError):data.save_normalization(p,mean,scale,subject='A01',source_sha256='a'*64,trial_ids=ids)
            data.save_normalization(p,mean,scale,subject='A01',source_sha256='a'*64,trial_ids=('A01:T:r01:t001',))
            before=p.read_bytes()
            with self.assertRaises(FileExistsError):data.save_normalization(p,mean+1,scale,subject='A01',source_sha256='a'*64,trial_ids=('A01:T:r01:t001',))
            self.assertEqual(p.read_bytes(),before)

if __name__=='__main__':unittest.main(verbosity=2)
