"""Supervisor-owned session02 tests. APIs fixed before worker launch."""
import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from published_covariance import data, masks

class DataAcceptance(unittest.TestCase):
    def test_native_struct_index_labels_channels_artifacts(self):
        # Deliberately non-sorted labels and large per-channel offsets expose
        # channel swaps, silent artifact removal, label shifts, and off-by-one.
        dtype = [('X','O'),('trial','O'),('y','O'),('fs','O'),('classes','O'),('artifacts','O')]
        runs = np.empty((1,2), dtype=object)
        for i in range(2):
            r = np.empty((1,1), dtype=dtype)
            starts = np.array([101,2101,4101,6101]) if i else np.array([],dtype=int)
            x = np.arange(8000)[:,None] + np.arange(25)[None,:]*100000.
            for k,v in dict(X=x,trial=starts[:,None],y=np.array([4,1,3,2] if i else [],dtype=int)[:,None],fs=np.array([[250]]),classes=np.array(['left','right','feet','tongue'],dtype=object),artifacts=np.array([1,0,1,0] if i else [],dtype=int)[:,None]).items():r[k][0,0]=v
            runs[0,i]=r
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'A01T.mat'; sio.savemat(path,{'data':runs})
            result=data.load_native_recording(path, subject='A01', session='T')
        expected=np.stack([x[s+375:s+1500,:22].T for s in starts])
        np.testing.assert_array_equal(result.values, expected)
        np.testing.assert_array_equal(result.labels,[3,0,2,1])
        self.assertEqual(len(set(result.trial_ids)),4)
        np.testing.assert_array_equal(result.artifacts,[True,False,True,False])

    def test_native_normalization_and_hidden_values(self):
        x=np.random.default_rng(71).normal(size=(12,22,1125))
        mean,scale=data.fit_native_normalization(x)
        perm=shuffle(np.arange(12),random_state=42)
        expected=np.empty_like(x)
        for c in range(22):
            sc=StandardScaler().fit(x[perm,c,:])
            np.testing.assert_array_equal(mean[c],sc.mean_)
            np.testing.assert_array_equal(scale[c],sc.scale_)
            expected[:,c,:]=sc.transform(x[:,c,:])
        full=np.ones((12,22),dtype=bool)
        np.testing.assert_array_equal(data.normalize(x,mean,scale,full),expected)
        mask=full.copy(); mask[:,[0,4,9]]=False
        clean=data.normalize(x,mean,scale,mask)
        for bad in [np.nan,np.inf,-np.inf,1e250]:
            z=x.copy(); z[~mask]=bad
            np.testing.assert_array_equal(data.normalize(z,mean,scale,mask),clean)
        np.testing.assert_array_equal(clean[mask],expected[mask])
        with self.assertRaises((ValueError,TypeError)):data.normalize(x,mean,scale,mask.astype(int))
        z=x.copy();z[0,1,0]=np.nan
        with self.assertRaises(ValueError):data.normalize(z,mean,scale,mask)
        with self.assertRaises(ValueError):data.normalize(x,mean,scale,np.zeros_like(mask))

    def test_exact_stratified_membership(self):
        labels=np.array([3,1,0,2]*11+[0,0,1,2,3])
        for seed in [0,1,2]:
            rng=np.random.Generator(np.random.PCG64(seed));val=[]
            for c in range(4):
                indices=np.flatnonzero(labels==c)
                val.extend(rng.permutation(indices)[:max(1,int(np.floor(.2*len(indices))))])
            train,validation=data.stratified_split(labels,seed)
            np.testing.assert_array_equal(validation,sorted(val))
            np.testing.assert_array_equal(train,sorted(set(range(len(labels)))-set(val)))
        with self.assertRaises(ValueError):data.stratified_split(np.arange(4),0)

    def test_static_mask_partitions(self):
        ids=[f'A01:T:{i:03}' for i in range(100)]
        for retained in [6,16]:
            pools=[]
            for part in ['train','validation','E']:
                m=masks.static_masks('A01',0,ids,part,retained,repeat=2,epoch=3)
                self.assertEqual(m.dtype,np.dtype(bool));self.assertEqual(m.shape,(100,22))
                np.testing.assert_array_equal(m.sum(axis=1),np.full(100,retained))
                np.testing.assert_array_equal(m,masks.static_masks('A01',0,ids,part,retained,repeat=2,epoch=3))
                pools.append({tuple(row) for row in m})
            for i in range(3):
                for j in range(i):self.assertFalse(pools[i]&pools[j])
        self.assertTrue(masks.static_masks('A01',0,ids,'train',22).all())
        for bad in [0,23,11]:
            with self.assertRaises(ValueError):masks.static_masks('A01',0,ids,'train',bad)

    def test_real_A01_native_array(self):
        # Admission tolerance fixed before adapter implementation: extraction
        # must be bitwise identical; normalization exact in the same runtime.
        source=ROOT/'.session-runs/published-checkpoint-covariance/assets/EEG-ATCNet/preprocess.py'
        node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='load_BCI2a_data')
        namespace={'np':np,'sio':sio}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
        folder=ROOT/'.session-runs/published-checkpoint-covariance/assets/recordings'
        expected,labels=namespace['load_BCI2a_data'](str(folder)+'/',1,True)
        result=data.load_native_recording(folder/'A01T.mat',subject='A01',session='T')
        np.testing.assert_array_equal(result.values,expected)
        np.testing.assert_array_equal(result.labels,labels)
        self.assertEqual(len(labels),288)

if __name__=='__main__':unittest.main(verbosity=2)
