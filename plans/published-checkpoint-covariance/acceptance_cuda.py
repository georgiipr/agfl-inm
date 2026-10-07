"""Supervisor-only native CUDA synthetic fit/save/replay admission through launcher.

Session06 adds native-smoke.json/npz to the existing explicit smoke --native mode.
This script never reads EEG trials or scores real outcomes.
"""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

def main():
    with tempfile.TemporaryDirectory(prefix='published-native-cuda-') as tmp:
        out=Path(tmp)/'smoke'
        command=['bash','scripts/run_published_covariance.sh','smoke','--native',
                 '--device','cuda:0','--output',str(out)]
        child=subprocess.run(command,cwd=ROOT,text=True,capture_output=True,timeout=600)
        print(child.stdout)
        if child.returncode: print(child.stderr,file=sys.stderr)
        assert child.returncode==0, 'Public native CUDA smoke failed'
        metadata=json.loads((out/'native-smoke.json').read_text())
        assert metadata['synthetic_fixture'] is True
        assert metadata['device']=='cuda:0'
        assert metadata['subjects']==[f'A{i:02d}' for i in range(1,10)]
        assert metadata['frozen_before_sha256']==metadata['frozen_after_sha256']
        history=metadata['history']
        assert len(history)>=2 and any(r['parameter_delta_norm']>0 for r in history[1:])
        with np.load(out/'native-smoke.npz',allow_pickle=False) as z:
            arrays={k:z[k].copy() for k in z.files}
        x=arrays['values']; mask=arrays['mask']
        assert x.ndim==3 and x.shape[1:]==(22,1125)
        assert mask.dtype==np.bool_ and mask.shape==x.shape[:2]
        for key in ('covariance_gradient','input_gradient'):
            assert np.isfinite(arrays[key]).all() and np.linalg.norm(arrays[key])>0,key
        assert np.array_equal(arrays['probabilities'],arrays['replay_probabilities'])
        import tensorflow as tf
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu,True)
        assert tf.config.list_physical_devices('GPU')
        from published_covariance.classifier import FrozenClassifier
        from published_covariance.completion import CovarianceCompleter
        with tf.device('/GPU:0'):
            classifier=FrozenClassifier('A01')
            completer=CovarianceCompleter(np.eye(22),trainable=False)
            completer.free.assign(arrays['selected_free'])
            completed=completer.complete(x,mask)
            replay=classifier(completed)
        assert 'GPU:0' in replay.device, replay.device
        np.testing.assert_array_equal(replay.numpy(),arrays['probabilities'])
        np.testing.assert_array_equal(completed.numpy()[mask],x[mask])
        # Independent NumPy conditional mean, using the serialized free state.
        free=arrays['selected_free']; factor=np.zeros((22,22),dtype=np.float64)
        factor[np.tril_indices(22)]=free
        diagonal=np.diag(factor).copy(); factor[np.diag_indices(22)]=np.logaddexp(0,diagonal)+1e-6
        covariance=factor@factor.T; ridge=.001*np.trace(covariance)/22
        expected=x.copy()
        for row in range(len(x)):
            obs=np.flatnonzero(mask[row]); missing=np.flatnonzero(~mask[row])
            expected[row,missing]=(covariance[np.ix_(missing,obs)]@np.linalg.solve(covariance[np.ix_(obs,obs)]+ridge*np.eye(len(obs)),x[row,obs])).astype(x.dtype)
        np.testing.assert_allclose(completed.numpy(),expected,atol=1e-6,rtol=1e-5)
        print(json.dumps({'passed':True,'real_fits':0,'real_E_predictions':0,
                          'native_subjects':9,'synthetic_epochs':len(history)-1,
                          'public_launcher':True,'replay_device':replay.device}))

if __name__=='__main__':main()
