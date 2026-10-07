"""Nonzero-target reconstruction oracle added after supervisor source review."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from pathlib import Path
import sys
import unittest
import numpy as np
import tensorflow as tf
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from published_covariance.completion import hidden_mse

class ReconstructionAcceptance(unittest.TestCase):
    def test_hidden_target_difference_and_observed_invariance(self):
        rng=np.random.default_rng(371)
        target=rng.normal(3,2,size=(2,22,7))
        pred=rng.normal(-1,1,size=target.shape)
        mask=np.ones((2,22),bool);mask[0,:6]=False;mask[1,10:]=False
        expected=np.mean((pred[~mask]-target[~mask])**2)
        self.assertAlmostEqual(float(hidden_mse(pred,target,mask)),expected,places=12)
        changed_target=target.copy();changed_target[mask]=1e6
        changed_pred=pred.copy();changed_pred[mask]=-1e6
        self.assertAlmostEqual(float(hidden_mse(changed_pred,changed_target,mask)),expected,places=12)
        variable=tf.Variable(pred,dtype=tf.float64)
        with tf.GradientTape() as tape:loss=hidden_mse(variable,target,mask)
        actual=tape.gradient(loss,variable).numpy()
        gradient=np.zeros_like(pred);gradient[~mask]=2*(pred[~mask]-target[~mask])/pred[~mask].size
        np.testing.assert_allclose(actual,gradient,atol=1e-12,rtol=1e-12)

if __name__=='__main__':unittest.main(verbosity=2)
