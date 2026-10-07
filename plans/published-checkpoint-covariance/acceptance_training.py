"""Supervisor synthetic epoch-zero restoration and covariance-only fit checks."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from pathlib import Path
import sys
import unittest
import random
import numpy as np
import tensorflow as tf
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from published_covariance.training import train_covariance
from published_covariance.completion import CovarianceCompleter

class ConstantFrozen:
    def __init__(self):
        inp=tf.keras.Input(shape=(22,1125))
        pooled=tf.keras.layers.GlobalAveragePooling1D()(inp)
        out=tf.keras.layers.Dense(4,kernel_initializer='zeros',bias_initializer='zeros',activation='softmax')(pooled)
        self.model=tf.keras.Model(inp,out);self.model.trainable=False
    def __call__(self,x,training=False):return self.model(x,training=False)

class TrainingAcceptance(unittest.TestCase):
    def test_epoch_zero_restore_validation_immutable_rng_scoped(self):
        rng=np.random.default_rng(34)
        train=rng.normal(size=(8,22,1125)).astype('float32')
        val=rng.normal(size=(8,22,1125)).astype('float32')
        before_val=val.copy();labels=np.arange(8)%4
        classifier=ConstantFrozen();before_weights=[v.numpy().copy() for v in classifier.model.weights]
        # The initializer is independently fitted on train alone.
        moment=np.einsum('nct,ndt->cd',train.astype('float64'),train.astype('float64'))/(8*1125)
        initial=CovarianceCompleter(moment).free.numpy().copy()
        py_state=random.getstate();np_state=np.random.get_state()
        result=train_covariance(classifier=classifier,train_values=train,train_labels=labels,
            train_ids=[f'A01:T:r01:t{i+1:03}' for i in range(8)],
            validation_values=val,validation_labels=labels,
            validation_ids=[f'A01:T:r02:t{i+1:03}' for i in range(8)],subject='A01',seed=0,
            max_epochs=2,min_epochs=0,patience=1,synthetic=True)
        self.assertEqual(result['selected_epoch'],0)
        self.assertGreaterEqual(len(result['history']),2)
        self.assertGreater(result['history'][1]['parameter_delta_norm'],0.)
        np.testing.assert_allclose(result['completer'].free.numpy(),initial,atol=1e-12,rtol=1e-12)
        np.testing.assert_array_equal(val,before_val)
        for a,b in zip(before_weights,classifier.model.weights):np.testing.assert_array_equal(a,b)
        self.assertEqual(random.getstate(),py_state)
        after=np.random.get_state();self.assertEqual(after[0],np_state[0]);np.testing.assert_array_equal(after[1],np_state[1]);self.assertEqual(after[2:],np_state[2:])

if __name__=='__main__':unittest.main(verbosity=2)
