"""Independent native forward/input-gradient/frozen-state checks for session03."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from pathlib import Path
import sys
import unittest
import numpy as np
import tensorflow as tf

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'.session-runs/published-checkpoint-covariance/assets/EEG-ATCNet'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(SOURCE))
import models as official
from published_covariance.classifier import FrozenClassifier
from published_covariance.data import load_native_recording,fit_native_normalization,normalize

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.experimental.enable_op_determinism()
tf.keras.utils.set_random_seed(30)

class ClassifierAcceptance(unittest.TestCase):
    def test_all_nine_synthetic_forward_gradient_frozen(self):
        x=tf.constant(np.random.default_rng(40).normal(size=(2,22,1125)).astype('float32'))
        for subject in range(1,10):
            with self.subTest(subject=subject):
                tf.keras.backend.clear_session()
                native=official.ATCNet_(n_classes=4)
                native.load_weights(str(SOURCE/f'results/saved models/run-1/subject-{subject}.h5'))
                wrapper=FrozenClassifier(f'A{subject:02}', assets_root=SOURCE.parent)
                with tf.GradientTape() as tape:
                    tape.watch(x);npred=native(x[:,None,:,:],training=False)
                    loss=tf.reduce_sum(npred[:,0])
                ng=tape.gradient(loss,x).numpy()
                before=[v.numpy().copy() for v in wrapper.model.weights]
                with tf.GradientTape() as tape:
                    tape.watch(x);wpred=wrapper(x,training=True)
                    loss=tf.reduce_sum(wpred[:,0])
                wg=tape.gradient(loss,x).numpy()
                np.testing.assert_allclose(wpred,npred,atol=1e-5,rtol=1e-4)
                np.testing.assert_array_equal(np.argmax(wpred,axis=1),np.argmax(npred,axis=1))
                np.testing.assert_allclose(wg,ng,atol=1e-5,rtol=1e-3)
                self.assertTrue(np.isfinite(wg).all());self.assertGreater(np.linalg.norm(wg),0)
                np.testing.assert_array_equal(wpred,wrapper(x,training=False))
                self.assertEqual(wrapper.model.trainable_variables,[])
                for v,b in zip(wrapper.model.weights,before):np.testing.assert_array_equal(v,b)

    def test_all_nine_T_only_native_forward(self):
        for subject in range(1,10):
            with self.subTest(subject=subject):
                tf.keras.backend.clear_session()
                r=load_native_recording(SOURCE.parent/f'recordings/A{subject:02}T.mat',f'A{subject:02}','T')
                mean,scale=fit_native_normalization(r.values)
                x=normalize(r.values[[0,71,143,287]],mean,scale,np.ones((4,22),bool)).astype('float32')
                native=official.ATCNet_(n_classes=4)
                native.load_weights(str(SOURCE/f'results/saved models/run-1/subject-{subject}.h5'))
                wrapper=FrozenClassifier(f'A{subject:02}',assets_root=SOURCE.parent)
                a=native(x[:,None,:,:],training=False).numpy();b=wrapper(tf.constant(x)).numpy()
                np.testing.assert_allclose(a,b,atol=1e-5,rtol=1e-4)
                np.testing.assert_array_equal(a.argmax(1),b.argmax(1))

if __name__=='__main__':unittest.main(verbosity=2)
