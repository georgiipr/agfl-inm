"""Independent conditional-completion numerical checks, fixed before session04."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from pathlib import Path
import sys
import unittest
import numpy as np
import tensorflow as tf
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from published_covariance.completion import CovarianceCompleter,zero_fill,hidden_mse
from published_covariance.training import select_checkpoint

class CompletionAcceptance(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(117)
        a=rng.normal(size=(22,22));self.s=a@a.T/22+np.eye(22)*.1
        self.x=rng.normal(size=(3,22,17)).astype('float32')
        self.mask=np.ones((3,22),bool);self.mask[0,16:]=False;self.mask[1,6:]=False

    def test_independent_numpy_solve_and_epoch_zero(self):
        fixed=CovarianceCompleter(self.s,trainable=False)
        learned=CovarianceCompleter(self.s,trainable=True)
        self.assertEqual(tuple(learned.free.shape),(253,));self.assertEqual(learned.free.dtype,tf.float64)
        sigma=self.s+np.eye(22)*1e-6
        np.testing.assert_allclose(learned.covariance.numpy(),sigma,atol=1e-12,rtol=1e-12)
        out=learned.complete(tf.constant(self.x),tf.constant(self.mask)).numpy()
        np.testing.assert_array_equal(out,fixed.complete(self.x,self.mask))
        for b in range(3):
            o=np.flatnonzero(self.mask[b]);h=np.flatnonzero(~self.mask[b])
            expected=sigma[np.ix_(h,o)]@np.linalg.solve(sigma[np.ix_(o,o)]+np.eye(len(o))*.001*np.diag(sigma).mean(),self.x[b,o])
            np.testing.assert_allclose(out[b,h],expected,atol=1e-6,rtol=1e-5)
        np.testing.assert_array_equal(out[self.mask],self.x[self.mask])

    def test_mask_safety_bypass_and_rejection(self):
        c=CovarianceCompleter(self.s)
        np.testing.assert_array_equal(c.complete(self.x,np.ones_like(self.mask)),self.x)
        expected=c.complete(self.x,self.mask)
        for bad in [np.nan,np.inf,-np.inf,1e30]:
            hidden=self.x.copy();hidden[~self.mask]=bad
            np.testing.assert_array_equal(c.complete(hidden,self.mask),expected)
            np.testing.assert_array_equal(zero_fill(hidden,self.mask),np.where(self.mask[...,None],self.x,0))
        with self.assertRaises((ValueError,tf.errors.InvalidArgumentError)):c.complete(self.x,np.zeros_like(self.mask))
        bad=self.x.copy();bad[0,0,0]=np.nan
        with self.assertRaises((ValueError,tf.errors.InvalidArgumentError)):c.complete(bad,self.mask)
        with self.assertRaises((TypeError,ValueError,tf.errors.InvalidArgumentError)):c.complete(self.x,self.mask.astype(int))

    def test_free_gradient_finite_difference(self):
        c=CovarianceCompleter(self.s)
        x=tf.constant(self.x,dtype=tf.float64);mask=tf.constant(self.mask)
        def objective():return tf.reduce_mean(tf.square(c.complete(x,mask)))
        with tf.GradientTape() as tape:loss=objective()
        gradient=tape.gradient(loss,c.free).numpy();initial=c.free.numpy().copy()
        self.assertTrue(np.isfinite(gradient).all());self.assertGreater(np.linalg.norm(gradient),0)
        for i in [0,1,17,120,252]:
            plus=initial.copy();minus=initial.copy();plus[i]+=1e-5;minus[i]-=1e-5
            c.free.assign(plus);a=float(objective());c.free.assign(minus);b=float(objective())
            self.assertAlmostEqual(gradient[i],(a-b)/2e-5,delta=1e-6)
        c.free.assign(initial)

    def test_hidden_loss_and_checkpoint_ties(self):
        pred=tf.Variable(np.ones((2,22,3)),dtype=tf.float64)
        target=tf.zeros_like(pred);mask=np.ones((2,22),bool);mask[0,0]=False
        self.assertEqual(float(hidden_mse(pred,target,mask)),1.)
        with tf.GradientTape() as tape:loss=hidden_mse(pred,target,np.ones_like(mask))
        grad=tape.gradient(loss,pred)
        self.assertEqual(float(loss),0.);self.assertIsNotNone(grad)
        np.testing.assert_array_equal(grad,np.zeros_like(pred.numpy()))
        rows=[{'epoch':2,'degraded_ba':.6,'degraded_log_loss':.5},{'epoch':0,'degraded_ba':.6,'degraded_log_loss':.5},{'epoch':1,'degraded_ba':.6,'degraded_log_loss':.7},{'epoch':3,'degraded_ba':.5,'degraded_log_loss':.1}]
        self.assertEqual(select_checkpoint(rows)['epoch'],0)
        rows.append({'epoch':4,'degraded_ba':.6,'degraded_log_loss':.4})
        self.assertEqual(select_checkpoint(rows)['epoch'],4)

if __name__=='__main__':unittest.main(verbosity=2)
