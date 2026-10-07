"""Run new reference subprocess on generated values, with no EEG outcome scores."""
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

def main():
    import tensorflow as tf
    from published_covariance.classifier import FrozenClassifier
    from published_covariance.native_reference import upstream_probabilities
    rng=np.random.Generator(np.random.PCG64(20261006))
    x=rng.normal(0,.2,(2,22,1125)).astype(np.float32)
    assets=ROOT/'.session-runs/published-checkpoint-covariance/assets'
    with tf.device('/CPU:0'):
        wrapper=FrozenClassifier('A01',assets_root=assets)
        actual=wrapper(x).numpy()
    native=upstream_probabilities('A01',x,assets,'cpu')
    np.testing.assert_allclose(actual,native,atol=1e-5,rtol=1e-4)
    np.testing.assert_array_equal(actual.argmax(1),native.argmax(1))
    print('PASS native reference subprocess; generated inputs only; max probability error',float(np.max(np.abs(actual-native))))

if __name__=='__main__':main()
