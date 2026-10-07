"""Generated-input CUDA libdevice/PTX compiler probe; zero EEG/data reads."""
import hashlib,json,os,shutil
from pathlib import Path
import numpy as np
import tensorflow as tf
root=Path(__file__).resolve().parents[2]
asset=root/'.venv-published-covariance/lib/python3.11/site-packages/nvidia/cuda_nvcc'
expected='--xla_gpu_cuda_data_dir='+str(asset)
assert os.environ.get('XLA_FLAGS')==expected
assert Path(shutil.which('ptxas')).resolve()==(asset/'bin/ptxas').resolve()
for gpu in tf.config.list_physical_devices('GPU'): tf.config.experimental.set_memory_growth(gpu,True)
assert tf.config.list_physical_devices('GPU')
values=np.linspace(.1,3.,128,dtype=np.float64)
def expression(x):return tf.math.lgamma(x)+tf.math.erf(x)+tf.math.sin(x)
with tf.device('/CPU:0'):
 cpu=tf.Variable(values)
 with tf.GradientTape() as tape: y=expression(cpu); loss=tf.reduce_sum(y)
 expected_y=y.numpy();expected_g=tape.gradient(loss,cpu).numpy()
@tf.function(jit_compile=True,autograph=False)
def compiled(x):
 with tf.GradientTape() as tape:
  tape.watch(x);y=expression(x);loss=tf.reduce_sum(y)
 return y,tape.gradient(loss,x)
with tf.device('/GPU:0'):
 x=tf.constant(values);actual,g=compiled(x)
assert 'GPU:0' in actual.device and 'GPU:0' in g.device
np.testing.assert_allclose(actual.numpy(),expected_y,atol=1e-10,rtol=1e-10)
np.testing.assert_allclose(g.numpy(),expected_g,atol=1e-10,rtol=1e-10)
assert np.isfinite(g.numpy()).all()
result={'passed':True,'generated_only':True,'ptxas_resolved':shutil.which('ptxas'),'PATH':os.environ['PATH'],'XLA_FLAGS':os.environ['XLA_FLAGS'],'gpu_device':actual.device,'maximum_output_error':float(np.max(np.abs(actual.numpy()-expected_y))),'maximum_gradient_error':float(np.max(np.abs(g.numpy()-expected_g))),'compiler_assets':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (asset/'bin/ptxas',asset/'nvvm/libdevice/libdevice.10.bc')}}
print(json.dumps(result,sort_keys=True))
