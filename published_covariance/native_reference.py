"""Run pinned unmodified EEG-ATCNet inference in an isolated child process."""
from __future__ import annotations
import json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import tensorflow as tf
from .classifier import _verified_checkpoint

def upstream_probabilities(subject, values, assets_root, device="cpu"):
    checkpoint=_verified_checkpoint(subject,assets_root)
    source=Path(assets_root).resolve(strict=True)/"EEG-ATCNet"
    x=tf.convert_to_tensor(values)
    if not x.dtype.is_floating or x.shape.rank!=3 or x.shape[1:]!=(22,1125): raise ValueError("native reference input must be [N,22,1125]")
    if device not in {"cpu","cuda:0"}: raise ValueError("device must be cpu or cuda:0")
    if device == "cuda:0":
        gpus=tf.config.list_physical_devices("GPU")
        if not gpus: raise RuntimeError("CUDA was requested but TensorFlow exposes no GPU")
        for gpu in gpus:
            try: tf.config.experimental.set_memory_growth(gpu,True)
            except RuntimeError as exc: raise RuntimeError("parent CUDA memory growth must be set before device initialization") from exc
    array=tf.cast(x,tf.float32).numpy()
    code="""import sys,numpy as np,tensorflow as tf
if sys.argv[5]=='cuda:0' and not tf.config.list_physical_devices('GPU'):
    raise RuntimeError('CUDA requested but child TensorFlow exposes no GPU')
for device in tf.config.list_physical_devices('GPU'):
    try: tf.config.experimental.set_memory_growth(device,True)
    except RuntimeError as exc: raise RuntimeError('child GPU memory growth unavailable before initialization') from exc
sys.path.insert(0,sys.argv[1])
import models
dev='/GPU:0' if sys.argv[5]=='cuda:0' else '/CPU:0'
with tf.device(dev):
    m=models.ATCNet_(n_classes=4)
    m.load_weights(sys.argv[2])
    x=np.load(sys.argv[3],allow_pickle=False)
    p=m(x[:,None,:,:],training=False).numpy()
    np.save(sys.argv[4],p,allow_pickle=False)
"""
    with tempfile.TemporaryDirectory(prefix="published-covariance-native-") as folder:
        inp=Path(folder)/"input.npy"; out=Path(folder)/"output.npy"
        np.save(inp,array,allow_pickle=False)
        result=subprocess.run([sys.executable,"-c",code,str(source),str(checkpoint),str(inp),str(out),device],capture_output=True,text=True,timeout=180)
        if result.returncode: raise RuntimeError("pinned author source inference failed: "+result.stderr[-2000:])
        return np.load(out,allow_pickle=False)
