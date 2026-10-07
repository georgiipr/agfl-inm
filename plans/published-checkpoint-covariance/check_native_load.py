"""Supervisor native-framework loading check, synthetic inputs only."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'.session-runs/published-checkpoint-covariance/assets/EEG-ATCNet'
sys.path.insert(0, str(SOURCE))
import models

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.experimental.enable_op_determinism()
tf.keras.utils.set_random_seed(102)
x = tf.constant(np.random.default_rng(103).normal(size=(2, 1, 22, 1125)).astype('float32'))
rows = []
for subject in range(1, 10):
    tf.keras.backend.clear_session()
    model = models.ATCNet_(n_classes=4)
    path = SOURCE/f'results/saved models/run-1/subject-{subject}.h5'
    model.load_weights(str(path))
    model.trainable = False
    before = [v.numpy().copy() for v in model.weights]
    with tf.GradientTape() as tape:
        tape.watch(x)
        probabilities = model(x, training=False)
        objective = -tf.math.log(probabilities[0, 0])
    gradient = tape.gradient(objective, x).numpy()
    replay = model(x, training=False).numpy()
    assert model.count_params() == 115172
    assert len(model.weights) == 195
    assert np.isfinite(probabilities.numpy()).all()
    assert np.array_equal(probabilities.numpy(), replay)
    assert np.isfinite(gradient).all() and np.linalg.norm(gradient) > 0
    assert all(np.array_equal(a, b.numpy()) for a, b in zip(before, model.weights))
    rows.append({'subject': f'A{subject:02}', 'parameters_including_buffers': model.count_params(),
                 'weights': len(model.weights), 'probability_sha256': hashlib.sha256(replay.tobytes()).hexdigest(),
                 'input_gradient_norm': float(np.linalg.norm(gradient)), 'exact_replay': True,
                 'unchanged_weights': True})
    print(json.dumps(rows[-1]), flush=True)
out = ROOT/'.session-runs/published-checkpoint-covariance/01/native-load.json'
with out.open('x') as f:
    json.dump({'runtime': sys.executable, 'tensorflow': tf.__version__, 'device': 'CPU',
               'synthetic_only': True, 'subjects': rows}, f, indent=2)
