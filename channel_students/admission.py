"""GPU admission with generated inputs; invoked only on the experiment cluster."""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from published_covariance.preflight import validate_runtime_and_attention
from published_covariance.masks import static_masks
from .artifacts import write_json
from .model import Student, Teacher, state_digest
from .protocol import ASSETS
from .study import recording
from .training import make_train_step


def admit(out, config, identity):
    receipt = out / "admission" / "gpu-checks.json"
    if receipt.exists():
        from .protocol import sha256
        import json
        value = json.loads(receipt.read_text())
        if value["identity_sha256"] != sha256(out / "identity.json") or value["passed"] is not True:
            raise ValueError("Admission receipt belongs to another experiment identity")
        return
    validate_runtime_and_attention(ASSETS)
    for subject in config["subjects"]:
        recording(subject, "T", identity)
    rng = np.random.Generator(np.random.PCG64(20261007))
    values = rng.normal(0, 0.1, (2, 22, 1125)).astype(np.float32)
    ids = ("A01:T:r99:t001", "A01:T:r99:t002")
    checks = []
    for retained in config["retained_counts"]:
        tf.keras.backend.clear_session()
        teacher = Teacher("A01", ASSETS) if config["distillation"]["enabled"] else None
        targets = teacher.logits(values) if teacher is not None else None
        student = Student("A01", 0, retained)
        mask = static_masks("A01", 0, ids, "train", retained, epoch=1,
                            namespace=config["mask_namespace"])
        expected = student.logits(values, mask, training=False).numpy()
        corrupt = np.where(mask[..., None], values, np.full_like(values, np.nan))
        np.testing.assert_array_equal(expected, student.logits(corrupt, mask, training=False).numpy())
        probe = tf.Variable(values)
        with tf.GradientTape() as tape:
            logits = student.logits(probe, mask, training=False)
            loss = tf.reduce_sum(tf.square(logits))
        gradient = tape.gradient(loss, probe)
        if gradient is None or not np.isfinite(gradient.numpy()).all():
            raise RuntimeError("Student input gradient is missing/nonfinite")
        hidden = ~np.broadcast_to(mask[..., None], values.shape)
        if np.any(gradient.numpy()[hidden] != 0) or not np.any(gradient.numpy()[~hidden] != 0):
            raise RuntimeError("Student accesses hidden signals or ignores every observed signal")
        before = state_digest(student.model)
        step = make_train_step(student, config)
        targets = targets if targets is not None else tf.zeros((2, 4), tf.float32)
        result = step(tf.convert_to_tensor(values), tf.constant([0, 1]),
                      tf.convert_to_tensor(mask), targets)
        # Exercise a smaller final batch with the same production trace.
        step(tf.convert_to_tensor(values[:1]), tf.constant([0]),
             tf.convert_to_tensor(mask[:1]), targets[:1])
        if before == state_digest(student.model) or "GPU:0" not in result[0].device:
            raise RuntimeError("Student update did not run on the GPU")
        if teacher is not None:
            teacher.verify_frozen()
        elif float(result[2].numpy()) != 0:
            raise RuntimeError("Q1 unexpectedly has teacher loss")
        checks.append({"retained": retained, "hidden_input_invariance": True,
                       "hidden_input_gradient_zero": True, "student_update": True,
                       "production_trace_variable_batch": True,
                       "teacher_present": teacher is not None,
                       "teacher_state_unchanged": True if teacher is not None else None})
        del student, teacher
    tf.keras.backend.clear_session()
    from .protocol import sha256
    write_json(receipt, {"schema": 1, "identity_sha256": sha256(out / "identity.json"),
                         "experiment": config["experiment"], "passed": True,
                         "native_checkpoint_attention_checks": True, "generated_input_checks": checks,
                         "E_signals_or_labels_opened": False})
