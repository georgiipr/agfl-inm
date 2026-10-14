"""Separate Q1/Q2 lifecycles with a complete fit barrier before E access."""
from __future__ import annotations

import json
import time
import numpy as np
import tensorflow as tf

from published_covariance.data import load_native_recording, fit_native_normalization, normalize, stratified_split
from published_covariance.masks import static_masks
from .artifacts import lock, write_json, write_npz, finish_task, verify_task
from .model import Student, Teacher, state_digest
from .protocol import ASSETS, sha256, tasks, initialization_seed, order_seed
from .training import fit, validation, selection, predict, metrics


def recording(subject, session, identity):
    record = load_native_recording(ASSETS / "recordings" / f"{subject}{session}.mat", subject, session)
    if record.metadata["source_sha256"] != identity["assets"]["recordings"][subject + session]:
        raise ValueError("Recording changed after admission")
    if len(record.values) != 288:
        raise ValueError("Keep all 288 native trials, including artifacts")
    return record


def teacher_targets(subject, record, train_indices, config):
    if not config["distillation"]["enabled"]:
        return None, {}, None
    # Replay the pretrained teacher's historical scaler. Student fitting below
    # uses its own strictly T-train-only scaler. Never compute validation/E KD targets.
    mean, scale = fit_native_normalization(record.values)
    values = normalize(record.values[train_indices], mean, scale,
                       np.ones((len(train_indices), 22), np.bool_)).astype(np.float32)
    tf.keras.backend.clear_session()
    teacher = Teacher(subject, ASSETS)
    logits = np.concatenate([teacher.logits(values[start:start + 64]).numpy()
                             for start in range(0, len(values), 64)])
    teacher.verify_frozen()
    metadata = {"teacher_checkpoint_sha256": teacher.classifier.checkpoint_sha256,
                "teacher_state_before": teacher.state_before,
                "teacher_state_after": state_digest(teacher.model),
                "teacher_target_partition": "T-train-only",
                "teacher_normalization_fit_ids": list(record.trial_ids)}
    del teacher
    return logits, metadata, (mean, scale)


def fit_name(subject, seed, retained):
    return f"{subject}-s{seed}-k{retained}"


def verify_fit(out, subject, seed, retained, config):
    folder = out / "fits" / fit_name(subject, seed, retained)
    meta = verify_task(folder, sha256(out / "identity.json"))
    if (meta["subject"], meta["seed"], meta["retained"], meta["experiment"]) != (subject, seed, retained, config["experiment"]):
        raise ValueError("Fit belongs to another student task")
    history = json.loads((folder / "history.json").read_text())
    if (selection(history)["epoch"] != meta["selected_epoch"] or
            len(history) - 1 != meta["epochs_completed"] or
            [r["epoch"] for r in history] != list(range(len(history)))):
        raise ValueError("Selected student does not match validation history")
    with np.load(folder / "training.npz", allow_pickle=False) as data:
        train_ids = data["train_ids"].astype(str).tolist()
        valid_ids = data["validation_ids"].astype(str).tolist()
        labels = data["all_labels"]
        train_indices, val_indices = stratified_split(labels, seed)
        all_ids = data["all_ids"].astype(str)
        if (not np.array_equal(data["train_indices"], train_indices)
                or not np.array_equal(data["validation_indices"], val_indices)
                or train_ids != all_ids[train_indices].tolist()
                or valid_ids != all_ids[val_indices].tolist()
                or set(train_ids) & set(valid_ids)):
            raise ValueError("Student split provenance is inconsistent")
        if any(not value.startswith(subject + ":T:") for value in all_ids):
            raise ValueError("Only this subject's T trials may fit/select a student")
        if data["mean"].shape != (22, 1125) or data["scale"].shape != (22, 1125) or not np.isfinite(data["mean"]).all() or not np.isfinite(data["scale"]).all() or (data["scale"] <= 0).any():
            raise ValueError("Invalid student scaler")
        rng = np.random.Generator(np.random.PCG64(order_seed(subject, seed, retained)))
        orders = np.stack([rng.permutation(len(train_ids)) for _ in range(len(history) - 1)])
        if not np.array_equal(data["epoch_orders"], orders):
            raise ValueError("Student batch order changed")
        masks = np.stack([static_masks(subject, seed, train_ids, "train", retained,
                                      epoch=e, namespace=config["mask_namespace"])
                          for e in range(1, len(history))])
        valid_masks = np.stack([static_masks(subject, seed, valid_ids, "validation", retained,
                                            repeat=r, namespace=config["mask_namespace"])
                               for r in range(config["mask_repeats"])])
        if not np.array_equal(data["train_masks"], masks) or not np.array_equal(data["validation_masks"], valid_masks):
            raise ValueError("Student masks differ from the paired protocol")
        if config["experiment"] == "q1":
            if "teacher_logits" in data.files or meta["teacher"]:
                raise ValueError("Q1 artifact contains teacher supervision")
            if any(r.get("distillation_kl", 0.) != 0 for r in history):
                raise ValueError("Q1 history contains a distillation objective")
        elif (data["teacher_logits"].shape != (len(train_ids), 4)
              or not np.isfinite(data["teacher_logits"]).all()
              or meta["teacher"]["teacher_target_partition"] != "T-train-only"
              or meta["teacher"]["teacher_state_before"] != meta["teacher"]["teacher_state_after"]):
            raise ValueError("Q2 teacher provenance is invalid")
    return meta


def fit_task(out, subject, seed, config, identity):
    with lock(out.parent / ("." + out.name + f"-{subject}-s{seed}.lock")):
        if (out / "fit-seal.json").exists():
            raise ValueError("Student fitting is closed after cohort sealing; use a fresh output for another experiment")
        record = recording(subject, "T", identity)
        tr, va = stratified_split(record.labels, seed)
        mean, scale = fit_native_normalization(record.values[tr])
        full = np.ones((len(record.values), 22), np.bool_)
        values = normalize(record.values, mean, scale, full).astype(np.float32)
        targets, teacher_meta, teacher_scaler = teacher_targets(subject, record, tr, config)
        for retained in config["retained_counts"]:
            name = fit_name(subject, seed, retained)
            final = out / "fits" / name
            if final.exists():
                verify_fit(out, subject, seed, retained, config)
                continue
            folder = out / "fits" / ("." + name + ".partial")
            if folder.exists():
                raise ValueError("An interrupted fit is preserved; choose a fresh output: " + name)
            folder.mkdir()
            tf.keras.backend.clear_session()
            student = Student(subject, seed, retained)
            initial_state = student.initial_state_sha256
            result = fit(student, values[tr], record.labels[tr], [record.trial_ids[i] for i in tr],
                         values[va], record.labels[va], [record.trial_ids[i] for i in va],
                         subject, seed, config, folder, targets)
            # Persist selected weights, then reconstruct and replay T validation.
            student.model.save_weights(str(folder / "student.weights.h5"), save_format="h5")
            selected_state = state_digest(student.model)
            parameter_count = student.model.count_params()
            trainable_count = int(sum(np.prod(v.shape) for v in student.model.trainable_variables))
            del student
            tf.keras.backend.clear_session()
            replay = Student(subject, seed, retained)
            if replay.initial_state_sha256 != initial_state:
                raise RuntimeError("Student initialization is not reproducible under this runtime")
            replay.model.load_weights(str(folder / "student.weights.h5"))
            replay_record, replay_probabilities = validation(
                replay, values[va], record.labels[va], result["validation_masks"],
                config["training"]["batch_size"], result["selected_epoch"])
            if state_digest(replay.model) != selected_state:
                raise RuntimeError("Selected student H5 tensor replay differs")
            np.testing.assert_allclose(replay_probabilities, result["selected_validation_probabilities"], atol=1e-6, rtol=1e-5)
            chosen = selection(result["history"])
            for field in ("validation_ba", "validation_log_loss"):
                if not np.isclose(replay_record[field], chosen[field], atol=1e-6, rtol=0):
                    raise RuntimeError("Selected student validation replay differs")
            arrays = {"mean": mean, "scale": scale, "train_indices": tr, "validation_indices": va,
                      "all_labels": record.labels, "all_ids": np.asarray(record.trial_ids),
                      "train_ids": np.asarray([record.trial_ids[i] for i in tr]),
                      "validation_ids": np.asarray([record.trial_ids[i] for i in va]),
                      "train_masks": result["train_masks"], "validation_masks": result["validation_masks"],
                      "epoch_orders": result["epoch_orders"],
                      "validation_probabilities": replay_probabilities}
            if targets is not None:
                arrays.update(teacher_logits=targets, teacher_mean=teacher_scaler[0], teacher_scale=teacher_scaler[1])
            write_npz(folder / "training.npz", **arrays)
            write_json(folder / "history.json", result["history"], replace=True)
            finish_task(folder, {"schema": 1, "identity_sha256": sha256(out / "identity.json"),
                                 "experiment": config["experiment"], "subject": subject, "seed": seed,
                                 "retained": retained, "selected_epoch": result["selected_epoch"],
                                 "epochs_completed": len(result["history"]) - 1,
                                 "initialization_seed": initialization_seed(subject, seed, retained),
                                 "initial_state_sha256": initial_state, "selected_state_sha256": selected_state,
                                 "parameter_count": parameter_count, "trainable_parameter_count": trainable_count,
                                 "normalization_fit_ids": [record.trial_ids[i] for i in tr],
                                 "source_T_sha256": record.metadata["source_sha256"],
                                 "training_seconds": result["training_seconds"], "teacher": teacher_meta,
                                 "selected_validation_replayed": True})
            folder.rename(final)
            verify_fit(out, subject, seed, retained, config)
            del replay


def expected_seal(out, config, *, audit_masks=True):
    entries = []
    for subject, seed in tasks(config):
        for retained in config["retained_counts"]:
            folder = out / "fits" / fit_name(subject, seed, retained)
            if audit_masks:
                verify_fit(out, subject, seed, retained, config)
            else:
                # Sealing already replayed every training mask. Subsequent E
                # tasks check exact sealed bytes without repeating that work.
                verify_task(folder, sha256(out / "identity.json"))
            entries.append({"task": folder.name, "receipt_sha256": sha256(folder / "receipt.json")})
    return {"schema": 1, "identity_sha256": sha256(out / "identity.json"),
            "student_fits": len(entries), "tasks": entries}


def seal_cohort(out, config):
    write_json(out / "fit-seal.json", expected_seal(out, config))


def require_seal(out, config):
    # Hash checks bind all selected weights/splits/masks before any E load.
    seal = json.loads((out / "fit-seal.json").read_text())
    if seal != expected_seal(out, config, audit_masks=False):
        raise ValueError("The complete declared student cohort must be sealed before E evaluation")
    return sha256(out / "fit-seal.json")


def evaluate_task(out, subject, seed, config, identity):
    seal_hash = require_seal(out, config)
    with lock(out.parent / ("." + out.name + f"-{subject}-s{seed}.lock")):
        name = f"{subject}-s{seed}"
        final = out / "evaluation" / name
        if final.exists():
            verify_evaluation(out, subject, seed, config)
            return
        folder = out / "evaluation" / ("." + name + ".partial")
        if folder.exists():
            raise ValueError("An interrupted E task is preserved; choose a fresh output")
        # E signals/labels are first opened here, after the whole-cohort barrier.
        record = recording(subject, "E", identity)
        folder.mkdir()
        rows = []
        for retained in config["retained_counts"]:
            fit_folder = out / "fits" / fit_name(subject, seed, retained)
            with np.load(fit_folder / "training.npz", allow_pickle=False) as data:
                mean, scale = data["mean"].copy(), data["scale"].copy()
            tf.keras.backend.clear_session()
            student = Student(subject, seed, retained)
            student.model.load_weights(str(fit_folder / "student.weights.h5"))
            before = state_digest(student.model)
            for repeat in range(config["mask_repeats"]):
                mask = static_masks(subject, seed, record.trial_ids, "E", retained, repeat=repeat,
                                    namespace=config["mask_namespace"])
                # Hidden raw values are removed before normalization and encoding.
                values = normalize(record.values, mean, scale, mask).astype(np.float32)
                predict(student, values[:2], mask[:2], config["training"]["batch_size"])
                started = time.perf_counter()
                probabilities = predict(student, values, mask, config["training"]["batch_size"])
                seconds = time.perf_counter() - started
                # Reloaded frozen-inference graph replay uses the same batch size.
                replay = predict(student, values, mask, config["training"]["batch_size"])
                np.testing.assert_allclose(replay, probabilities, atol=1e-6, rtol=1e-5)
                score = metrics(probabilities, record.labels)
                filename = f"static_{retained}-r{repeat}.npz"
                write_npz(folder / filename, ids=np.asarray(record.trial_ids), labels=record.labels,
                          mask=mask, probabilities=probabilities)
                rows.append({"subject": subject, "seed": seed, "retained": retained,
                             "repeat": repeat, "inference_seconds": seconds, **score})
            if before != state_digest(student.model):
                raise RuntimeError("Student changed during E inference")
            del student
        write_json(folder / "rows.json", rows)
        finish_task(folder, {"schema": 1, "identity_sha256": sha256(out / "identity.json"),
                             "subject": subject, "seed": seed, "experiment": config["experiment"],
                             "fit_seal_sha256": seal_hash, "source_E_sha256": record.metadata["source_sha256"],
                             "prediction_replayed": True, "teacher_used_at_inference": False})
        folder.rename(final)
        verify_evaluation(out, subject, seed, config)


def verify_evaluation(out, subject, seed, config):
    folder = out / "evaluation" / f"{subject}-s{seed}"
    meta = verify_task(folder, sha256(out / "identity.json"))
    if (meta["subject"], meta["seed"], meta["experiment"]) != (subject, seed, config["experiment"]):
        raise ValueError("Evaluation belongs to another task")
    if meta["fit_seal_sha256"] != sha256(out / "fit-seal.json") or meta["teacher_used_at_inference"]:
        raise ValueError("E evaluation is not bound to sealed students alone")
    rows = json.loads((folder / "rows.json").read_text())
    if {(r["retained"], r["repeat"]) for r in rows} != {
            (k, r) for k in config["retained_counts"] for r in range(config["mask_repeats"])} or len(rows) != len(config["retained_counts"]) * config["mask_repeats"]:
        raise ValueError("Evaluation has missing/duplicate/foreign cells")
    ids_reference, labels_reference = None, None
    for row in rows:
        with np.load(folder / f"static_{row['retained']}-r{row['repeat']}.npz", allow_pickle=False) as data:
            ids, y = data["ids"].astype(str).tolist(), data["labels"]
            if len(ids) != 288 or len(set(ids)) != 288 or any(not i.startswith(subject + ":E:") for i in ids):
                raise ValueError("E trial identities changed")
            if ids_reference is not None and (ids != ids_reference or not np.array_equal(y, labels_reference)):
                raise ValueError("E labels/IDs differ across conditions")
            ids_reference, labels_reference = ids, y.copy()
            expected_mask = static_masks(subject, seed, ids, "E", row["retained"], repeat=row["repeat"],
                                         namespace=config["mask_namespace"])
            if not np.array_equal(expected_mask, data["mask"]):
                raise ValueError("Saved E mask differs from the paired covariance protocol")
            score = metrics(data["probabilities"], y)
            for field in ("balanced_accuracy", "accuracy", "log_loss", "n_trials"):
                if not np.isclose(row[field], score[field], atol=1e-12, rtol=0):
                    raise ValueError("E metric differs from saved probabilities")
    return rows
