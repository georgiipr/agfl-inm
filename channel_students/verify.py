"""Independent NumPy-only verification of student results; no model calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from .artifacts import manifest, safe_path, write_json
from .protocol import sha256


def check(condition, message):
    if not condition:
        raise ValueError(message)


def numeric_manifest(out):
    ignored = {"audit.json", "report.json", "report.md", "verification.json"}
    return {name: digest for name, digest in manifest(out).items() if Path(name).name not in ignored}


def score(probabilities, labels):
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels)
    check(y.shape == (len(p),) and set(y.tolist()) == {0, 1, 2, 3}, "Invalid test labels")
    check(p.shape == (len(y), 4) and np.isfinite(p).all() and (p >= 0).all(), "Invalid probabilities")
    np.testing.assert_allclose(p.sum(1), 1, atol=1e-5, rtol=0)
    predicted = p.argmax(1)
    return np.asarray([np.mean([np.mean(predicted[y == c] == c) for c in range(4)]),
                       np.mean(predicted == y),
                       -np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)))])


def verify(out):
    out = safe_path(out)
    ident_path = out / "identity.json"
    identity = json.loads(ident_path.read_text())
    config = identity["configuration"]
    report = json.loads((out / "report.json").read_text())
    audit = json.loads((out / "audit.json").read_text())
    check(report["complete"] is True and audit["passed"] is True, "Incomplete student report/audit")
    check(report["configuration"] == config and report["experiment"] == config["experiment"], "Report configuration changed")
    check(report["identity_sha256"] == audit["identity_sha256"] == sha256(ident_path), "Report identity changed")
    inventory = numeric_manifest(out)
    check(inventory == audit["numeric_artifacts"], "Audited artifact bytes/inventory changed")
    seal = json.loads((out / "fit-seal.json").read_text())
    expected_tasks = [(s, seed) for s in config["subjects"] for seed in config["seeds"]]
    expected_fits = {f"{s}-s{seed}-k{k}" for s, seed in expected_tasks for k in config["retained_counts"]}
    check({row["task"] for row in seal["tasks"]} == expected_fits and len(seal["tasks"]) == len(expected_fits), "Fit seal is incomplete")
    check(seal["identity_sha256"] == sha256(ident_path) and seal["student_fits"] == len(expected_fits), "Fit seal identity/count differs")
    admission = json.loads((out / "admission" / "gpu-checks.json").read_text())
    check(admission["passed"] and admission["identity_sha256"] == sha256(ident_path), "Missing identity-bound GPU admission")
    check(admission["E_signals_or_labels_opened"] is False, "Admission must not score E")
    values, fits = {}, []
    for subject, seed in expected_tasks:
        task = out / "evaluation" / f"{subject}-s{seed}"
        receipt = json.loads((task / "receipt.json").read_text())
        check(receipt["files"] == manifest(task), "Prediction task bytes changed")
        check(receipt["identity_sha256"] == sha256(ident_path) and receipt["fit_seal_sha256"] == sha256(out / "fit-seal.json"), "Prediction task identity/seal changed")
        check(receipt["source_E_sha256"] == identity["assets"]["recordings"][subject + "E"] and receipt["teacher_used_at_inference"] is False, "Prediction source/inference provenance differs")
        rows = json.loads((task / "rows.json").read_text())
        check(len(rows) == len(config["retained_counts"]) * 5 and {(r["retained"], r["repeat"]) for r in rows} == {
            (k, r) for k in config["retained_counts"] for r in range(5)}, "Evaluation cells incomplete")
        expected_files = {f"static_{k}-r{r}.npz" for k in config["retained_counts"] for r in range(5)} | {"rows.json"}
        check(set(receipt["files"]) == expected_files, "Unexpected prediction task inventory")
        for retained in config["retained_counts"]:
            folder = out / "fits" / f"{subject}-s{seed}-k{retained}"
            fit = json.loads((folder / "receipt.json").read_text())
            seal_row = next(r for r in seal["tasks"] if r["task"] == folder.name)
            check(sha256(folder / "receipt.json") == seal_row["receipt_sha256"], "Fit receipt changed after sealing")
            check(fit["files"] == manifest(folder) and fit["selected_validation_replayed"] is True, "Fit bytes/replay receipt differs")
            check(fit["identity_sha256"] == sha256(ident_path) and fit["source_T_sha256"] == identity["assets"]["recordings"][subject + "T"], "Fit source identity differs")
            history = json.loads((folder / "history.json").read_text())
            chosen = min(history, key=lambda r: (-r["validation_ba"], r["validation_log_loss"], r["epoch"]))
            check(chosen["epoch"] == fit["selected_epoch"] and len(history) - 1 == fit["epochs_completed"], "Validation selection differs")
            with np.load(folder / "training.npz", allow_pickle=False) as data:
                tr, va = data["train_indices"], data["validation_indices"]
                all_ids = data["all_ids"].astype(str)
                check(len(all_ids) == 288 and len(tr) == 232 and len(va) == 56, "Native training/validation trial counts changed")
                check(not set(tr.tolist()) & set(va.tolist()) and set(tr.tolist()) | set(va.tolist()) == set(range(288)), "Training/validation indices overlap or omit trials")
                check(fit["normalization_fit_ids"] == all_ids[tr].tolist(), "Student statistics include nontraining trials")
                np.testing.assert_array_equal(data["train_ids"].astype(str), all_ids[tr])
                np.testing.assert_array_equal(data["validation_ids"].astype(str), all_ids[va])
                check(all(i.startswith(subject + ":T:") for i in all_ids), "Student used non-T trial identities")
                check(data["train_masks"].shape == (len(history) - 1, 232, 22) and np.all(data["train_masks"].sum(-1) == retained), "Student observed an undeclared channel count")
                check(data["validation_masks"].shape == (5, 56, 22) and np.all(data["validation_masks"].sum(-1) == retained), "Validation mask dimensions/count differ")
                v_scores = np.mean([score(p, data["all_labels"][va]) for p in data["validation_probabilities"]], axis=0)
                np.testing.assert_allclose([chosen["validation_ba"], chosen["validation_log_loss"]], v_scores[[0, 2]], atol=1e-6, rtol=0)
                if config["experiment"] == "q1":
                    check("teacher_logits" not in data.files and not fit["teacher"], "Q1 contains teacher supervision")
                    check(all(r.get("distillation_kl", 0) == 0 for r in history), "Q1 has distillation loss")
                else:
                    check(data["teacher_logits"].shape == (232, 4) and np.isfinite(data["teacher_logits"]).all(), "Q2 teacher targets are not training-only")
                    check(fit["teacher"]["teacher_state_before"] == fit["teacher"]["teacher_state_after"], "Teacher changed during fitting")
                    checkpoint = f"results/saved models/run-1/subject-{int(subject[1:])}.h5"
                    check(fit["teacher"]["teacher_checkpoint_sha256"] == identity["assets"]["files"][checkpoint], "Teacher checkpoint differs")
            fits.append(fit)
            repeats = []
            for repeat in range(5):
                with np.load(task / f"static_{retained}-r{repeat}.npz", allow_pickle=False) as data:
                    ids = data["ids"].astype(str)
                    labels = data["labels"]
                    check(len(ids) == 288 and len(set(ids)) == 288 and all(i.startswith(subject + ":E:") for i in ids), "E trial identities changed")
                    check(data["mask"].shape == (288, 22) and data["mask"].dtype == np.bool_ and np.all(data["mask"].sum(1) == retained), "Invalid E availability mask")
                    result = score(data["probabilities"], labels)
                    row = next(r for r in rows if (r["retained"], r["repeat"]) == (retained, repeat))
                    np.testing.assert_allclose([row["balanced_accuracy"], row["accuracy"], row["log_loss"]], result, atol=1e-12, rtol=0)
                    repeats.append(result)
            values[(subject, seed, retained)] = np.mean(repeats, axis=0)
    fields = ("balanced_accuracy", "accuracy", "log_loss")
    participant = {(s, k): np.mean([values[(s, seed, k)] for seed in config["seeds"]], axis=0)
                   for s in config["subjects"] for k in config["retained_counts"]}
    for subject in config["subjects"]:
        for retained in config["retained_counts"]:
            np.testing.assert_allclose([report["participants"][subject][str(retained)][f] for f in fields], participant[(subject, retained)], atol=1e-12, rtol=0)
    for retained in config["retained_counts"]:
        expected = np.mean([participant[(s, retained)] for s in config["subjects"]], axis=0)
        np.testing.assert_allclose([report["cohort"][str(retained)][f] for f in fields], expected, atol=1e-12, rtol=0)
    check(report["fits"] == fits, "Report fit metadata changed")
    check(report["student_fits"] == len(expected_fits) == len(fits) and report["evaluation_cells"] == len(expected_tasks) * len(config["retained_counts"]) * 5, "Report counts changed")
    return {"passed": True, "experiment": config["experiment"], "student_fits": len(fits),
            "evaluation_cells": len(expected_tasks) * len(config["retained_counts"]) * 5, "identity_sha256": sha256(ident_path),
            "report_sha256": sha256(out / "report.json"), "numeric_artifacts": inventory,
            "verification": "Independent saved-probability metric and aggregation recomputation; no model calls"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.output)
    write_json(args.output / "verification.json", result)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
