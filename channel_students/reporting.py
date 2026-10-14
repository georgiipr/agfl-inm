"""Complete-cohort student reports, recomputed from saved per-trial predictions."""
from __future__ import annotations

import json
import numpy as np

from .artifacts import write_json, safe_path
from .protocol import tasks, sha256
from .study import require_seal, verify_evaluation


def numeric_manifest(out):
    ignored = {"audit.json", "report.json", "report.md", "verification.json"}
    result = {}
    for path in sorted(out.rglob("*")):
        safe_path(path)
        if path.is_file() and path.name not in ignored:
            result[str(path.relative_to(out))] = sha256(path)
    return result


def summarize(out, config):
    require_seal(out, config)
    rows, fits = [], []
    for subject, seed in tasks(config):
        rows.extend(verify_evaluation(out, subject, seed, config))
        for retained in config["retained_counts"]:
            folder = out / "fits" / f"{subject}-s{seed}-k{retained}"
            fits.append(json.loads((folder / "receipt.json").read_text()))
    participants = {}
    fields = ("balanced_accuracy", "accuracy", "log_loss")
    for subject in config["subjects"]:
        participants[subject] = {}
        for retained in config["retained_counts"]:
            by_seed = [{field: float(np.mean([r[field] for r in rows
                        if (r["subject"], r["seed"], r["retained"]) == (subject, seed, retained)]))
                        for field in fields} for seed in config["seeds"]]
            participants[subject][str(retained)] = {field: float(np.mean([r[field] for r in by_seed]))
                                                   for field in fields}
    cohort = {str(k): {field: float(np.mean([participants[s][str(k)][field] for s in config["subjects"]]))
                       for field in fields} for k in config["retained_counts"]}
    for retained in config["retained_counts"]:
        cohort[str(retained)]["measured_inference_seconds_per_trial"] = float(np.mean([
            r["inference_seconds"] / r["n_trials"] for r in rows if r["retained"] == retained]))
    disclosures = {
        "interpretation": "Exploratory comparison; all declared participants retained. No untouched-test or confirmatory claim.",
        "student_normalization": "Fitted on this seed's T training trials only",
        "teacher_used_for_student_fitting": config["distillation"]["enabled"],
        "teacher_used_at_inference": False,
        "teacher_normalization": "Historical all-T scaler, including the later student-validation trials" if config["experiment"] == "q2" else "Not used for student fitting",
        "teacher_checkpoint_grade": "B" if config["experiment"] == "q2" else "Not used for student fitting",
        "teacher_E_selection_excluded": False if config["experiment"] == "q2" else None,
        "prior_project_E_outcome_exposure": "unknown",
        "input": "22 electrode identity slots; exactly 16 or 6 observed signals, remaining slots zero; no completion",
        "mask_generalization": "Varying whole-trial subsets; train/validation/E subset pools reserved separately",
        "comparison_to_covariance": "Students use T-train-only normalization; published frozen classifier retains historical all-T normalization. Classifier fitting budgets and trainable parameter counts differ.",
    }
    result = {"schema": 1, "complete": True, "experiment": config["experiment"],
              "subjects": config["subjects"], "seeds": config["seeds"],
              "student_fits": len(fits), "evaluation_cells": len(rows),
              "identity_sha256": sha256(out / "identity.json"),
              "configuration": config, "cohort": cohort, "participants": participants,
              "fits": fits, "disclosures": disclosures}
    write_json(out / "audit.json", {"passed": True, "identity_sha256": result["identity_sha256"],
                                   "student_fits": len(fits), "evaluation_cells": len(rows),
                                   "numeric_artifacts": numeric_manifest(out)})
    write_json(out / "report.json", result)
    lines = [f"# {config['experiment'].upper()} reduced-channel student report", "", disclosures["interpretation"], "",
             "Q1 trains from class labels; Q2 adds frozen full-channel teacher guidance. Both students classify observed signals directly.", "",
             f"Completed {len(fits)} student fits and {len(rows)} evaluation cells. Average repeats, then seeds, then participants equally.", "",
             "| Available channels | Balanced accuracy (%) | Accuracy (%) | Log loss | Inference ms/trial |",
             "|---|---:|---:|---:|---:|"]
    for k, score in cohort.items():
        lines.append(f"| {k} | {score['balanced_accuracy'] * 100:.2f} | {score['accuracy'] * 100:.2f} | {score['log_loss']:.4f} | {score['measured_inference_seconds_per_trial'] * 1000:.4f} |")
    lines += ["", "## Participant results", "", "| Participant | Channels | Balanced accuracy (%) | Accuracy (%) |", "|---|---:|---:|---:|"]
    for s, conditions in participants.items():
        for k, score in conditions.items():
            lines.append(f"| {s} | {k} | {score['balanced_accuracy'] * 100:.2f} | {score['accuracy'] * 100:.2f} |")
    lines += ["", "## Fitting and selection", "", "| Participant | Seed | Channels | Selected epoch | Completed epochs | Parameters | Trainable parameters | Training seconds |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for f in fits:
        lines.append(f"| {f['subject']} | {f['seed']} | {f['retained']} | {f['selected_epoch']} | {f['epochs_completed']} | {f['parameter_count']} | {f['trainable_parameter_count']} | {f['training_seconds']:.1f} |")
    kd = config["distillation"]
    lines += ["", "## Protocol", "", f"Distillation enabled: {kd['enabled']}; weight: {kd['weight']}; temperature: {kd['temperature']}.",
              "Q1 objective: class cross entropy plus native ATCNet kernel regularization. Q2 adds weight × temperature² × KL(teacher || student).",
              "Both students start from identical random initialization per participant/seed/channel count. No teacher weights initialize either student.",
              "Validation uses class-label balanced accuracy, then log loss, then earlier epoch. Test outcomes never select a checkpoint.",
              "Inference timing includes warmed batched model calls, transfers and probability materialization; it excludes normalization and a separate verification replay. It is environment-specific.",
              "", "## Interpretation and provenance", ""]
    lines += [f"- {key}: {value}" for key, value in disclosures.items()]
    text = "\n".join(lines) + "\n"
    path = out / "report.md"
    if path.exists() and path.read_text() != text:
        raise ValueError("Existing human report differs")
    if not path.exists():
        with safe_path(path).open("x") as stream:
            stream.write(text)
    return result
