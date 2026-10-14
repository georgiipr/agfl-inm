"""Compare three completed studies without fitting, model calls or rewriting them."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from .artifacts import safe_path, write_json
from .protocol import ROOT, sha256
from .verify import verify, check, score


def compare(covariance, q1, q2):
    for output in (q1, q2):
        receipt = verify(output)
        check(receipt == json.loads((output / "verification.json").read_text()), "A completed independently verified student report is required")
    identities = {key: json.loads((path / "identity.json").read_text())
                  for key, path in (("covariance", covariance), ("q1", q1), ("q2", q2))}
    cfg1, cfg2 = identities["q1"]["configuration"], identities["q2"]["configuration"]
    check(cfg1["experiment"] == "q1" and cfg2["experiment"] == "q2", "Student report experiment selectors are swapped")
    ignored = {"experiment", "study_id", "distillation", "output"}
    check({k: v for k, v in cfg1.items() if k not in ignored} == {k: v for k, v in cfg2.items() if k not in ignored},
          "Q1/Q2 architecture, cohort, masks or training budget differ; refuse an unmatched distillation claim")
    cov_ident = identities["covariance"]
    audit = json.loads((covariance / "audit.json").read_text())
    check(audit["passed"] is True and audit["identity_sha256"] == sha256(covariance / "identity.json"), "Covariance report lacks its audit")
    for name, digest in audit["numeric_artifacts"].items():
        check(sha256(safe_path(covariance / name)) == digest, "Covariance artifact changed: " + name)
    cov_report = json.loads((covariance / "report.json").read_text())
    cov_verification = json.loads((covariance / "independent-verification.json").read_text())
    check(cov_report["complete"] is True and cov_verification["passed"] is True and cov_verification["report_sha256"] == sha256(covariance / "report.json"), "Complete independently verified covariance report required")
    for subject in cfg1["subjects"]:
        for session in ("T", "E"):
            check(cov_ident["recording_bytes_sha256"][subject + session] == identities["q1"]["assets"]["recordings"][subject + session] == identities["q2"]["assets"]["recordings"][subject + session], "Native recording sources differ")
    paired = {"covariance_learned": {}, "q1": {}, "q2": {}}
    initializations = []
    for subject in cfg1["subjects"]:
        for retained in cfg1["retained_counts"]:
            values = {name: [] for name in paired}
            for seed in cfg1["seeds"]:
                fit1 = q1 / "fits" / f"{subject}-s{seed}-k{retained}"
                fit2 = q2 / "fits" / f"{subject}-s{seed}-k{retained}"
                receipts = [json.loads((f / "receipt.json").read_text()) for f in (fit1, fit2)]
                check(receipts[0]["initial_state_sha256"] == receipts[1]["initial_state_sha256"], "Q1/Q2 initial weights differ")
                check(receipts[0]["parameter_count"] == receipts[1]["parameter_count"], "Q1/Q2 student architectures differ")
                initializations.append(receipts[0]["initial_state_sha256"])
                with np.load(fit1 / "training.npz", allow_pickle=False) as f1, np.load(fit2 / "training.npz", allow_pickle=False) as f2, np.load(covariance / "fits" / f"{subject}-s{seed}.npz", allow_pickle=False) as cov_fit:
                    for field in ("train_indices", "validation_indices", "train_ids", "validation_ids"):
                        np.testing.assert_array_equal(f1[field], f2[field])
                        np.testing.assert_array_equal(f1[field], cov_fit[field])
                    for field in ("mean", "scale", "validation_masks"):
                        np.testing.assert_array_equal(f1[field], f2[field])
                    common_epochs = min(len(f1["epoch_orders"]), len(f2["epoch_orders"]))
                    np.testing.assert_array_equal(f1["epoch_orders"][:common_epochs], f2["epoch_orders"][:common_epochs])
                    np.testing.assert_array_equal(f1["train_masks"][:common_epochs], f2["train_masks"][:common_epochs])
                repeats = {name: [] for name in paired}
                for repeat in range(cfg1["mask_repeats"]):
                    files = {
                        "covariance_learned": covariance / "evaluation" / f"{subject}-s{seed}-covariance_learned-static_{retained}-r{repeat}.npz",
                        "q1": q1 / "evaluation" / f"{subject}-s{seed}" / f"static_{retained}-r{repeat}.npz",
                        "q2": q2 / "evaluation" / f"{subject}-s{seed}" / f"static_{retained}-r{repeat}.npz",
                    }
                    reference = None
                    for name, path in files.items():
                        with np.load(path, allow_pickle=False) as data:
                            if reference is None:
                                reference = {field: data[field].copy() for field in ("ids", "labels", "mask")}
                            else:
                                for field in reference:
                                    np.testing.assert_array_equal(data[field], reference[field])
                            repeats[name].append(score(data["probabilities"], data["labels"]))
                for name in paired:
                    values[name].append(np.mean(repeats[name], axis=0))
            for name in paired:
                paired[name][(subject, retained)] = np.mean(values[name], axis=0)
    cohort = {name: {str(k): dict(zip(("balanced_accuracy", "accuracy", "log_loss"),
                       np.mean([cells[(s, k)] for s in cfg1["subjects"]], axis=0).tolist()))
                     for k in cfg1["retained_counts"]} for name, cells in paired.items()}
    effects = {}
    for label, left, right in (("q1_minus_covariance", "q1", "covariance_learned"),
                               ("q2_minus_covariance", "q2", "covariance_learned"),
                               ("distillation_q2_minus_q1", "q2", "q1")):
        participant = {s: float(np.mean([paired[left][(s, k)][0] - paired[right][(s, k)][0]
                                        for k in cfg1["retained_counts"]])) for s in cfg1["subjects"]}
        vector = np.asarray(list(participant.values()))
        rng = np.random.Generator(np.random.PCG64(cfg1["bootstrap"]["seed"]))
        interval = np.quantile(rng.choice(vector, size=(cfg1["bootstrap"]["draws"], len(vector)), replace=True).mean(1), [.025, .975])
        effects[label] = {"mean": float(vector.mean()), "bootstrap_95": interval.tolist(), "participants": participant}
    return {"schema": 1, "complete": True, "subjects": cfg1["subjects"], "seeds": cfg1["seeds"],
            "cohort": cohort, "paired_effects": effects, "paired_initializations": initializations,
            "source_identities": {name: sha256(path / "identity.json") for name, path in (("covariance", covariance), ("q1", q1), ("q2", q2))},
            "interpretation": "Exploratory paired comparison. Different adaptation/training budgets: covariance updates completion only; students update the classifier. Student scaling uses T-train only, frozen published teacher/classifier uses historical all-T scaling. Q2 inherits unresolved external teacher checkpoint selection. No untouched-test claim."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--covariance", type=Path, required=True)
    parser.add_argument("--q1", type=Path, required=True)
    parser.add_argument("--q2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = safe_path(args.output)
    if out.parent != ROOT / "results" or not out.name.startswith("channel-comparison-") or out.exists():
        raise ValueError("Choose a fresh direct results/channel-comparison-* directory")
    result = compare(safe_path(args.covariance), safe_path(args.q1), safe_path(args.q2))
    out.mkdir()
    write_json(out / "report.json", result)
    lines = ["# Covariance, supervised student and distilled student", "", result["interpretation"], "",
             "All approaches use identical E trials and masks. Student pairs also share splits, initial weights and the overlapping training schedule.", "",
             "| Approach | Available channels | Balanced accuracy (%) | Accuracy (%) | Log loss |", "|---|---:|---:|---:|---:|"]
    for name, cells in result["cohort"].items():
        for k, value in cells.items():
            lines.append(f"| {name} | {k} | {value['balanced_accuracy'] * 100:.2f} | {value['accuracy'] * 100:.2f} | {value['log_loss']:.4f} |")
    lines += ["", "## Paired effects averaged over 16 and 6 channels", "",
              "| Comparison | BA difference (pp) | Participant-bootstrap 95% interval (pp) |", "|---|---:|---:|"]
    for name, value in result["paired_effects"].items():
        lo, hi = value["bootstrap_95"]
        lines.append(f"| {name} | {value['mean'] * 100:+.2f} | [{lo * 100:+.2f}, {hi * 100:+.2f}] |")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"complete": True, "report": str(out / "report.md")}))


if __name__ == "__main__":
    main()
