"""Independent hierarchical aggregation for published covariance cells."""
from __future__ import annotations
import math
import numpy as np

ARMS = ("zero", "covariance_fixed", "covariance_learned")
CONDITIONS = ("full_22", "static_16", "static_6")

def summarize_cells(rows):
    rows = list(rows)
    keyset = set()
    by = {}
    for r in rows:
        required = {"subject", "seed", "arm", "condition", "repeat", "balanced_accuracy", "accuracy", "log_loss", "n_trials"}
        if not isinstance(r, dict) or not required <= r.keys(): raise ValueError("malformed result cell")
        key = tuple(r[k] for k in ("subject", "seed", "arm", "condition", "repeat"))
        if key in keyset: raise ValueError(f"duplicate result cell: {key}")
        keyset.add(key)
        if r["arm"] not in ARMS or r["condition"] not in CONDITIONS: raise ValueError("unknown arm or condition")
        if not all(math.isfinite(float(r[k])) for k in ("balanced_accuracy", "accuracy", "log_loss")) or int(r["n_trials"]) < 1: raise ValueError("invalid numeric cell")
        by[key] = r
    subjects = sorted({str(r["subject"]) for r in rows})
    seeds = sorted({int(r["seed"]) for r in rows})
    expected = {(s, seed, arm, cond, rep) for s in subjects for seed in seeds for arm in ARMS for cond in CONDITIONS for rep in range(1 if cond == "full_22" else 5)}
    complete = bool(subjects) and keyset == expected and len(subjects) == 9 and seeds == [0, 1, 2]
    result = {"complete": complete, "checkpoint_provenance_grade": "B", "prior_project_E_outcome_exposure": "unknown", "external_checkpoint_E_selection_excluded": False, "interpretation": "Exploratory published-checkpoint robustness; grade B selection lineage and prior E exposure unknown; no confirmatory claim.", "cell_count": len(rows)}
    if not complete: return result
    def mean_for(subject, arm, condition):
        vals = []
        for seed in seeds:
            reps = range(1 if condition == "full_22" else 5)
            vals.append(np.mean([by[(subject, seed, arm, condition, rep)]["balanced_accuracy"] for rep in reps]))
        return float(np.mean(vals))
    cohort = {}
    for arm in ARMS:
        degraded = [by[k]["balanced_accuracy"] for k in by if k[2] == arm and k[3] != "full_22"]
        full = float(np.mean([by[k]["balanced_accuracy"] for k in by if k[2] == arm and k[3] == "full_22"]))
        conditions={}
        for cond in CONDITIONS:
            cells=[by[k] for k in by if k[2]==arm and k[3]==cond]
            ba=float(np.mean([r["balanced_accuracy"] for r in cells]))
            conditions[cond]={"balanced_accuracy":ba,"accuracy":float(np.mean([r["accuracy"] for r in cells])),"log_loss":float(np.mean([r["log_loss"] for r in cells])),"degradation_from_full_ba":float(full-ba)}
        cohort[arm] = {"degraded_ba": float(np.mean(degraded)), "full_22_ba":full,"conditions":conditions}
    diffs = {s: float(np.mean([mean_for(s, "covariance_learned", c) - mean_for(s, "covariance_fixed", c) for c in ("static_16", "static_6")])) for s in subjects}
    vals = np.asarray([diffs[s] for s in subjects], dtype=np.float64)
    learned_zero={s:float(np.mean([mean_for(s,"covariance_learned",c)-mean_for(s,"zero",c) for c in ("static_16","static_6")])) for s in subjects}
    rng = np.random.Generator(np.random.PCG64(20261006))
    ci = np.quantile(rng.choice(vals, size=(2000, len(vals)), replace=True).mean(axis=1), [0.025, 0.975])
    result.update({"cohort": cohort, "participants": {s: {"learned_minus_fixed": diffs[s],"learned_minus_zero":learned_zero[s]} for s in subjects}, "secondary_learned_minus_zero": {"mean":float(np.mean(list(learned_zero.values()))),"participants":learned_zero}, "primary_learned_minus_fixed": {"mean": float(vals.mean()), "bootstrap_95": [float(ci[0]), float(ci[1])], "draws": 2000, "seed": 20261006, "unit": "participant"}})
    return result
