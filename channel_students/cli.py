"""Required experiment selector; each mode has independent settings and output."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .protocol import ROOT, load_configuration, tasks


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m channel_students")
    parser.add_argument("mode", choices=("plan", "admit", "fit-task", "seal-cohort", "evaluate-task", "report"))
    parser.add_argument("--experiment", choices=("q1", "q2"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--subjects", nargs="+", choices=tuple(f"A{i:02d}" for i in range(1, 10)))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--retained-counts", nargs="+", type=int, choices=(16, 6))
    parser.add_argument("--subject")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    args = parser.parse_args(argv)
    config, config_path = load_configuration(args.experiment, args.config, args.subjects, args.seeds, args.retained_counts)
    declared_tasks = tasks(config)
    if args.mode == "plan":
        print(json.dumps({"experiment": args.experiment, "student_fits": len(declared_tasks) * len(config["retained_counts"]),
                          "evaluation_cells": len(declared_tasks) * len(config["retained_counts"]) * config["mask_repeats"],
                          "teacher_supervision": config["distillation"]["enabled"],
                          "output": str(args.output or config["output"]),
                          "tasks": [{"subject": s, "seed": k} for s, k in declared_tasks]}, indent=2))
        return 0
    if args.mode in ("fit-task", "evaluate-task") and (args.subject, args.seed) not in declared_tasks:
        parser.error("Task must belong to the declared participant/seed cohort")
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("GPU TensorFlow is required; CPU fallback is forbidden")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.set_soft_device_placement(False)
    tf.random.set_seed(20261007)
    from .artifacts import lock, open_study
    from .protocol import ASSETS, identity
    if args.mode == "admit":
        from published_covariance.assets import prepare_assets
        with lock(ROOT / "assets" / ".atcnet-student-assets.lock"):
            prepare_assets(ASSETS, offline=True)
    expected = identity(config, config_path)
    out = open_study(config, args.output, expected, create=args.mode == "admit")
    admission = out / "admission" / "gpu-checks.json"
    with tf.device("/GPU:0"):
        if args.mode == "admit":
            from .admission import admit
            admit(out, config, expected)
        else:
            from .protocol import sha256
            admitted = json.loads(admission.read_text())
            if admitted["passed"] is not True or admitted["identity_sha256"] != sha256(out / "identity.json"):
                raise ValueError("Cluster GPU admission is missing or stale")
            from .study import fit_task, seal_cohort, evaluate_task
            if args.mode == "fit-task":
                fit_task(out, args.subject, args.seed, config, expected)
            elif args.mode == "seal-cohort":
                seal_cohort(out, config)
            elif args.mode == "evaluate-task":
                evaluate_task(out, args.subject, args.seed, config, expected)
            elif args.mode == "report":
                from .reporting import summarize
                result = summarize(out, config)
                print(json.dumps({"complete": result["complete"], "report": str(out / "report.md")}))
    return 0
