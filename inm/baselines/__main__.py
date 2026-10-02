"""Dependency-light command line for declaring and inspecting baseline studies."""
from __future__ import annotations

import argparse
import json
import sys

from .protocol import arms, load_config, tasks


def _parser():
    parser = argparse.ArgumentParser(description="Baseline-first EEG accuracy study")
    parser.add_argument("--config", default="configs/baselines.json",
                        help="study JSON; relative paths inside it resolve from its directory")
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--plan", action="store_true", help="print the validated fit matrix")
    operation.add_argument("--preflight", action="store_true", help="check execution prerequisites")
    operation.add_argument("--task-index", type=int, help="execute one declared task")
    operation.add_argument("--smoke", action="store_true", help="run a synthetic CPU smoke fit")
    operation.add_argument("--summarize-only", action="store_true", help="summarize saved study results")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None,
                        help="execution device (default: cuda; smoke always uses cpu)")
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ValueError as error:
        parser.error(str(error))
    if args.plan:
        task_list, arm_list = tasks(cfg), arms(cfg)
        print(f"Study: {cfg['name']} ({cfg['schema_name']})")
        print(f"Config: {cfg['config_path']}")
        print(f"Protocol: {cfg['protocol']} — {cfg['protocol_description']}")
        print(f"Data: {cfg['data_dir']}")
        print(f"Output: {cfg['output_dir']}")
        print(f"Tasks: {len(task_list)}; fits: {len(task_list) * len(arm_list)}")
        for arm in arm_list:
            print(f"Arm {arm['name']}: coverage={','.join(arm['coverage'])}; family={arm['family']}")
        print("Degraded mask repeats: " + str(cfg["mask_repeats"]))
        print("Subject bootstrap repeats: " + str(cfg["bootstrap_repeats"]))
        if cfg["protocol"] == "cross_session" and cfg["external_labels_dir"] is None:
            print("External E-session labels: required explicitly by the loader; not inferred")
        return 0
    if args.preflight:
        from .diagnostics import inspect_environment
        report = inspect_environment(cfg["config_path"])
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["readiness"] == "ready" else 1
    if args.task_index is not None and not 0 <= args.task_index < len(tasks(cfg)):
        parser.error(f"--task-index must be in 0..{len(tasks(cfg)) - 1}")
    if args.smoke:
        from .study import run_smoke, smoke_output_path
        smoke_output = str(smoke_output_path(cfg))
        try:
            result = run_smoke(cfg, smoke_output)
        except Exception as error:
            print(f"Synthetic smoke failed: {type(error).__name__}: {error}", file=sys.stderr)
            return 2
        print(json.dumps({"status": result["status"], "synthetic": result["synthetic"],
                          "output_dir": smoke_output, "arms": [arm["arm"] for arm in result["arms"]]},
                         indent=2, sort_keys=True))
        return 0
    if args.task_index is not None:
        from .study import run_task
        try:
            result = run_task(cfg, args.task_index, device=args.device or "cuda")
        except Exception as error:
            print(f"Task failed: {type(error).__name__}: {error}", file=sys.stderr)
            return 2
        print(json.dumps({"status": result["status"], "synthetic": result["synthetic"],
                          "subject": result["subject"], "seed": result["seed"],
                          "arms": [arm["arm"] for arm in result["arms"]]}, indent=2, sort_keys=True))
        return 0
    if args.summarize_only:
        from .reporting import summarize
        try:
            progress = summarize(cfg)
        except Exception as error:
            print(f"Summarization failed: {type(error).__name__}: {error}", file=sys.stderr)
            return 2
        print(json.dumps(progress, indent=2, sort_keys=True))
        return 0 if progress["complete"] else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
