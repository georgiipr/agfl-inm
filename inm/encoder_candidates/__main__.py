"""Dependency-light command line for the declared encoder-candidate study."""
from __future__ import annotations

import argparse
import json
import sys

from .protocol import arms, inspect_inputs, load_config, tasks


def _parser():
    parser = argparse.ArgumentParser(description="Fixed five-arm EEG encoder-candidate study")
    parser.add_argument("--config", default="configs/encoder-candidates.json",
                        help="study JSON; relative paths resolve from its directory")
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--plan", action="store_true", help="print the fixed fit matrix")
    operation.add_argument("--preflight", action="store_true", help="inventory files and packages only")
    operation.add_argument("--task-index", type=int, help="execute one declared real task")
    operation.add_argument("--smoke", action="store_true", help="run a synthetic CPU smoke")
    operation.add_argument("--summarize-only", action="store_true", help="summarize saved results")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-dir", help="separate output for synthetic smoke only")
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config, allow_existing_output=(
            args.task_index is not None or args.smoke or args.summarize_only))
        if args.output_dir and not args.smoke:
            raise ValueError("--output-dir is only supported with --smoke")
        if args.plan:
            task_count = len(tasks(cfg))
            arm_ids = arms(cfg)
            print(f"Study: {cfg['name']} ({cfg['schema_name']})")
            print(f"Config: {cfg['config_path']}")
            print(f"Tasks: {task_count}; neural fits: {task_count * len(arm_ids)}; "
                  f"CPU probe fits: {task_count * 4}; Tucker fits: 0")
            print("Arms: " + ", ".join(arm_ids))
            print(f"Partitions: {'/'.join(cfg['partitions'])}; training regime: {cfg['training_regime']}")
            print(f"Output: {cfg['output_dir']}")
            print("No fits or measurements performed.")
            return 0
        if args.preflight:
            report = inspect_inputs(cfg)
            print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
            return 0 if report["status"] == "ready" else 1
        if args.task_index is not None and not 0 <= args.task_index < len(tasks(cfg)):
            raise ValueError(f"--task-index must be in 0..{len(tasks(cfg)) - 1}")
        if args.smoke:
            from .study import run_smoke
            result = run_smoke(cfg, args.output_dir)
            print(json.dumps({"status": result["status"], "synthetic": True,
                              "task": result["subject"], "seed": result["seed"],
                              "fits": len(result["fits"]),
                              "checks": result["checks"]}, indent=2, sort_keys=True))
            return 0
        if args.task_index is not None:
            from .study import run_task
            result = run_task(cfg, args.task_index, device=args.device,
                              synthetic_fixture=cfg["synthetic"])
            print(json.dumps({"status": result["status"], "subject": result["subject"],
                              "seed": result["seed"], "fits": len(result["fits"]),
                              "study_id": result["study_id"]}, indent=2, sort_keys=True))
            return 0
        if args.summarize_only:
            from .reporting import summarize
            result = summarize(cfg)
            print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
            return 0 if result["complete"] else 1
    except (ValueError, OSError, RuntimeError, NotImplementedError) as error:
        print(f"encoder-candidates: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
