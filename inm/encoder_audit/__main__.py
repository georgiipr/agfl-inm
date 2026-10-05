"""Dependency-light plan/inventory and lazy audit execution commands."""
import argparse
import json
import tempfile

from .protocol import ROOT, load_config, tasks


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read-only encoder investigation")
    parser.add_argument("--config", default=str(ROOT / "configs/encoder-audit.json"))
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan", action="store_true")
    action.add_argument("--inventory", action="store_true")
    action.add_argument("--task-index", type=int)
    action.add_argument("--smoke", action="store_true")
    action.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-dir", help="Separate output for synthetic smoke")
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        if args.output_dir and not args.smoke:
            raise ValueError("--output-dir is only supported with --smoke")
        if args.plan:
            count = len(tasks(cfg))
            print(f"Tasks: {count}; CPU probe fits: {count * 4}; neural refits: 0")
            print("Partitions: train/validation only; exploratory reused validation")
            print(f"Output: {cfg['output_dir']}")
            print("No fits or measurements performed.")
            return 0
        if args.inventory:
            from .inventory import inspect_inputs
            result = inspect_inputs(cfg)
            print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
            return 0 if result["status"] == "ready" else 1
        if args.task_index is not None:
            if not 0 <= args.task_index < len(tasks(cfg)):
                raise ValueError(f"task-index must be in 0..{len(tasks(cfg)) - 1}")
            from .study import run_task
            result = run_task(cfg, args.task_index, device=args.device)
            print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
            return 0 if result.get("status") == "complete" else 1
        if args.smoke:
            output = args.output_dir or tempfile.mkdtemp(prefix="agfl-encoder-audit-smoke-")
            from .study import run_smoke
            print(json.dumps(run_smoke(cfg, output), indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.summarize_only:
            from .reporting import summarize
            print(json.dumps(summarize(cfg), indent=2, sort_keys=True, allow_nan=False))
            return 0
    except (ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
