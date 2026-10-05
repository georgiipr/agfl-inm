"""Public dependency-light planner and explicit replay/smoke entry points."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

from .protocol import ARM_IDS, CONDITIONS, STRATEGY_IDS, inspect_inputs, load_config, tasks


def _parser():
    parser = argparse.ArgumentParser(prog="python -m inm.completion_transformer")
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--task-index", type=int)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--output-dir", help="required fresh output directory for synthetic smoke")
    parser.add_argument("--resume-smoke", action="store_true", help="verify/resume the same synthetic smoke identity")
    return parser


def _synthetic_context(subject: int, seed: int, *, factor_epochs: int = 1):
    import numpy as np
    from types import SimpleNamespace
    from inm.encoder_candidates.models import build_model

    rng = np.random.default_rng(55801 + seed)
    labels_train = np.tile(np.arange(4, dtype=np.int64), 2)
    labels_validation = np.arange(4, dtype=np.int64)
    train = SimpleNamespace(raw=rng.normal(size=(8, 22, 4, 250)).astype(np.float32),
                            labels=labels_train, sample_ids=[f"synthetic:train:{i}" for i in range(8)])
    validation = SimpleNamespace(raw=rng.normal(size=(4, 22, 4, 250)).astype(np.float32),
                                 labels=labels_validation,
                                 sample_ids=[f"synthetic:validation:{i}" for i in range(4)])
    models = {arm: build_model(arm, seed=subject * 100 + seed).eval().requires_grad_(False) for arm in ARM_IDS}
    return {"prepared": SimpleNamespace(subject=subject, seed=seed, train=train, validation=validation,
                  data_id="synthetic", split_id="synthetic"),
            "models": models, "factor_epochs": factor_epochs}


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        if not args.smoke and (args.output_dir is not None or args.resume_smoke):
            raise ValueError("--output-dir and --resume-smoke are valid only with --smoke")
        cfg = load_config(args.config, allow_existing_output=True)
        if args.plan:
            planned = tasks(cfg)
            print(json.dumps({"status": "planned", "tasks": len(planned), "task_rows": planned,
                "cells_per_task": len(ARM_IDS) * len(STRATEGY_IDS),
                "evaluation_rows_per_cell": sum(int(item["repeats"]) for item in CONDITIONS),
                "neural_fits": 0, "tucker_calibrations": len(planned)}, indent=2))
            return 0
        if args.preflight:
            inventory = inspect_inputs(cfg)
            print(json.dumps(inventory, indent=2, sort_keys=True))
            return 0 if inventory["status"] == "inventory_complete" else 2
        from .reporting import summarize
        if args.summarize_only:
            report = summarize(cfg)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0 if report["complete"] or cfg["synthetic"] else 2
        from .study import run_task
        if args.smoke:
            if not args.output_dir:
                raise ValueError("--smoke requires --output-dir naming an explicit fresh directory")
            output = Path(args.output_dir).expanduser().resolve()
            if output.exists() and any(output.iterdir()) and not args.resume_smoke:
                raise FileExistsError("synthetic smoke output must be fresh and empty")
            cfg = copy.deepcopy(cfg)
            cfg["subjects"], cfg["seeds"], cfg["synthetic"] = [1], [0], True
            cfg["output_dir"] = str(output)
            cfg["config_path"] = str(Path(args.config).expanduser().resolve())
            result = run_task(cfg, 0, synthetic_context=_synthetic_context(1, 0, factor_epochs=1))
            report = summarize(cfg)
            print(json.dumps({"status": result["status"], "synthetic": True,
                              "output_dir": str(output), "cells": len(result["cells"]),
                              "report_status": report["status"], "synthetic_excluded": 1}, indent=2))
            return 0
        if cfg["synthetic"]:
            raise ValueError("real --task-index is disabled for synthetic declarations; use --smoke")
        result = run_task(cfg, args.task_index)
        print(json.dumps({"status": result["status"], "subject": result["subject"],
                          "seed": result["seed"], "cells": len(result["cells"])}, indent=2))
        return 0
    except Exception as exc:
        print(f"completion-transformer: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
