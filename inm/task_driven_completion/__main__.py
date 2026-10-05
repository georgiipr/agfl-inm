"""Dependency-light planning and explicit CPU smoke/task/report operations."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from .protocol import inspect_inputs, load_config, plan


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m inm.task_driven_completion')
    parser.add_argument('--config',required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--plan',action='store_true')
    mode.add_argument('--preflight',action='store_true')
    mode.add_argument('--smoke',action='store_true')
    mode.add_argument('--task-index',type=int)
    mode.add_argument('--summarize-only',action='store_true')
    parser.add_argument('--output-dir')
    parser.add_argument('--resume-smoke',action='store_true')
    args = parser.parse_args(argv)
    try:
        if not args.smoke and (args.output_dir is not None or args.resume_smoke):
            raise ValueError('--output-dir and --resume-smoke require --smoke')
        cfg = load_config(args.config,allow_existing_output=True)
        if args.plan or args.preflight:
            result = plan(cfg) if args.plan else inspect_inputs(cfg)
            print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
            return 0 if args.plan or result['status'] == 'inventory_complete' else 2
        from .study import run_task, synthetic_context
        from .reporting import summarize
        if args.smoke:
            if args.output_dir is None:
                raise ValueError('--smoke requires an explicit --output-dir')
            output = Path(args.output_dir).expanduser().resolve()
            if args.resume_smoke:
                if not (output/'study-identity.json').is_file() or not (output/'tasks/A01_seed_0/task.json').is_file():
                    raise ValueError('--resume-smoke requires an existing completed smoke')
            elif output.exists() and (not output.is_dir() or any(output.iterdir())):
                raise ValueError('Smoke requires a fresh empty output directory')
            cfg = copy.deepcopy(cfg)
            cfg.update(subjects=[1],seeds=[0],synthetic=True,output_dir=str(output))
            result = run_task(cfg,0,synthetic_context=synthetic_context())
            report = summarize(cfg)
            print(json.dumps(dict(status=result['status'],synthetic=True,output_dir=str(output),
                cells=len(result['cells']),strategies=list(result['fits']),supervised_completion_fits=2,
                report_status=report['status'],synthetic_tasks_excluded=len(report['synthetic_tasks_excluded'])),indent=2))
            return 0
        if args.summarize_only:
            result = summarize(cfg)
            print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
            return 0 if result['complete'] else 2
        if cfg['synthetic']:
            raise ValueError('Use --smoke for synthetic execution')
        result = run_task(cfg,args.task_index)
        print(json.dumps({k:result[k] for k in ('status','subject','seed','supervised_completion_fits')},indent=2))
        return 0
    except Exception as exc:
        print(f'task-driven-completion: {type(exc).__name__}: {exc}',file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
