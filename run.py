"""AGFL-inm cluster entry point. Local execution is not part of preparation."""
from __future__ import annotations
import argparse
from inm.protocol import arms, load_config, tasks


def main():
    parser = argparse.ArgumentParser(description='EEGNet-derived tensor/availability study')
    parser.add_argument('--config', default='configs/study.json')
    parser.add_argument('--plots', action='store_true', help='Optional saved-table plots with --summarize-only')
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument('--task-index', type=int, help='Subject/seed task, 0..26 for the supplied study')
    operation.add_argument('--summarize-only', action='store_true')
    operation.add_argument('--plan', action='store_true', help='Print the declared array tasks without training')
    args = parser.parse_args()
    if args.plots and not args.summarize_only:
        parser.error('--plots is used with --summarize-only after results are available')
    cfg = load_config(args.config)
    if args.plan:
        for index, (subject, seed) in enumerate(tasks(cfg)):
            print(f'{index}: A{subject:02d}, seed={seed}, classifier fits={len(arms(cfg))}')
        print(f'Total classifier fits: {len(tasks(cfg)) * len(arms(cfg))}')
    elif args.summarize_only:
        from inm.study import summarize_existing
        result = summarize_existing(cfg)
        print(f"Completed fits: {result['completed_fits']}/{result['expected_fits']}; status={result['status']}")
        if args.plots:
            from pathlib import Path
            from inm.plots import plot_report
            print(plot_report(Path(cfg['output_dir']) / 'report'))
    else:
        from inm.study import run_task
        run_task(cfg, args.task_index)


if __name__ == '__main__':
    main()
