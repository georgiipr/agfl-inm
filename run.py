"""Entry point for the four-model channel-availability study."""
from __future__ import annotations
import argparse
from inm.protocol import load_config, models, tasks


def main():
    parser = argparse.ArgumentParser(description='EEGNet / Signal Transformer, each with and without tensors; built-in MHA')
    parser.add_argument('--config', default='configs/study.json')
    parser.add_argument('--plots', action='store_true', help='Saved-table plots with --summarize-only')
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument('--task-index', type=int, help='Participant/seed task; each trains the configured models once')
    operation.add_argument('--task-count', action='store_true', help='Print the declared participant/seed task count')
    operation.add_argument('--summarize-only', action='store_true')
    operation.add_argument('--plan', action='store_true', help='Print the declared models and tasks without training')
    args = parser.parse_args()
    if args.plots and not args.summarize_only:
        parser.error('--plots is used with --summarize-only after results are available')
    cfg = load_config(args.config)
    if args.task_count:
        print(len(tasks(cfg)))
    elif args.plan:
        print('Models: ' + ', '.join(cfg['models']))
        for index, (subject, seed) in enumerate(tasks(cfg)):
            print(f'{index}: A{subject:02d}, seed={seed}, model runs={len(models(cfg))}')
        print(f'Total model runs: {len(tasks(cfg)) * len(models(cfg))}; train on 22 channels once per run')
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
