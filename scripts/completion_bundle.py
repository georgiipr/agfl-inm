#!/usr/bin/env python3
"""Portable inspection of immutable evidence; never relax scientific replay guards."""
import argparse
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'evidence/learned-covariance-v2'
STUDY = 'results/task-driven-completion-cuda-v2'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def bundle(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    archive = directory / manifest['archive']
    require(archive.parent == directory and archive.name == 'study-bundle.tar.gz', 'Unexpected archive path')
    require(sha(archive.read_bytes()) == manifest['archive_sha256'], 'Archive checksum mismatch')
    files = {}
    with tarfile.open(archive, 'r:gz') as stream:
        for member in stream:
            path = PurePosixPath(member.name)
            require(member.isfile() and not path.is_absolute() and '..' not in path.parts
                    and path.parts[0] in {'results', 'verification'}, 'Unsafe archive member')
            require(member.name not in files, 'Duplicate archive member')
            files[member.name] = stream.extractfile(member).read()
    require(set(files) == set(manifest['files_sha256']), 'Archive inventory mismatch')
    for name, data in files.items():
        require(sha(data) == manifest['files_sha256'][name], f'Checksum mismatch: {name}')
    for name, expected in manifest['source_files_sha256'].items():
        path = ROOT / name
        require(path.is_file() and sha(path.read_bytes()) == expected, f'Scientific source differs: {name}')
    return manifest, files


def verify(directory):
    import numpy as np
    manifest, files = bundle(directory)
    evidence = json.loads(files[f'{STUDY}/report/evidence.json'])
    values = defaultdict(list)
    total = 0
    for subject in range(1, 10):
        for seed in range(3):
            prefix = f'{STUDY}/tasks/A{subject:02d}_seed_{seed}'
            task = json.loads(files[f'{prefix}/task.json'])
            require(task['status'] == 'complete' and task['synthetic'] is False
                    and task['study_id'] == manifest['study_id'], 'Incomplete or foreign task')
            require((task['subject'], task['seed']) == (subject, seed), 'Wrong task coordinates')
            require(not set(task['train_sample_ids']) & set(task['validation_sample_ids']), 'Overlapping partitions')
            for name, expected in task['artifacts'].items():
                require(sha(files[f'{prefix}/{name}']) == expected, f'Task checksum differs: {name}')
            require(len(task['cells']) == 105, 'Incomplete cell grid')
            keys = set()
            for row in task['cells']:
                key = (row['strategy'], row['condition'], row['repeat'])
                require(key not in keys, 'Duplicate cell'); keys.add(key)
                with np.load(io.BytesIO(files[f"{prefix}/{row['artifact']}"]), allow_pickle=False) as data:
                    p, y, mask = data['probabilities'].astype(float), data['labels'], data['mask']
                    require(data['sample_ids'].tolist() == task['validation_sample_ids']
                            and y.tolist() == task['validation_labels'], 'Wrong sample IDs/labels')
                    count = int(row['condition'].rsplit('_', 1)[1])
                    require(mask.dtype == np.bool_ and np.all(mask.sum(1) == count), 'Wrong electrode count')
                    require(set(y.tolist()) == {0, 1, 2, 3} and p.shape == (len(y), 4)
                            and np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
                            and np.allclose(p.sum(1), 1, atol=1e-6), 'Invalid probabilities')
                    pred = p.argmax(1)
                    ba = float(np.mean([(pred[y == c] == c).mean() for c in range(4)]))
                    ll = float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())
                for metric, value in [('balanced_accuracy', ba), ('log_loss', ll)]:
                    require(abs(value-row[metric]) < 1e-12, 'Saved metric differs from predictions')
                    values[subject, seed, row['strategy'], row['condition'], metric].append(value)
                total += 1
    participant = defaultdict(list)
    for (subject, seed, strategy, condition, metric), items in values.items():
        participant[subject, strategy, condition, metric].append(float(np.mean(items)))
    participant = {k: float(np.mean(v)) for k, v in participant.items()}
    cohort = defaultdict(list)
    for (subject, strategy, condition, metric), value in participant.items():
        cohort[strategy, condition, metric].append(value)
    cohort = {k: float(np.mean(v)) for k, v in cohort.items()}
    for row in evidence['cohort_conditions']:
        require(abs(cohort[row['strategy'], row['condition'], row['metric']]-row['value']) < 1e-12,
                'Cohort aggregation mismatch')
    conditions = ['random_static_16', 'dynamic_random_16', 'random_static_6', 'dynamic_random_6']
    for row in evidence['contrasts']:
        candidate, control = row['contrast'].removeprefix('primary_').removeprefix('secondary_').split('_minus_')
        gains = [np.mean([participant[s, candidate, c, 'balanced_accuracy'] -
                          participant[s, control, c, 'balanced_accuracy'] for c in conditions]) for s in range(1, 10)]
        interval = np.quantile(np.mean(np.random.default_rng(20261005).choice(gains, (2000, 9)), axis=1), [.025, .975])
        require(abs(np.mean(gains)-row['mean_gain']) < 1e-12 and
                np.allclose(interval, row['ci_95'], atol=1e-12, rtol=1e-12), 'Contrast or bootstrap mismatch')
    print(f'PASS: 27 real tasks, 54 completer fits, {total} prediction cells; hashes, metrics and intervals verified.')
    print('This verifies saved evidence without refitting or rerunning GPU inference. Validation is not independent confirmation.\n')
    print('| Completion | 16 retained BA | 6 retained BA | Overall degraded BA |')
    print('|---|---:|---:|---:|')
    for strategy in ['zero', 'tucker_frozen', 'tucker_learned', 'covariance_frozen', 'covariance_learned']:
        a = np.mean([cohort[strategy, c, 'balanced_accuracy'] for c in conditions[:2]])
        b = np.mean([cohort[strategy, c, 'balanced_accuracy'] for c in conditions[2:]])
        print(f'| {strategy} | {100*a:.2f}% | {100*b:.2f}% | {50*(a+b):.2f}% |')
    return manifest, files


def unpack(directory):
    manifest, files = bundle(directory)
    targets = {}
    for name, data in files.items():
        # Review receipts have a separate local destination, never in scientific outputs.
        rel = name if name.startswith('results/') else '.session-runs/shared-completion/' + name
        target = ROOT / rel
        require(not any(p.is_symlink() for p in [target, *target.parents] if p != ROOT.parent), 'Symlink destination refused')
        require(not target.exists() or (target.is_file() and target.read_bytes() == data), f'Refusing overwrite: {rel}')
        targets[target] = data
    for path, data in targets.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                stream.write(data)
    print(f'Unpacked/verified {len(targets)} immutable files; existing different files were not overwritten.')


def prepare(args):
    manifest, _ = bundle(args.bundle)
    target, output = args.config_out.resolve(), args.output.resolve()
    require(not target.exists() and not output.exists(), 'Config and output must both be fresh')
    require(output.parent == ROOT/'results' and output.name.startswith('task-driven-completion-cuda-'),
            'Output must be a new results/task-driven-completion-cuda-* directory')
    require(args.data_dir.resolve() == Path(manifest['historical_data_dir']),
            'Historical inputs require the canonical data path; see the reproduction guide/container mounts')
    cfg = json.loads((ROOT/'configs/task-driven-completion-cuda-v2.json').read_text())
    cfg.update(name=output.name, data_dir=str(args.data_dir.resolve()), output_dir=str(output),
        candidate_source_dir=str(ROOT/'results/encoder-candidates-reproducible-v1'),
        split_source_dir=str(ROOT/'results/baselines-reproducible-v1'))
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x') as stream:
        json.dump(cfg, stream, indent=2); stream.write('\n')
    print(f'Created {target}. Fresh output: {output}. Scientific settings unchanged.')


def gate(args):
    sys.path.insert(0, str(ROOT))
    from inm.task_driven_completion_cuda.protocol import load_config, study_identity
    cfg = load_config(args.config, allow_existing_output=True)
    manifest, _ = bundle(args.bundle)
    for package, expected in manifest['packages'].items():
        require(importlib.metadata.version(package) == expected, f'Historical replay needs {package}=={expected}')
    require(Path(cfg['data_dir']) == Path(manifest['historical_data_dir']), 'Historical data path mismatch')
    for name, expected in manifest['recording_sha256'].items():
        require(sha((Path(cfg['data_dir'])/name).read_bytes()) == expected, f'Recording checksum differs: {name}')
    require(Path(cfg['output_dir']) != ROOT/STUDY or args.action == 'check-env', 'Original archived output is immutable; prepare a fresh config')
    identity = study_identity(cfg)
    if args.action in ('check-env', 'check-run'):
        print(f'PASS: historical package, recording and source checks. Current study ID: {identity["study_id"]}')
        return
    path = Path(cfg['output_dir'])/'tasks/A01_seed_0/task.json'
    audit_path = args.state/'pilot-audit.stdout.log'
    require(path.is_file() and audit_path.is_file(), 'Pilot and its independent audit must complete first')
    task, audit = json.loads(path.read_text()), json.loads(audit_path.read_text())
    require(task['status'] == 'complete' and not task['synthetic'] and task['study_id'] == identity['study_id'], 'Pilot identity mismatch')
    require(audit['status'] == 'passed' and audit['study_id'] == identity['study_id'] and
            audit['replayed_cells'] == 105 and audit['new_fits'] == 0 and
            all(audit[k] is True for k in ['exact_prediction_replay', 'backbone_matches_source',
                'selected_states_loaded', 'independently_recomputed_metrics', 'independently_recomputed_contrasts']), 'Pilot audit failed')
    receipt = dict(decision='proceed_to_cohort', study_id=identity['study_id'],
        task_sha256=sha(path.read_bytes()), audit_sha256=sha(audit_path.read_bytes()),
        basis='Operational replay only; accuracy is not an advancement gate')
    target = args.state/'pilot-review.json'
    if args.action == 'accept-pilot':
        with target.open('x') as stream: json.dump(receipt, stream, indent=2)
    else:
        require(json.loads(target.read_text()) == receipt, 'Pilot acceptance receipt changed')
    print('PASS: operational pilot review; scientific settings remain frozen.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['verify', 'unpack', 'prepare-config', 'check-env', 'check-run', 'accept-pilot', 'check-pilot'])
    parser.add_argument('--bundle', type=Path, default=DEFAULT)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--config-out', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--state', type=Path)
    args = parser.parse_args()
    try:
        if args.action == 'verify': verify(args.bundle)
        elif args.action == 'unpack': unpack(args.bundle)
        elif args.action == 'prepare-config':
            require(all([args.config_out, args.output, args.data_dir]), 'Provide --config-out, --output and --data-dir')
            prepare(args)
        else:
            require(args.config is not None, 'Provide --config')
            if args.action not in ('check-env', 'check-run'): require(args.state is not None, 'Provide --state')
            gate(args)
        return 0
    except (ValueError, OSError, KeyError, ImportError, tarfile.TarError) as exc:
        print(f'Completion review stopped: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
