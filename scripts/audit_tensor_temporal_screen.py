"""Read-only independent numerical audit; writes its report only under /tmp."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import tempfile

if not __debug__:
    raise RuntimeError('Run this audit without Python optimization flags')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from inm.encoder_candidates.data import prepare_subject
from inm.tensor_temporal.models import restore_model
from inm.tensor_temporal.study import load_config


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def metrics(y, probabilities):
    y = np.asarray(y)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    assert probabilities.shape == (len(y), 4)
    assert np.isfinite(probabilities).all() and (probabilities >= 0).all()
    assert np.allclose(probabilities.sum(1), 1, atol=1e-6, rtol=0)
    labels = probabilities.argmax(1)
    return {'balanced_accuracy': float(np.mean([np.mean(labels[y == c] == c) for c in range(4)])),
            'accuracy': float(np.mean(labels == y)),
            'log_loss': float(-np.log(probabilities[np.arange(len(y)), y].clip(1e-12, 1)).mean())}


@torch.no_grad()
def predict(model, raw, batch=32, masks=None):
    rows = []
    for start in range(0, len(raw), batch):
        x = torch.tensor(np.asarray(raw[start:start + batch]), dtype=torch.float32)
        mask = (torch.ones(x.shape[:3], dtype=torch.bool) if masks is None else
                torch.tensor(masks[start:start + batch], dtype=torch.bool))
        rows.append(model(x, mask).softmax(-1).numpy())
    return np.concatenate(rows)


def close_metrics(actual, saved):
    for key, value in actual.items():
        assert math.isclose(value, saved[key], abs_tol=1e-10, rel_tol=0), (key, value, saved[key])


def audit(cfg, requested):
    assert requested and len(requested) == len(set(requested)) and set(requested).issubset(cfg['subjects'])
    output = Path(cfg['output_dir'])
    manifest = read(output / 'study.json')
    identity = manifest['identity']
    assert digest(identity) == manifest['study_id']
    assert identity['config'] == {k: v for k, v in cfg.items() if k != 'config_path'}
    assert identity['config_sha256'] == sha(cfg['config_path'])
    assert identity['python'] == platform.python_version()
    for package, version in identity['packages'].items():
        assert importlib.metadata.version(package) == version, package
    source_before = {name: sha(ROOT / name) for name in identity['source_sha256']}
    assert source_before == identity['source_sha256']
    actual_sources = {str(p.relative_to(ROOT)) for p in ROOT.glob('*.py')} | {
        str(p.relative_to(ROOT)) for tree in ('agfl', 'inm') for p in (ROOT / tree).rglob('*.py')}
    assert actual_sources == set(source_before)
    for filename, expected in identity['input_sha256'].items():
        assert sha(filename) == expected, filename
    report = {'study_id': manifest['study_id'], 'source_files': len(source_before),
              'input_files': len(identity['input_sha256']), 'subjects': [], 'pending': [],
              'no_test_scoring': True, 'replayed_partitions': ['train', 'validation'],
              'masked_logits_replayed': False}
    aggregate_rows, robustness_rows = [], []
    for subject in requested:
        directory = output / 'tasks' / f'A{subject:02d}_seed_0'
        if not (directory / 'complete.json').exists():
            report['pending'].append(subject)
            continue
        print(f'Auditing A{subject:02d}', flush=True)
        complete = read(directory / 'complete.json')
        provenance = read(directory / 'data_provenance.json')
        assert complete['study_id'] == manifest['study_id']
        assert complete['subject'] == subject and complete['seed'] == 0
        assert complete['data_sha256'] == sha(directory / 'data_provenance.json')
        assert set(complete['results_sha256']) == set(cfg['arms'])
        with tempfile.TemporaryDirectory(prefix='agfl-independent-audit-', dir='/tmp') as temporary:
            prepared = prepare_subject(cfg, subject, 0, temporary)
        assert provenance['data_id'] == prepared.data_id and provenance['split_id'] == prepared.split_id
        assert provenance['normalization'] == prepared.normalization
        for partition in ('train', 'validation'):
            part = getattr(prepared, partition)
            assert provenance['partitions'][partition]['sample_ids'] == list(part.sample_ids)
            assert provenance['partitions'][partition]['labels'] == hashlib.sha256(part.labels.tobytes()).hexdigest()
            assert provenance['partitions'][partition]['raw'] == hashlib.sha256(part.raw.tobytes()).hexdigest()
        subject_report = {'subject': subject, 'arms': [], 'reference_replication': {}}
        paired_masks = None
        for arm in cfg['arms']:
            ad = directory / arm
            result = read(ad / 'result.json')
            assert complete['results_sha256'][arm] == sha(ad / 'result.json')
            assert result['identity'] == {'study_id': manifest['study_id'], 'arm': arm, 'synthetic': False, 'data': provenance}
            assert set(result['artifact_sha256']) == {'checkpoint.pt', 'history.json', 'predictions.npz', 'robustness.json'}
            for name, expected in result['artifact_sha256'].items():
                assert sha(ad / name) == expected, str(ad / name)
            history = read(ad / 'history.json')
            assert [r['epoch'] for r in history] == list(range(1, len(history) + 1))
            assert result['epochs_run'] == len(history)
            selected = max(history, key=lambda row: (row['clean_validation_metrics']['balanced_accuracy'],
                        -row['clean_validation_metrics']['log_loss'], -row['epoch']))
            assert result['selected_epoch'] == selected['epoch']
            assert [r['epoch'] for r in history if r['selected']] == [selected['epoch']]
            assert result['train_metrics'] == selected['clean_train_metrics']
            assert result['validation_metrics'] == selected['clean_validation_metrics']
            checkpoint = torch.load(ad / 'checkpoint.pt', map_location='cpu', weights_only=True)
            assert checkpoint['identity'] == result['identity']
            assert checkpoint['selected_epoch'] == selected['epoch']
            assert checkpoint['normalization'] == prepared.normalization
            model = restore_model(arm, checkpoint['constructor'], checkpoint['state_dict']).eval()
            assert result['parameters'] == sum(p.numel() for p in model.parameters() if p.requires_grad)
            replay = {}
            with np.load(ad / 'predictions.npz', allow_pickle=False) as saved:
                assert set(saved.files) == {f'{p}_{n}' for p in ('train', 'validation')
                                           for n in ('labels', 'probabilities', 'sample_ids')}
                for partition in ('train', 'validation'):
                    part = getattr(prepared, partition)
                    assert np.array_equal(saved[partition + '_labels'], part.labels)
                    assert np.array_equal(saved[partition + '_sample_ids'], part.sample_ids)
                    probability = predict(model, part.raw, cfg['training']['batch_size'])
                    expected = saved[partition + '_probabilities']
                    assert np.array_equal(probability, expected), (subject, arm, partition, float(np.max(np.abs(probability - expected))))
                    measured = metrics(part.labels, probability)
                    close_metrics(measured, result[partition + '_metrics'])
                    replay[partition] = measured
            robust = read(ad / 'robustness.json')
            masks = [(r['scenario'], r['repeat'], r['mask_sha256']) for r in robust]
            assert len(robust) == 21 and len(set((s, r) for s, r, _ in masks)) == 21
            if paired_masks is not None:
                assert masks == paired_masks
            paired_masks = masks
            for condition in cfg['conditions']:
                values = [r for r in robust if r['scenario'] == condition['name']]
                assert sorted(r['repeat'] for r in values) == list(range(condition['repeats']))
                from inm.availability import make_mask_bank, mask_bank_digest, CHANNEL_IDS
                for row in values:
                    bank = make_mask_bank(len(prepared.validation.raw), 4, condition['retained'],
                        condition['pattern'], seed=0, partition='validation', subject=f'A{subject:02d}',
                        repeat=row['repeat'], sample_ids=prepared.validation.sample_ids, channel_ids=CHANNEL_IDS)
                    assert mask_bank_digest(bank) == row['mask_sha256']
                    masked_probability = predict(model, prepared.validation.raw,
                        cfg['training']['batch_size'], masks=bank)
                    close_metrics(metrics(prepared.validation.labels, masked_probability), row)
                mean = {metric: float(np.mean([r[metric] for r in values])) for metric in ('balanced_accuracy', 'log_loss')}
                robustness_rows.append({'subject': subject, 'seed': 0, 'arm': arm, 'scenario': condition['name'], **mean})
                if condition['name'] == 'full_22':
                    close_metrics(replay['validation'], values[0])
            subject_report['arms'].append({'arm': arm, 'parameters': result['parameters'],
                'selected_epoch': result['selected_epoch'], 'epochs_run': len(history),
                'train': replay['train'], 'validation': replay['validation'], 'bitwise_replay': True})
            aggregate_rows.append({'subject': subject, 'arm': arm, **replay})
            if arm == 'spatial_transformer':
                old = ROOT / 'results/encoder-candidates-reproducible-v1/tasks' / f'A{subject:02d}_seed_0/ARMS/spatial_transformer'
                old_result = read(old / 'result.json')
                old_checkpoint = torch.load(old / 'checkpoint.pt', map_location='cpu', weights_only=True)
                old_history = read(old / 'history.json')
                keys = ('epoch', 'learning_rate', 'optimization_loss', 'clean_train_metrics', 'clean_validation_metrics', 'selected')
                comparison = {
                    'same_selected_epoch': old_result['selected_epoch'] == result['selected_epoch'],
                    'same_validation_metrics': old_result['clean_validation'] == result['validation_metrics'],
                    'same_training_history': [{k: r[k] for k in keys} for r in old_history] == [{k: r[k] for k in keys} for r in history],
                    'same_selected_weights': all(torch.equal(v, old_checkpoint['state_dict'][k]) for k, v in checkpoint['state_dict'].items())}
                with np.load(old / 'predictions.npz') as old_probs, np.load(ad / 'predictions.npz') as new_probs:
                    earlier = old_probs['validation_probabilities']
                    current = new_probs['validation_probabilities']
                    comparison['same_validation_probabilities'] = np.array_equal(earlier, current)
                    comparison['same_validation_predicted_classes'] = np.array_equal(earlier.argmax(1), current.argmax(1))
                    comparison['max_absolute_probability_difference'] = float(np.abs(earlier - current).max())
                comparison['validation_balanced_accuracy_difference'] = result['validation_metrics']['balanced_accuracy'] - old_result['clean_validation']['balanced_accuracy']
                comparison['validation_log_loss_difference'] = result['validation_metrics']['log_loss'] - old_result['clean_validation']['log_loss']
                comparison['same_epochs_trained'] = len(history) == len(old_history)
                comparison['same_validation_confusion'] = result['validation_metrics']['confusion_matrix'] == old_result['clean_validation']['confusion_matrix']
                weight_differences = {k: float((value - old_checkpoint['state_dict'][k]).abs().max())
                                      for k, value in checkpoint['state_dict'].items()}
                comparison['largest_weight_or_buffer_differences'] = dict(sorted(weight_differences.items(), key=lambda item: -item[1])[:10])
                comparison['historical_thread_policy'] = 'Not pinned by historical runner; eight-thread epoch-one replay matches historical A01 exactly.'
                comparison['current_thread_policy'] = 'One CPU thread, recorded in study identity.'
                comparison['interpretation'] = ('Historical comparison is descriptive because CPU reduction threading differs. '
                    'A01 one-epoch isolated diagnostic reproduced new values exactly at one thread and historical values exactly at eight threads. '
                    'New-study checkpoint and probability replay remain required to match bitwise.')
                subject_report['reference_replication'] = comparison
        report['subjects'].append(subject_report)
    report['robustness'] = robustness_rows
    report['cohort'] = []
    report['contrasts'] = []
    if len(report['subjects']) == len(cfg['subjects']):
        for arm in cfg['arms']:
            subset = [r for r in aggregate_rows if r['arm'] == arm]
            report['cohort'].append({'arm': arm, 'participants': len(subset),
                'train_balanced_accuracy': float(np.mean([r['train']['balanced_accuracy'] for r in subset])),
                'validation_balanced_accuracy': float(np.mean([r['validation']['balanced_accuracy'] for r in subset])),
                'validation_log_loss': float(np.mean([r['validation']['log_loss'] for r in subset]))})
        pairs = [cfg['comparisons']['primary']] + cfg['comparisons']['secondary']
        for candidate, reference in pairs:
            values = {(r['subject'], r['arm']): r['validation']['balanced_accuracy'] for r in aggregate_rows}
            delta = np.array([values[s, candidate] - values[s, reference] for s in cfg['subjects']])
            rng = np.random.default_rng(cfg['comparisons']['bootstrap_seed'])
            bootstrap = np.mean(delta[rng.integers(len(delta), size=(cfg['comparisons']['bootstrap_repeats'], len(delta)))], axis=1)
            report['contrasts'].append({'candidate': candidate, 'reference': reference,
                'mean_balanced_accuracy_difference': float(delta.mean()), 'positive_participants': int((delta > 0).sum()),
                'ci95_participant_bootstrap': np.quantile(bootstrap, [.025, .975]).tolist()})
    report_path = output / 'report/summary.json'
    if report_path.exists():
        official = read(report_path)
        assert official['study_id'] == manifest['study_id'] and official['synthetic'] is False
        if report['cohort']:
            assert official['status'] == 'complete'
            for cohort in report['cohort']:
                other = next(r for r in official['cohort'] if r['arm'] == cohort['arm'])
                assert all(cohort[k] == other[k] for k in cohort)
            for contrast in report['contrasts']:
                other = next(r for r in official['contrasts'] if r['candidate'] == contrast['candidate'] and r['reference'] == contrast['reference'])
                assert all(contrast[k] == other[k] for k in contrast)
            assert official['robustness'] == robustness_rows
    assert source_before == {name: sha(ROOT / name) for name in source_before}
    report['masked_logits_replayed_for_completed_subjects'] = bool(report['subjects'])
    report['masked_logits_replayed'] = len(report['subjects']) == len(requested)
    report['status'] = 'passed' if len(report['subjects']) == len(requested) else 'partial'
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subjects', type=int, nargs='+')
    parser.add_argument('--output', default='/tmp/agfl-tensor-temporal-independent-audit.json')
    args = parser.parse_args()
    destination = Path(args.output).resolve()
    if not destination.is_relative_to('/tmp'):
        raise ValueError('Audit output must stay under /tmp')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    configuration = load_config(ROOT / 'configs/tensor-temporal-screen.json')
    result = audit(configuration, args.subjects or configuration['subjects'])
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'subjects': len(result['subjects']), 'pending': result['pending'], 'output': str(destination)}))
