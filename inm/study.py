"""Cluster experiment: shared training calibration, paired availability arms."""
from __future__ import annotations
import copy
import fcntl
import hashlib
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from agfl.datasets.splits import get_split
from agfl.reproducibility import seed_everything
from agfl.storage import run_directory
from .availability import (CHANNEL_IDS, availability_metadata, evaluation_scenarios,
                           make_mask_bank, mask_bank_digest, training_mask_bank)
from .data import load_subject
from .model import EEGWindowEncoder, EEGPretrainModel, FeatureClassifier
from .protocol import arms, digest, read_json, source_identity, task_name, tasks, write_json
from .tensor_attention import Tucker2
from .training import fit, metrics, predict


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_torch(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def initialize(cfg):
    output = Path(cfg['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    identity = {'config': cfg, 'source': source_identity()}
    study_id = digest(identity)
    with (output / '.manifest.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = output / 'study.json'
        if path.exists():
            previous = read_json(path)
            if previous.get('study_id') != study_id:
                raise RuntimeError('Code, packages, Python, or settings changed. Use a NEW output_dir; old results will not be reused.')
        else:
            write_json(path, {**identity, 'study_id': study_id, 'tasks': tasks(cfg),
                             'arms': arms(cfg), 'availability': availability_metadata()})
        report = output / 'report'
        report.mkdir(exist_ok=True)
        write_json(report / 'study_manifest.json', read_json(path))
    return output, study_id


def _seed(seed):
    seed_everything(seed, deterministic=True, threads=int(os.environ.get('SLURM_CPUS_PER_TASK', '4')))


def _register_dataset(output, subject, bundle):
    # All seeds of one participant must use the same source and preprocessed trials.
    with (output / '.manifest.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = output / 'datasets.json'
        records = read_json(path) if path.is_file() else {}
        key = f'A{subject:02d}'
        value = {'fingerprint': bundle.fingerprint, 'sources': bundle.metadata['sources'],
                 'trials': len(bundle), 'class_counts': bundle.metadata['class_counts'],
                 'skipped': bundle.metadata['skipped']}
        if key in records and records[key] != value:
            raise RuntimeError(f'{key} dataset changed between tasks. Use a new output directory.')
        records[key] = value
        write_json(path, records)
        write_json(output / 'report' / 'datasets.json', records)


def _full(n, windows):
    return torch.ones(n, 22, windows, dtype=torch.bool)


def _calibrate(cfg, bundle, split, directory, identity, seed, device):
    """Full TRAIN sensor calibration shared by all classifier-availability arms."""
    cache = directory / 'calibration.pt'
    record = directory / 'calibration.json'
    if cache.is_file() and record.is_file():
        previous = read_json(record)
        if previous.get('identity') != identity or previous.get('cache_sha256') != file_sha256(cache):
            raise RuntimeError('Calibration identity/checksum changed. Use a new output directory.')
        # This is our own provenance-checked local artifact, never an arbitrary checkpoint.
        stage = torch.load(cache, map_location='cpu', weights_only=False)
        if stage['identity'] != identity:
            raise RuntimeError('Calibration payload identity mismatch')
        return stage, previous['cache_sha256']
    _seed(seed)
    train, val = split['train'], split['validation']
    raw = bundle.x
    mean = raw[train].mean(axis=(0, 2), keepdims=True, dtype=np.float64)
    std = raw[train].std(axis=(0, 2), keepdims=True, dtype=np.float64)
    std = np.maximum(std, 1e-12)
    raw = torch.from_numpy(((raw - mean) / std).astype(np.float32))
    encoder = EEGWindowEncoder(**cfg['encoder'])
    pretrain = EEGPretrainModel(encoder, **cfg['classifier'])
    windows = cfg['encoder']['windows']
    print(f'{directory.name}: training shared EEGNet-derived encoder', flush=True)
    selection = fit(pretrain, raw[train], bundle.y[train], raw[val], bundle.y[val],
                    lambda epoch: _full(len(train), windows), _full(len(val), windows),
                    cfg['training']['encoder'], seed=seed, device=device,
                    history_path=directory / 'encoder_history.json', description='Common encoder')
    encoder.freeze().to(device)
    feature_batches = []
    with torch.no_grad():
        batch_size = cfg['training']['encoder']['batch_size']
        for start in range(0, len(raw), batch_size):
            feature_batches.append(encoder(raw[start:start + batch_size].to(device)).cpu())
    features = torch.cat(feature_batches)
    feature_mean = features[train].double().mean(dim=(0, 2), keepdim=True).float()
    feature_std = features[train].double().std(dim=(0, 2), correction=0, keepdim=True).float().clamp_min(1e-6)
    features = (features - feature_mean) / feature_std
    if not torch.isfinite(features).all():
        raise FloatingPointError('Nonfinite frozen EEG features')
    _seed(seed)
    tensor = Tucker2(channels=22, features=encoder.features, **cfg['tensor']).to(device)
    print(f'{directory.name}: fitting training-only Tucker factors', flush=True)
    factor_history = tensor.fit(features[train].to(device), _full(len(train), windows).to(device))
    write_json(directory / 'factor_history.json', factor_history)
    stage = {'identity': identity, 'features': features, 'encoder': encoder.cpu().state_dict(),
             'encoder_selection': selection, 'tensor': tensor.cpu().state_dict(),
             'raw_mean': torch.from_numpy(mean), 'raw_std': torch.from_numpy(std),
             'feature_mean': feature_mean, 'feature_std': feature_std,
             'train_feature_mean': features[train].mean(dim=(0, 2)),
             'feature_count': encoder.features}
    save_torch(cache, stage)
    checksum = file_sha256(cache)
    write_json(record, {'identity': identity, 'cache_sha256': checksum,
                       'encoder_selection': selection, 'feature_count': encoder.features,
                       'calibration': 'full training channels; frozen for every downstream arm',
                       'raw_mean': mean.tolist(), 'raw_std': std.tolist(),
                       'feature_mean': feature_mean.tolist(), 'feature_std': feature_std.tolist()})
    return stage, checksum


def _evaluation_banks(cfg, bundle, split, subject, seed):
    banks = []
    for partition in ('validation', 'test'):
        indices = split[partition]
        for scenario in evaluation_scenarios():
            repeats = 1 if scenario['retained'] == 22 else cfg['mask_repeats']
            for repeat in range(repeats):
                mask = make_mask_bank(len(indices), cfg['encoder']['windows'],
                                      scenario['retained'], scenario['pattern'], seed=seed,
                                      partition=partition, subject=f'A{subject:02d}', repeat=repeat,
                                      sample_ids=[bundle.sample_ids[i] for i in indices])
                row = {'partition': partition, 'scenario': scenario['name'],
                       'pattern': scenario['pattern'], 'retained': scenario['retained'],
                       'excluded_fraction': (22 - scenario['retained']) / 22,
                       'mask_repeat': repeat, 'mask_sha256': mask_bank_digest(mask)}
                banks.append((row, torch.from_numpy(mask)))
    return banks


@torch.no_grad()
def _reconstruction_scores(tensor, train_mean, features, split, banks, batch_size, device):
    """Hidden reference values are accessed ONLY for post-fit scoring."""
    scores = {}
    tensor.to(device)
    for row, mask in banks:
        key = (row['partition'], row['scenario'], row['mask_repeat'])
        if row['retained'] == 22:
            scores[key] = {'tensor': None, 'mean': None}
            continue
        part = features[split[row['partition']]]
        numerator = {'tensor': 0., 'mean': 0.}
        denominator = 0.
        for start in range(0, len(part), batch_size):
            truth = part[start:start + batch_size].to(device)
            available = mask[start:start + batch_size].to(device)
            observed = torch.where(available[..., None], truth, torch.zeros_like(truth))
            reconstruction = tensor.reconstruct(observed, available)
            estimates = {'tensor': reconstruction, 'mean': train_mean[None, :, None, :].to(device)}
            hidden = ~available[..., None]
            denominator += float(torch.where(hidden, truth.double().square(), 0.).sum())
            for name, estimate in estimates.items():
                numerator[name] += float(torch.where(hidden, (estimate.double() - truth.double()).square(), 0.).sum())
        scores[key] = {name: (value / denominator) ** .5 if denominator > 0 else None
                       for name, value in numerator.items()}
    tensor.cpu()
    return scores


def _valid_result(path, identity, arm, cfg):
    if not path.is_file():
        return False
    value = read_json(path)
    if value.get('identity') != identity or value.get('arm') != arm:
        raise RuntimeError(f'Result identity mismatch at {path}; use a new output directory.')
    expected = {(partition, item['name'], repeat)
                for partition in ('validation', 'test') for item in evaluation_scenarios()
                for repeat in range(1 if item['retained'] == 22 else cfg['mask_repeats'])}
    rows = value.get('metrics', [])
    return (value.get('status') == 'complete' and len(rows) == len(expected)
            and {(r['partition'], r['scenario'], r['mask_repeat']) for r in rows} == expected
            and all(all(np.isfinite(r[k]) for k in ('accuracy', 'balanced_accuracy', 'f1_macro')) for r in rows))


def run_task(cfg, task_index, device='cuda', *, validation_mask_patterns=None):
    from .reporting import summarize
    if not 0 <= task_index < len(tasks(cfg)):
        raise ValueError(f'task-index must be in 0..{len(tasks(cfg)) - 1}')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('The declared study needs the allocated GPU; CUDA is unavailable.')
    output, study_id = initialize(cfg)
    subject, seed = tasks(cfg)[task_index]
    _seed(seed)
    directory = output / 'artifacts' / task_name(subject, seed)
    with run_directory(directory):
        data_config = {**cfg['data'], 'subjects': [subject]}
        bundle = load_subject(data_config)
        if bundle.metadata['channel_names'] != list(CHANNEL_IDS) or bundle.metadata['sampling_rate'] != 250.:
            raise ValueError('Expected canonical 22 BCI2a EEG channels sampled at 250 Hz.')
        _register_dataset(output, subject, bundle)
        split = get_split(bundle, cfg['split'], seed, directory / 'splits')
        write_json(directory / 'split.json', split)
        write_json(directory / 'dataset.json', bundle.metadata)
        identity = {'study_id': study_id, 'subject': subject, 'seed': seed,
                    'dataset_fingerprint': bundle.fingerprint, 'split_id': split['split_id']}
        stage, checksum = _calibrate(cfg, bundle, split, directory, identity, seed, device)
        identity = {**identity, 'calibration_sha256': checksum}
        tensor = Tucker2(channels=22, features=stage['feature_count'], **cfg['tensor'])
        tensor.load_state_dict(stage['tensor'])
        windows = cfg['encoder']['windows']
        banks = _evaluation_banks(cfg, bundle, split, subject, seed)
        selection_banks = [('full', _full(len(split['validation']), windows))]
        if validation_mask_patterns:
            selection_banks = []
            for row, mask in banks:
                if row['partition'] == 'validation' and row['scenario'] in validation_mask_patterns:
                    selection_banks.append((row['scenario'], mask))
            if not selection_banks or not any(name == 'full_22' for name, _ in selection_banks):
                raise ValueError('Validation selection needs full_22 and at least one held-out loss pattern')
        # All fitted state is frozen before any degraded validation/test scoring.
        reconstruction = _reconstruction_scores(tensor, stage['train_feature_mean'], stage['features'],
                                                split, banks, cfg['training']['classifier']['batch_size'], device)
        train, val = split['train'], split['validation']
        train_ids = [bundle.sample_ids[i] for i in train]
        failures = []
        for number, arm in enumerate(arms(cfg), 1):
            arm_dir = directory / 'ARMS' / arm['name']
            arm_dir.mkdir(parents=True, exist_ok=True)
            result_path = arm_dir / 'result.json'
            if _valid_result(result_path, identity, arm, cfg):
                print(f'{directory.name} {number}/{len(arms(cfg))}: verified complete {arm["name"]}', flush=True)
                continue
            if digest({'config': cfg, 'source': source_identity()}) != study_id:
                raise RuntimeError('Source/environment changed during the study; stop and use a new output directory.')
            _seed(seed)
            print(f'{directory.name} fit {number}/{len(arms(cfg))}: {arm["name"]}', flush=True)
            write_json(arm_dir / 'config.json', {'identity': identity, 'arm': arm, 'config': cfg})
            model = None
            started = time.monotonic()
            try:
                model = FeatureClassifier(arm['attention'], arm['representation'],
                                          channels=22, windows=windows, features=stage['feature_count'],
                                          **cfg['classifier'],
                                          rank_channels=cfg['tensor']['rank_channels'],
                                          rank_features=cfg['tensor']['rank_features'],
                                          tensor=copy.deepcopy(tensor) if arm['representation'].startswith('tensor_') else None,
                                          train_feature_mean=stage['train_feature_mean'],
                                          attention_options=cfg['attentions'][arm['attention']])
                train_masks = lambda epoch: training_mask_bank(
                    len(train), windows, regime=arm['regime'], seed=seed, epoch=epoch,
                    subject=f'A{subject:02d}', sample_ids=train_ids)
                selection = fit(model, stage['features'][train], bundle.y[train],
                                stage['features'][val], bundle.y[val], train_masks, _full(len(val), windows),
                                cfg['training']['classifier'], seed=seed, device=device,
                                history_path=arm_dir / 'history.json', description=arm['name'],
                                validation_masks=selection_banks)
                rows = []
                for row, mask in banks:
                    indices = split[row['partition']]
                    probabilities = predict(model, stage['features'][indices], mask,
                                            cfg['training']['classifier']['batch_size'], device)
                    score = metrics(bundle.y[indices], probabilities)
                    rec = reconstruction[(row['partition'], row['scenario'], row['mask_repeat'])]
                    rec_name = ('tensor' if arm['representation'].startswith('tensor_') else
                                'mean' if arm['representation'] == 'mean_completion' else None)
                    rows.append({**row, **score, 'reconstruction_nrmse': rec[rec_name] if rec_name else None})
                result = {'status': 'complete', 'study_id': study_id, 'identity': identity,
                          'subject': subject, 'seed': seed, 'arm': arm,
                          'pairing': {'feature_sha256': _feature_digest(stage['features']),
                                      'calibration_sha256': checksum,
                                      'mask_sha256': {row['partition'] + ':' + row['scenario'] + ':' + str(row['mask_repeat']):
                                                      row['mask_sha256'] for row, _ in banks}},
                          'selection': selection, 'metrics': rows,
                          'token_axis': model.token_axis, 'attention_tokens': model.num_tokens,
                          'classifier_trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                          'elapsed_seconds': time.monotonic() - started}
                write_json(result_path, result)
                full = next(r for r in rows if r['partition'] == 'test' and r['scenario'] == 'full_22')
                print(f'  Full test: accuracy={full["accuracy"]:.2%}, balanced={full["balanced_accuracy"]:.2%}', flush=True)
            except Exception as error:
                failures.append(arm['name'])
                write_json(arm_dir / 'error.json', {'study_id': study_id, 'arm': arm,
                           'error': str(error), 'traceback': traceback.format_exc()})
                traceback.print_exc()
            finally:
                if model is not None:
                    model.cpu()
                    del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                summarize(output, cfg, study_id)
        write_json(directory / 'task_status.json', {'study_id': study_id, 'subject': subject,
                   'seed': seed, 'status': 'failed' if failures else 'complete', 'failed_arms': failures})
        progress = summarize(output, cfg, study_id)
        print(f'Report: {output / "report"}. Completed fits: {progress["completed_fits"]}/{progress["expected_fits"]}', flush=True)
        if failures:
            raise RuntimeError(f'{len(failures)} arms failed; resubmit this task after reviewing error.json.')


def _feature_digest(features):
    contiguous = features.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(str(tuple(contiguous.shape)).encode())
    digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def summarize_existing(cfg):
    from .reporting import summarize
    output = Path(cfg['output_dir'])
    manifest = read_json(output / 'study.json')
    if manifest['config'] != cfg:
        raise ValueError('Summary configuration differs from the recorded study.')
    return summarize(output, cfg, manifest['study_id'])
