"""Train each full-trial model once; sweep availability after epoch selection."""
from __future__ import annotations
import fcntl
import hashlib
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from eeg_models.datasets.splits import get_split
from eeg_models.models import build_model
from eeg_models.reproducibility import seed_everything
from eeg_models.storage import run_directory
from .availability import (CHANNEL_IDS, availability_metadata, evaluation_scenarios,
                           make_mask_bank, mask_bank_digest)
from .data import load_subject
from .protocol import digest, models, n_windows, read_json, source_identity, task_name, tasks, write_json
from eeg_models.models._shared.tensor import Tucker2
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
            if read_json(path).get('study_id') != study_id:
                raise RuntimeError('Source, environment or configuration changed. Use a NEW output_dir; old results will not be reused.')
        else:
            write_json(path, {**identity, 'study_id': study_id, 'tasks': tasks(cfg),
                             'models': models(cfg), 'availability': availability_metadata()})
        report = output / 'report'
        report.mkdir(exist_ok=True)
        write_json(report / 'study_manifest.json', read_json(path))
    return output, study_id


def _seed(seed):
    seed_everything(seed, deterministic=True, threads=int(os.environ.get('SLURM_CPUS_PER_TASK', '4')))


def _register_dataset(output, subject, bundle):
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
    """Fit train-only signal statistics and shared raw-window Tucker factors.

    No classifier or encoder is fitted here. Both backbones remain trainable
    end to end; paired variants share exactly the same data calibration.
    """
    cache, record = directory / 'calibration.pt', directory / 'calibration.json'
    if cache.is_file() and record.is_file():
        previous = read_json(record)
        if previous.get('identity') != identity or previous.get('cache_sha256') != file_sha256(cache):
            raise RuntimeError('Calibration identity/checksum changed. Use a new output directory.')
        stage = torch.load(cache, map_location='cpu', weights_only=False)
        if stage['identity'] != identity:
            raise RuntimeError('Calibration payload identity mismatch')
        raw = torch.from_numpy(((bundle.x - stage['raw_mean'].numpy()) /
                                stage['raw_std'].numpy()).astype(np.float32))
        return raw, stage, previous['cache_sha256']
    train = split['train']
    mean = bundle.x[train].mean(axis=(0, 2), keepdims=True, dtype=np.float64)
    std = np.maximum(bundle.x[train].std(axis=(0, 2), keepdims=True, dtype=np.float64), 1e-12)
    raw = torch.from_numpy(((bundle.x - mean) / std).astype(np.float32))
    stage = {'identity': identity, 'raw_mean': torch.from_numpy(mean),
             'raw_std': torch.from_numpy(std), 'tensor': None}
    if any(model['tensor'] for model in models(cfg)):
        _seed(seed)
        tensor = Tucker2(channels=22, features=cfg['data']['filter_window_samples'], **cfg['tensor']).to(device)
        train_windows = raw[train].reshape(len(train), 22, n_windows(cfg), -1).to(device)
        print(f'{directory.name}: fitting train-only Tucker signal factors', flush=True)
        history = tensor.fit(train_windows, _full(len(train), n_windows(cfg)).to(device))
        write_json(directory / 'factor_history.json', history)
        stage['tensor'] = tensor.cpu().state_dict()
        del tensor, train_windows
    save_torch(cache, stage)
    checksum = file_sha256(cache)
    write_json(record, {'identity': identity, 'cache_sha256': checksum,
                       'calibration': 'training-only channel statistics and raw-window Tucker factors; no frozen encoder',
                       'raw_mean': mean.tolist(), 'raw_std': std.tolist(),
                       'tensor_fitted': stage['tensor'] is not None,
                       'tensor_input': [22, n_windows(cfg), cfg['data']['filter_window_samples']]})
    return raw, stage, checksum


def _evaluation_banks(cfg, bundle, split, subject, seed):
    banks = []
    for partition in ('validation', 'test'):
        indices = split[partition]
        for scenario in evaluation_scenarios():
            for repeat in range(1 if scenario['retained'] == 22 else cfg['mask_repeats']):
                mask = make_mask_bank(len(indices), n_windows(cfg), scenario['retained'], scenario['pattern'],
                                      seed=seed, partition=partition, subject=f'A{subject:02d}', repeat=repeat,
                                      sample_ids=[bundle.sample_ids[i] for i in indices])
                row = {'partition': partition, 'scenario': scenario['name'], 'pattern': scenario['pattern'],
                       'retained': scenario['retained'], 'excluded_fraction': (22 - scenario['retained']) / 22,
                       'mask_repeat': repeat, 'mask_sha256': mask_bank_digest(mask)}
                banks.append((row, torch.from_numpy(mask)))
    return banks


@torch.no_grad()
def _reconstruction_scores(tensor, raw, split, banks, cfg, device):
    """Hidden reference samples are targets ONLY for post-fit scoring."""
    scores = {}
    tensor.to(device)
    for row, mask in banks:
        key = (row['partition'], row['scenario'], row['mask_repeat'])
        if row['retained'] == 22:
            scores[key] = None
            continue
        indices = split[row['partition']]
        numerator, denominator = 0., 0.
        batch_size = cfg['training']['batch_size']
        for start in range(0, len(indices), batch_size):
            truth = raw[indices[start:start + batch_size]].to(device).reshape(-1, 22, n_windows(cfg),
                                                                            cfg['data']['filter_window_samples'])
            available = mask[start:start + batch_size].to(device)
            observed = torch.where(available[..., None], truth, torch.zeros_like(truth))
            estimate = tensor.reconstruct(observed, available)
            hidden = ~available[..., None]
            denominator += float(torch.where(hidden, truth.double().square(), 0.).sum())
            numerator += float(torch.where(hidden, (estimate.double() - truth.double()).square(), 0.).sum())
        scores[key] = (numerator / denominator) ** .5 if denominator > 0 else None
    tensor.cpu()
    return scores


def _valid_result(path, identity, model, cfg):
    if not path.is_file():
        return False
    from .reporting import _expected_cells, _validated_checkpoint, _validated_result
    value = read_json(path)
    if value.get('identity') != identity or value.get('model') != model:
        raise RuntimeError(f'Result identity mismatch at {path}; use a new output directory.')
    try:
        _validated_result(value, identity['subject'], identity['seed'], model,
                          identity['study_id'], _expected_cells(cfg))
        _validated_checkpoint(path, value)
    except (OSError, ValueError, TypeError, KeyError):
        return False
    checkpoint = path.parent / 'checkpoint.pt'
    if not checkpoint.is_file() or value.get('checkpoint_sha256') != file_sha256(checkpoint):
        raise RuntimeError(f'Completed result has a missing/changed checkpoint at {path.parent}')
    return True


def _train_or_restore(model, directory, identity, definition, cfg, raw, bundle, split, stage, seed, device):
    """A saved selected checkpoint avoids retraining if evaluation was interrupted."""
    checkpoint, record = directory / 'checkpoint.pt', directory / 'checkpoint.json'
    if checkpoint.is_file() and record.is_file():
        saved = read_json(record)
        if (saved.get('identity') != identity or saved.get('model') != definition
                or saved.get('checkpoint_sha256') != file_sha256(checkpoint)):
            raise RuntimeError('Checkpoint identity/checksum mismatch; use a new output directory')
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if payload['identity'] != identity or payload['model'] != definition:
            raise RuntimeError('Checkpoint payload identity mismatch')
        model.load_state_dict(payload['state_dict'])
        model.to(device).eval()
        return payload['selection'], saved['checkpoint_sha256']
    train, val = split['train'], split['validation']
    selection = fit(model, raw[train], bundle.y[train], raw[val], bundle.y[val], cfg['training'],
                    seed=seed, device=device, windows=n_windows(cfg),
                    history_path=directory / 'history.json', description=definition['name'])
    payload = {'identity': identity, 'model': definition, 'selection': selection,
               'state_dict': {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
               'model_options': cfg['model_options'][definition['backbone']], 'tensor_options': cfg['tensor'],
               'window_samples': cfg['data']['filter_window_samples'], 'metadata': bundle.metadata,
               'raw_mean': stage['raw_mean'], 'raw_std': stage['raw_std']}
    save_torch(checkpoint, payload)
    checksum = file_sha256(checkpoint)
    write_json(record, {'identity': identity, 'model': definition, 'selection': selection,
                        'checkpoint_sha256': checksum})
    return selection, checksum


def _run_models(cfg, output, study_id, directory, subject, seed, device):
    from .reporting import summarize
    bundle = load_subject({**cfg['data'], 'subjects': [subject]})
    if bundle.metadata['channel_names'] != list(CHANNEL_IDS) or bundle.metadata['sampling_rate'] != 250.:
        raise ValueError('Expected canonical 22 BCI2a EEG channels sampled at 250 Hz.')
    _register_dataset(output, subject, bundle)
    split = get_split(bundle, cfg['split'], seed, directory / 'splits')
    write_json(directory / 'split.json', split)
    write_json(directory / 'dataset.json', bundle.metadata)
    identity = {'study_id': study_id, 'subject': subject, 'seed': seed,
                'dataset_fingerprint': bundle.fingerprint, 'split_id': split['split_id']}
    raw, stage, checksum = _calibrate(cfg, bundle, split, directory, identity, seed, device)
    identity = {**identity, 'calibration_sha256': checksum}
    banks = _evaluation_banks(cfg, bundle, split, subject, seed)
    reconstruction = None
    failures = []
    definitions = models(cfg)
    for number, definition in enumerate(definitions, 1):
        model_dir = directory / 'MODELS' / definition['name']
        model_dir.mkdir(parents=True, exist_ok=True)
        result_path = model_dir / 'result.json'
        model = None
        started = time.monotonic()
        try:
            if _valid_result(result_path, identity, definition, cfg):
                print(f'{directory.name} {number}/{len(definitions)}: verified complete {definition["name"]}', flush=True)
                continue
            if digest({'config': cfg, 'source': source_identity()}) != study_id:
                raise RuntimeError('Source/environment changed during the study; use a new output directory.')
            # Matched baseline/tensor initialization and batch order. Completion
            # adds only buffers and is bypassed on all full-channel training batches.
            _seed(seed)
            print(f'{directory.name} model {number}/{len(definitions)}: {definition["name"]}', flush=True)
            write_json(model_dir / 'config.json', {'identity': identity, 'model': definition, 'config': cfg})
            model = build_model(definition['name'], bundle.metadata,
                                cfg['model_options'][definition['backbone']],
                                window_samples=cfg['data']['filter_window_samples'], tensor_options=cfg['tensor'])
            if definition['tensor']:
                model.signal_input.completion.load_state_dict(stage['tensor'])
            selection, checkpoint_hash = _train_or_restore(model, model_dir, identity, definition, cfg, raw,
                                                           bundle, split, stage, seed, device)
            # Everything is fitted/selected before the degraded sweep. Test
            # labels are read only by metrics, never by training or factor fitting.
            if definition['tensor'] and reconstruction is None:
                tensor = Tucker2(channels=22, features=cfg['data']['filter_window_samples'], **cfg['tensor'])
                tensor.load_state_dict(stage['tensor'])
                reconstruction = _reconstruction_scores(tensor, raw, split, banks, cfg, device)
                del tensor
            rows = []
            for row, mask in banks:
                indices = split[row['partition']]
                probabilities = predict(model, raw[indices], mask, cfg['training']['batch_size'], device)
                key = (row['partition'], row['scenario'], row['mask_repeat'])
                rows.append({**row, **metrics(bundle.y[indices], probabilities),
                             'reconstruction_nrmse': reconstruction[key] if definition['tensor'] else None})
            if digest({'config': cfg, 'source': source_identity()}) != study_id:
                raise RuntimeError('Source/environment changed during this model run; no complete result published.')
            result = {'schema_version': 3, 'status': 'complete', 'study_id': study_id, 'identity': identity,
                      'subject': subject, 'seed': seed, 'model': definition, 'selection': selection,
                      'metrics': rows, 'checkpoint_sha256': checkpoint_hash,
                      'token_axis': model.token_axis, 'attention_tokens': model.num_tokens,
                      'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                      'elapsed_seconds': time.monotonic() - started}
            write_json(result_path, result)
            full = next(row for row in rows if row['partition'] == 'test' and row['scenario'] == 'full_22')
            print(f'  Full test: accuracy={full["accuracy"]:.2%}, balanced={full["balanced_accuracy"]:.2%}', flush=True)
        except Exception as error:
            failures.append(definition['name'])
            write_json(model_dir / 'error.json', {'study_id': study_id, 'model': definition,
                       'error': str(error), 'traceback': traceback.format_exc()})
            traceback.print_exc()
        finally:
            if model is not None:
                model.cpu()
                del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            summarize(output, cfg, study_id)
    return failures


def run_task(cfg, task_index, device='cuda'):
    from .reporting import summarize
    if not 0 <= task_index < len(tasks(cfg)):
        raise ValueError(f'task-index must be in 0..{len(tasks(cfg)) - 1}')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('The study needs the allocated GPU; CUDA is unavailable.')
    output, study_id = initialize(cfg)
    subject, seed = tasks(cfg)[task_index]
    _seed(seed)
    directory = output / 'artifacts' / task_name(subject, seed)
    with run_directory(directory):
        try:
            failures = _run_models(cfg, output, study_id, directory, subject, seed, device)
            write_json(directory / 'task_status.json', {'study_id': study_id, 'subject': subject, 'seed': seed,
                       'status': 'failed' if failures else 'complete', 'failed_models': failures})
            if failures:
                raise RuntimeError(f'{len(failures)} models failed; review error.json and resubmit this task.')
        except Exception as error:
            write_json(directory / 'error.json', {'study_id': study_id, 'subject': subject, 'seed': seed,
                       'error': str(error), 'traceback': traceback.format_exc()})
            write_json(directory / 'task_status.json', {'study_id': study_id, 'subject': subject,
                       'seed': seed, 'status': 'failed', 'error': str(error)})
            raise
        finally:
            progress = summarize(output, cfg, study_id)
            print(f'Report: {output / "report"}. Model runs: {progress["completed_fits"]}/{progress["expected_fits"]}', flush=True)


def summarize_existing(cfg):
    from .reporting import summarize
    output = Path(cfg['output_dir'])
    manifest = read_json(output / 'study.json')
    if manifest['config'] != cfg:
        raise ValueError('Summary configuration differs from the recorded study.')
    return summarize(output, cfg, manifest['study_id'])
