"""Bounded, train/validation-only supervised spectral Tucker comparison with verified artifacts.

The configuration is a frozen declaration, not a hyperparameter search API.
Real data always uses nine participants, three seeds, and the full training budget.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import tempfile
import time

from inm.encoder_candidates.protocol import FIXED as CANDIDATE_FIXED
from inm.encoder_candidates.protocol import _same_fixed, _unique_object, _reject_constant

ROOT = Path(__file__).resolve().parents[2]
ARM_IDS = ('spectral_tucker_frozen', 'spectral_tucker_supervised')
SCHEMA = 'agfl-supervised-tucker-v1'
FIXED = {key: copy.deepcopy(CANDIDATE_FIXED[key]) for key in
         ('preprocessing', 'training', 'selection', 'conditions')}
FIXED.update(schema_name=SCHEMA, partitions=['train', 'validation'], training_regime='full',
             arms=list(ARM_IDS), calibration_epochs=30,
             comparisons={'primary': ['spectral_tucker_supervised', 'spectral_tucker_frozen'],
                 'secondary': [],
                 'screening_mean_gain': 0.02, 'screening_positive_participants': 6,
                 'bootstrap_repeats': 2000, 'bootstrap_seed': 20261004})
PATH_KEYS = ('data_dir', 'split_source_dir', 'output_dir')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def state_sha256(state, keys=None):
    """Portable hash of tensor names, shapes, dtypes and exact numeric bytes."""
    values = {}
    for name in sorted(state if keys is None else keys):
        tensor = state[name].detach().cpu().contiguous()
        values[name] = {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                        'sha256': hashlib.sha256(tensor.numpy().tobytes()).hexdigest()}
    return digest(values)


def calibration_sha256(calibration):
    return digest({key: value for key, value in calibration.items()
                   if key not in ('arm_id', 'supervised_factors')})


def read_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=_unique_object,
                      parse_constant=_reject_constant)


def load_config(path):
    path = Path(path).expanduser().resolve()
    cfg = read_json(path)
    required = set(FIXED) | set(PATH_KEYS) | {'name', 'subjects', 'seeds', 'synthetic'}
    if not isinstance(cfg, dict) or set(cfg) != required:
        raise ValueError('Config must have exactly the declared keys; unknown settings forbidden')
    if type(cfg['synthetic']) is not bool:
        raise ValueError('synthetic must be Boolean')
    for key, expected in FIXED.items():
        value = cfg[key]
        if cfg['synthetic'] and key == 'training':
            if not isinstance(value, dict) or set(value) != set(expected):
                raise ValueError('Synthetic training must preserve all setting keys')
            normalized = copy.deepcopy(value)
            for budget in ('maximum_epochs', 'minimum_epochs', 'patience', 'batch_size', 'warmup_epochs'):
                lower = 0 if budget in ('patience', 'warmup_epochs') else 1
                if type(value[budget]) is not int or not lower <= value[budget] <= expected[budget]:
                    raise ValueError('Synthetic budgets may only shrink the real budget')
                normalized[budget] = expected[budget]
            if value['minimum_epochs'] > value['maximum_epochs'] or value['warmup_epochs'] >= value['maximum_epochs']:
                raise ValueError('Invalid synthetic epoch budget')
            _same_fixed(normalized, expected, key)
        elif cfg['synthetic'] and key == 'calibration_epochs':
            if type(value) is not int or not 1 <= value <= expected:
                raise ValueError('Synthetic calibration epochs must be in 1..30')
        else:
            _same_fixed(value, expected, key)
    if not isinstance(cfg['name'], str) or not cfg['name'].strip():
        raise ValueError('name must be a nonempty string')
    for key, expected in (('subjects', list(range(1, 10))), ('seeds', [0, 1, 2])):
        value = cfg[key]
        if (not isinstance(value, list) or not value or any(type(x) is not int for x in value)
                or len(set(value)) != len(value) or not set(value).issubset(expected)
                or (not cfg['synthetic'] and value != expected)):
            raise ValueError(f'{key} must match the fixed cohort; only synthetic fixtures may shrink subjects')
    for key in PATH_KEYS:
        if not isinstance(cfg[key], str) or not cfg[key].strip():
            raise ValueError(f'{key} must be a path')
        cfg[key] = str((path.parent / Path(cfg[key]).expanduser()).resolve())
    output = Path(cfg['output_dir'])
    if output == ROOT or ROOT.is_relative_to(output):
        raise ValueError('Output must not contain the repository root')
    for key in ('data_dir', 'split_source_dir'):
        source = Path(cfg[key])
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError('Output overlaps immutable inputs')
    if cfg['synthetic'] and 'synthetic' not in output.name:
        raise ValueError('Synthetic output basename must contain synthetic')
    if not cfg['synthetic'] and 'synthetic' in output.name:
        raise ValueError('Real output cannot use a synthetic output identity')
    if output.exists() and not output.is_dir():
        raise ValueError('output_dir is not a directory')
    cfg['config_path'] = str(path)
    return cfg


def study_identity(cfg):
    source_paths = sorted(set(ROOT.glob('*.py')) | set((ROOT / 'agfl').rglob('*.py')) |
                          set((ROOT / 'inm').rglob('*.py')))
    packages = {}
    for name in ('numpy', 'scipy', 'torch', 'scikit-learn', 'mne', 'tqdm'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    inputs = {}
    if not cfg['synthetic']:
        paths = [Path(cfg['split_source_dir']) / 'study.json']
        for subject in cfg['subjects']:
            paths.append(Path(cfg['data_dir']) / f'A{subject:02d}T.gdf')
            for seed in cfg['seeds']:
                task = Path(cfg['split_source_dir']) / 'artifacts' / f'A{subject:02d}_seed_{seed}'
                paths += [task / 'split.json', task / 'dataset.json']
        inputs = {str(p): file_sha256(p) for p in paths}
    identity = {'schema': SCHEMA, 'config': {k: v for k, v in cfg.items() if k != 'config_path'},
                'config_sha256': file_sha256(cfg['config_path']),
                'source_sha256': {str(p.relative_to(ROOT)): file_sha256(p) for p in source_paths},
                'packages': packages, 'python': platform.python_version(),
                'execution': {'device': 'cpu', 'torch_threads': 1}, 'input_sha256': inputs}
    return {'study_id': digest(identity), 'identity': identity}


def ensure_study(cfg):
    from inm.encoder_candidates.training import _atomic_json
    identity = study_identity(cfg)
    output = Path(cfg['output_dir'])
    manifest = output / 'study.json'
    if manifest.exists():
        if read_json(manifest) != identity:
            raise ValueError('Study source/config/packages/inputs changed; use a fresh output identity')
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError('Output has artifacts without its study manifest; use a fresh output identity')
        _atomic_json(manifest, identity)
    return identity


def _prepared_identity(prepared):
    import numpy as np
    arrays = {}
    for name in ('train', 'validation'):
        part = getattr(prepared, name)
        arrays[name] = {'raw': hashlib.sha256(np.asarray(part.raw).tobytes()).hexdigest(),
                        'labels': hashlib.sha256(np.asarray(part.labels).tobytes()).hexdigest(),
                        'sample_ids': list(part.sample_ids)}
    return {'subject': prepared.subject, 'seed': prepared.seed, 'split_id': prepared.split_id,
            'data_id': prepared.data_id, 'normalization': prepared.normalization,
            'provenance': prepared.provenance, 'partitions': arrays}


def _verify_fit(directory, expected_identity=None):
    directory = Path(directory)
    names = {'checkpoint.pt', 'history.json', 'predictions.npz', 'robustness.json', 'result.json'}
    if not directory.is_dir() or set(p.name for p in directory.iterdir()) != names:
        raise ValueError(f'Incomplete or unrecognized fit artifacts: {directory}; use a fresh output identity')
    result = read_json(directory / 'result.json')
    if expected_identity is not None and result['identity'] != expected_identity:
        raise ValueError('Fit identity differs from the requested data/config/source')
    if set(result['artifact_sha256']) != names - {'result.json'}:
        raise ValueError('Fit artifact manifest is incomplete')
    for name, expected in result['artifact_sha256'].items():
        if file_sha256(directory / name) != expected:
            raise ValueError(f'Artifact checksum mismatch: {directory / name}')
    history = read_json(directory / 'history.json')
    if (not isinstance(history, list) or not history
            or [row['epoch'] for row in history] != list(range(1, len(history) + 1))
            or result['epochs_run'] != len(history)):
        raise ValueError('Training history is incomplete or epochs are not contiguous')
    selected = max(history, key=lambda row: (row['clean_validation_metrics']['balanced_accuracy'],
                   -row['clean_validation_metrics']['log_loss'], -row['epoch']))
    if (result['selected_epoch'] != selected['epoch']
            or [row['epoch'] for row in history if row['selected']] != [selected['epoch']]
            or result['train_metrics'] != selected['clean_train_metrics']
            or result['validation_metrics'] != selected['clean_validation_metrics']):
        raise ValueError('Selected checkpoint does not follow declared validation selection')
    data = result['identity']['data']
    if (result['arm'] != result['identity']['arm'] or result['subject'] != data['subject']
            or result['seed'] != data['seed']):
        raise ValueError('Result arm/subject/seed metadata is inconsistent')
    import torch
    payload = torch.load(directory / 'checkpoint.pt', map_location='cpu', weights_only=True)
    if (payload['identity'] != result['identity'] or payload['selected_epoch'] != selected['epoch']
            or payload['normalization'] != data['normalization']):
        raise ValueError('Checkpoint metadata differs from fit identity or selected epoch')
    if payload.get('format') != SCHEMA:
        raise ValueError('Wrong checkpoint schema')
    calibration = result['calibration']
    if (calibration['arm_id'] != result['arm']
            or calibration['supervised_factors'] != (result['arm'] == 'spectral_tucker_supervised')
            or calibration['training_trials'] != len(data['partitions']['train']['sample_ids'])
            or calibration['factor_epochs'] != len(calibration['factor_history'])
            or calibration_sha256(calibration) != result['calibration_sha256']):
        raise ValueError('Calibration metadata is inconsistent')
    for key in ('initial_state_sha256', 'initial_factor_sha256', 'calibration_sha256'):
        if payload[key] != result[key]:
            raise ValueError('Checkpoint initial/calibration identity mismatch')
    selected_hash = state_sha256(payload['state_dict'], ('U', 'V'))
    if (selected_hash != result['selected_factor_sha256']
            or result['factors_changed'] != (selected_hash != result['initial_factor_sha256'])
            or (result['arm'] == 'spectral_tucker_frozen' and result['factors_changed'])):
        raise ValueError('Factor training policy is inconsistent')
    if (not bool(payload['state_dict']['calibrated'].item())
            or payload['state_dict']['calibration_trials'].item() != calibration['training_trials']
            or payload['state_dict']['factor_fit_steps'].item() != calibration['factor_epochs']):
        raise ValueError('Checkpoint calibration buffers differ from actual training calibration')
    return result


def fit_arm(prepared, cfg, arm, directory, identity=None):
    """Fit one fixed arm; validate checkpoints and refuse incomplete artifact reuse."""
    import numpy as np
    import torch
    from inm.encoder_candidates.training import (_scoped_fit_rng, _evaluate, _atomic_json,
        _atomic_torch_save, _state_copy, learning_rate_for_epoch)
    from inm.encoder_candidates.diagnostics import _atomic_npz, evaluate_selected
    from .models import build_model, restore_model
    if arm not in ARM_IDS:
        raise ValueError('Unknown arm')
    if cfg != load_config(cfg['config_path']):
        raise ValueError('In-memory configuration differs from its strict declaration')
    torch.set_num_threads(1)
    identity = identity or study_identity(cfg)
    fit_identity = {'study_id': identity['study_id'], 'arm': arm, 'synthetic': cfg['synthetic'],
                    'data': _prepared_identity(prepared)}
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        result = _verify_fit(directory, fit_identity)
        # Reuse verifies checkpoint predictions against current prepared data as well.
        _reload_verify(directory, prepared, cfg, arm, result)
        return result
    for name in ('train', 'validation'):
        part = getattr(prepared, name)
        if (part.raw.shape != (len(part.labels), 22, 4, 250) or not np.isfinite(part.raw).all()
                or not np.isin(part.labels, range(4)).all()
                or np.any(np.bincount(part.labels, minlength=4) == 0)
                or len(part.sample_ids) != len(part.labels) or len(set(part.sample_ids)) != len(part.labels)):
            raise ValueError('Invalid prepared data shape, finite values, classes, or sample IDs')
    if set(prepared.train.sample_ids) & set(prepared.validation.sample_ids):
        raise ValueError('Training and validation sample IDs overlap')
    device = torch.device('cpu')
    settings = cfg['training']
    train_x = torch.tensor(np.asarray(prepared.train.raw), dtype=torch.float32)
    val_x = torch.tensor(np.asarray(prepared.validation.raw), dtype=torch.float32)
    train_y = torch.tensor(np.asarray(prepared.train.labels), dtype=torch.long)
    batch = settings['batch_size']
    full = torch.ones((22, 4), dtype=torch.bool)
    started = time.monotonic()
    # Reserve directory before starting expensive work. Interrupted fits fail closed.
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_json(directory / 'in_progress.json', fit_identity)
    with _scoped_fit_rng(prepared.seed, device) as rng_metadata:
        model = build_model(arm, seed=prepared.seed).to(device)
        cal_start = time.monotonic()
        with _scoped_fit_rng(prepared.seed + 300007, device):
            calibration = model.calibrate(train_x, tucker_epochs=cfg['calibration_epochs'])
        calibration_seconds = time.monotonic() - cal_start
        initial_state_sha256 = state_sha256(model.state_dict())
        initial_factor_sha256 = state_sha256(model.state_dict(), ('U', 'V'))
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'],
                                      weight_decay=settings['weight_decay'])
        criterion = torch.nn.CrossEntropyLoss()
        history, best_key, best_state, best_epoch, stale = [], None, None, None, 0
        for epoch in range(1, settings['maximum_epochs'] + 1):
            lr = learning_rate_for_epoch(epoch, settings)
            for group in optimizer.param_groups:
                group['lr'] = lr
            model.train()
            order = np.random.default_rng((prepared.seed + 100003 * epoch + 700001) % (2**63 - 1)).permutation(len(train_x))
            total_loss = 0.0
            for start in range(0, len(order), batch):
                indices = order[start:start + batch]
                optimizer.zero_grad(set_to_none=True)
                logits = model(train_x[indices], full.unsqueeze(0).expand(len(indices), -1, -1))
                loss = criterion(logits, train_y[indices])
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip_norm'], error_if_nonfinite=True)
                optimizer.step()
                model.clip_weights()
                total_loss += float(loss.detach()) * len(indices)
            tm, _ = _evaluate(model, train_x, prepared.train.labels, batch, device)
            vm, _ = _evaluate(model, val_x, prepared.validation.labels, batch, device)
            key = (vm['balanced_accuracy'], -vm['log_loss'])
            improved = best_key is None or key > best_key
            if improved:
                best_key, best_state, best_epoch, stale = key, _state_copy(model), epoch, 0
            else:
                stale += 1
            history.append({'epoch': epoch, 'learning_rate': lr, 'optimization_loss': total_loss / len(train_x),
                            'clean_train_metrics': tm, 'clean_validation_metrics': vm})
            if epoch == 1 or epoch % 25 == 0:
                print(f'A{prepared.subject:02d} {arm} epoch {epoch}: validation BA={vm["balanced_accuracy"]:.4f}', flush=True)
            if epoch >= settings['minimum_epochs'] and not improved and stale >= settings['patience']:
                break
        model.load_state_dict(best_state, strict=True)
        model.eval()
        selected_factor_sha256 = state_sha256(model.state_dict(), ('U', 'V'))
        if arm == 'spectral_tucker_frozen' and selected_factor_sha256 != initial_factor_sha256:
            raise RuntimeError('Frozen factors changed during classifier training')
        tm, tp = _evaluate(model, train_x, prepared.train.labels, batch, device)
        vm, vp = _evaluate(model, val_x, prepared.validation.labels, batch, device)
        for row in history:
            row['selected'] = row['epoch'] == best_epoch
        checkpoint = {'format': SCHEMA, 'identity': fit_identity, 'constructor': model.constructor_settings(),
                      'state_dict': _state_copy(model), 'selected_epoch': best_epoch,
                      'normalization': prepared.normalization, 'rng_policy': rng_metadata,
                      'initial_state_sha256': initial_state_sha256,
                      'initial_factor_sha256': initial_factor_sha256,
                      'calibration_sha256': calibration_sha256(calibration)}
        _atomic_torch_save(directory / 'checkpoint.pt', checkpoint)
        _atomic_json(directory / 'history.json', history)
        _atomic_npz(directory / 'predictions.npz', train_labels=prepared.train.labels,
                    validation_labels=prepared.validation.labels, train_probabilities=tp, validation_probabilities=vp,
                    train_sample_ids=np.asarray(prepared.train.sample_ids),
                    validation_sample_ids=np.asarray(prepared.validation.sample_ids))
        payload = torch.load(directory / 'checkpoint.pt', map_location='cpu', weights_only=True)
        restored = restore_model(arm, payload['constructor'], payload['state_dict']).eval()
        _, rtp = _evaluate(restored, train_x, prepared.train.labels, batch, device)
        _, rvp = _evaluate(restored, val_x, prepared.validation.labels, batch, device)
        if not np.array_equal(tp, rtp) or not np.array_equal(vp, rvp):
            raise RuntimeError('Checkpoint reload changed probabilities')
        robustness = evaluate_selected(restored, prepared, cfg)
        _atomic_json(directory / 'robustness.json', robustness)
        result = {'identity': fit_identity, 'arm': arm, 'subject': prepared.subject, 'seed': prepared.seed,
                  'selected_epoch': best_epoch, 'epochs_run': len(history), 'train_metrics': tm, 'validation_metrics': vm,
                  'parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                  'calibration': calibration, 'calibration_seconds': calibration_seconds,
                  'initial_state_sha256': initial_state_sha256,
                  'initial_factor_sha256': initial_factor_sha256,
                  'selected_factor_sha256': selected_factor_sha256,
                  'factors_changed': selected_factor_sha256 != initial_factor_sha256,
                  'calibration_sha256': calibration_sha256(calibration),
                  'elapsed_seconds': time.monotonic() - started, 'checkpoint_reload_exact': True,
                  'rng_policy': rng_metadata,
                  'artifact_sha256': {name: file_sha256(directory / name) for name in
                                      ('checkpoint.pt', 'history.json', 'predictions.npz', 'robustness.json')}}
        _atomic_json(directory / 'result.json', result)
        (directory / 'in_progress.json').unlink()
        return result


def _reload_verify(directory, prepared, cfg, arm, result):
    import numpy as np
    import torch
    from .models import restore_model
    from inm.encoder_candidates.training import _evaluate
    payload = torch.load(directory / 'checkpoint.pt', map_location='cpu', weights_only=True)
    if payload['identity'] != result['identity'] or payload['selected_epoch'] != result['selected_epoch']:
        raise ValueError('Checkpoint metadata does not match result')
    model = restore_model(arm, payload['constructor'], payload['state_dict']).eval()
    with np.load(directory / 'predictions.npz', allow_pickle=False) as saved:
        for part in ('train', 'validation'):
            data = getattr(prepared, part)
            metrics, probs = _evaluate(model, torch.tensor(data.raw), data.labels,
                                       cfg['training']['batch_size'], torch.device('cpu'))
            if (not np.array_equal(saved[part + '_labels'], data.labels)
                    or not np.array_equal(saved[part + '_sample_ids'], data.sample_ids)
                    or not np.array_equal(saved[part + '_probabilities'], probs)
                    or metrics != result[part + '_metrics']):
                raise ValueError('Checkpoint reload or saved prediction alignment failed')


def _synthetic_prepared(subject, seed):
    import numpy as np
    from inm.encoder_candidates.data import PreparedData, PartitionData
    rng = np.random.default_rng(seed + subject * 1000)
    def part(name, repeats):
        y = np.tile(np.arange(4, dtype=np.int64), repeats)
        x = rng.normal(size=(len(y), 22, 4, 250)).astype(np.float32)
        x[:, :4] += y[:, None, None, None] * 0.1
        return PartitionData(x, y, tuple(f'synthetic-{subject}-{name}-{i}' for i in range(len(y))))
    return PreparedData(subject, seed, part('train', 4), part('validation', 2),
                        {'synthetic': True}, 'synthetic-split', f'synthetic-{subject}-{seed}', {'synthetic': True})


def task_directory(cfg, subject, seed=0):
    return Path(cfg['output_dir']) / 'tasks' / f'A{subject:02d}_seed_{seed}'


def run_task(cfg, task_index, *, synthetic=False):
    from inm.encoder_candidates.training import _atomic_json
    from inm.encoder_candidates.data import prepare_subject
    if synthetic != cfg['synthetic']:
        raise ValueError('Synthetic configuration and --synthetic flag must agree')
    tasks = [(s, seed) for s in cfg['subjects'] for seed in cfg['seeds']]
    if not 0 <= task_index < len(tasks):
        raise ValueError('Task index outside declared cohort')
    identity = ensure_study(cfg)
    subject, seed = tasks[task_index]
    directory = task_directory(cfg, subject, seed)
    directory.mkdir(parents=True, exist_ok=True)
    if synthetic:
        prepared = _synthetic_prepared(subject, seed)
    else:
        # The old loader writes metadata; stage it outside established artifacts first.
        with tempfile.TemporaryDirectory(prefix='agfl-supervised-tucker-data-') as temporary:
            prepared = prepare_subject(cfg, subject, seed, temporary)
    metadata = _prepared_identity(prepared)
    path = directory / 'data_provenance.json'
    if path.exists() and read_json(path) != metadata:
        raise ValueError('Existing task data identity changed')
    if not path.exists():
        _atomic_json(path, metadata)
    for arm in ARM_IDS:
        print(f'A{subject:02d} seed {seed}: {arm}', flush=True)
        fit_arm(prepared, cfg, arm, directory / arm, identity)
    verify_pair([_verify_fit(directory / arm) for arm in ARM_IDS])
    completion = {'study_id': identity['study_id'], 'subject': subject, 'seed': seed,
                  'data_sha256': file_sha256(path),
                  'results_sha256': {arm: file_sha256(directory / arm / 'result.json') for arm in ARM_IDS}}
    completion_path = directory / 'complete.json'
    if completion_path.exists() and read_json(completion_path) != completion:
        raise ValueError('Existing task completion changed')
    _atomic_json(completion_path, completion)
    return completion


def verify_task(cfg, subject, seed, identity):
    directory = task_directory(cfg, subject, seed)
    complete = read_json(directory / 'complete.json')
    if (complete['study_id'] != identity['study_id'] or complete['subject'] != subject
            or complete['seed'] != seed or set(complete['results_sha256']) != set(ARM_IDS)
            or complete['data_sha256'] != file_sha256(directory / 'data_provenance.json')):
        raise ValueError('Task completion identity mismatch')
    data = read_json(directory / 'data_provenance.json')
    results = []
    for arm in ARM_IDS:
        if file_sha256(directory / arm / 'result.json') != complete['results_sha256'][arm]:
            raise ValueError('Task result checksum mismatch')
        expected = {'study_id': identity['study_id'], 'arm': arm, 'synthetic': cfg['synthetic'], 'data': data}
        result = _verify_fit(directory / arm, expected)
        if result['calibration']['factor_epochs'] != cfg['calibration_epochs']:
            raise ValueError('Actual factor fitting budget differs from declaration')
        results.append(result)
    verify_pair(results)
    return results


def verify_pair(results):
    if {result['arm'] for result in results} != set(ARM_IDS) or len(results) != 2:
        raise ValueError('Both paired arms are required')
    for field in ('identity',):
        data = [result[field]['data'] for result in results]
        if data[0] != data[1]:
            raise ValueError('Paired arms have different data identities')
    for field in ('initial_state_sha256', 'initial_factor_sha256', 'calibration_sha256'):
        if results[0][field] != results[1][field]:
            raise ValueError('Paired arm initialization/calibration mismatch: ' + field)


def aggregate_complete(cfg, rows, robustness):
    """Equal participants after equal declared seeds; incomplete evidence has no means."""
    import numpy as np
    empty = {'cohort': [], 'contrasts': [], 'participant_scores': [], 'seed_contrasts': [],
             'robustness_cohort': [], 'robustness_contrasts': []}
    expected = {(subject, seed, arm) for subject in cfg['subjects']
                for seed in cfg['seeds'] for arm in ARM_IDS}
    keys = [(r['subject'], r['seed'], r['arm']) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate participant/seed/arm score')
    if set(keys) != expected:
        return empty
    conditions = [condition['name'] for condition in cfg['conditions']]
    robust_expected = {(*key, scenario) for key in expected for scenario in conditions}
    robust_keys = [(r['subject'], r['seed'], r['arm'], r['scenario']) for r in robustness]
    if len(set(robust_keys)) != len(robust_keys):
        raise ValueError('Duplicate robustness score')
    if set(robust_keys) != robust_expected:
        return empty
    result = copy.deepcopy(empty)
    metrics = ('train_balanced_accuracy', 'validation_balanced_accuracy', 'validation_accuracy',
               'validation_log_loss', 'train_validation_gap')
    for arm in ARM_IDS:
        for subject in cfg['subjects']:
            selected = [r for r in rows if r['subject'] == subject and r['arm'] == arm]
            result['participant_scores'].append({'subject': subject, 'arm': arm,
                'seeds': len(selected), **{key: float(np.mean([r[key] for r in selected])) for key in metrics}})
        selected = [r for r in result['participant_scores'] if r['arm'] == arm]
        result['cohort'].append({'arm': arm, 'participants': len(selected),
            'seeds_per_participant': len(cfg['seeds']),
            **{key: float(np.mean([r[key] for r in selected])) for key in metrics}})
        for scenario in conditions:
            by_participant = [{key: float(np.mean([r[key] for r in robustness if r['arm'] == arm
                and r['subject'] == subject and r['scenario'] == scenario]))
                for key in ('balanced_accuracy', 'log_loss')} for subject in cfg['subjects']]
            result['robustness_cohort'].append({'arm': arm, 'scenario': scenario,
                'participants': len(by_participant), 'seeds_per_participant': len(cfg['seeds']),
                **{key: float(np.mean([r[key] for r in by_participant]))
                   for key in ('balanced_accuracy', 'log_loss')}})
    candidate, reference = cfg['comparisons']['primary']
    def contrast(values, metric, scenario=None):
        differences = np.asarray(values, dtype=float)
        rng = np.random.default_rng(cfg['comparisons']['bootstrap_seed'])
        sampled = rng.integers(0, len(differences), size=(cfg['comparisons']['bootstrap_repeats'], len(differences)))
        boot = differences[sampled].mean(axis=1)
        value = {'candidate': candidate, 'reference': reference, 'metric': metric,
            'difference_direction': 'candidate_minus_reference',
            'mean_difference': float(differences.mean()),
            'ci95_participant_bootstrap': np.quantile(boot, [.025, .975]).tolist(),
            'participant_differences': dict(zip(map(str, cfg['subjects']), differences.tolist()))}
        if metric == 'balanced_accuracy':
            value['positive_participants'] = int((differences > 0).sum())
            value['passes_prespecified_screen'] = bool(differences.mean() >= cfg['comparisons']['screening_mean_gain']
                and (differences > 0).sum() >= cfg['comparisons']['screening_positive_participants'])
        if scenario is not None:
            value['scenario'] = scenario
        return value
    for metric in ('balanced_accuracy', 'log_loss'):
        key = 'validation_' + metric
        values = []
        for subject in cfg['subjects']:
            pair = {r['arm']: r for r in result['participant_scores'] if r['subject'] == subject}
            values.append(pair[candidate][key] - pair[reference][key])
            for seed in cfg['seeds']:
                seed_pair = {r['arm']: r for r in rows if r['subject'] == subject and r['seed'] == seed}
                result['seed_contrasts'].append({'subject': subject, 'seed': seed, 'metric': metric,
                    'candidate': candidate, 'reference': reference,
                    'difference': seed_pair[candidate][key] - seed_pair[reference][key]})
        result['contrasts'].append(contrast(values, metric))
        for scenario in conditions:
            values = []
            for subject in cfg['subjects']:
                pair = {arm: float(np.mean([r[metric] for r in robustness if r['subject'] == subject
                    and r['arm'] == arm and r['scenario'] == scenario])) for arm in ARM_IDS}
                values.append(pair[candidate] - pair[reference])
            value = contrast(values, metric, scenario)
            value.pop('passes_prespecified_screen', None)  # Full input is the only primary screen.
            result['robustness_contrasts'].append(value)
    return result


def summarize(cfg):
    """Recompute full-input metrics from numeric predictions; never report partial cohort means."""
    import numpy as np
    from inm.encoder_candidates.training import _metrics, _atomic_json
    identity = ensure_study(cfg)
    rows, missing, robustness = [], [], []
    for subject in cfg['subjects']:
        for seed in cfg['seeds']:
            directory = task_directory(cfg, subject, seed)
            if not (directory / 'complete.json').exists():
                missing.append({'subject': subject, 'seed': seed, 'reason':
                                'partial task artifacts without completion' if directory.exists() and any(directory.iterdir())
                                else 'missing task completion'})
                continue
            results = verify_task(cfg, subject, seed, identity)
            paired_ids, paired_masks = None, None
            for result in results:
                arm = result['arm']
                with np.load(directory / arm / 'predictions.npz', allow_pickle=False) as saved:
                    ids = {part: saved[part + '_sample_ids'].tolist() for part in ('train', 'validation')}
                    if paired_ids is not None and ids != paired_ids:
                        raise ValueError('Arms have unpaired sample order')
                    paired_ids = ids
                    row = {'subject': subject, 'seed': seed, 'arm': arm,
                           'selected_epoch': result['selected_epoch'], 'epochs_run': result['epochs_run'],
                           'parameters': result['parameters'], 'elapsed_seconds': result['elapsed_seconds']}
                    for part in ('train', 'validation'):
                        expected = result['identity']['data']['partitions'][part]
                        if (saved[part + '_sample_ids'].tolist() != expected['sample_ids']
                                or hashlib.sha256(saved[part + '_labels'].tobytes()).hexdigest() != expected['labels']):
                            raise ValueError('Saved labels or sample IDs differ from verified data identity')
                        metrics = _metrics(saved[part + '_labels'], saved[part + '_probabilities'])
                        if metrics != result[part + '_metrics']:
                            raise ValueError('Saved metrics differ from recomputed predictions')
                        for metric in ('balanced_accuracy', 'accuracy', 'log_loss'):
                            row[part + '_' + metric] = metrics[metric]
                    row['train_validation_gap'] = row['train_balanced_accuracy'] - row['validation_balanced_accuracy']
                    rows.append(row)
                robust = read_json(directory / arm / 'robustness.json')
                expected_conditions = {(condition['name'], repeat) for condition in cfg['conditions']
                                       for repeat in range(condition['repeats'])}
                if (len(robust) != len(expected_conditions) or
                        {(r['scenario'], r['repeat']) for r in robust} != expected_conditions):
                    raise ValueError('Robustness scenarios/repeat identities differ from declaration')
                masks = [(r['scenario'], r['repeat'], r['mask_sha256']) for r in robust]
                if paired_masks is not None and masks != paired_masks:
                    raise ValueError('Robustness masks are not paired across arms')
                paired_masks = masks
                for condition in cfg['conditions']:
                    selected = [r for r in robust if r['scenario'] == condition['name']]
                    if len(selected) != condition['repeats']:
                        raise ValueError('Incomplete robustness repeats')
                    robustness.append({'subject': subject, 'seed': seed, 'arm': arm,
                                       'scenario': condition['name'],
                                       'balanced_accuracy': float(np.mean([r['balanced_accuracy'] for r in selected])),
                                       'log_loss': float(np.mean([r['log_loss'] for r in selected]))})
    complete = not missing
    report = {'format': SCHEMA, 'study_id': identity['study_id'], 'synthetic': cfg['synthetic'],
              'status': 'complete' if complete else 'partial', 'missing': missing,
              'rows': rows, 'robustness': robustness, 'cohort': [], 'contrasts': [], 'participant_scores': [], 'seed_contrasts': [],
              'robustness_cohort': [], 'robustness_contrasts': [],
              'limitations': ['Three paired split/training repetitions averaged within each participant; exploratory validation-selected screen.',
                              'No test evaluation or independent confirmation.',
                              'Bootstrap intervals describe participant sampling, not model selection uncertainty.']}
    if complete:
        report.update(aggregate_complete(cfg, rows, robustness))
    output = Path(cfg['output_dir']) / 'report'
    _atomic_json(output / 'summary.json', report)
    for name, values in (('scores.csv', rows), ('robustness.csv', robustness),
                         ('cohort.csv', report['cohort']), ('contrasts.csv', report['contrasts']),
                         ('participant_scores.csv', report['participant_scores']),
                         ('seed_contrasts.csv', report['seed_contrasts']),
                         ('robustness_cohort.csv', report['robustness_cohort']),
                         ('robustness_contrasts.csv', report['robustness_contrasts'])):
        path = output / name
        with path.open('w', newline='') as stream:
            if values:
                writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for value in values for key in value)))
                writer.writeheader()
                writer.writerows(values)
    return report


def verify_pilot(cfg):
    """Gate execution on calibrated pairing, artifact integrity and numerical replay."""
    import torch
    from inm.encoder_candidates.data import prepare_subject
    from inm.encoder_candidates.diagnostics import evaluate_selected
    from inm.encoder_candidates.training import _atomic_json
    from .models import restore_model
    torch.set_num_threads(1)
    identity = ensure_study(cfg)
    subject, seed = cfg['subjects'][0], cfg['seeds'][0]
    results = verify_task(cfg, subject, seed, identity)
    if cfg['synthetic']:
        prepared = _synthetic_prepared(subject, seed)
    else:
        with tempfile.TemporaryDirectory(prefix='agfl-supervised-pilot-data-') as temporary:
            prepared = prepare_subject(cfg, subject, seed, temporary)
    directory = task_directory(cfg, subject, seed)
    if _prepared_identity(prepared) != read_json(directory / 'data_provenance.json'):
        raise ValueError('Pilot replay data differs from saved identity')
    paired_masks = None
    for result in results:
        arm = result['arm']
        _reload_verify(directory / arm, prepared, cfg, arm, result)
        payload = torch.load(directory / arm / 'checkpoint.pt', map_location='cpu', weights_only=True)
        model = restore_model(arm, payload['constructor'], payload['state_dict']).eval()
        actual = evaluate_selected(model, prepared, cfg)
        if actual != read_json(directory / arm / 'robustness.json'):
            raise ValueError('Pilot robustness replay differs from saved evidence')
        masks = [(row['scenario'], row['repeat'], row['mask_sha256']) for row in actual]
        if paired_masks is not None and paired_masks != masks:
            raise ValueError('Pilot availability masks are not paired')
        paired_masks = masks
    review = {'study_id': identity['study_id'], 'status': 'passed',
              'completion_sha256': file_sha256(directory / 'complete.json'),
              'checks': ['paired calibrated initialization', 'frozen factors unchanged',
                         'calibration training trial and epoch buffers', 'checkpoint probability replay',
                         'all validation availability conditions replay', 'paired masks'],
              'accuracy_gate': False, 'synthetic': cfg['synthetic']}
    _atomic_json(Path(cfg['output_dir']) / 'pilot_review.json', review)
    return review


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'configs/supervised-tucker.json'))
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--plan', action='store_true')
    actions.add_argument('--task-index', type=int)
    actions.add_argument('--summarize-only', action='store_true')
    actions.add_argument('--verify-pilot', action='store_true')
    parser.add_argument('--synthetic', action='store_true')
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.plan:
        print(json.dumps({'tasks': len(cfg['subjects']) * len(cfg['seeds']),
                          'fits': len(cfg['subjects']) * len(cfg['seeds']) * len(ARM_IDS),
                          'arms': list(ARM_IDS), 'output_dir': cfg['output_dir'],
                          'synthetic': cfg['synthetic'], 'training': cfg['training']}, indent=2))
        return 0
    import torch
    torch.set_num_threads(1)
    if args.verify_pilot:
        verify_pilot(cfg)
        print('Pilot artifact, calibration, pairing and all-condition numerical replay checks passed.')
    elif args.summarize_only:
        report = summarize(cfg)
        print(json.dumps({'status': report['status'], 'fits': len(report['rows']), 'cohort': report['cohort']}, indent=2))
    else:
        run_task(cfg, args.task_index, synthetic=args.synthetic)
    return 0
