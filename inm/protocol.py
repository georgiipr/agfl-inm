"""Four-model study configuration, atomic records and source provenance."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from eeg_models.models import MODEL_KEYS, model_definition
from eeg_models.config import merge

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def source_identity():
    import importlib.metadata
    import os
    import platform
    files = sorted([*ROOT.glob('*.py'), *(ROOT / 'eeg_models').rglob('*.py'),
                    *(ROOT / 'inm').rglob('*.py'), *(ROOT / 'configs').rglob('*.json')])
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    versions = {name: importlib.metadata.version(name)
                for name in ('torch', 'numpy', 'scipy', 'scikit-learn', 'mne', 'tqdm')}
    launcher = os.environ.get('AGFL_INM_LAUNCHER')
    launcher_record = None
    if launcher:
        path = Path(launcher)
        launcher_record = {'name': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                           'content': path.read_text()}
    return {'source_sha256': digest(hashes), 'files_sha256': hashes,
            'packages': versions, 'python': platform.python_version(), 'launcher': launcher_record}


def _read_preset(path, seen=None):
    path = Path(path).resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError('Configuration inheritance contains a cycle')
    seen.add(path)
    contents = path.read_bytes()
    value = json.loads(contents)
    if not isinstance(value, dict):
        raise ValueError('Configuration must be a JSON object')
    parent = value.pop('extends', None)
    sources = []
    if parent is not None:
        if not isinstance(parent, str):
            raise ValueError('extends must name a JSON configuration')
        base, sources = _read_preset(path.parent / parent, seen)
        value = merge(base, value)
    sources.append({'name': path.name, 'sha256': hashlib.sha256(contents).hexdigest()})
    return value, sources


def _positive(value, name, *, zero=False):
    if type(value) is not int or value < (0 if zero else 1):
        raise ValueError(f'{name} must be a {"nonnegative" if zero else "positive"} integer')


def _number(value, name, *, minimum=0, strict=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or (value <= minimum if strict else value < minimum)):
        raise ValueError(f'Invalid numeric setting: {name}')


def load_config(path):
    cfg, sources = _read_preset(path)
    required = {'schema_version', 'name', 'subjects', 'seeds', 'data', 'split', 'models',
                'model_options', 'training', 'tensor', 'mask_repeats', 'bootstrap_repeats', 'output_dir'}
    if set(cfg) != required or cfg['schema_version'] != 3:
        raise ValueError('Use the four-model schema in configs/study.json; older studies need a separate output directory.')
    if (not cfg['subjects'] or len(set(cfg['subjects'])) != len(cfg['subjects'])
            or any(type(s) is not int or not 1 <= s <= 9 for s in cfg['subjects'])):
        raise ValueError('subjects must be unique integers in 1..9')
    if (not cfg['seeds'] or len(set(cfg['seeds'])) != len(cfg['seeds'])
            or any(type(s) is not int or not 0 <= s < 2**31 for s in cfg['seeds'])):
        raise ValueError('seeds must be unique nonnegative integers')
    if not cfg['models'] or len(set(cfg['models'])) != len(cfg['models']) or not set(cfg['models']) <= set(MODEL_KEYS):
        raise ValueError(f'models must be a nonempty unique selection from {MODEL_KEYS}')
    if set(cfg['model_options']) != {'eegnet', 'signal_transformer'}:
        raise ValueError('model_options must describe the two backbones, shared by their tensor variants')
    data = cfg['data']
    if (data['filter_scope'] != 'window' or data['sessions'] != ['T']
            or data['normalization'] != 'train_channel'):
        raise ValueError('This protocol uses window-local filtering and train-only channel normalization of T sessions')
    _positive(data['window'], 'data.window')
    _positive(data['filter_window_samples'], 'data.filter_window_samples')
    if data['filter_window_samples'] < 128 or data['window'] % data['filter_window_samples'] or n_windows(cfg) < 3:
        raise ValueError('Trials must contain at least three availability windows of at least 128 samples')
    if data['artifact_policy'] not in ('include', 'exclude'):
        raise ValueError('Invalid artifact policy')
    _number(data['offset_seconds'], 'data.offset_seconds')
    if not 0 < data['lowcut'] < data['highcut'] < 125:
        raise ValueError('EEG bandpass must lie below the 250 Hz Nyquist frequency')
    if cfg['split'] != {'protocol': 'stratified', 'train': 0.6, 'validation': 0.2, 'test': 0.2}:
        raise ValueError('Change protocol version explicitly before changing the split design')
    for key in ('mask_repeats', 'bootstrap_repeats'):
        _positive(cfg[key], key)
    tensor = cfg['tensor']
    for key in ('rank_channels', 'rank_features', 'fit_epochs', 'fit_batch_size'):
        _positive(tensor[key], f'tensor.{key}')
    if tensor['rank_channels'] > 22 or tensor['rank_features'] > data['filter_window_samples']:
        raise ValueError('Tensor ranks exceed the signal dimensions')
    for key in ('ridge', 'fit_lr'):
        _number(tensor[key], f'tensor.{key}', strict=True)
    options = cfg['training']
    for key in ('epochs', 'batch_size', 'minimum_epochs'):
        _positive(options[key], f'training.{key}')
    for key in ('warmup_epochs', 'patience'):
        _positive(options[key], f'training.{key}', zero=True)
    if options['warmup_epochs'] >= options['epochs'] or options['minimum_epochs'] > options['epochs']:
        raise ValueError('Warmup/minimum epochs must fit within the training budget')
    if options['loss'] not in ('cross_entropy', 'focal') or type(options['class_weights']) is not bool:
        raise ValueError('Invalid supervised loss settings')
    for key in ('learning_rate', 'focal_gamma'):
        _number(options[key], f'training.{key}', strict=True)
    _number(options['weight_decay'], 'training.weight_decay')
    if options['gradient_clip'] is not None:
        _number(options['gradient_clip'], 'training.gradient_clip', strict=True)
    cfg = deepcopy(cfg)
    # Paths resolve from the project launch directory, including inherited presets.
    cfg['data']['data_dir'] = str(Path(data['data_dir']).expanduser().resolve())
    if data['labels_dir'] is not None:
        cfg['data']['labels_dir'] = str(Path(data['labels_dir']).expanduser().resolve())
    cfg['output_dir'] = str(Path(cfg['output_dir']).expanduser().resolve())
    cfg['config_sources'] = sources
    return cfg


def n_windows(cfg):
    return cfg['data']['window'] // cfg['data']['filter_window_samples']


def tasks(cfg):
    return [(subject, seed) for subject in cfg['subjects'] for seed in cfg['seeds']]


def models(cfg):
    return [model_definition(key) for key in cfg['models']]


def task_name(subject, seed):
    return f'A{subject:02d}_seed_{seed}'
