"""Declared experiment matrix, atomic records, and source/configuration identity."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

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
    import platform
    files = sorted([*ROOT.glob('*.py'), *ROOT.glob('*.sbatch'),
                    *(ROOT / 'agfl').rglob('*.py'), *(ROOT / 'inm').rglob('*.py')])
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    versions = {name: importlib.metadata.version(name)
                for name in ('torch', 'numpy', 'scipy', 'scikit-learn', 'mne', 'tqdm')}
    return {'source_sha256': digest(hashes), 'files_sha256': hashes,
            'packages': versions, 'python': platform.python_version()}


def load_config(path):
    cfg = read_json(path)
    required = {'schema_version', 'name', 'subjects', 'seeds', 'data', 'split', 'encoder',
                'classifier', 'training', 'tensor', 'attentions', 'representations',
                'mha_controls', 'regimes', 'mask_repeats', 'bootstrap_repeats', 'output_dir'}
    if set(cfg) != required or cfg['schema_version'] != 2:
        raise ValueError('Unsupported study configuration; use configs/study.json as the schema.')
    if not cfg['subjects'] or len(set(cfg['subjects'])) != len(cfg['subjects']) or any(type(s) is not int or not 1 <= s <= 9 for s in cfg['subjects']):
        raise ValueError('subjects must be unique integers in 1..9')
    if not cfg['seeds'] or len(set(cfg['seeds'])) != len(cfg['seeds']) or any(type(s) is not int or not 0 <= s < 2**31 for s in cfg['seeds']):
        raise ValueError('seeds must be unique nonnegative integers')
    if (set(cfg['attentions']) != {'mha', 'performer'}
            or cfg['representations'] != ['baseline', 'tensor_core']
            or cfg['mha_controls'] != ['tensor_completion', 'linear_core', 'mean_completion']
            or cfg['regimes'] != ['full', 'mixed']):
        raise ValueError('The declared experiment requires MHA/Performer, paired routes, MHA controls, and both regimes.')
    if (cfg['data']['filter_scope'] != 'window' or cfg['data']['sessions'] != ['T']
            or cfg['data']['normalization'] != 'train_channel'):
        raise ValueError('This version uses independent-window filtering of T-session trials.')
    if cfg['encoder']['windows'] < 3 or cfg['encoder']['window_samples'] < 128:
        raise ValueError('Use at least three windows, each of at least 128 samples.')
    if cfg['data']['window'] != cfg['encoder']['windows'] * cfg['encoder']['window_samples']:
        raise ValueError('Trial size must equal windows * window_samples')
    if cfg['data']['filter_window_samples'] != cfg['encoder']['window_samples']:
        raise ValueError('Filter and EEGNet windows must coincide exactly.')
    if cfg['encoder']['channels'] != 22 or cfg['classifier']['num_classes'] != 4:
        raise ValueError('This study requires the 22 EEG channels and four classes of BCI2a.')
    if cfg['split'] != {'protocol': 'stratified', 'train': 0.6, 'validation': 0.2, 'test': 0.2}:
        raise ValueError('Change protocol version explicitly before changing the split design.')
    for key in ('mask_repeats', 'bootstrap_repeats'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f'{key} must be positive')
    if cfg['tensor']['ridge'] <= 0:
        raise ValueError('Tensor core ridge penalty must be positive.')
    for stage in ('encoder', 'classifier'):
        options = cfg['training'][stage]
        for key in ('epochs', 'batch_size', 'minimum_epochs'):
            if type(options[key]) is not int or options[key] < 1:
                raise ValueError(f'{stage}.{key} must be a positive integer')
        if not 0 <= options['warmup_epochs'] < options['epochs'] or options['minimum_epochs'] > options['epochs']:
            raise ValueError('Warmup/minimum epochs must fit within the training budget.')
        if options['loss'] not in ('cross_entropy', 'focal') or options['learning_rate'] <= 0 or options['patience'] < 0:
            raise ValueError('Invalid supervised training options.')
    cfg['data']['data_dir'] = str(Path(cfg['data']['data_dir']).expanduser().resolve())
    cfg['output_dir'] = str(Path(cfg['output_dir']).expanduser().resolve())
    return cfg


def tasks(cfg):
    return [(subject, seed) for subject in cfg['subjects'] for seed in cfg['seeds']]


def arms(cfg):
    return [{'attention': attention, 'representation': representation, 'regime': regime,
             'name': f'{attention}__{representation}__{regime}'}
            for attention in cfg['attentions']
            for representation in (cfg['representations'] + (cfg['mha_controls'] if attention == 'mha' else []))
            for regime in cfg['regimes']]


def task_name(subject, seed):
    return f'A{subject:02d}_seed_{seed}'
