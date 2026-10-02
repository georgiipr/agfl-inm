"""Entry point for the reduced MHA comparison in legacy frozen features."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .protocol import load_config as load_legacy_config
from .protocol import tasks


def load_config(path):
    """Load the follow-up declaration and adapt the unchanged legacy runtime."""
    path = Path(path).expanduser().resolve()
    declaration = json.loads(path.read_text())
    if declaration.get('schema_name') != 'agfl-legacy-tensor-followup-v1':
        raise ValueError('Expected agfl-legacy-tensor-followup-v1 configuration')
    base_path = (path.parent / declaration['base_config']).resolve()
    cfg = load_legacy_config(base_path)
    for key in ('subjects', 'seeds'):
        values = declaration[key]
        if (not values or len(set(values)) != len(values)
                or any(type(value) is not int or
                       (not 1 <= value <= 9 if key == 'subjects' else value < 0)
                       for value in values)):
            raise ValueError(f'{key} must be a nonempty list of unique valid integers')
    allowed = {'baseline', 'tensor_completion', 'tensor_core', 'mean_completion', 'linear_core'}
    initial = declaration['representations']
    controls = declaration['available_controls'] if declaration.get('include_controls', False) else []
    if not initial or len(set(initial)) != len(initial) or set(initial) - allowed:
        raise ValueError('representations must be unique supported legacy feature routes')
    if len(set(controls)) != len(controls) or set(controls) - allowed or set(initial) & set(controls):
        raise ValueError('controls must be unique routes separate from representations')
    if declaration['attention'] != 'mha' or declaration['regimes'] != ['full', 'mixed']:
        raise ValueError('The tensor follow-up declares MHA and full/mixed regimes only')
    expected_patterns = {'full_22', 'random_static_16', 'dynamic_random_16',
                         'random_static_6', 'dynamic_random_6'}
    selection = declaration['selection']
    if (selection.get('policy') != 'validation_robust_balanced_accuracy_then_log_loss'
            or set(selection.get('patterns', [])) != expected_patterns):
        raise ValueError('Use the declared full plus held-out random validation mask patterns')
    tensor = declaration['tensor']
    for key in ('rank_channels', 'rank_features', 'ridge'):
        if key not in tensor:
            raise ValueError(f'tensor.{key} must remain explicit')
    if tensor['ridge'] <= 0 or min(tensor['rank_channels'], tensor['rank_features']) < 1:
        raise ValueError('Tucker rank and ridge must be positive')
    candidates = tensor.get('candidate_settings', [])
    for candidate in candidates:
        if min(candidate['rank_channels'], candidate['rank_features']) < 1 or candidate['ridge'] <= 0:
            raise ValueError('Candidate ranks and ridge values must be positive')
    cfg['attentions'] = {'mha': cfg['attentions']['mha']}
    cfg['subjects'] = list(declaration['subjects'])
    cfg['seeds'] = list(declaration['seeds'])
    cfg['representations'] = list(initial)
    cfg['mha_controls'] = list(controls)
    cfg['regimes'] = list(declaration['regimes'])
    cfg['tensor'].update({key: tensor[key] for key in ('rank_channels', 'rank_features', 'ridge')})
    cfg['output_dir'] = str((path.parent / declaration['output_dir']).resolve())
    cfg['tensor_followup'] = {
        'schema_name': declaration['schema_name'], 'name': declaration['name'],
        'selection': declaration['selection'], 'candidate_settings': candidates,
        'available_controls': declaration['available_controls'],
        'controls_output_dir': str((path.parent / declaration['controls_output_dir']).resolve()),
        'matrix': {'attention': 'mha', 'representations': initial + controls,
                   'regimes': declaration['regimes']},
    }
    return cfg


def matrix(cfg):
    from .protocol import arms
    return arms(cfg)


def paired_identity(feature_sha256, calibration_sha256, mask_sha256):
    """Identity tuple required to compare two representation rows fairly."""
    return {'feature_sha256': feature_sha256, 'calibration_sha256': calibration_sha256,
            'mask_sha256': mask_sha256}


def feature_digest(features):
    value = features.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def run_task(cfg, task_index, device='cuda'):
    from .study import run_task as run_legacy_task
    patterns = cfg['tensor_followup']['selection']['patterns']
    return run_legacy_task(cfg, task_index, device, validation_mask_patterns=patterns)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Reduced legacy MHA tensor follow-up')
    parser.add_argument('--config', default='configs/tensor-followup.json')
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument('--plan', action='store_true')
    operation.add_argument('--task-index', type=int)
    operation.add_argument('--summarize-only', action='store_true')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--include-controls', action='store_true',
                        help='Add mean_completion and linear_core to this run identity')
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.include_controls:
        controls = cfg['tensor_followup']['available_controls']
        cfg['mha_controls'] = list(controls)
        cfg['tensor_followup']['matrix']['representations'] += list(controls)
        cfg['output_dir'] = cfg['tensor_followup']['controls_output_dir']
    if args.plan:
        work = tasks(cfg)
        count = len(matrix(cfg))
        print(f"Tasks: {len(work)}; fits/task: {count}; total fits: {len(work) * count}")
        print(f"Output: {cfg['output_dir']}")
        print('Arms: ' + ', '.join(arm['name'] for arm in matrix(cfg)))
    elif args.summarize_only:
        from .study import summarize_existing
        result = summarize_existing(cfg)
        print(f"Completed fits: {result['completed_fits']}/{result['expected_fits']}; status={result['status']}")
    else:
        run_task(cfg, args.task_index, args.device)


if __name__ == '__main__':
    main()
