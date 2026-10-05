"""Read-only selected-state replay, independent of the saved replay assertion.

No calibration or optimization occurs here. Real audits prepare only historical
train/validation inputs in disposable external staging and verify the original
full-input replay before evaluating the saved completion states.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

from .execution import DEVICE, scoped_cuda_rng, execution_metadata
from .protocol import CONDITIONS, STRATEGY_IDS, load_config, study_identity, tasks, validate_output_path


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def prediction_metrics(labels, probabilities):
    """Independent four-class recall mean and clipped categorical log loss."""
    y, p = np.asarray(labels), np.asarray(probabilities, dtype=np.float64)
    _require(y.ndim == 1 and y.dtype.kind in 'iu' and len(y) > 0 and
             set(y.tolist()) == {0, 1, 2, 3}, 'Audit requires all four classes')
    _require(p.shape == (len(y), 4) and np.isfinite(p).all() and
             np.all((p >= 0) & (p <= 1)) and np.allclose(p.sum(1), 1, atol=1e-6),
             'Audit probabilities are malformed')
    predicted = np.argmax(p, axis=1)
    recalls = [float(np.mean(predicted[y == label] == label)) for label in range(4)]
    return {'balanced_accuracy': sum(recalls) / 4,
            'log_loss': float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())}


def task_contrasts(cells):
    """Task-level degraded BA contrasts; no cohort or synthetic promotion."""
    from .reporting import CONTRASTS
    expected = {(s, c['name'], r) for s in STRATEGY_IDS for c in CONDITIONS
                for r in range(c['repeats'])}
    keys = [(r['strategy'], r['condition'], r['repeat']) for r in cells]
    _require(len(keys) == len(expected) and set(keys) == expected, 'Audit incomplete cell grid')
    averages = {}
    for strategy in STRATEGY_IDS:
        condition_means = []
        for condition in CONDITIONS[1:]:
            scores = [r['balanced_accuracy'] for r in cells
                      if r['strategy'] == strategy and r['condition'] == condition['name']]
            condition_means.append(sum(scores) / len(scores))
        averages[strategy] = sum(condition_means) / len(condition_means)
    return [{'contrast': name, 'gain': averages[candidate] - averages[control]}
            for name, candidate, control in CONTRASTS]


def _replay(path, task, prepared, supplied_backbone):
    import torch
    from .adapters import CompletionAdapter
    from .completion import restore_completer
    from .training import validate_partition, validation_bank

    for partition in ('train', 'validation'):
        part = getattr(prepared, partition)
        _, y, ids = validate_partition(part.raw, part.labels, part.sample_ids)
        _require(list(ids) == task[f'{partition}_sample_ids'] and
                 y.tolist() == task[f'{partition}_labels'], 'Audit partition IDs/labels differ')
    _require(prepared.data_id == task['data_id'] and prepared.split_id == task['split_id'],
             'Audit data/split identity differs')
    # A valid shape and a rehashed snapshot cannot substitute for the original.
    source_state = supplied_backbone.state_dict()
    with np.load(path / 'backbone.npz', allow_pickle=False) as saved:
        _require(set(saved.files) == set(source_state) and all(
            np.array_equal(saved[k], source_state[k].detach().cpu().numpy()) for k in saved.files),
            'Audit backbone snapshot differs from immutable source backbone')
        saved_state = {k: torch.from_numpy(saved[k].copy()) for k in saved.files}
    backbone = copy.deepcopy(supplied_backbone)
    backbone.load_state_dict(saved_state, strict=True)
    backbone.cpu().eval().requires_grad_(False)
    from inm.task_driven_completion.training import predict as cpu_predict
    cpu_x, _, _ = validate_partition(prepared.validation.raw, prepared.validation.labels,
                                     prepared.validation.sample_ids)
    cpu_full = cpu_predict(CompletionAdapter(backbone), cpu_x,
                           np.ones((len(cpu_x),22,4),dtype=np.bool_))
    with np.load(path / 'full-input-reference.npz', allow_pickle=False) as saved:
        _require(np.array_equal(cpu_full, saved['cpu_probabilities']),
                 'Audit CPU prefit reference differs from source')
    backbone.to(DEVICE).eval().requires_grad_(False)
    x, y, ids = validate_partition(prepared.validation.raw, prepared.validation.labels,
                                   prepared.validation.sample_ids)
    x = x.to(DEVICE)
    banks = validation_bank(ids, task['subject'], task['seed'])
    recorded = {(r['strategy'], r['condition'], r['repeat']): r for r in task['cells']}
    recomputed = []
    for strategy in STRATEGY_IDS:
        with np.load(path / strategy / 'selected.npz', allow_pickle=False) as saved:
            completer = restore_completer(strategy, dict(saved)).to(DEVICE)
        adapter = CompletionAdapter(backbone, strategy, completer).eval()
        for condition, repeat, mask in banks:
            chunks = []
            # Match the fixed evaluation batch size, avoiding shared predict/evaluate code.
            with torch.no_grad():
                for start in range(0, len(x), 32):
                    logits = adapter(x[start:start + 32], torch.from_numpy(mask[start:start + 32]).to(DEVICE))
                    chunks.append(torch.softmax(logits, dim=-1).cpu().numpy())
            probabilities = np.concatenate(chunks)
            if condition == 'full_22':
                _require(np.allclose(cpu_full, probabilities, rtol=1e-4, atol=1e-6),
                         'Audit CUDA full-input probabilities disagree with CPU')
            row = recorded[strategy, condition, repeat]
            with np.load(path / row['artifact'], allow_pickle=False) as saved:
                _require(np.array_equal(probabilities, saved['probabilities']),
                         f'Audit selected-state predictions differ: {strategy}/{condition}/{repeat}')
            metrics = prediction_metrics(y.numpy(), probabilities)
            _require(all(np.isclose(metrics[k], row[k], rtol=1e-12, atol=1e-12) for k in metrics),
                     'Audit independently recomputed metrics differ')
            recomputed.append(dict(strategy=strategy, condition=condition, repeat=repeat, **metrics))
    _require(all(torch.equal(v.cpu(), saved_state[k]) for k, v in backbone.state_dict().items()),
             'Audit backbone changed during replay')
    actual_contrasts, recorded_contrasts = task_contrasts(recomputed), task_contrasts(task['cells'])
    _require(all(np.isclose(a['gain'], b['gain'], rtol=1e-12, atol=1e-12)
                 for a, b in zip(actual_contrasts, recorded_contrasts)), 'Audit contrasts differ')
    return dict(cells=recomputed, task_contrasts=actual_contrasts,
                replayed_cells=len(recomputed), backbone_matches_source=True,
                selected_states_loaded=True, exact_prediction_replay=True,
                independently_recomputed_metrics=True, independently_recomputed_contrasts=True,
                cuda_full_input_vs_cpu=dict(passed=True, rtol=1e-4, atol=1e-6))


def audit_task(cfg, task_index, *, synthetic_context=None):
    """Audit a complete task without modifying its output or fitting anything.

    A supplied synthetic context must reproduce its saved digest. When omitted,
    synthetic audit reconstructs the public smoke fixture. Real inputs always
    come from the strict historical readers, never caller-supplied replacements.
    """
    from .completion import scoped_cpu_rng
    from .study import _json, _verify_complete_task, _context_digest, historical_hashes
    planned = tasks(cfg)
    _require(type(task_index) is int and 0 <= task_index < len(planned), 'Invalid audit task index')
    output, identity = validate_output_path(cfg), study_identity(cfg)
    _require((output / 'study-identity.json').is_file() and
             _json(output / 'study-identity.json') == identity, 'Audit current identity differs from output')
    spec = planned[task_index]
    path = output / 'tasks' / f"A{spec['subject']:02d}_seed_{spec['seed']}"
    task = _verify_complete_task(path, identity, synthetic=cfg['synthetic'])
    _require((task['subject'], task['seed']) == (spec['subject'], spec['seed']), 'Audit task identity differs')
    with scoped_cuda_rng(spec['seed'] + 930001):
        _require(task.get('execution') == execution_metadata(), 'Audit CUDA hardware/runtime differs')
        if cfg['synthetic']:
            if synthetic_context is None:
                from .study import synthetic_context as smoke_context
                synthetic_context = smoke_context(spec['subject'], spec['seed'])
            _require(task['synthetic_context_sha256'] == _context_digest(synthetic_context),
                     'Audit synthetic context differs')
            replay = _replay(path, task, synthetic_context['prepared'],
                             synthetic_context['models']['spatial_transformer'])
        else:
            _require(synthetic_context is None, 'Real audit refuses supplied synthetic context')
            from inm.completion_transformer.replay import prepare_task, replay_full_input, restore_backbones
            # Historical reader requires staging under its configured output root.
            # Redirect only that reader to an external temporary directory; study
            # identity and completed-task output continue to use the original cfg.
            with tempfile.TemporaryDirectory(prefix='agfl-completion-audit-') as staging:
                reader_cfg = dict(cfg, output_dir=staging)
                prepared, record = prepare_task(reader_cfg, spec['subject'], spec['seed'],
                                                Path(staging) / 'prepared')
                _require(task['historical_input_sha256'] == historical_hashes(cfg, record, spec['subject']),
                         'Audit historical input hashes differ')
                original_replay = replay_full_input(record, prepared)
                replay = _replay(path, task, prepared, restore_backbones(record)['spatial_transformer'])
                replay['original_full_input_replay'] = original_replay
    return dict(schema_name='agfl-task-driven-completion-cuda-audit-v1', status='passed',
                study_id=identity['study_id'], subject=spec['subject'], seed=spec['seed'],
                synthetic=cfg['synthetic'], device=DEVICE, calibration_device='cpu', threads=1, cuda_verified=True,
                new_fits=0, cohort_estimates=None, **replay)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m inm.task_driven_completion_cuda.audit')
    parser.add_argument('--config', required=True)
    parser.add_argument('--task-index', type=int, default=0)
    parser.add_argument('--synthetic-smoke', action='store_true')
    parser.add_argument('--output-dir')
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config, allow_existing_output=True)
        if args.synthetic_smoke:
            _require(args.output_dir is not None and args.task_index == 0,
                     'Synthetic smoke audit requires --output-dir and task index 0')
            cfg.update(subjects=[1], seeds=[0], synthetic=True,
                       output_dir=str(Path(args.output_dir).expanduser().resolve()))
        else:
            _require(args.output_dir is None, '--output-dir requires --synthetic-smoke')
        print(json.dumps(audit_task(cfg, args.task_index), indent=2, sort_keys=True, allow_nan=False))
        return 0
    except Exception as exc:
        print(f'task-driven-completion-audit: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
