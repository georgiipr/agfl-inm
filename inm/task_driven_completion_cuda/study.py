"""Isolated task execution, original historical replay gates and strict artifacts."""
from __future__ import annotations

import json
import copy
from pathlib import Path
import numpy as np

from inm.completion_transformer.study import _atomic_json, _atomic_npz, _historical_hashes
from .protocol import (STRATEGY_IDS, LEARNED_STRATEGY_IDS, digest, file_sha256,
                       study_identity, tasks, validate_output_path)

from .execution import DEVICE, scoped_cuda_rng, execution_metadata

SCHEMA = 'agfl-task-driven-completion-cuda-task-v1'


def _json(path):
    from .protocol import _unique_object, _reject_constant
    return json.loads(Path(path).read_text(), object_pairs_hook=_unique_object, parse_constant=_reject_constant)


def initialize_output(cfg):
    output = validate_output_path(cfg)
    identity = study_identity(cfg)
    marker = output / 'study-identity.json'
    if marker.exists():
        if _json(marker) != identity:
            raise ValueError('Changed source/config/packages/history identity; use a fresh output directory')
        if any(p.name not in {'study-identity.json', 'tasks', 'report'} for p in output.iterdir()):
            raise ValueError('Foreign files in study output')
        task_root = output / 'tasks'
        allowed_tasks = {f"A{r['subject']:02d}_seed_{r['seed']}" for r in tasks(cfg)}
        if task_root.exists() and (not task_root.is_dir() or any(
                not p.is_dir() or p.name not in allowed_tasks for p in task_root.iterdir())):
            raise ValueError('Foreign task path in study output')
    else:
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError('Foreign or partial output without study identity')
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(marker, identity)
    return identity


def _verify_complete_task(path, identity, *, synthetic=None):
    path = Path(path)
    if not (path / 'task.json').is_file():
        raise ValueError('Partial task: task.json missing; retry forbidden')
    task = _json(path / 'task.json')
    if (task.get('schema_name') != SCHEMA or task.get('status') != 'complete' or
            task.get('study_id') != identity['study_id'] or
            task.get('execution_source_sha256') != identity['source_sha256']):
        raise ValueError('Failed, partial or foreign task; retry forbidden')
    if synthetic is not None and task.get('synthetic') is not synthetic:
        raise ValueError('Synthetic status mismatch')
    hashes = task.get('artifacts')
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError('Missing artifact hashes')
    actual = {p.relative_to(path).as_posix() for p in path.rglob('*') if p.is_file() and p != path / 'task.json'}
    if actual != set(hashes):
        raise ValueError('Unhashed or missing task artifacts')
    for name, expected in hashes.items():
        rel = Path(name)
        target = path / rel
        if rel.is_absolute() or '..' in rel.parts or target.is_symlink() or not target.resolve().is_relative_to(path.resolve()):
            raise ValueError('Unsafe artifact path')
        if file_sha256(target) != expected:
            raise ValueError(f'Artifact checksum mismatch: {name}')
    from .reporting import verify_task_artifacts
    verify_task_artifacts(path, task)
    return task


def historical_hashes(cfg, record, subject):
    hashes = _historical_hashes(record)
    # Original recordings are immutable inputs too, even on completed-task resume.
    raw = Path(cfg['data_dir']) / f'A{subject:02d}T.gdf'
    hashes[str(raw)] = file_sha256(raw)
    return hashes


def synthetic_context(subject=1, seed=0):
    """Small, explicitly synthetic public smoke; all fixed strategies and masks."""
    from types import SimpleNamespace
    from inm.encoder_candidates.transformer import SpatialTransformer
    from .completion import scoped_cpu_rng
    with scoped_cpu_rng(55801 + seed):
        rng = np.random.default_rng(55801 + seed)
        def partition(name):
            return SimpleNamespace(raw=rng.normal(size=(4,22,4,250)).astype(np.float32),
                labels=np.arange(4, dtype=np.int64), sample_ids=[f'synthetic:{name}:{i}' for i in range(4)])
        prepared = SimpleNamespace(train=partition('train'), validation=partition('validation'),
            data_id='synthetic', split_id='synthetic')
        model = SpatialTransformer(seed=subject * 100 + seed).eval().requires_grad_(False)
    return {'prepared': prepared, 'models': {'spatial_transformer': model},
            'budget': {'maximum_epochs': 1, 'minimum_epochs': 1, 'patience': 1}}


def _context_digest(context):
    import hashlib
    h = hashlib.sha256()
    for name in ('train', 'validation'):
        part = getattr(context['prepared'], name)
        for arr in (part.raw, part.labels, np.asarray(part.sample_ids, dtype=np.str_)):
            arr = np.asarray(arr)
            h.update(str((arr.shape, arr.dtype.str)).encode()); h.update(arr.tobytes())
    for key, value in context['models']['spatial_transformer'].state_dict().items():
        h.update(key.encode()); h.update(value.detach().cpu().numpy().tobytes())
    h.update(digest(context.get('budget')).encode())
    return h.hexdigest()


def run_task(cfg, task_index, *, synthetic_context=None, strategy_order=None):
    with scoped_cuda_rng(920001):
        return _run_task(cfg, task_index, synthetic_context=synthetic_context, strategy_order=strategy_order)


def _run_task(cfg, task_index, *, synthetic_context=None, strategy_order=None):
    planned = tasks(cfg)
    if type(task_index) is not int or not 0 <= task_index < len(planned):
        raise ValueError('Invalid task index')
    synthetic = cfg['synthetic']
    if synthetic != (synthetic_context is not None):
        raise ValueError('Synthetic context is mandatory only for synthetic tasks')
    order = tuple(strategy_order or STRATEGY_IDS)
    if len(order) != len(STRATEGY_IDS) or set(order) != set(STRATEGY_IDS):
        raise ValueError('Every fixed strategy must run exactly once')
    if not synthetic and strategy_order is not None:
        raise ValueError('Real strategy order is fixed')
    identity = initialize_output(cfg)
    spec = planned[task_index]
    subject, seed = spec['subject'], spec['seed']
    path = Path(cfg['output_dir']) / 'tasks' / f'A{subject:02d}_seed_{seed}'
    if path.exists():
        prior = _verify_complete_task(path, identity, synthetic=synthetic)
        if prior.get('execution') != execution_metadata():
            raise ValueError('Changed CUDA hardware/runtime execution identity')
        if (prior['subject'], prior['seed']) != (subject, seed):
            raise ValueError('Task identity differs from planned subject/seed')
        if synthetic:
            if prior['synthetic_context_sha256'] != _context_digest(synthetic_context):
                raise ValueError('Changed synthetic data/backbone/budget')
        else:
            from inm.completion_transformer.replay import verify_historical_task
            record = verify_historical_task(cfg, subject, seed)
            if prior['historical_input_sha256'] != historical_hashes(cfg, record, subject):
                raise ValueError('Changed historical input hashes')
        return prior
    path.mkdir(parents=True, exist_ok=False)
    status = {'schema_name': SCHEMA, 'status': 'running', 'study_id': identity['study_id'],
        'subject': subject, 'seed': seed, 'synthetic': synthetic, 'cells': [],
        'execution': execution_metadata(), 'execution_source_sha256': identity['source_sha256']}
    _atomic_json(path / 'task.json', status)
    try:
        import torch
        from .completion import (fit_initialization, paired_completers, completion_state_arrays,
                                 restore_completer, scoped_cpu_rng)
        from .adapters import CompletionAdapter
        from .training import (fit_completion, evaluate, validation_bank, validate_partition,
                               training_settings)
        with scoped_cpu_rng(seed + 920001):
            if synthetic:
                prepared = synthetic_context['prepared']
                models = synthetic_context['models']
                if set(models) != {'spatial_transformer'}:
                    raise ValueError('Synthetic context requires only the Transformer')
                budget = synthetic_context.get('budget')
                replay = {'synthetic_untrained_backbone': True}
                status['synthetic_context_sha256'] = _context_digest(synthetic_context)
            else:
                from inm.completion_transformer.replay import prepare_task, replay_full_input, restore_backbones
                prepared, historical = prepare_task(cfg, subject, seed, path / 'prepared')
                # This verifies BOTH original prediction sets BEFORE any calibration/fitting.
                replay = replay_full_input(historical, prepared)
                models = restore_backbones(historical)
                budget = None
                status['historical_input_sha256'] = historical_hashes(cfg, historical, subject)
            settings = training_settings(synthetic=synthetic, budget=budget)
            train, validation = prepared.train, prepared.validation
            tx, ty, tids = validate_partition(train.raw, train.labels, train.sample_ids)
            vx, vy, vids = validate_partition(validation.raw, validation.labels, validation.sample_ids)
            if set(tids) & set(vids):
                raise ValueError('Training and validation sample IDs overlap')
            backbone = copy.deepcopy(models['spatial_transformer']).cpu().eval().requires_grad_(False)
            backbone_before = {k: v.detach().clone() for k,v in backbone.state_dict().items()}
            _atomic_npz(path / 'backbone.npz', **{k:v.numpy() for k,v in backbone_before.items()})
            # Compare unchanged full-input CPU and CUDA execution BEFORE calibration.
            from inm.task_driven_completion.training import predict as cpu_predict
            full_mask = np.ones((len(vx), 22, 4), dtype=np.bool_)
            cpu_full = cpu_predict(CompletionAdapter(backbone), vx, full_mask)
            backbone.to(DEVICE)
            from .training import predict
            gpu_full = predict(CompletionAdapter(backbone), vx, full_mask)
            if not np.allclose(cpu_full, gpu_full, rtol=1e-4, atol=1e-6):
                raise RuntimeError('CUDA full-input probabilities disagree with CPU prefit reference')
            replay['cuda_full_input_vs_cpu'] = dict(passed=True, rtol=1e-4, atol=1e-6,
                max_abs_probability_error=float(np.max(np.abs(cpu_full-gpu_full))))
            _atomic_npz(path / 'full-input-reference.npz', cpu_probabilities=cpu_full,
                        cuda_probabilities=gpu_full, labels=vy.numpy(), sample_ids=np.asarray(vids))
            initialization, factor_history = fit_initialization(tx, seed)
            _atomic_npz(path / 'initialization.npz', **{k:v.numpy() for k,v in initialization.items()})
            _atomic_json(path / 'factor-history.json', factor_history)
            completers = {k: v.to(DEVICE) for k,v in paired_completers(initialization).items()}
            banks = validation_bank(vids, subject, seed)
            full_reference = evaluate(CompletionAdapter(backbone), vx, vy.numpy(), banks[:1])[0]['probabilities']
            rows, fits = [], {}
            for strategy in order:
                completer = completers[strategy]
                if strategy in LEARNED_STRATEGY_IDS:
                    fit = fit_completion(backbone, strategy, completer, train, validation,
                        subject=subject, seed=seed, synthetic=synthetic, budget=budget)
                    _atomic_npz(path / f'{strategy}/training-schedule.npz',
                        masks=fit['training_masks'], orders=fit['training_orders'],
                        sample_ids=np.asarray(tids), labels=ty.numpy())
                else:
                    fit = {'initial': completion_state_arrays(completer),
                        'selected': completion_state_arrays(completer), 'history': [], 'selected_epoch': None}
                for state in ('initial', 'selected'):
                    _atomic_npz(path / f'{strategy}/{state}.npz', **fit[state])
                _atomic_json(path / f'{strategy}/history.json', fit['history'])
                fits[strategy] = {'selected_epoch': fit['selected_epoch'], 'epochs': max(0, len(fit['history'])-1)}
                evaluated = evaluate(CompletionAdapter(backbone, strategy, completer), vx, vy.numpy(), banks)
                with np.load(path / f'{strategy}/selected.npz', allow_pickle=False) as saved:
                    restored = restore_completer(strategy, dict(saved)).to(DEVICE)
                reproduced = evaluate(CompletionAdapter(backbone, strategy, restored), vx, vy.numpy(), banks)
                for row, check in zip(evaluated, reproduced):
                    if not np.array_equal(row['probabilities'], check['probabilities']):
                        raise RuntimeError('Selected numeric state does not exactly replay')
                    if row['condition'] == 'full_22' and not np.array_equal(row['probabilities'], full_reference):
                        raise RuntimeError('Full input changed across completion strategies')
                    name = f"cells/{strategy}__{row['condition']}__r{row['repeat']}.npz"
                    _atomic_npz(path / name, probabilities=row['probabilities'], labels=vy.numpy(),
                        sample_ids=np.asarray(vids), mask=row['mask'])
                    rows.append({k:v for k,v in row.items() if k not in {'probabilities', 'mask'}} |
                        {'strategy':strategy, 'backbone':'spatial_transformer', 'artifact':name})
            if any(not torch.equal(v.detach().cpu(), backbone_before[k]) for k,v in backbone.state_dict().items()):
                raise RuntimeError('Backbone state changed')
            status.update(status='complete', replay=replay, data_id=prepared.data_id, split_id=prepared.split_id,
                validation_sample_ids=list(vids), validation_labels=vy.tolist(), train_sample_ids=list(tids),
                train_labels=ty.tolist(), cells=rows, fits=fits, training_settings=settings,
                synthetic_budget=budget, factor_epochs=len(factor_history), backbone_fits=0,
                supervised_completion_fits=2, exact_selected_state_replay=True, backbone_unchanged=True)
            status['artifacts'] = {p.relative_to(path).as_posix():file_sha256(p)
                for p in sorted(path.rglob('*')) if p.is_file() and p != path / 'task.json'}
            _atomic_json(path / 'task.json', status)
            return _verify_complete_task(path, identity, synthetic=synthetic)
    except BaseException as exc:
        status.update(status='failed', failure_type=type(exc).__name__, failure=str(exc))
        _atomic_json(path / 'task.json', status)
        raise
