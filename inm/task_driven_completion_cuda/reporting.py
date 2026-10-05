"""Numeric artifact verification and participant-paired exploratory summaries."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from .protocol import CONDITIONS, STRATEGY_IDS, LEARNED_STRATEGY_IDS, study_identity, tasks, validate_output_path


def _require(value, message):
    if not value:
        raise ValueError(message)


def _close(a, b):
    return bool(np.isfinite(a) and np.isfinite(b) and np.isclose(a, b, rtol=1e-12, atol=1e-12))


def verify_task_artifacts(path, task):
    from inm.encoder_candidates.training import _metrics
    from inm.availability import mask_bank_digest
    from .completion import completion_state_arrays, paired_completers, restore_completer
    from .training import validation_bank, training_schedule, training_settings, selection_key
    from .study import _json
    import torch
    path = Path(path)
    _require(type(task.get('synthetic')) is bool, 'Missing synthetic flag')
    required = {'backbone.npz','initialization.npz','factor-history.json','full-input-reference.npz'}
    required.update(f'{s}/{name}' for s in STRATEGY_IDS for name in ('initial.npz','selected.npz','history.json'))
    required.update(f'{s}/training-schedule.npz' for s in LEARNED_STRATEGY_IDS)
    required.update(f"cells/{s}__{c['name']}__r{r}.npz" for s in STRATEGY_IDS
        for c in CONDITIONS for r in range(c['repeats']))
    declared = set(task.get('artifacts',{}))
    _require(required.issubset(declared) and all(name in required or
        (not task['synthetic'] and name.startswith('prepared/')) for name in declared),
        'Missing required or unknown numeric artifacts')
    execution = task.get('execution', {})
    _require(execution.get('device') == 'cuda:0' and execution.get('calibration_device') == 'cpu'
        and execution.get('tf32') is False and execution.get('amp') is False
        and execution.get('deterministic_algorithms') is True, 'Missing CUDA execution provenance')
    _require(task.get('replay', {}).get('cuda_full_input_vs_cpu', {}).get('passed') is True,
             'Missing CPU/CUDA full-input prefit gate')
    with np.load(path / 'full-input-reference.npz', allow_pickle=False) as saved:
        _require(set(saved.files) == {'cpu_probabilities','cuda_probabilities','labels','sample_ids'}
            and np.isfinite(saved['cpu_probabilities']).all()
            and np.isfinite(saved['cuda_probabilities']).all()
            and np.allclose(saved['cpu_probabilities'], saved['cuda_probabilities'], rtol=1e-4, atol=1e-6),
            'CPU/CUDA prefit reference disagrees')
        reference_full = saved['cuda_probabilities'].copy()
        _require(saved['sample_ids'].tolist() == task['validation_sample_ids']
            and saved['labels'].tolist() == task['validation_labels'], 'Prefit reference IDs differ')
    settings = training_settings(synthetic=task['synthetic'], budget=task.get('synthetic_budget'))
    _require(task.get('training_settings') == settings, 'Training settings drift')
    _require(task.get('backbone_fits') == 0 and task.get('supervised_completion_fits') == 2 and
        task.get('exact_selected_state_replay') is True and task.get('backbone_unchanged') is True,
        'Missing frozen-backbone/numeric replay checks')
    ids, tids = task.get('validation_sample_ids'), task.get('train_sample_ids')
    labels, tlabels = np.asarray(task.get('validation_labels')), np.asarray(task.get('train_labels'))
    for names, y in ((ids, labels), (tids, tlabels)):
        _require(isinstance(names, list) and bool(names) and all(isinstance(i,str) for i in names)
            and len(names) == len(set(names)) and y.shape == (len(names),) and y.dtype.kind in 'iu'
            and np.all((y >= 0) & (y < 4)), 'Invalid partition IDs/labels')
    _require(not set(ids) & set(tids), 'Partition IDs overlap')
    from inm.encoder_candidates.transformer import SpatialTransformer
    from .completion import scoped_cpu_rng
    with scoped_cpu_rng(0):
        backbone = SpatialTransformer(seed=0).state_dict()
    with np.load(path / 'backbone.npz',allow_pickle=False) as saved:
        _require(set(saved.files) == set(backbone), 'Malformed backbone snapshot')
        _require(all(saved[k].shape == tuple(backbone[k].shape) and saved[k].dtype == backbone[k].numpy().dtype
            and np.isfinite(saved[k]).all() for k in saved.files), 'Invalid backbone snapshot arrays')
    cells = task.get('cells', [])
    expected = {(s,c['name'],r) for s in STRATEGY_IDS for c in CONDITIONS for r in range(c['repeats'])}
    _require(len(cells) == len(expected) and {(r.get('strategy'),r.get('condition'),r.get('repeat')) for r in cells} == expected,
             'Incomplete/duplicate/unknown strategy-condition-repeat grid')
    masks = {(c,r):m for c,r,m in validation_bank(ids,task['subject'],task['seed'])}
    full = None
    for row in cells:
        strategy, condition, repeat = row['strategy'],row['condition'],row['repeat']
        name = f'cells/{strategy}__{condition}__r{repeat}.npz'
        _require(row.get('artifact') == name and row.get('backbone') == 'spatial_transformer', 'Unexpected cell artifact/backbone')
        with np.load(path / name, allow_pickle=False) as saved:
            _require(set(saved.files) == {'probabilities','labels','sample_ids','mask'}, 'Unexpected cell arrays')
            p, m = saved['probabilities'], saved['mask']
            _require(saved['sample_ids'].tolist() == ids and np.array_equal(saved['labels'],labels), 'Unpaired IDs/labels')
            _require(m.dtype == np.bool_ and np.array_equal(m,masks[condition,repeat]) and
                row['mask_sha256'] == mask_bank_digest(m), 'Unpaired or nonprotocol validation masks')
            _require(p.shape == (len(ids),4) and p.dtype.kind == 'f' and np.isfinite(p).all() and
                np.all((p >= 0) & (p <= 1)) and np.allclose(p.sum(1),1,atol=1e-6,rtol=1e-6), 'Malformed probabilities')
            metrics = _metrics(labels,p)
            _require(all(_close(row[k],metrics[k]) for k in ('balanced_accuracy','log_loss')), 'Metrics do not match numeric predictions')
            if condition == 'full_22':
                if full is None:
                    full = p.copy()
                _require(np.array_equal(full,p) and np.array_equal(reference_full,p), 'Full-input invariant violated')
    factor_history = _json(path / 'factor-history.json')
    _require(task.get('factor_epochs') == 30 and isinstance(factor_history,list) and len(factor_history) == 30,
             'Calibration must retain fixed 30-epoch history')
    for epoch, row in enumerate(factor_history, 1):
        _require(isinstance(row,dict) and set(row) == {'epoch','objective','reconstruction_mse','core_penalty'}
            and type(row['epoch']) is int and row['epoch'] == epoch,
            'Malformed calibration history schema or epoch sequence')
        _require(all(type(row[k]) in (int,float) and np.isfinite(row[k]) and row[k] >= 0
            for k in ('objective','reconstruction_mse','core_penalty')),
            'Nonfinite or negative calibration history')
    with np.load(path / 'initialization.npz',allow_pickle=False) as saved:
        _require(set(saved.files) == {'U','V','second_moment','training_trials'},'Invalid calibration state')
        initialization = {k:torch.from_numpy(saved[k].copy()) for k in saved.files}
        _require(initialization['training_trials'].item() == len(tids),'Wrong calibration training count')
    expected_initial = paired_completers(initialization)
    schedules = []
    _require(set(task.get('fits',{})) == set(STRATEGY_IDS),'Missing fits')
    for strategy in STRATEGY_IDS:
        initial_arrays = None
        for state in ('initial','selected'):
            with np.load(path / f'{strategy}/{state}.npz',allow_pickle=False) as saved:
                arrays = dict(saved)
            restore_completer(strategy, arrays)
            if state == 'initial':
                initial_arrays = arrays
                target = completion_state_arrays(expected_initial[strategy])
                _require(set(arrays) == set(target) and all(np.array_equal(arrays[k],target[k]) for k in arrays),
                         'Initial states differ from paired calibration')
            elif strategy not in LEARNED_STRATEGY_IDS:
                _require(all(np.array_equal(arrays[k],initial_arrays[k]) for k in arrays),'Frozen completion changed')
            else:
                # Anchors must remain those from the task-wide initialization.
                _require(all(np.array_equal(arrays[k],initial_arrays[k]) for k in arrays if k.startswith('initial_')),
                         'Learned anchor drift')
        history = _json(path / f'{strategy}/history.json')
        fit = task['fits'][strategy]
        if strategy not in LEARNED_STRATEGY_IDS:
            _require(history == [] and fit == {'selected_epoch':None,'epochs':0},'Frozen arm received supervised selection')
            continue
        n = len(history)-1
        _require(settings['minimum_epochs'] <= n <= settings['maximum_epochs'] and fit['epochs'] == n,
                 'Invalid training history budget')
        _require([r.get('epoch') for r in history] == list(range(n+1)), 'History epochs incomplete')
        _require(all(set(r) == {'epoch','mean_degraded_balanced_accuracy','mean_degraded_log_loss','updates','loss'}
            and type(r['epoch']) is int and type(r['updates']) is int and r['updates'] >= 0
            and 0 <= r['mean_degraded_balanced_accuracy'] <= 1 and r['mean_degraded_log_loss'] >= 0
            and (r['loss'] is None or (type(r['loss']) in (int,float) and np.isfinite(r['loss']) and r['loss'] >= 0))
            for r in history), 'Malformed numeric training history')
        _require(history[0]['updates'] == 0 and history[0]['loss'] is None, 'Epoch zero must be untrained')
        keys = [(-r['mean_degraded_balanced_accuracy'],r['mean_degraded_log_loss'],r['epoch']) for r in history]
        _require(all(np.isfinite(k[:2]).all() for k in keys), 'Nonfinite history')
        selected = min(keys)
        _require(fit['selected_epoch'] == selected[2], 'Selected epoch violates BA/logloss/earlier rule')
        current = selection_key([r for r in cells if r['strategy'] == strategy], fit['selected_epoch'])
        _require(_close(current[0],selected[0]) and _close(current[1],selected[1]),'Selected predictions disagree with selected history')
        best, stale, stop = keys[0], 0, settings['maximum_epochs']
        for epoch, key in enumerate(keys[1:],1):
            if key < best:
                best, stale = key, 0
            else:
                stale += 1
            if epoch >= settings['minimum_epochs'] and stale >= settings['patience']:
                stop = epoch
                break
        _require(n == stop, 'History violates fixed early stopping')
        with np.load(path / f'{strategy}/training-schedule.npz',allow_pickle=False) as saved:
            _require(set(saved.files) == {'masks','orders','sample_ids','labels'} and saved['sample_ids'].tolist() == tids
                and np.array_equal(saved['labels'],tlabels),'Unpaired training IDs/labels')
            tm, orders = saved['masks'],saved['orders']
            _require(tm.dtype == np.bool_ and tm.shape == (n,len(tids),22,4) and orders.shape == (n,len(tids)), 'Malformed training schedule')
            for epoch in range(1,n+1):
                expected_mask, expected_order = training_schedule(tids,task['subject'],task['seed'],epoch)
                _require(np.array_equal(tm[epoch-1],expected_mask) and np.array_equal(orders[epoch-1],expected_order),
                         'Training schedule differs from fixed partition-disjoint schedule')
                updates = sum(not expected_mask[expected_order[start:start+settings['batch_size']]].all()
                    for start in range(0,len(tids),settings['batch_size']))
                _require(history[epoch]['updates'] == updates and (history[epoch]['loss'] is None) == (updates == 0),
                         'History update count disagrees with full-batch skip policy')
            schedules.append((tm.copy(),orders.copy()))
    common = min(len(s[0]) for s in schedules)
    _require(all(np.array_equal(schedules[0][j][:common],schedules[1][j][:common]) for j in (0,1)),
             'Learned arms have unpaired schedule prefixes')


CONTRASTS = (
    ('primary_tucker_learned_minus_tucker_frozen','tucker_learned','tucker_frozen'),
    ('secondary_tucker_learned_minus_covariance_frozen','tucker_learned','covariance_frozen'),
    ('secondary_tucker_learned_minus_covariance_learned','tucker_learned','covariance_learned'),
    ('secondary_covariance_learned_minus_covariance_frozen','covariance_learned','covariance_frozen'),
    ('secondary_tucker_learned_minus_zero','tucker_learned','zero'))


def aggregate(task_rows, subjects, seeds, *, complete=False):
    """Average repeats, seeds, then participants equally; preserve partial rows."""
    _require(not any(t.get('synthetic',False) for t in task_rows), 'Synthetic data cannot enter real aggregation')
    planned = {(subject,seed) for subject in subjects for seed in seeds}
    actual = [(t['subject'],t['seed']) for t in task_rows]
    _require(len(actual) == len(set(actual)) and set(actual).issubset(planned), 'Unexpected or duplicate task grid')
    complete = bool(complete and planned and set(actual) == planned)
    means, seed_rows, participant_rows = {}, [], []
    for task in task_rows:
        expected = {(s,c['name'],r) for s in STRATEGY_IDS for c in CONDITIONS for r in range(c['repeats'])}
        actual_cells = [(r['strategy'],r['condition'],r['repeat']) for r in task['cells']]
        _require(len(actual_cells) == len(expected) and set(actual_cells) == expected,'Incomplete or duplicate cell grid')
        for strategy in STRATEGY_IDS:
            for condition in CONDITIONS:
                rows = [r for r in task['cells'] if r['strategy'] == strategy and r['condition'] == condition['name']]
                _require(len(rows) == condition['repeats'],'Incomplete repeat coverage')
                for metric in ('balanced_accuracy','log_loss'):
                    value = float(np.mean([r[metric] for r in rows]))
                    key = (task['subject'],task['seed'],strategy,condition['name'],metric)
                    _require(key not in means,'Duplicate task')
                    means[key] = value
                    seed_rows.append(dict(subject=key[0],seed=key[1],strategy=strategy,condition=condition['name'],metric=metric,value=value,
                        repeat_std=float(np.std([r[metric] for r in rows]))))
    cohort = []
    for strategy in STRATEGY_IDS:
        for condition in CONDITIONS:
            for metric in ('balanced_accuracy','log_loss'):
                values = []
                for subject in subjects:
                    vals = [means[(subject,s,strategy,condition['name'],metric)] for s in seeds
                            if (subject,s,strategy,condition['name'],metric) in means]
                    if vals:
                        value = float(np.mean(vals)); values.append(value)
                        participant_rows.append(dict(subject=subject,strategy=strategy,condition=condition['name'],metric=metric,
                            value=value,seeds=len(vals),seed_std=float(np.std(vals))))
                cohort.append(dict(strategy=strategy,condition=condition['name'],metric=metric,
                    value=float(np.mean(values)) if complete else None,
                    participant_std=float(np.std(values)) if complete else None,participants=len(values)))
    contrasts, participant_contrasts = [], []
    for name, candidate, control in CONTRASTS:
        values = []
        for subject in subjects:
            vals = []
            for seed in seeds:
                pairs = [(subject,seed,s,c['name'],'balanced_accuracy') for c in CONDITIONS[1:] for s in (candidate,control)]
                if all(k in means for k in pairs):
                    vals.append(float(np.mean([means[pairs[i]]-means[pairs[i+1]] for i in range(0,len(pairs),2)])))
            if vals:
                value = float(np.mean(vals)); values.append(value)
                participant_contrasts.append(dict(subject=subject,contrast=name,gain=value,seeds=len(vals),seed_std=float(np.std(vals))))
        ci = None
        if complete:
            rng = np.random.default_rng(20261005)
            bootstrap = np.mean(rng.choice(values,size=(2000,len(values)),replace=True),axis=1)
            ci = np.quantile(bootstrap,[.025,.975]).tolist()
        contrasts.append(dict(contrast=name,mean_gain=float(np.mean(values)) if complete else None,
            positive_participants=sum(v>0 for v in values) if complete else None,ci_95=ci,participants=len(values)))
    return dict(seed_conditions=seed_rows,participant_conditions=participant_rows,cohort_conditions=cohort,
                participant_contrasts=participant_contrasts,contrasts=contrasts)


def summarize(cfg):
    from .execution import scoped_cuda_rng
    with scoped_cuda_rng(940001):
        return _summarize(cfg)


def _summarize(cfg):
    from .execution import execution_metadata
    from .study import _json, _verify_complete_task, _atomic_json, historical_hashes
    output = validate_output_path(cfg)
    identity = study_identity(cfg)
    _require((output/'study-identity.json').is_file() and _json(output/'study-identity.json') == identity,
             'Current identity does not match report output')
    valid, issues, excluded = [], [], []
    for spec in tasks(cfg):
        path = output/'tasks'/f"A{spec['subject']:02d}_seed_{spec['seed']}"
        try:
            task = _verify_complete_task(path,identity,synthetic=cfg['synthetic'])
            _require(task.get('execution') == execution_metadata(), 'Report CUDA hardware/runtime differs')
            _require((task['subject'],task['seed']) == (spec['subject'],spec['seed']),'Task identity mismatch')
            if task['synthetic']:
                excluded.append(spec); continue
            from inm.completion_transformer.replay import verify_historical_task
            historical = verify_historical_task(cfg,spec['subject'],spec['seed'])
            _require(task['historical_input_sha256'] == historical_hashes(cfg,historical,spec['subject']), 'Changed historical hashes')
            valid.append(task)
        except Exception as exc:
            issues.append({**spec,'reason':str(exc)})
    complete = not cfg['synthetic'] and len(valid) == 27 and not issues and cfg['subjects'] == list(range(1,10)) and cfg['seeds'] == [0,1,2]
    values = aggregate(valid,cfg['subjects'],cfg['seeds'],complete=complete)
    primary = values['contrasts'][0]
    screen = bool(complete and primary['mean_gain'] >= .02 and primary['positive_participants'] >= 6)
    evidence = dict(schema_name='agfl-task-driven-completion-cuda-report-v1',study_id=identity['study_id'],
        status='complete' if complete else 'partial',complete=complete,synthetic=False,valid_tasks=len(valid),
        expected_tasks=len(tasks(cfg)),synthetic_tasks_excluded=excluded,issues=issues,
        execution=execution_metadata(), execution_source_sha256=identity['source_sha256'],
        primary_screening_rule_met=screen,**values,
        bootstrap={'unit':'paired_participant','resamples':2000,'seed':20261005,'exploratory_unadjusted':True},
        limitations=['Historical and completion checkpoints reuse validation selection; no independent confirmation.',
          'No causal recovery or architecture superiority claim.', 'Screening is separate from strong-control comparisons.',
          'Incomplete real coverage suppresses cohort means; synthetic evidence is excluded.'])
    _atomic_json(output/'report/evidence.json',evidence)
    from inm.completion_transformer.reporting import _csv, _atomic_text
    for name in ('seed_conditions','participant_conditions','cohort_conditions','participant_contrasts'):
        rows = evidence[name]
        _csv(output/f'report/{name}.csv',rows,list(rows[0]) if rows else ['status'])
    repeats = [{**r,'subject':t['subject'],'seed':t['seed']} for t in valid for r in t['cells']]
    _csv(output/'report/per_repeat.csv',repeats,list(repeats[0]) if repeats else ['status'])
    summary = [f"# Task-driven completion\n\nStatus: {evidence['status']}. Verified real tasks: {len(valid)}/27.",
        'Primary: learned Tucker minus frozen Tucker, averaged over the four degraded validation conditions.',
        'Average mask repeats, then seeds within participant, then participants equally.']
    if complete:
        summary += ['| Contrast | Mean gain (pp) | 95% participant interval (pp) | Positive participants |',
                    '|---|---:|---:|---:|']
        for row in evidence['contrasts']:
            summary.append(f"| {row['contrast']} | {100*row['mean_gain']:.3f} | "
                f"[{100*row['ci_95'][0]:.3f}, {100*row['ci_95'][1]:.3f}] | {row['positive_participants']}/9 |")
        summary.append(f'Primary exploratory screen (at least 2 pp and 6/9 positive participants): {screen}.')
    else:
        summary.append('Incomplete coverage: cohort estimates and intervals are withheld. See per-repeat and participant tables for retained evidence.')
    _atomic_text(output/'report/summary.md','\n\n'.join(summary+evidence['limitations'])+'\n')
    return evidence
