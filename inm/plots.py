"""Optional publication figures from saved summary tables, never model runs.

Curves display subject-averaged means without uncertainty bands: the accuracy
table contains subject SD, which is not a confidence interval. Only paired
gain figures draw the reported subject-bootstrap 95% confidence intervals.
"""
from __future__ import annotations

import csv
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile


ATTENTIONS = ('mha', 'performer')
LABELS = {'mha': 'MHA', 'performer': 'Performer'}
COLORS = {'mha': '#0072B2', 'performer': '#009E73'}
PATTERNS = ('random_static', 'spatial_static', 'dynamic_random', 'dynamic_spatial')
PATTERN_LABELS = {'random_static': 'Static random loss',
                  'spatial_static': 'Static spatial loss',
                  'dynamic_random': 'Changing random availability',
                  'dynamic_spatial': 'Changing spatial availability'}
REGIMES = ('full', 'mixed')
REGIME_LABELS = {'full': 'Training with all channels',
                 'mixed': 'Training with mixed availability'}
COUNTS = (22, 16, 11, 6)
REMOVED = tuple(100.0 * (22 - count) / 22 for count in COUNTS)
CONTROL_LABELS = {'baseline': 'Baseline', 'tensor_core': 'Tensor core',
                  'tensor_completion': 'Tensor completion',
                  'linear_core': 'Linear core', 'mean_completion': 'Mean completion'}


def _atomic_json(path, value):
    descriptor, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _number(row, field):
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _complete(row, fields):
    return (row is not None and str(row.get('complete', '')).lower() in ('true', '1')
            and all(_number(row, field) is not None for field in fields))


def _index(rows, fields, *, robust=False):
    indexed = {}
    for row in rows:
        if row.get('partition') != 'test' or (robust and row.get('kind') != 'robustness'):
            continue
        key = tuple(row.get(field) for field in fields)
        # A repeated identity is invalid, not an opportunity to choose a score.
        indexed[key] = None if key in indexed else row
    return indexed


def _scenario(pattern, count):
    return 'full_22' if count == 22 else f'{pattern}_{count}'


def _curve_keys(regime, pairs):
    return list(dict.fromkeys((attention, representation, regime, _scenario(pattern, count))
                             for attention, representation in pairs
                             for pattern in PATTERNS for count in COUNTS))


def _group_status(name, indexed, keys, fields):
    good = sum(_complete(indexed.get(key), fields) for key in keys)
    ready = good == len(keys)
    return {'name': name, 'status': 'ready' if ready else 'skipped',
            'expected_rows': len(keys), 'complete_rows': good,
            'reason': None if ready else
            'Required complete test rows or finite summary values/intervals are missing; no figure generated.'}


def _axis_style(axis, pattern, *, curves):
    axis.set_title(PATTERN_LABELS[pattern], fontsize=11)
    axis.grid(axis='y', alpha=.25, linewidth=.7)
    axis.set_axisbelow(True)
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    if curves:
        axis.set_xticks(REMOVED, ['0', '27.27', '50', '72.73'])
        axis.set_xlabel('Channels removed (%)')
        axis.set_ylabel('Test balanced accuracy (%)')
        axis.set_ylim(0, 100)
        axis.set_xlim(-2, 75)
        axis.axhline(25, color='#666666', linestyle=':', linewidth=.8, alpha=.7)


def _save(figure, folder, name):
    outputs = []
    for extension in ('png', 'pdf'):
        destination = folder / f'{name}.{extension}'
        descriptor, temporary = tempfile.mkstemp(prefix=f'.{name}.', suffix=f'.{extension}', dir=folder)
        os.close(descriptor)
        try:
            figure.savefig(temporary, dpi=180, bbox_inches='tight', format=extension)
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        outputs.append(str(destination.relative_to(folder.parent)))
    return outputs


def _primary_curves(plt, Line2D, indexed, regime):
    figure, axes = plt.subplots(2, 2, figsize=(11.4, 8.2))
    for axis, pattern in zip(axes.flat, PATTERNS):
        for attention in ATTENTIONS:
            for representation, style, marker in (('baseline', '--', 'o'), ('tensor_core', '-', 's')):
                values = [_number(indexed[(attention, representation, regime, _scenario(pattern, count))],
                                  'balanced_accuracy_mean_percent') for count in COUNTS]
                axis.plot(REMOVED, values, color=COLORS[attention], linestyle=style,
                          marker=marker, markersize=4, linewidth=1.7)
        _axis_style(axis, pattern, curves=True)
    handles = [Line2D([0], [0], color=COLORS[attention], linewidth=2, label=LABELS[attention])
               for attention in ATTENTIONS]
    handles.extend([Line2D([0], [0], color='#333333', linestyle='--', marker='o', label='Baseline'),
                    Line2D([0], [0], color='#333333', linestyle='-', marker='s', label='Tensor core')])
    figure.suptitle(f'EEGNet-derived spatial attention: {REGIME_LABELS[regime]}', fontsize=13)
    figure.legend(handles=handles, loc='lower center', ncol=4, frameon=False,
                  bbox_to_anchor=(.5, .018))
    figure.text(.5, .005, 'Equal-weight subject means; seeds and mask repeats averaged within subjects. '
                'Dotted horizontal line: 25% reference.', ha='center', fontsize=8)
    figure.tight_layout(rect=(0, .105, 1, .95))
    return figure


def _control_curves(plt, indexed, regime):
    figure, axes = plt.subplots(2, 2, figsize=(11.4, 8.2))
    colors = ('#555555', '#D55E00', '#0072B2', '#009E73', '#CC79A7')
    markers = ('o', 's', '^', 'D', 'v')
    for axis, pattern in zip(axes.flat, PATTERNS):
        for (representation, label), color, marker in zip(CONTROL_LABELS.items(), colors, markers):
            values = [_number(indexed[('mha', representation, regime, _scenario(pattern, count))],
                              'balanced_accuracy_mean_percent') for count in COUNTS]
            axis.plot(REMOVED, values, color=color, marker=marker, markersize=4,
                      linestyle='--' if representation == 'baseline' else '-', linewidth=1.7, label=label)
        _axis_style(axis, pattern, curves=True)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.suptitle(f'MHA representation controls: {REGIME_LABELS[regime]}', fontsize=13)
    figure.legend(handles, labels, loc='lower center', ncol=3, frameon=False,
                  bbox_to_anchor=(.5, .018))
    figure.text(.5, .005, 'Equal-weight subject means; no confidence intervals are inferred from subject SD.',
                ha='center', fontsize=8)
    figure.tight_layout(rect=(0, .105, 1, .95))
    return figure


def _paired_gains(plt, indexed, regime):
    figure, axes = plt.subplots(2, 2, figsize=(11.4, 8.2), sharey=True)
    bounds = [0.]
    for axis, pattern in zip(axes.flat, PATTERNS):
        for position, attention in enumerate(ATTENTIONS):
            row = indexed[(attention, regime, pattern)]
            value = _number(row, 'balanced_accuracy_gain_pp')
            low = _number(row, 'balanced_accuracy_ci95_low_pp')
            high = _number(row, 'balanced_accuracy_ci95_high_pp')
            if low > high:
                raise ValueError('A saved confidence interval has reversed bounds')
            bounds.extend((value, low, high))
            axis.bar(position, value, width=.65, color=COLORS[attention], alpha=.85, zorder=2)
            # Draw absolute interval limits: unlike yerr, this remains valid if
            # a percentile bootstrap interval does not contain the point mean.
            axis.vlines(position, low, high, color='#222222', linewidth=1.2, zorder=3)
            axis.hlines((low, high), position - .11, position + .11,
                        color='#222222', linewidth=1.2, zorder=3)
        _axis_style(axis, pattern, curves=False)
        axis.axhline(0, color='#444444', linewidth=.9)
        axis.set_xticks(range(len(ATTENTIONS)), [LABELS[key] for key in ATTENTIONS], rotation=15)
        axis.set_ylabel('Tensor core − baseline\nbalanced accuracy (percentage points)')
    padding = max(1., .12 * (max(bounds) - min(bounds)))
    axes.flat[0].set_ylim(min(bounds) - padding, max(bounds) + padding)
    figure.suptitle(f'Paired robustness gain: {REGIME_LABELS[regime]}', fontsize=13)
    figure.text(.5, .024, 'Each score equally weights 22, 16, 11 and 6 retained channels. Positive favors tensor core.',
                ha='center', fontsize=8)
    figure.text(.5, .007, 'Whiskers: pointwise 95% subject-bootstrap confidence intervals; no multiple-comparison correction.',
                ha='center', fontsize=8)
    figure.tight_layout(rect=(0, .065, 1, .95))
    return figure


def plot_report(report: Path) -> dict:
    """Plot only complete figure groups from accuracy/paired summary CSV files.

    Return/save a manifest even when nothing is complete. This optional API
    does not train, load checkpoints, import matplotlib until it is needed, or
    install dependencies. The experiment runner never calls it automatically.
    """
    report = Path(report)
    report.mkdir(parents=True, exist_ok=True)
    filenames = ('accuracy_table.csv', 'paired_tensor_gain.csv')
    status = {'schema_version': 1, 'primary_metric': 'test_balanced_accuracy',
              'status': 'incomplete', 'files': [], 'groups': [],
              'uncertainty': 'Paired-gain figures use saved 95% subject-bootstrap intervals; curves show means only.'}
    with (report / '.summary.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
        missing = [name for name in filenames if not (report / name).is_file()]
        if missing:
            status['reason'] = f'Missing summary tables: {", ".join(missing)}'
            _atomic_json(report / 'plots_status.json', status)
            return status
        contents = {name: (report / name).read_bytes() for name in filenames}
    status['source_tables_sha256'] = {name: hashlib.sha256(data).hexdigest()
                                      for name, data in contents.items()}
    tables = {name: list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))
              for name, data in contents.items()}
    status['table_rows'] = {name: len(rows) for name, rows in tables.items()}
    accuracy = _index(tables['accuracy_table.csv'], ('attention', 'representation', 'regime', 'scenario'))
    paired = _index(tables['paired_tensor_gain.csv'], ('attention', 'regime', 'pattern'), robust=True)
    fields = ('balanced_accuracy_gain_pp', 'balanced_accuracy_ci95_low_pp', 'balanced_accuracy_ci95_high_pp')
    for regime in REGIMES:
        pairs = [(attention, representation) for attention in ATTENTIONS
                 for representation in ('baseline', 'tensor_core')]
        status['groups'].append(_group_status(f'primary_curves_{regime}', accuracy,
                                _curve_keys(regime, pairs), ('balanced_accuracy_mean_percent',)))
        pairs = [('mha', representation) for representation in CONTROL_LABELS]
        status['groups'].append(_group_status(f'mha_controls_{regime}', accuracy,
                                _curve_keys(regime, pairs), ('balanced_accuracy_mean_percent',)))
        keys = [(attention, regime, pattern) for attention in ATTENTIONS for pattern in PATTERNS]
        status['groups'].append(_group_status(f'paired_robustness_{regime}', paired, keys, fields))
    ready = [group for group in status['groups'] if group['status'] == 'ready']
    if not ready:
        status['reason'] = 'No complete figure group yet. No blank charts were generated.'
        _atomic_json(report / 'plots_status.json', status)
        return status
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as error:
        status.update(status='dependency_unavailable',
                      reason='Optional plotting requires matplotlib in the existing environment.')
        _atomic_json(report / 'plots_status.json', status)
        raise RuntimeError('Optional plots require matplotlib. Use the existing environment that '
                           'generated AGFL plots, or omit --plots and use the CSV tables. '
                           'No dependency was installed.') from error
    folder = report / 'plots'
    folder.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'pdf.fonttype': 42, 'ps.fonttype': 42}):
        for group in ready:
            name = group['name']
            regime = name.rsplit('_', 1)[-1]
            if name.startswith('primary_curves_'):
                figure = _primary_curves(plt, Line2D, accuracy, regime)
            elif name.startswith('mha_controls_'):
                figure = _control_curves(plt, accuracy, regime)
            else:
                figure = _paired_gains(plt, paired, regime)
            try:
                generated = _save(figure, folder, name)
                status['files'].extend(generated)
                group.update(status='generated', files=generated)
            finally:
                plt.close(figure)
    status['status'] = 'complete' if len(ready) == len(status['groups']) else 'partial'
    status['reason'] = ('All requested figure groups are complete.' if status['status'] == 'complete'
                        else 'Only complete figure groups were generated; see skipped groups.')
    _atomic_json(report / 'plots_status.json', status)
    return status
