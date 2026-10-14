"""Optional plots from complete saved summary tables; no model execution."""
from __future__ import annotations
import csv
import fcntl
import hashlib
import io
import json
import math
from pathlib import Path
from .availability import DEGRADED_PATTERNS
from .reporting import _atomic_text
from eeg_models.models import MODEL_LABELS, model_definition

COUNTS = (22, 16, 11, 6)
REMOVED = tuple(100 * (22 - count) / 22 for count in COUNTS)
COLORS = {'eegnet': '#666666', 'eegnet_covariance_frozen': '#0072B2',
          'eegnet_covariance_learned': '#009E73', 'signal_transformer': '#999999',
          'signal_transformer_covariance_frozen': '#E69F00',
          'signal_transformer_covariance_learned': '#CC79A7'}
PATTERN_LABELS = {
    'random_static': 'Static random loss', 'spatial_static': 'Static spatial loss',
    'dynamic_random': 'Dynamic random loss', 'dynamic_spatial': 'Dynamic spatial loss',
}


def _number(row, field):
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f'Nonfinite saved value: {field}')
    return value


def _index(rows, fields):
    indexed = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in indexed:
            raise ValueError(f'Duplicate summary row: {key}')
        indexed[key] = row
    return indexed


def _scenario(pattern, count):
    return 'full_22' if count == 22 else f'{pattern}_{count}'


def _save(figure, folder, name):
    files = []
    for extension in ('png', 'pdf'):
        path = folder / f'{name}.{extension}'
        temporary = path.with_suffix(f'.{extension}.tmp')
        figure.savefig(temporary, format=extension, dpi=180, bbox_inches='tight')
        temporary.replace(path)
        files.append(str(path.relative_to(folder.parent)))
    return files


def _curves(plt, indexed, partition, model_keys):
    figure, axes = plt.subplots(2, 2, figsize=(11.4, 8.2), sharey=True)
    for axis, pattern in zip(axes.flat, DEGRADED_PATTERNS):
        for key in model_keys:
            values = [_number(indexed[(partition, key, _scenario(pattern, count))],
                              'balanced_accuracy_mean_percent') for count in COUNTS]
            completion = model_definition(key)['completion']
            style, marker = {'none': ('--', 'o'), 'covariance_frozen': ('-', 's'),
                             'covariance_learned': ('-.', '^')}[completion]
            axis.plot(REMOVED, values, color=COLORS[key],
                      linestyle=style, marker=marker,
                      linewidth=1.7, markersize=4, label=MODEL_LABELS[key])
        axis.axhline(25, color='#888888', linestyle=':', linewidth=.8)
        axis.set_title(PATTERN_LABELS[pattern])
        axis.set_xticks(REMOVED, [f'{value:.1f}\n({count} retained)' for value, count in zip(REMOVED, COUNTS)])
        axis.set_xlabel('Missing channels (%)')
        axis.set_ylabel('Balanced accuracy (%)')
        axis.set_ylim(0, 100)
        axis.grid(alpha=.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.suptitle(f'{partition.capitalize()} availability: EEGNet / Signal Transformer with built-in MHA')
    figure.legend(handles, labels, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(.5, .018))
    figure.text(.5, .005, 'Equal-weight participant means; seeds and repeats averaged within participants. '
                'Dotted line: 25% reference.', ha='center', fontsize=8)
    figure.tight_layout(rect=(0, .15, 1, .95))
    return figure


def _gains(plt, indexed, partition, contrasts):
    figure, axes = plt.subplots(2, 2, figsize=(13.4, 9.2))
    for axis, pattern in zip(axes.flat, DEGRADED_PATTERNS):
        labels = []
        for position, contrast in enumerate(contrasts):
            row = indexed[(partition, contrast, 'robustness', pattern)]
            value = _number(row, 'balanced_accuracy_gain_pp')
            axis.bar(position, value, color=COLORS[row['comparison_model']], width=.65)
            backbone = 'EEGNet' if row['backbone'] == 'eegnet' else 'Signal Transformer'
            reference = model_definition(row['reference_model'])['completion'].removeprefix('covariance_')
            comparison = model_definition(row['comparison_model'])['completion'].removeprefix('covariance_')
            labels.append(f'{backbone}\n{comparison} − {"baseline" if reference == "none" else reference}')
            if row['balanced_accuracy_ci95_low_pp'] and row['balanced_accuracy_ci95_high_pp']:
                low = _number(row, 'balanced_accuracy_ci95_low_pp')
                high = _number(row, 'balanced_accuracy_ci95_high_pp')
                if low > high:
                    raise ValueError('Reversed saved bootstrap interval')
                axis.vlines(position, low, high, color='#222222')
                axis.hlines((low, high), position - .1, position + .1, color='#222222')
        axis.axhline(0, color='#444444', linewidth=.8)
        axis.set_title(PATTERN_LABELS[pattern])
        axis.set_xticks(range(len(contrasts)), labels, rotation=25, ha='right', fontsize=8)
        axis.set_ylabel('Comparison − reference balanced accuracy (pp)')
        axis.grid(axis='y', alpha=.2)
    figure.suptitle(f'{partition.capitalize()} paired covariance robustness gains')
    figure.text(.5, .025, 'Equal weights for 22, 16, 11 and 6 channels. Positive favors the comparison model.',
                ha='center', fontsize=8)
    figure.text(.5, .007, 'Whiskers: pointwise 95% subject-bootstrap intervals; unavailable for one participant.',
                ha='center', fontsize=8)
    figure.tight_layout(rect=(0, .07, 1, .95))
    return figure


def plot_report(report: Path) -> dict:
    report = Path(report)
    report.mkdir(parents=True, exist_ok=True)
    status = {'schema_version': 4, 'status': 'incomplete', 'files': [], 'groups': [],
              'primary_metric': 'balanced_accuracy', 'curves_uncertainty': 'means only',
              'gain_uncertainty': 'pointwise 95% subject-bootstrap intervals where available'}
    filenames = ('accuracy_table.csv', 'paired_completion_gain.csv')
    # Serialize plotting as well as table reading. This avoids simultaneous
    # workers publishing two versions of the same PNG/PDF or status record.
    with (report / '.plots.lock').open('a+') as plotting_lock:
        fcntl.flock(plotting_lock, fcntl.LOCK_EX)
        with (report / '.summary.lock').open('a+') as summary_lock:
            fcntl.flock(summary_lock, fcntl.LOCK_SH)
            missing = [name for name in filenames if not (report / name).is_file()]
            if missing:
                status['reason'] = f'Missing summary tables: {", ".join(missing)}'
                _atomic_text(report / 'plots_status.json', json.dumps(status, indent=2) + '\n')
                return status
            contents = {name: (report / name).read_bytes() for name in filenames}
        status['source_tables_sha256'] = {name: hashlib.sha256(value).hexdigest() for name, value in contents.items()}
        tables = {name: list(csv.DictReader(io.StringIO(value.decode('utf-8-sig')))) for name, value in contents.items()}
        accuracy = _index(tables['accuracy_table.csv'], ('partition', 'model', 'scenario'))
        paired = _index(tables['paired_completion_gain.csv'], ('partition', 'contrast', 'kind', 'pattern', 'scenario'))
        paired_robust = {key[:4]: value for key, value in paired.items() if key[2] == 'robustness'}
        model_keys = list(dict.fromkeys(row['model'] for row in tables['accuracy_table.csv']))
        contrasts = list(dict.fromkeys(row['contrast'] for row in tables['paired_completion_gain.csv']))
        ready = []
        for partition in ('validation', 'test'):
            for kind, indexed, keys, selected in (
                ('availability', accuracy,
                 [(partition, model, _scenario(pattern, count)) for model in model_keys
                  for pattern in DEGRADED_PATTERNS for count in COUNTS], model_keys),
                ('completion_gain', paired_robust,
                 [(partition, contrast, 'robustness', pattern) for contrast in contrasts
                  for pattern in DEGRADED_PATTERNS], contrasts),
            ):
                missing_keys = [key for key in keys if key not in indexed or indexed[key]['complete'] != 'True']
                group = {'name': f'{kind}_{partition}',
                         'status': 'ready' if keys and not missing_keys else 'incomplete',
                         'missing_cells': [list(key) for key in missing_keys]}
                if not keys:
                    group['reason'] = 'No configured completion pairs' if kind == 'completion_gain' else 'No saved model rows'
                status['groups'].append(group)
                if group['status'] == 'ready':
                    ready.append((group, kind, indexed, partition, selected))
        if ready:
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
            except ImportError as error:
                status.update(status='dependency_unavailable', reason='Optional plots require matplotlib in the active environment')
                _atomic_text(report / 'plots_status.json', json.dumps(status, indent=2) + '\n')
                raise RuntimeError(status['reason']) from error
            folder = report / 'plots'
            folder.mkdir(exist_ok=True)
            with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10,
                                 'pdf.fonttype': 42, 'ps.fonttype': 42}):
                for group, kind, indexed, partition, selected in ready:
                    figure = (_curves(plt, indexed, partition, selected) if kind == 'availability'
                              else _gains(plt, indexed, partition, selected))
                    try:
                        files = _save(figure, folder, group['name'])
                        status['files'].extend(files)
                        group.update(status='generated', files=files)
                    finally:
                        plt.close(figure)
        applicable = [group for group in status['groups'] if 'reason' not in group]
        status['status'] = ('complete' if applicable and all(group['status'] == 'generated' for group in applicable)
                            else 'partial' if status['files'] else 'incomplete')
        status['reason'] = 'Only complete configured comparisons are plotted; see groups for pending or unconfigured pairs.'
        _atomic_text(report / 'plots_status.json', json.dumps(status, indent=2) + '\n')
        return status
