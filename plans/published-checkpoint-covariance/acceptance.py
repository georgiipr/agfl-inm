"""Independent checkpoint admission gate; subsequent gates added before workers."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / '.session-runs/published-checkpoint-covariance/assets'

def checkpoint_gate():
    import h5py
    import numpy as np
    audit = json.loads((ROOT/'docs/published-covariance/checkpoint-audit.json').read_text())
    assert audit['admission'] == 'compatible', 'Unresolved admission cannot pass'
    assert audit['provenance_grade'] in ['A', 'B']
    assert audit['accuracy_used_for_selection'] is False
    assert audit['new_e_fitted_transform'] is False
    assert audit['hidden_channel_local_preprocessing'] is True
    assert audit['native_recipe_resolved'] is True
    source = ASSETS/audit['source_directory']
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    assert revision == audit['source_revision'] and len(revision) == 40
    assert (source/'LICENSE').is_file()
    assert [r['subject'] for r in audit['subjects']] == [f'A{i:02d}' for i in range(1, 10)]
    signatures = []
    for row in audit['subjects']:
        path = ASSETS/row['asset_path']
        assert path.resolve().is_relative_to(ASSETS.resolve())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
        assert row['compatible'] is True
        assert row['training_sessions'] and row['selection_sessions'] and row['normalization_source']
        assert row['provenance_grade'] in ['A', 'B']
        tensors = []
        with h5py.File(path, 'r') as f:
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset):
                    array = obj[()]
                    assert np.isfinite(array).all(), name
                    tensors.append((name, list(array.shape), str(array.dtype)))
            f.visititems(visit)
        assert tensors, 'No actual tensors inspected'
        assert len(tensors) == row['tensor_count']
        assert sum(int(np.prod(t[1])) for t in tensors) == row['tensor_elements']
        signatures.append(sorted((t[1], t[2]) for t in tensors))
        print(row['subject'], row['sha256'], len(tensors), row['tensor_elements'])
    assert all(s == signatures[0] for s in signatures), 'Subject architectures differ'
    print('PASS: nine real checkpoint hashes, finite tensors and structural admission metadata; narrative source review also required')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--session', choices=['01'])
    args = p.parse_args()
    checkpoint_gate()
    if args.session is None:
        checks = ['acceptance_data.py', 'acceptance_data_boundaries.py',
                  'acceptance_classifier.py', 'acceptance_completion.py',
                  'acceptance_training.py', 'acceptance_reconstruction.py',
                  'acceptance_reporting.py', 'acceptance_launcher.py',
                  'acceptance_execution.py', 'acceptance_execution_faults.py',
                  'acceptance_identity.py', 'acceptance_native_reference.py',
                  'acceptance_launcher_sequence.py']
        for name in checks:
            print('RUN', name, flush=True)
            subprocess.run([sys.executable, str(Path(__file__).with_name(name))],
                           cwd=ROOT, check=True, timeout=1200)
