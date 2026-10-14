"""Frozen published-checkpoint protocol and portable source/runtime provenance."""
from __future__ import annotations
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/published-covariance.json'
ASSETS = ROOT / 'assets/published-covariance'
ARMS = ('zero', 'covariance_fixed', 'covariance_learned')
CONDITIONS = {'full_22': 1, 'static_16': 5, 'static_6': 5}
TASKS = [(f'A{i:02d}', seed) for i in range(1, 10) for seed in range(3)]


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def runtime(device):
    # TensorFlow 2.2 cluster modules can use Python 3.7. pkg_resources is already
    # provided by setuptools in that environment; no new package is installed.
    try:
        from importlib import metadata
        packages = {d.metadata.get('Name', 'unknown'): d.version for d in metadata.distributions()}
    except ImportError:
        import pkg_resources
        packages = {d.project_name: d.version for d in pkg_resources.working_set}
    from .tf_compat import compatibility_record
    gpus = tf.config.list_physical_devices('GPU')
    details = []
    for gpu in gpus:
        if hasattr(tf.config.experimental, 'get_device_details'):
            details.append(tf.config.experimental.get_device_details(gpu))
        else:
            details.append({'name': gpu.name, 'device_type': gpu.device_type})
    try:
        build = tf.sysconfig.get_build_info()
    except AttributeError:
        build = {}
    return {'python': platform.python_version(), 'python_executable': sys.executable,
            'python_flags': {'optimize': sys.flags.optimize},
            'platform': platform.platform(), 'packages': dict(sorted(packages.items())),
            'tensorflow_build': build, 'gpu_details': details,
            'requested_device': device, 'attention_compatibility': compatibility_record(),
            'runtime_flags': {k: os.environ.get(k) for k in (
                'TF_DETERMINISTIC_OPS', 'TF_ENABLE_ONEDNN_OPTS', 'TF_CUDNN_DETERMINISTIC',
                'OMP_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS')}}


def source_manifest():
    paths = [*sorted((ROOT / 'published_covariance').glob('*.py')),
             ROOT / 'configs/published-covariance.json',
             *sorted((ROOT / 'docs/published-covariance').glob('*'))]
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths if p.is_file()}


def identity(device, synthetic):
    config = json.loads(CONFIG.read_text())
    value = {'study_id': config['study_id'], 'configuration': config,
             'configuration_sha256': sha256(CONFIG), 'source': source_manifest(),
             'runtime': runtime(device), 'synthetic_fixture': bool(synthetic)}
    launcher = os.environ.get('AGFL_INM_LAUNCHER')
    if launcher:
        path = Path(launcher)
        value['launcher'] = {'sha256': sha256(path), 'content': path.read_text()}
    if not synthetic:
        origin = ASSETS / 'origin.json'
        origin_data = json.loads(origin.read_text())
        if (origin_data.get('repository') != 'https://github.com/Altaheri/EEG-ATCNet'
                or origin_data.get('revision') != '65162fb359ea46a2f62c885a9987247ab491ee7a'):
            raise RuntimeError('Published checkpoint origin does not identify the pinned revision')
        value['origin_manifest_sha256'] = sha256(origin)
        required = {'models.py', 'attention_models.py', 'preprocess.py', 'LICENSE', 'README.md'}
        required.update(f'results/saved models/run-1/subject-{i}.h5' for i in range(1, 10))
        pinned = json.loads((ROOT / 'docs/published-covariance/checkpoints-origin.json').read_text())
        source = {}
        for relative in sorted(required):
            expected = pinned['files'][relative]
            if origin_data['files'].get(relative) != expected or sha256(ASSETS / 'EEG-ATCNet' / relative) != expected:
                raise RuntimeError('Published asset bytes differ: ' + relative)
            source[relative] = expected
        value['published_source_and_checkpoint_sha256'] = source
        expected_recordings = {}
        for line in (ROOT / 'docs/published-covariance/recordings-sha256.txt').read_text().splitlines():
            expected, name = line.split()
            stem = Path(name).stem
            receipt = ASSETS / 'recordings' / (stem + '-origin.json')
            record = json.loads(receipt.read_text())
            if record.get('sha256') != expected or sha256(ASSETS / 'recordings' / name) != expected:
                raise RuntimeError('Native recording bytes differ: ' + name)
            expected_recordings[stem] = expected
        if set(expected_recordings) != {f'A{i:02d}{s}' for i in range(1, 10) for s in ('T', 'E')}:
            raise RuntimeError('All nine T/E recording identities are required')
        value['recording_bytes_sha256'] = expected_recordings
        value['native_alignment_receipt_sha256'] = sha256(
            ROOT / 'docs/published-covariance/native-recording-alignment.json')
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
