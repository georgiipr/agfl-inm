"""Download and verify pinned sources, checkpoints and native MAT recordings.

Uses the Python standard library. This preparation performs no model execution,
statistics fitting or scoring. Existing files must match their pinned hashes.
"""
from pathlib import Path
import hashlib
import http.client
import json
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSETS = ROOT / 'assets/published-covariance'
RECORDING_BASE = 'https://lampx.tugraz.at/~bci/database/001-2014/'
LEGACY_RECORDING_BASE = 'https://bnci-horizon-2020.eu/database/data-sets/001-2014/'


def _connect_ipv4(address, timeout=None, source_address=None):
    """Avoid unsupported IPv6 sockets during asset preparation only."""
    last_error = None
    for family, kind, protocol, _, endpoint in socket.getaddrinfo(
            address[0], address[1], socket.AF_INET, socket.SOCK_STREAM):
        connection = None
        try:
            connection = socket.socket(family, kind, protocol)
            connection.settimeout(timeout)
            if source_address:
                connection.bind(source_address)
            connection.connect(endpoint)
            return connection
        except OSError as error:
            last_error = error
            if connection is not None:
                connection.close()
    if last_error is not None:
        raise last_error
    raise OSError('No IPv4 address available for asset host: ' + address[0])


class _IPv4Connection:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._create_connection = _connect_ipv4


class _IPv4HTTPConnection(_IPv4Connection, http.client.HTTPConnection):
    pass


class _IPv4HTTPSConnection(_IPv4Connection, http.client.HTTPSConnection):
    pass


class _IPv4HTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, request):
        return self.do_open(_IPv4HTTPConnection, request)


class _IPv4HTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, request):
        return self.do_open(_IPv4HTTPSConnection, request,
                            context=self._context, check_hostname=self._check_hostname)


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _download(path, url, expected, offline=False):
    if path.is_symlink():
        raise ValueError('Refusing symlinked asset: ' + str(path))
    if path.exists():
        if _hash(path) != expected:
            raise ValueError('Existing asset has different bytes: ' + str(path))
        return
    if offline:
        raise FileNotFoundError(
            'Required local asset is missing: ' + str(path)
            + '. Prepare assets on a machine with internet access before submitting the experiment.')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.download-' + str(os.getpid()))
    for attempt in range(1, 4):
        print('Downloading ' + path.name + ' (attempt ' + str(attempt)
              + '/3): ' + url, flush=True)
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'AGFL-inm-reproduction/1'})
            opener = urllib.request.build_opener(_IPv4HTTPHandler, _IPv4HTTPSHandler)
            with opener.open(request, timeout=120) as response, temporary.open('xb') as stream:
                length = response.headers.get('Content-Length', '')
                total = int(length) if length.isdigit() else None
                received = 0
                next_progress = 8 * 1024 * 1024
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    stream.write(block)
                    received += len(block)
                    if received >= next_progress:
                        progress = str(round(received / (1024 * 1024), 1)) + ' MiB'
                        if total:
                            progress += ' / ' + str(round(total / (1024 * 1024), 1)) + ' MiB'
                        print(path.name + ': ' + progress, flush=True)
                        next_progress = received + 8 * 1024 * 1024
                if total is not None and received != total:
                    raise OSError('Incomplete download of ' + path.name)
            if _hash(temporary) != expected:
                raise ValueError('Downloaded asset does not match the pinned SHA256: ' + url)
            temporary.replace(path)
            return
        except (urllib.error.URLError, OSError, http.client.HTTPException) as error:
            print('Download failed for ' + path.name + ': ' + str(error), flush=True)
            if attempt == 3:
                raise RuntimeError(
                    'Asset download failed after three attempts: ' + url
                    + '. Download the pinned file on a machine that can reach the server'
                    + ' and transfer it to ' + str(path) + ' before preparing again.') from error
            time.sleep(2 * attempt)
        finally:
            if temporary.exists():
                temporary.unlink()


def _manifest(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Existing origin manifest differs: ' + str(path))
    else:
        with path.open('x') as stream:
            stream.write(json.dumps(value, sort_keys=True, indent=2) + '\n')


def prepare_assets(assets=None, *, offline=False):
    root = DEFAULT_ASSETS if assets is None else Path(assets).expanduser().absolute()
    if root != DEFAULT_ASSETS:
        raise ValueError('Use the declared assets/published-covariance directory')
    for parent in [root, *root.parents]:
        if parent.is_symlink():
            raise ValueError('Asset paths cannot contain symlinks')
    root.mkdir(parents=True, exist_ok=True)
    docs = ROOT / 'docs/published-covariance'
    origin = json.loads((docs / 'checkpoints-origin.json').read_text())
    revision = origin['revision']
    source_files = ['models.py', 'attention_models.py', 'preprocess.py', 'LICENSE', 'README.md']
    source_files += [f'results/saved models/run-1/subject-{i}.h5' for i in range(1, 10)]
    for relative in source_files:
        url = 'https://raw.githubusercontent.com/Altaheri/EEG-ATCNet/' + revision + '/' + urllib.parse.quote(relative)
        _download(root / 'EEG-ATCNet' / relative, url, origin['files'][relative], offline=offline)
        print('Verified published asset: ' + relative, flush=True)
    _manifest(root / 'origin.json', origin)
    records = root / 'recordings'
    records.mkdir(exist_ok=True)
    for line in (docs / 'recordings-sha256.txt').read_text().splitlines():
        expected, name = line.split()
        url = RECORDING_BASE + name
        path = records / name
        _download(path, url, expected, offline=offline)
        receipt = records / (path.stem + '-origin.json')
        value = {'url': url, 'sha256': expected, 'bytes': path.stat().st_size}
        if receipt.exists():
            # Preserve an earlier verified origin rather than rewriting history
            # when only the download endpoint changes.
            previous = json.loads(receipt.read_text())
            permitted = [dict(value, url=base + name)
                         for base in (RECORDING_BASE, LEGACY_RECORDING_BASE)]
            if previous not in permitted:
                raise ValueError('Existing recording origin differs: ' + str(receipt))
        else:
            _manifest(receipt, value)
        print('Verified native recording: ' + name, flush=True)
    print('Asset preparation complete. No classifier or outcome was evaluated.', flush=True)
