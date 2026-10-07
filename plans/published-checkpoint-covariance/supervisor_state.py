"""Supervisor-only immutable inventories and bounded-session receipts."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / '.session-runs/published-checkpoint-covariance'

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def inventory():
    result = {}
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = sorted(d for d in dirs if d not in {'.git', '__pycache__', '.venv', '.review-venv', '.local-history', '.venv-published-covariance'} and not (Path(base)/d).is_symlink() and Path(base)/d != STATE)
        for name in sorted(files):
            p = Path(base)/name
            if p.is_symlink():
                result[str(p.relative_to(ROOT))] = {'symlink': os.readlink(p)}
            elif not name.endswith('.pyc'):
                result[str(p.relative_to(ROOT))] = {'sha256': sha(p), 'size': p.stat().st_size}
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['snapshot', 'verify'])
    parser.add_argument('path', type=Path)
    args = parser.parse_args()
    if args.mode == 'snapshot':
        data = {'utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'files': inventory()}
        with args.path.open('x') as f:
            json.dump(data, f, indent=2, sort_keys=True)
        print(json.dumps({'files': len(data['files']), 'inventory_sha256': sha(args.path)}))
    else:
        data = json.loads(args.path.read_text())
        errors = []
        for name, old in data['files'].items():
            p = ROOT/name
            if 'symlink' in old:
                valid = p.is_symlink() and os.readlink(p) == old['symlink']
            else:
                valid = p.is_file() and not p.is_symlink() and sha(p) == old['sha256']
            if not valid:
                errors.append(name)
        print(json.dumps({'protected_files': len(data['files']), 'changed': errors}))
        raise SystemExit(bool(errors))

if __name__ == '__main__':
    main()
