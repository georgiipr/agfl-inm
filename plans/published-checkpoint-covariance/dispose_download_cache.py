"""Remove only reviewed, hash-verified duplicate wheel downloads from this task."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
MANIFEST=ROOT/'.session-runs/published-checkpoint-covariance/runtime-cuda/disposable-download-cache.json'

def main():
    p=argparse.ArgumentParser();p.add_argument('--apply',action='store_true');a=p.parse_args()
    manifest=json.loads(MANIFEST.read_text()); base=Path(manifest['cache_root']).resolve(strict=True)
    assert base==Path('/home/kalexu97/.cache/pip/http-v2')
    paths=[]
    for row in manifest['entries']:
        assert row['mtime_utc']>='2026-10-06T14:00:00'
        for name,digest in ((row['path'],row['sha256']),(row['metadata_path'],row['metadata_sha256'])):
            if name is None:continue
            path=Path(name)
            assert not path.is_symlink() and path.is_file()
            assert path.resolve(strict=True).is_relative_to(base)
            with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==digest
            paths.append(path)
    result={'cache_files':len(paths),'download_bytes':manifest['bytes'],'applied':a.apply}
    if a.apply:
        for path in paths:path.unlink()
    print(json.dumps(result))

if __name__=='__main__':main()
