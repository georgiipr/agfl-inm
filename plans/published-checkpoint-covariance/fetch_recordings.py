"""Supervisor download of declared official native recordings; no scoring."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'.session-runs/published-checkpoint-covariance/assets/recordings'
for subject in range(1,10):
    for session in ['T','E']:
        name=f'A{subject:02}{session}'
        path=DEST/(name+'.mat'); receipt=DEST/(name+'-origin.json')
        if path.exists():
            assert receipt.exists(), f'Partial/unverified download {path}'
            assert hashlib.sha256(path.read_bytes()).hexdigest()==json.loads(receipt.read_text())['sha256']
            print(name, 'verified existing',flush=True)
            continue
        assert not receipt.exists()
        url=f'https://bnci-horizon-2020.eu/database/data-sets/001-2014/{name}.mat'
        command=['curl','--fail','--silent','--show-error','--location','--max-time','180','--output',str(path),'--write-out','%{url_effective}',url]
        started=dt.datetime.now(dt.timezone.utc).isoformat()
        result=subprocess.run(command,capture_output=True,text=True,timeout=200)
        record={'command':command,'url':url,'redirect_url':result.stdout,'stderr':result.stderr,'exit':result.returncode,'retrieved_utc':started}
        if result.returncode==0:
            record.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
        with receipt.open('x') as f:json.dump(record,f,indent=2)
        print(name,result.returncode,record.get('bytes'),flush=True)
        if result.returncode:raise SystemExit(result.returncode)
