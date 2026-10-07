"""Supervisor recording identity check. No classifier or outcome computation."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import numpy as np
import scipy.io as sio
import mne

ROOT=Path(__file__).resolve().parents[2]
ASSETS=ROOT/'.session-runs/published-checkpoint-covariance/assets/recordings'
GDF=ROOT.parent/'ml'
LABELS=ROOT/'.session-runs/covariance-confirmation/official-inputs'
OUTPUT=ROOT/'.session-runs/published-checkpoint-covariance/02/native-recording-alignment.json'
EXPECTED_CHANNELS=['EEG-Fz','EEG-0','EEG-1','EEG-2','EEG-3','EEG-4','EEG-5','EEG-C3','EEG-6','EEG-Cz','EEG-7','EEG-C4','EEG-8','EEG-9','EEG-10','EEG-11','EEG-12','EEG-13','EEG-14','EEG-Pz','EEG-15','EEG-16']
rows=[]
for subject in range(1,10):
    for session in ['T','E']:
        stem=f'A{subject:02}{session}';p=ASSETS/(stem+'.mat')
        origin=json.loads((ASSETS/(stem+'-origin.json')).read_text())
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        assert digest==origin['sha256']
        assert origin['url']==f'https://bnci-horizon-2020.eu/database/data-sets/001-2014/{stem}.mat'
        raw=mne.io.read_raw_gdf(GDF/(stem+'.gdf'),preload=False,verbose='ERROR')
        assert raw.info['sfreq']==250 and raw.ch_names[:22]==EXPECTED_CHANNELS
        onsets=np.rint(raw.annotations.onset[raw.annotations.description=='768']*250).astype(int)
        if session=='T':
            labels=np.array([int(v)-769 for v in raw.annotations.description if v in ['769','770','771','772']])
        else:
            labels=sio.loadmat(LABELS/(stem+'.mat'))['classlabel'].reshape(-1).astype(int)-1
        values=[];native_labels=[];native_ids=[];flagged=0;max_error=0.;trial_no=0
        for run_no,run in enumerate(sio.loadmat(p)['data'][0]):
            v=run[0,0]
            assert v['fs'].item()==250
            for within,(start,label,flag) in enumerate(zip(v['trial'].ravel(),v['y'].ravel(),v['artifacts'].ravel())):
                assert int(start)+1750<=len(v['X'])
                native=v['X'][int(start)+375:int(start)+1500,:22].T
                observed=raw.get_data(picks=list(range(22)),start=int(onsets[trial_no])+376,stop=int(onsets[trial_no])+1501)*1e6
                # Fixed before adapter checks in 02/start.json.
                np.testing.assert_allclose(native,observed,atol=1e-8,rtol=1e-10)
                max_error=max(max_error,float(np.max(np.abs(native-observed))))
                values.append(native);native_labels.append(int(label)-1);flagged+=int(flag!=0)
                native_ids.append(f'A{subject:02}:{session}:r{run_no+1:02}:t{within+1:03}')
                trial_no+=1
        np.testing.assert_array_equal(native_labels,labels)
        assert trial_no==len(onsets)==288
        arrays=np.stack(values)
        row={'session':stem,'trials':trial_no,'artifact_flags_retained':flagged,'excluded':0,
             'max_abs_error_microvolts':max_error,'native_values_sha256':hashlib.sha256(arrays.tobytes()).hexdigest(),
             'labels_sha256':hashlib.sha256(np.array(native_labels,dtype=np.int64).tobytes()).hexdigest(),
             'trial_ids_sha256':hashlib.sha256(json.dumps(native_ids).encode()).hexdigest(),
             'MAT_sha256':digest,'GDF_sha256':hashlib.sha256((GDF/(stem+'.gdf')).read_bytes()).hexdigest()}
        rows.append(row);print(json.dumps(row),flush=True)
        raw.close()
with OUTPUT.open('x') as f:json.dump({'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'no_classifier_or_outcomes':True,'tolerances':{'atol_microvolts':1e-8,'rtol':1e-10},'rows':rows},f,indent=2)
