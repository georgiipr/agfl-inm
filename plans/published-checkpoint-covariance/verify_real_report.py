"""Independent numeric verdict admission, using saved real per-trial probabilities.

No classifier calls, fitting, checkpoint selection or production report helpers.
Run after the complete production task audits and summarize command.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

SUBJECTS=[f'A{i:02d}' for i in range(1,10)]
ARMS=['zero','covariance_fixed','covariance_learned']
CONDITIONS={'full_22':1,'static_16':5,'static_6':5}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('output',type=Path); args=ap.parse_args()
    out=args.output.resolve(strict=True)
    identity=json.loads((out/'identity.json').read_text())
    assert identity['synthetic_fixture'] is False,'Synthetic fixtures are not real measurements'
    report=json.loads((out/'report.json').read_text())
    assert report['complete'] is True
    values={}; artifact_hashes={}; native_count=0; fits=[]
    for subject in SUBJECTS:
        with np.load(out/'evaluation'/f'{subject}-E-input.npz',allow_pickle=False) as z:
            labels=z['labels'].copy();ids=z['ids'].copy()
        assert len(labels)==288 and set(labels.tolist())=={0,1,2,3}
        assert len(set(ids.tolist()))==288
        assert all(':E:' in str(x) for x in ids)
        with np.load(out/'evaluation'/f'{subject}-E-native-reference.npz',allow_pickle=False) as z:
            native=z['probabilities'].copy()
            np.testing.assert_array_equal(z['labels'],labels)
            np.testing.assert_array_equal(z['ids'],ids)
        native_count+=1; full=None
        for seed in range(3):
            fit=out/'fits'/f'{subject}-s{seed}.npz'
            meta=json.loads(fit.with_suffix('.json').read_text())
            assert hashlib.sha256(fit.read_bytes()).hexdigest()==meta['npz_sha256']
            fits.append((subject,seed,meta['selected_epoch'],meta['epochs_completed']))
            for arm in ARMS:
                for condition,repeats in CONDITIONS.items():
                    repeat_values=[]
                    for repeat in range(repeats):
                        p=out/'evaluation'/f'{subject}-s{seed}-{arm}-{condition}-r{repeat}.npz'
                        artifact_hashes[str(p.relative_to(out))]=hashlib.sha256(p.read_bytes()).hexdigest()
                        with np.load(p,allow_pickle=False) as z:
                            probs=z['probabilities'].copy(); mask=z['mask'].copy()
                            np.testing.assert_array_equal(z['labels'],labels)
                            np.testing.assert_array_equal(z['ids'],ids)
                        assert probs.shape==(288,4) and np.isfinite(probs).all() and (probs>=0).all()
                        np.testing.assert_allclose(probs.sum(1),1,atol=1e-5,rtol=0)
                        assert mask.dtype==np.bool_ and mask.shape==(288,22)
                        assert np.all(mask.sum(1)==int(condition.split('_')[-1]))
                        predicted=probs.argmax(1)
                        ba=np.mean([np.mean(predicted[labels==c]==c) for c in range(4)])
                        accuracy=np.mean(predicted==labels)
                        logloss=-np.mean(np.log(np.clip(probs.astype(np.float64)[np.arange(288),labels],1e-12,1)))
                        repeat_values.append([ba,accuracy,logloss])
                        if condition=='full_22':
                            if full is None:full=probs
                            np.testing.assert_array_equal(probs,full)
                            np.testing.assert_allclose(probs,native,atol=1e-5,rtol=1e-4)
                            np.testing.assert_array_equal(predicted,native.argmax(1))
                    values[(subject,seed,arm,condition)]=np.mean(repeat_values,axis=0)
    # First repeats above, then seeds within each participant, finally participants.
    participant={(s,a,c):np.mean([values[(s,k,a,c)] for k in range(3)],axis=0)
                 for s in SUBJECTS for a in ARMS for c in CONDITIONS}
    effects={s:np.mean([participant[(s,'covariance_learned',c)][0]-participant[(s,'covariance_fixed',c)][0]
                       for c in ('static_16','static_6')]) for s in SUBJECTS}
    effects_zero={s:np.mean([participant[(s,'covariance_learned',c)][0]-participant[(s,'zero',c)][0]
                            for c in ('static_16','static_6')]) for s in SUBJECTS}
    for a in ARMS:
        for c in CONDITIONS:
            expected=np.mean([participant[(s,a,c)] for s in SUBJECTS],axis=0)
            actual=report['cohort'][a]['conditions'][c]
            np.testing.assert_allclose([actual['balanced_accuracy'],actual['accuracy'],actual['log_loss']],expected,atol=1e-12,rtol=0)
    vector=np.asarray([effects[s] for s in SUBJECTS]); rng=np.random.Generator(np.random.PCG64(20261006))
    interval=np.quantile(vector[rng.integers(0,9,size=(2000,9))].mean(1),[.025,.975])
    np.testing.assert_allclose(report['primary_learned_minus_fixed']['mean'],vector.mean(),atol=1e-12,rtol=0)
    np.testing.assert_allclose(report['primary_learned_minus_fixed']['bootstrap_95'],interval,atol=1e-12,rtol=0)
    for s in SUBJECTS:
        np.testing.assert_allclose(report['participants'][s]['learned_minus_fixed'],effects[s],atol=1e-12,rtol=0)
        np.testing.assert_allclose(report['participants'][s]['learned_minus_zero'],effects_zero[s],atol=1e-12,rtol=0)
    assert len(artifact_hashes)==891 and len(fits)==27 and native_count==9
    result={'passed':True,'real_covariance_fits':27,'arm_evaluations':81,'native_references':9,
            'evaluation_cells':891,'participants':{s:{'learned_minus_fixed':float(effects[s]),'learned_minus_zero':float(effects_zero[s])} for s in SUBJECTS},
            'learned_minus_fixed_mean':float(vector.mean()),'bootstrap_95':interval.tolist(),
            'epoch_zero_selections':sum(t[2]==0 for t in fits),'epoch_budget_saturations':sum(t[3]>=100 for t in fits),
            'identity_sha256':hashlib.sha256((out/'identity.json').read_bytes()).hexdigest(),
            'report_sha256':hashlib.sha256((out/'report.json').read_bytes()).hexdigest(),
            'numeric_artifacts':artifact_hashes,'interpretation':'Exploratory grade B, prior E exposure unknown'}
    print(json.dumps(result,sort_keys=True,indent=2))

if __name__=='__main__':main()
