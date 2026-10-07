"""Independent hierarchy/coverage/bootstrap check for the declared reporting API."""
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from published_covariance.reporting import summarize_cells

class ReportingAcceptance(unittest.TestCase):
    def cells(self):
        rows=[];deltas=np.array([-.04,.02,.01,-.01,.08,.03,-.02,.05,.06])
        for i in range(9):
            for seed in range(3):
                for arm in ['zero','covariance_fixed','covariance_learned']:
                    for condition in ['full_22','static_16','static_6']:
                        for repeat in range(1 if condition=='full_22' else 5):
                            base=.35+.03*i+.01*seed+.002*repeat+(.02 if condition=='static_16' else 0)
                            ba=.7+.01*i if condition=='full_22' else base+({'zero':0,'covariance_fixed':.05,'covariance_learned':.05+deltas[i]}[arm])
                            rows.append({'subject':f'A{i+1:02}','seed':seed,'arm':arm,'condition':condition,'repeat':repeat,'balanced_accuracy':ba,'accuracy':ba-.01,'log_loss':1.5-ba,'n_trials':20*(i+1)**2})
        return rows,deltas

    def test_hierarchy_paired_bootstrap_negative_effects(self):
        cells,deltas=self.cells();report=summarize_cells(cells)
        self.assertTrue(report['complete'])
        contrast=report['primary_learned_minus_fixed']
        self.assertAlmostEqual(contrast['mean'],float(deltas.mean()),places=12)
        rng=np.random.Generator(np.random.PCG64(20261006))
        ci=np.quantile(rng.choice(deltas,size=(2000,9),replace=True).mean(axis=1),[.025,.975])
        np.testing.assert_allclose(contrast['bootstrap_95'],ci,atol=1e-12,rtol=1e-12)
        self.assertLess(report['participants']['A01']['learned_minus_fixed'],0)
        expected=np.mean([r['balanced_accuracy'] for r in cells if r['arm']=='covariance_learned' and r['condition']!='full_22'])
        self.assertAlmostEqual(report['cohort']['covariance_learned']['degraded_ba'],expected,places=12)

    def test_missing_and_duplicate_cells_fail_closed(self):
        cells,_=self.cells();report=summarize_cells(cells[:-1])
        self.assertFalse(report['complete']);self.assertFalse(report.get('cohort'))
        self.assertFalse(report.get('primary_learned_minus_fixed'))
        with self.assertRaises(ValueError):summarize_cells(cells+[cells[0]])

if __name__=='__main__':unittest.main(verbosity=2)
