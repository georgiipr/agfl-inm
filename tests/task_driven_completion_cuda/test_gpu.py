"""Required CUDA acceptance: missing GPU is an error, never a skip.

Run outside the sandbox with the project virtualenv. All inputs are synthetic.
CPU/GPU completion and derivative tolerances: rtol=1e-4, atol=1e-6.
"""
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from inm.task_driven_completion_cuda.execution import scoped_cuda_rng, execution_metadata
from inm.task_driven_completion_cuda.completion import fit_initialization, paired_completers
from inm.task_driven_completion_cuda.adapters import CompletionAdapter
from inm.task_driven_completion_cuda.study import synthetic_context
from inm.task_driven_completion_cuda.training import fit_completion

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/task-driven-completion-cuda-v2.json'


def snapshot():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state().clone(),
        torch.cuda.get_rng_state(0).clone(), torch.get_num_threads(),
        torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
        torch.backends.mha.get_fastpath_enabled(),
        os.environ.get('CUBLAS_WORKSPACE_CONFIG'))


class GPUChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with scoped_cuda_rng(42):
            cls.hardware = execution_metadata()
        cls.context = synthetic_context()
        cls.prepared = cls.context['prepared']
        cls.initial, _ = fit_initialization(torch.from_numpy(cls.prepared.train.raw), 0)

    def test_cpu_gpu_outputs_gradients_hidden_values_and_observation_identity(self):
        x = torch.from_numpy(self.prepared.train.raw[:2].copy())
        m = torch.ones(2,22,4,dtype=torch.bool)
        m[0,6:,0] = False; m[0,16:,1:] = False; m[1,:6,:] = False
        weights = torch.linspace(-.7,.9,x.numel()).reshape_as(x)
        with scoped_cuda_rng(7):
            for name, cpu in paired_completers(self.initial).items():
                gpu = copy.deepcopy(cpu).to('cuda:0')
                cx, gx = x.clone().requires_grad_(), x.cuda().requires_grad_()
                a, b = cpu(cx,m), gpu(gx,m.cuda())
                torch.testing.assert_close(a, b.cpu(), rtol=1e-4, atol=1e-6)
                self.assertTrue(torch.equal(b[m.cuda()], gx[m.cuda()]))
                for fill in (float('nan'),float('inf'),float('-inf')):
                    poisoned = torch.where(m[...,None], x, torch.full_like(x,fill)).cuda()
                    self.assertTrue(torch.equal(b, gpu(poisoned,m.cuda())))
                self.assertTrue(torch.equal(gpu(gx,torch.ones_like(m).cuda()),gx))
                (a*weights).sum().backward()
                (b*weights.cuda()).sum().backward()
                torch.testing.assert_close(cx.grad,gx.grad.cpu(),rtol=1e-4,atol=1e-6)
                for cp,gp in zip(cpu.parameters(),gpu.parameters()):
                    self.assertIsNotNone(gp.grad)
                    self.assertGreater(gp.grad.abs().max().item(),0)
                    torch.testing.assert_close(cp.grad,gp.grad.cpu(),rtol=1e-4,atol=1e-6)
                if name.startswith('covariance'):
                    self.assertEqual(gpu.free.dtype,torch.float64)

    def test_classification_gradient_and_frozen_backbone(self):
        with scoped_cuda_rng(17):
            backbone = copy.deepcopy(self.context['models']['spatial_transformer']).cuda()
            before = {k:v.clone() for k,v in backbone.state_dict().items()}
            x = torch.from_numpy(self.prepared.train.raw).cuda()
            mask = torch.ones(4,22,4,dtype=torch.bool,device='cuda:0'); mask[:,6:,:] = False
            for name in ('covariance_learned','tucker_learned'):
                model = paired_completers(self.initial)[name].cuda()
                adapter = CompletionAdapter(backbone,name,model).train()
                loss = F.cross_entropy(adapter(x,mask),torch.arange(4,device='cuda:0'))
                loss.backward()
                self.assertGreater(sum(p.grad.abs().sum().item() for p in model.parameters()),0)
                self.assertTrue(all(p.grad is None and not p.requires_grad for p in backbone.parameters()))
                self.assertTrue(all(not m.training for m in backbone.modules()))
                self.assertTrue(all(torch.equal(before[k],v) for k,v in backbone.state_dict().items()))
                model.zero_grad(set_to_none=True)
                full = adapter(x.requires_grad_(),torch.ones_like(mask))
                full.sum().backward()
                self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_scope_restores_all_state_success_exception_and_missing_gpu(self):
        caller_fastpath = torch.backends.mha.get_fastpath_enabled()
        try:
            for ambient_fastpath in (True, False):
                torch.backends.mha.set_fastpath_enabled(ambient_fastpath)
                for fail in (False,True):
                    with self.subTest(ambient_fastpath=ambient_fastpath, fail=fail):
                        before = snapshot()
                        try:
                            with scoped_cuda_rng(99):
                                random.random(); np.random.rand(); torch.rand(2); torch.rand(2,device='cuda:0')
                                self.assertTrue(torch.are_deterministic_algorithms_enabled())
                                self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                                self.assertFalse(torch.backends.cudnn.allow_tf32)
                                self.assertFalse(torch.backends.mha.get_fastpath_enabled())
                                self.assertIs(execution_metadata()['mha_fastpath_enabled'], False)
                                for inner_fail in (False, True):
                                    outer = snapshot()
                                    try:
                                        with scoped_cuda_rng(100):
                                            self.assertFalse(torch.backends.mha.get_fastpath_enabled())
                                            random.random(); np.random.rand(); torch.rand(2)
                                            torch.rand(2,device='cuda:0')
                                            if inner_fail:
                                                raise RuntimeError('nested injected')
                                    except RuntimeError as exc:
                                        self.assertEqual(str(exc), 'nested injected')
                                    self.assert_state(outer,snapshot())
                                if fail:
                                    raise RuntimeError('injected')
                        except RuntimeError as exc:
                            self.assertEqual(str(exc),'injected')
                        self.assert_state(before,snapshot())
                before = snapshot()
                with patch('torch.cuda.is_available',return_value=False):
                    with self.assertRaisesRegex(RuntimeError,'fallback'):
                        with scoped_cuda_rng(1):
                            self.fail('CPU fallback')
                self.assert_state(before,snapshot())
        finally:
            torch.backends.mha.set_fastpath_enabled(caller_fastpath)

    def test_metadata_enforces_actual_fastpath_setting(self):
        before = snapshot()
        with scoped_cuda_rng(18):
            self.assertIs(execution_metadata()['mha_fastpath_enabled'], False)
            torch.backends.mha.set_fastpath_enabled(True)
            with self.assertRaisesRegex(RuntimeError, 'execution settings'):
                execution_metadata()
            # A nested scope must disable the changed caller setting and restore it.
            with scoped_cuda_rng(19):
                self.assertFalse(torch.backends.mha.get_fastpath_enabled())
                self.assertIs(execution_metadata()['mha_fastpath_enabled'], False)
            self.assertTrue(torch.backends.mha.get_fastpath_enabled())
        self.assert_state(before,snapshot())

    def assert_state(self,a,b):
        self.assertEqual(a[0],b[0]); self.assertEqual(a[1][0],b[1][0])
        np.testing.assert_array_equal(a[1][1],b[1][1]); self.assertEqual(a[1][2:],b[1][2:])
        self.assertTrue(torch.equal(a[2],b[2])); self.assertTrue(torch.equal(a[3],b[3]))
        self.assertEqual(a[4:],b[4:])

    def test_repeated_fits_changed_rng_arm_order_and_masks(self):
        def fit(name):
            with scoped_cuda_rng(8):
                backbone = copy.deepcopy(self.context['models']['spatial_transformer']).cuda()
                model = paired_completers(self.initial)[name].cuda()
                return fit_completion(backbone,name,model,self.prepared.train,self.prepared.validation,
                    subject=1,seed=0,synthetic=True,budget=self.context['budget'])
        first = {name:fit(name) for name in ('covariance_learned','tucker_learned')}
        random.seed(88); np.random.seed(89); torch.random.default_generator.manual_seed(90)
        with scoped_cuda_rng(91):
            torch.rand(99,device='cuda:0')
        second = {name:fit(name) for name in ('tucker_learned','covariance_learned')}
        for name in first:
            a,b = first[name],second[name]
            self.assertEqual(a['history'],b['history'])
            self.assertEqual(a['selected_epoch'],b['selected_epoch'])
            for field in ('training_masks','training_orders'):
                np.testing.assert_array_equal(a[field],b[field])
            for key in a['selected']:
                np.testing.assert_array_equal(a['selected'][key],b['selected'][key])
            self.assertGreater(a['history'][1]['updates'],0)
        np.testing.assert_array_equal(first['covariance_learned']['training_masks'],first['tucker_learned']['training_masks'])
        np.testing.assert_array_equal(first['covariance_learned']['training_orders'],first['tucker_learned']['training_orders'])

    def test_public_fresh_smoke_resume_independent_audit_and_identity_refusal(self):
        with tempfile.TemporaryDirectory(prefix='agfl-cuda-acceptance-') as directory:
            output = Path(directory)/'smoke'
            base = [sys.executable,'-m','inm.task_driven_completion_cuda','--config',str(CONFIG)]
            def call(args,expected=0):
                result = subprocess.run(args,cwd=ROOT,text=True,capture_output=True,timeout=240)
                self.assertEqual(result.returncode,expected,result.stdout+'\n'+result.stderr)
                return json.loads(result.stdout) if expected == 0 else result
            fresh = call(base+['--smoke','--output-dir',str(output)])
            self.assertEqual(fresh['cells'],105)
            self.assertEqual(fresh['synthetic_tasks_excluded'],1)
            task_path = output/'tasks/A01_seed_0/task.json'
            task = json.loads(task_path.read_text())
            self.assertIs(task['execution']['mha_fastpath_enabled'], False)
            call(base+['--smoke','--output-dir',str(output),'--resume-smoke'])
            audit = call([sys.executable,'-m','inm.task_driven_completion_cuda.audit','--config',str(CONFIG),
                '--synthetic-smoke','--output-dir',str(output)])
            self.assertEqual(audit['device'],'cuda:0'); self.assertEqual(audit['new_fits'],0)
            self.assertEqual(audit['replayed_cells'],105); self.assertTrue(audit['exact_prediction_replay'])
            cfg = json.loads(CONFIG.read_text()); cfg['name'] += '-changed'
            for key in ('data_dir','candidate_source_dir','split_source_dir','output_dir'):
                cfg[key] = str((CONFIG.parent/cfg[key]).resolve())
            changed = Path(directory)/'changed.json'; changed.write_text(json.dumps(cfg))
            failed = call([sys.executable,'-m','inm.task_driven_completion_cuda','--config',str(changed),
                '--smoke','--output-dir',str(output),'--resume-smoke'],2)
            self.assertIn('identity',failed.stderr)
            # Completed outputs cannot hide a changed or absent backend setting.
            for fastpath in (True, None):
                changed_task = copy.deepcopy(task)
                if fastpath is None:
                    changed_task['execution'].pop('mha_fastpath_enabled')
                else:
                    changed_task['execution']['mha_fastpath_enabled'] = fastpath
                task_path.write_text(json.dumps(changed_task))
                failed = call(base+['--smoke','--output-dir',str(output),'--resume-smoke'],2)
                self.assertIn('execution identity', failed.stderr)
                failed = call([sys.executable,'-m','inm.task_driven_completion_cuda.audit',
                    '--config',str(CONFIG),'--synthetic-smoke','--output-dir',str(output)],2)
                self.assertIn('hardware/runtime differs', failed.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
