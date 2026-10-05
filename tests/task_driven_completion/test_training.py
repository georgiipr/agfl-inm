"""Actual short CPU fits exercise determinism, selection and partition isolation."""
import random
import unittest
from unittest.mock import patch
import numpy as np
import torch
from inm.availability import subset_partition
from inm.task_driven_completion.completion import fit_initialization, paired_completers, scoped_cpu_rng
from inm.task_driven_completion.study import synthetic_context
from inm.task_driven_completion.training import (fit_completion, selection_key,
    training_schedule, training_settings, validation_bank)


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = synthetic_context()
        cls.prepared = cls.context['prepared']
        cls.initial, _ = fit_initialization(torch.from_numpy(cls.prepared.train.raw),0)

    def fit(self,strategy,models=None,**kw):
        return fit_completion(self.context['models']['spatial_transformer'],strategy,
            (models or paired_completers(self.initial))[strategy],self.prepared.train,self.prepared.validation,
            subject=1,seed=0,synthetic=True,budget=self.context['budget'],**kw)

    def test_repeated_synthetic_optimizer_fits_ignore_ambient_rng_and_arm_order(self):
        first = {s:self.fit(s) for s in ('covariance_learned','tucker_learned')}
        random.seed(998); np.random.seed(997); torch.manual_seed(996)
        second = {s:self.fit(s) for s in ('tucker_learned','covariance_learned')}
        for strategy in first:
            a,b = first[strategy],second[strategy]
            self.assertEqual(a['history'],b['history'])
            self.assertEqual(a['selected_epoch'],b['selected_epoch'])
            np.testing.assert_array_equal(a['training_masks'],b['training_masks'])
            np.testing.assert_array_equal(a['training_orders'],b['training_orders'])
            for key in a['selected']:
                np.testing.assert_array_equal(a['selected'][key],b['selected'][key])
        np.testing.assert_array_equal(first['covariance_learned']['training_masks'],first['tucker_learned']['training_masks'])

    def test_training_masks_partition_and_per_id_order_stability(self):
        ids = self.prepared.train.sample_ids
        seen_retained, seen_dynamic = set(),set()
        for epoch in range(1,15):
            mask, order = training_schedule(ids,1,0,epoch)
            reversed_mask,_ = training_schedule(ids[::-1],1,0,epoch)
            np.testing.assert_array_equal(mask,reversed_mask[::-1])
            self.assertEqual(sorted(order.tolist()),list(range(len(ids))))
            for sample in mask:
                retained = int(sample[:,0].sum()); seen_retained.add(retained)
                if retained < 22:
                    seen_dynamic.add(not np.array_equal(sample[:,0],sample[:,1]))
                    for window in range(4):
                        self.assertEqual(subset_partition(sample[:,window]),'train')
        self.assertEqual(seen_retained,{22,16,6}); self.assertEqual(seen_dynamic,{True,False})
        for condition,repeat,mask in validation_bank(self.prepared.validation.sample_ids,1,0):
            if condition != 'full_22':
                for sample in mask:
                    for window in range(4):
                        self.assertEqual(subset_partition(sample[:,window]),'validation')

    def test_epoch_zero_eligible_and_wholly_full_batch_skipped(self):
        def full_schedule(ids,subject,seed,epoch):
            return np.ones((len(ids),22,4),dtype=bool),np.arange(len(ids))
        with patch('inm.task_driven_completion.training.training_schedule',side_effect=full_schedule):
            result = self.fit('tucker_learned')
        self.assertEqual(result['selected_epoch'],0)
        self.assertEqual(result['history'][1]['updates'],0)
        for key in result['initial']:
            np.testing.assert_array_equal(result['initial'][key],result['selected'][key])
        def rows(ba,ll):
            return [dict(condition='random_static_16',balanced_accuracy=ba,log_loss=ll)]
        self.assertLess(selection_key(rows(.6,1),0),selection_key(rows(.5,.2),1))
        self.assertLess(selection_key(rows(.6,.8),1),selection_key(rows(.6,1),0))
        self.assertLess(selection_key(rows(.6,.8),0),selection_key(rows(.6,.8),1))

    def test_rng_threads_restore_on_training_exception(self):
        py, num, tor, threads = random.getstate(),np.random.get_state(),torch.get_rng_state().clone(),torch.get_num_threads()
        with patch('inm.task_driven_completion.training.evaluate',side_effect=RuntimeError('injected')):
            with self.assertRaisesRegex(RuntimeError,'injected'):
                self.fit('covariance_learned')
        self.assertEqual(py,random.getstate())
        np.testing.assert_array_equal(num[1],np.random.get_state()[1])
        self.assertEqual(num[2:],np.random.get_state()[2:])
        self.assertTrue(torch.equal(tor,torch.get_rng_state()))
        self.assertEqual(threads,torch.get_num_threads())

    def test_real_budget_fixed_and_invalid_synthetic_budget_rejected(self):
        self.assertEqual(training_settings()['maximum_epochs'],100)
        self.assertEqual(training_settings()['minimum_epochs'],10)
        with self.assertRaises(ValueError):
            training_settings(budget=self.context['budget'])
        with self.assertRaises(ValueError):
            training_settings(synthetic=True,budget={'maximum_epochs':1})
        with self.assertRaises(ValueError):
            training_settings(synthetic=True,budget={'maximum_epochs':True,'minimum_epochs':1,'patience':1})

    def test_train_validation_ids_must_be_disjoint(self):
        from types import SimpleNamespace
        original = self.prepared.validation
        validation = SimpleNamespace(raw=original.raw,labels=original.labels,sample_ids=self.prepared.train.sample_ids)
        with self.assertRaisesRegex(ValueError,'overlap'):
            fit_completion(self.context['models']['spatial_transformer'],'covariance_learned',
                paired_completers(self.initial)['covariance_learned'],self.prepared.train,validation,
                subject=1,seed=0,synthetic=True,budget=self.context['budget'])
