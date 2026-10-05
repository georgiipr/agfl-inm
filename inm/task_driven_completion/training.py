"""Deterministic CPU completion optimization through a frozen Transformer."""
from __future__ import annotations

import copy
import hashlib
import numpy as np
import torch
from torch.nn import functional as F

from inm.availability import CHANNEL_IDS, make_mask_bank, mask_bank_digest
from inm.encoder_candidates.training import _metrics
from .adapters import CompletionAdapter
from .completion import (completion_state_arrays, reconstruction_loss,
                         restore_completer, scoped_cpu_rng)
from .protocol import CONDITIONS, FIXED, LEARNED_STRATEGY_IDS


def training_settings(*, synthetic=False, budget=None):
    settings = copy.deepcopy(FIXED['protocol']['training'])
    if budget is not None:
        if not synthetic or not isinstance(budget, dict) or set(budget) != {'maximum_epochs', 'minimum_epochs', 'patience'}:
            raise ValueError('Only synthetic runs may inject an explicit three-field epoch budget')
        if any(type(v) is not int or v < 1 for v in budget.values()):
            raise ValueError('Synthetic epoch budgets must be positive integers')
        if budget['minimum_epochs'] > budget['maximum_epochs'] or budget['maximum_epochs'] > 100:
            raise ValueError('Invalid synthetic epoch budget')
        settings.update(budget)
    return settings


def _seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:8], 'little')


def training_schedule(sample_ids, subject, seed, epoch):
    """Per-ID choices and partition-disjoint subsets; no ambient/arm RNG use."""
    ids = tuple(map(str, sample_ids))
    if not ids or len(set(ids)) != len(ids) or type(epoch) is not int or epoch < 1:
        raise ValueError('Training requires unique sample IDs and a positive epoch')
    masks = []
    for sample_id in ids:
        rng = np.random.default_rng(_seed('choices', subject, seed, epoch, sample_id))
        retained = int(rng.choice([22, 16, 6]))
        pattern = 'full' if retained == 22 else str(rng.choice(['random_static', 'dynamic_random']))
        masks.append(make_mask_bank(1, 4, retained, pattern, seed=int(seed), partition='train',
            subject=f'A{subject:02d}', repeat=epoch, sample_ids=[sample_id], channel_ids=CHANNEL_IDS)[0])
    order = np.random.default_rng(_seed('order', subject, seed, epoch)).permutation(len(ids))
    return np.stack(masks), order


def validation_bank(sample_ids, subject, seed):
    ids = tuple(map(str, sample_ids))
    return [(condition['name'], repeat, make_mask_bank(len(ids), 4, condition['retained'],
        condition['pattern'], seed=int(seed), partition='validation', subject=f'A{subject:02d}',
        repeat=repeat, sample_ids=ids, channel_ids=CHANNEL_IDS))
        for condition in CONDITIONS for repeat in range(condition['repeats'])]


def validate_partition(raw, labels, sample_ids):
    x = torch.as_tensor(np.array(raw, dtype=np.float32, copy=True))
    y = np.asarray(labels)
    ids = tuple(map(str, sample_ids))
    if (tuple(x.shape) != (len(ids), 22, 4, 250) or not len(ids) or
            y.shape != (len(ids),) or y.dtype.kind not in 'iu' or
            np.any((y < 0) | (y > 3)) or len(set(ids)) != len(ids) or not torch.isfinite(x).all()):
        raise ValueError('Malformed finite raw data, labels, or unique sample IDs')
    return x, torch.as_tensor(y.astype(np.int64)), ids


def predict(adapter, x, mask, batch_size=32):
    adapter.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            chunks.append(adapter(x[start:start+batch_size],
                torch.as_tensor(mask[start:start+batch_size])).softmax(-1).numpy())
    result = np.concatenate(chunks)
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite classification probabilities')
    return result


def evaluate(adapter, x, labels, banks):
    rows = []
    for condition, repeat, mask in banks:
        probabilities = predict(adapter, x, mask)
        metric = _metrics(np.asarray(labels), probabilities)
        rows.append({'condition': condition, 'repeat': repeat, 'mask': mask,
            'mask_sha256': mask_bank_digest(mask), 'probabilities': probabilities,
            'balanced_accuracy': metric['balanced_accuracy'], 'log_loss': metric['log_loss']})
    return rows


def selection_key(rows, epoch):
    degraded = [row for row in rows if row['condition'] != 'full_22']
    return (-float(np.mean([r['balanced_accuracy'] for r in degraded])),
            float(np.mean([r['log_loss'] for r in degraded])), int(epoch))


def fit_completion(backbone, strategy, completer, train, validation, *, subject, seed,
                   synthetic=False, budget=None):
    """Return initialized/selected numeric states, history and exact mask schedule."""
    if strategy not in LEARNED_STRATEGY_IDS:
        raise ValueError('Only learned completion arms receive supervised training')
    settings = training_settings(synthetic=synthetic, budget=budget)
    with scoped_cpu_rng(int(seed) + 910001):
        x, y, ids = validate_partition(train.raw, train.labels, train.sample_ids)
        vx, vy, vids = validate_partition(validation.raw, validation.labels, validation.sample_ids)
        if set(ids) & set(vids):
            raise ValueError('Training and validation sample IDs overlap')
        before = {k: v.detach().clone() for k, v in backbone.state_dict().items()}
        initial = completion_state_arrays(completer)
        adapter = CompletionAdapter(backbone, strategy, completer)
        banks = validation_bank(vids, subject, seed)
        rows = evaluate(adapter, vx, vy.numpy(), banks)
        full = rows[0]['probabilities'].copy()
        best_key = selection_key(rows, 0)
        selected = completion_state_arrays(completer)
        history = [{'epoch': 0, 'mean_degraded_balanced_accuracy': -best_key[0],
                    'mean_degraded_log_loss': best_key[1], 'updates': 0, 'loss': None}]
        optimizer = torch.optim.Adam(completer.parameters(), lr=settings['learning_rate'], weight_decay=0)
        stale = 0
        schedules, orders = [], []
        for epoch in range(1, settings['maximum_epochs'] + 1):
            mask, order = training_schedule(ids, subject, seed, epoch)
            schedules.append(mask)
            orders.append(order)
            adapter.train()
            losses, updates = [], 0
            for start in range(0, len(x), settings['batch_size']):
                indices = order[start:start+settings['batch_size']]
                xb, yb, mb = x[indices], y[indices], torch.as_tensor(mask[indices])
                if mb.all().item():
                    continue
                optimizer.zero_grad(set_to_none=True)
                logits = adapter(xb, mb)
                completed = completer(xb, mb)
                loss = (F.cross_entropy(logits, yb) + settings['reconstruction_weight'] *
                    reconstruction_loss(completed, xb, mb) + settings['anchor_weight'] * completer.anchor_penalty())
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training objective')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(completer.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
                optimizer.step()
                completer.post_step()
                losses.append(float(loss.detach()))
                updates += 1
            rows = evaluate(adapter, vx, vy.numpy(), banks)
            if not np.array_equal(full, rows[0]['probabilities']):
                raise RuntimeError('Full-input probabilities changed during completion training')
            key = selection_key(rows, epoch)
            history.append({'epoch': epoch, 'mean_degraded_balanced_accuracy': -key[0],
                'mean_degraded_log_loss': key[1], 'updates': updates,
                'loss': float(np.mean(losses)) if losses else None})
            if key < best_key:
                best_key, selected, stale = key, completion_state_arrays(completer), 0
            else:
                stale += 1
            if epoch >= settings['minimum_epochs'] and stale >= settings['patience']:
                break
        if any(not torch.equal(before[k], v) for k, v in backbone.state_dict().items()):
            raise RuntimeError('Frozen backbone tensors changed')
        if any(p.requires_grad or p.grad is not None for p in backbone.parameters()):
            raise RuntimeError('Frozen backbone acquired gradients')
        completer.load_state_dict(restore_completer(strategy, selected).state_dict())
        return {'initial': initial, 'selected': selected, 'history': history,
            'selected_epoch': best_key[2], 'settings': settings,
            'training_masks': np.stack(schedules), 'training_orders': np.stack(orders)}
