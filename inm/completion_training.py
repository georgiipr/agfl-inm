"""Fixed-budget learned covariance calibration on training trials only.

The selected full-channel backbone remains frozen, including BatchNorm and
dropout. No validation or test arrays are accepted by this fitting function.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from eeg_models.optimization import ClassificationLoss
from .availability import make_mask_bank, mask_bank_digest
from .protocol import n_windows, write_json


def _seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:8], 'little')


def training_schedule(sample_ids, subject, seed, epoch, cfg):
    """Deterministic choices by original trial ID; subsets belong only to train."""
    options = cfg['covariance']['learned']
    masks = []
    for sample_id in sample_ids:
        rng = np.random.default_rng(_seed('covariance-choices-v1', subject, seed, epoch, sample_id))
        retained = int(rng.choice(options['training_retained_counts']))
        pattern = 'full' if retained == 22 else str(rng.choice(options['training_patterns']))
        masks.append(make_mask_bank(1, n_windows(cfg), retained, pattern,
                     seed=seed, partition='train', subject=f'A{subject:02d}',
                     repeat=epoch, sample_ids=[sample_id])[0])
    order = np.random.default_rng(_seed('covariance-order-v1', subject, seed, epoch)).permutation(len(sample_ids))
    return np.stack(masks), order


def fit_completion(model, train_x, train_y, sample_ids, observed_input, cfg,
                   *, subject, seed, device, directory, partition='train'):
    """Fit only covariance coordinates; retain the declared final epoch.

    observed_input(mask) must apply the shared outage-aware preprocessing to
    these training indices. Hidden full-input targets enter reconstruction loss
    only; they never enter the model's observed-input computation.
    """
    if partition != 'train':
        raise ValueError('Learned covariance accepts training trials only')
    options = cfg['covariance']['learned']
    if options['selection'] != 'fixed_final_epoch':
        raise ValueError('Degraded validation must not select completion weights')
    sample_ids = tuple(map(str, sample_ids))
    if (len(sample_ids) != len(train_x) or len(set(sample_ids)) != len(sample_ids)
            or len(train_x) != len(train_y) or not len(train_x)):
        raise ValueError('Training signals, labels and unique sample IDs must match')
    completer = model.signal_input.completion
    if completer is None or not completer.learned or not bool(completer.fitted):
        raise ValueError('Expected training-initialized learned covariance')
    model.to(device).eval()
    prefix = 'signal_input.completion.'
    backbone_before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                       if not key.startswith(prefix)}
    flags = [(parameter, parameter.requires_grad) for name, parameter in model.named_parameters()
             if not name.startswith(prefix)]
    for parameter, _ in flags:
        parameter.requires_grad_(False)
        parameter.grad = None
    completer.free.requires_grad_(True)
    with torch.no_grad():
        full_before = model(train_x[:min(8, len(train_x))].to(device)).detach().cpu().clone()
    counts = np.bincount(np.asarray(train_y), minlength=4)
    if np.any(counts == 0):
        raise ValueError('Training partition lacks a class')
    weights = (torch.tensor(counts.sum() / (4 * counts), dtype=torch.float32, device=device)
               if cfg['training']['class_weights'] else None)
    criterion = ClassificationLoss(weights, cfg['training']['loss'], cfg['training']['focal_gamma'])
    optimizer = torch.optim.Adam([completer.free], lr=options['learning_rate'], weight_decay=0.)
    target_y = torch.as_tensor(train_y, dtype=torch.long)
    history, schedules, orders = [], [], []
    directory = Path(directory)
    try:
        with tqdm(range(1, options['epochs'] + 1), desc='learned covariance', unit='epoch',
                  dynamic_ncols=True, mininterval=5., disable=None) as progress:
            for epoch in progress:
                masks, order = training_schedule(sample_ids, subject, seed, epoch, cfg)
                available = torch.from_numpy(masks)
                observed = observed_input(available)
                schedules.append(masks)
                orders.append(order)
                losses, classification, reconstruction, anchors = [], [], [], []
                for start in range(0, len(order), options['batch_size']):
                    indices = order[start:start + options['batch_size']]
                    mask = available[indices].to(device)
                    if bool(mask.all()):
                        continue
                    x = observed[indices].to(device)
                    y = target_y[indices].to(device)
                    truth = train_x[indices].to(device).reshape(
                        -1, 22, n_windows(cfg), cfg['availability']['window_samples'])
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(x, mask)
                    completed = completer.complete(x.reshape_as(truth), mask)
                    hidden = ~mask[..., None].expand_as(truth)
                    reconstruction_loss = (completed[hidden] - truth[hidden]).square().mean()
                    classification_loss = criterion(logits, y)
                    anchor_loss = completer.anchor_penalty()
                    loss = (classification_loss + options['reconstruction_weight'] * reconstruction_loss
                            + options['anchor_weight'] * anchor_loss)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError(f'Nonfinite covariance objective at epoch {epoch}')
                    loss.backward()
                    if completer.free.grad is None:
                        raise RuntimeError('Covariance objective has no parameter gradient')
                    torch.nn.utils.clip_grad_norm_([completer.free], options['gradient_clip'],
                                                  error_if_nonfinite=True)
                    optimizer.step()
                    completer.covariance()
                    losses.append(float(loss.detach()))
                    classification.append(float(classification_loss.detach()))
                    reconstruction.append(float(reconstruction_loss.detach()))
                    anchors.append(float(anchor_loss.detach()))
                if not losses:
                    raise RuntimeError('No degraded training batches in covariance calibration epoch')
                history.append({'epoch': epoch, 'updates': len(losses), 'loss': float(np.mean(losses)),
                                'classification_loss': float(np.mean(classification)),
                                'hidden_training_mse': float(np.mean(reconstruction)),
                                'anchor_loss': float(np.mean(anchors)),
                                'mask_sha256': mask_bank_digest(masks),
                                'observed_input_sha256': hashlib.sha256(
                                    memoryview(observed.numpy()).cast('B')).hexdigest(),
                                'order_sha256': hashlib.sha256(order.tobytes()).hexdigest()})
                write_json(directory / 'completion_history.json', history)
                progress.set_postfix(loss=f'{np.mean(losses):.4f}')
        with torch.no_grad():
            full_after = model(train_x[:min(8, len(train_x))].to(device)).detach().cpu()
        unchanged = all(torch.equal(backbone_before[key], value.detach().cpu())
                        for key, value in model.state_dict().items() if not key.startswith(prefix))
        if (not unchanged or not torch.equal(full_before, full_after)
                or any(parameter.grad is not None for parameter, _ in flags)):
            raise RuntimeError('Covariance calibration changed the frozen backbone or full-input route')
        schedule_path = directory / 'completion_training.npz'
        temporary = schedule_path.with_suffix('.npz.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, masks=np.stack(schedules), orders=np.stack(orders),
                                sample_ids=np.asarray(sample_ids), labels=np.asarray(train_y))
        temporary.replace(schedule_path)
        record = {'partition': 'train', 'selection': 'fixed_final_epoch',
                  'epochs_trained': len(history), 'selected_epoch': options['epochs'],
                  'settings': options, 'optimizer': 'Adam', 'weight_decay': 0., 'scheduler': None,
                  'classification_loss': cfg['training']['loss'],
                  'class_weights': None if weights is None else weights.cpu().tolist(),
                  'training_trials': len(train_x), 'training_sample_ids': list(sample_ids),
                  'schedule_sha256': hashlib.sha256(schedule_path.read_bytes()).hexdigest(),
                  'history_sha256': hashlib.sha256((directory / 'completion_history.json').read_bytes()).hexdigest(),
                  'backbone_unchanged': unchanged, 'full_input_unchanged': True,
                  'validation_used': False, 'test_used': False}
        write_json(directory / 'completion_fit.json', record)
        return record
    finally:
        for parameter, required in flags:
            parameter.requires_grad_(required)
        completer.free.requires_grad_(False)
        completer.free.grad = None
        model.eval()
