"""Matched supervised fitting; checkpoint decisions use validation only."""
from __future__ import annotations
import copy
import math
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from eeg_models.optimization import ClassificationLoss, gradients_are_finite
from eeg_models.metrics import classification_metrics
from .protocol import CHECKPOINT_SELECTION, write_json


def metrics(targets, probabilities):
    result = classification_metrics(targets, probabilities)
    recalls = [row['recall'] for row in result['per_class']]
    if any(value is None for value in recalls):
        raise ValueError('Balanced accuracy requires all four classes in this declared partition.')
    result['balanced_accuracy'] = float(np.mean(recalls))
    result['f1_macro'] = result['f1']
    return result


@torch.no_grad()
def predict(model, x, mask, batch_size, device):
    model.eval()
    probabilities = []
    for start in range(0, len(x), batch_size):
        part = x[start:start + batch_size].to(device)
        available = mask[start:start + batch_size].to(device)
        logits = model(part) if bool(available.all()) else model(part, available)
        if logits.shape != (len(part), 4) or not bool(torch.isfinite(logits).all()):
            raise FloatingPointError('Invalid or nonfinite four-class logits')
        probabilities.append(logits.softmax(-1).cpu().numpy())
    return np.concatenate(probabilities)


@torch.no_grad()
def _validation_metrics(model, x, y, batch_size, device, criterion):
    """Use the training loss/weights on full-channel validation, as in the baseline."""
    model.eval()
    probabilities, loss_sum = [], 0.
    targets = torch.as_tensor(y, dtype=torch.long)
    for start in range(0, len(x), batch_size):
        part = x[start:start + batch_size].to(device)
        logits = model(part)
        if logits.shape != (len(part), 4) or not bool(torch.isfinite(logits).all()):
            raise FloatingPointError('Invalid or nonfinite full-channel validation logits')
        loss = criterion(logits, targets[start:start + batch_size].to(device))
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite full-channel validation loss')
        loss_sum += float(loss) * len(part)
        probabilities.append(logits.softmax(-1).cpu().numpy())
    probabilities = np.concatenate(probabilities)
    result = metrics(np.asarray(y), probabilities)
    result['loss'] = loss_sum / len(x)
    # Retain unweighted log loss as a descriptive metric, not a tie breaker.
    result['log_loss'] = float(-np.log(np.clip(
        probabilities[np.arange(len(y)), np.asarray(y)], 1e-12, 1)).mean())
    return result


def fit(model, train_x, train_y, val_x, val_y, options,
        *, seed, device, history_path, description):
    """Fit the entire model on 22 channels; only full validation selects epochs."""
    model.to(device)
    counts = np.bincount(np.asarray(train_y), minlength=4)
    if np.any(counts == 0):
        raise ValueError('Training partition lacks a class')
    weights = (torch.tensor(counts.sum() / (4 * counts), dtype=torch.float32, device=device)
               if options['class_weights'] else None)
    criterion = ClassificationLoss(weights, options['loss'], options['focal_gamma'])
    if options['checkpoint_criterion'] != 'loss' or options['checkpoint_tiebreaker'] != 'none':
        raise ValueError('This recipe selects the lowest full-channel validation loss without a tie breaker')
    # The original seeded PyTorch RandomSampler advances one generator across
    # epochs. DataLoader also consumes its iterator seed from that generator.
    loader = DataLoader(
        TensorDataset(train_x, torch.as_tensor(train_y, dtype=torch.long)),
        batch_size=options['batch_size'], shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=torch.device(device).type == 'cuda', drop_last=False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=options['learning_rate'], weight_decay=options['weight_decay'])
    epochs, warmup = options['epochs'], options['warmup_epochs']
    def factor(epoch):
        if epoch < warmup:
            return 0.1 + 0.9 * epoch / max(1, warmup)
        return (1 + math.cos(math.pi * min(1., (epoch - warmup) / max(1, epochs - warmup)))) / 2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    history, best_loss, best_state, best_epoch = [], None, None, None
    started = time.monotonic()
    with tqdm(range(epochs), desc=description, unit='epoch', dynamic_ncols=True,
              mininterval=5., disable=None) as progress:
        for epoch in progress:
            model.train()
            loss_sum, correct = 0., 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = criterion(logits, y)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f'Nonfinite loss at epoch {epoch + 1}')
                loss.backward()
                if not gradients_are_finite(model.parameters()):
                    raise FloatingPointError(f'Nonfinite gradients at epoch {epoch + 1}')
                if options['gradient_clip'] is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), options['gradient_clip'], error_if_nonfinite=True)
                optimizer.step()
                if hasattr(model, 'clip_weights'):
                    model.clip_weights()
                loss_sum += float(loss.detach()) * len(x)
                correct += int((logits.argmax(-1) == y).sum())
            validation = _validation_metrics(model, val_x, val_y,
                                             options['batch_size'], device, criterion)
            # Strict loss improvement only: exact ties retain the earlier epoch.
            if best_loss is None or validation['loss'] < best_loss:
                best_loss, best_epoch = validation['loss'], epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            history.append({'epoch': epoch + 1, 'train_loss': loss_sum / len(train_x),
                            'train_accuracy': correct / len(train_x), 'validation': validation,
                            'learning_rate': optimizer.param_groups[0]['lr']})
            write_json(history_path, history)
            progress.set_postfix(loss=f'{loss_sum / len(train_x):.4f}',
                                 val_loss=f"{validation['loss']:.4f}",
                                 val_bacc=f"{validation['balanced_accuracy']:.1%}")
            scheduler.step()
            if (options['patience'] and epoch + 1 >= options['minimum_epochs']
                    and epoch + 1 - best_epoch >= options['patience']):
                break
    if best_state is None:
        raise RuntimeError('No validation-selected weights')
    model.load_state_dict(best_state)
    model.eval()
    return {'best_epoch': best_epoch, 'epochs_trained': len(history),
            'selected_validation': copy.deepcopy(history[best_epoch - 1]['validation']),
            'selection': CHECKPOINT_SELECTION,
            'criterion': options['checkpoint_criterion'], 'tiebreaker': options['checkpoint_tiebreaker'],
            'class_weights': None if weights is None else weights.cpu().tolist(),
            'batch_order': {'sampler': 'pytorch_random_sampler', 'seed': seed, 'num_workers': 0},
            'cpu_threads': torch.get_num_threads(),
            'elapsed_seconds': time.monotonic() - started}
