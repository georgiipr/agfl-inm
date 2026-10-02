"""Matched supervised fitting; checkpoint decisions use validation only."""
from __future__ import annotations
import copy
import math
import time
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from agfl.optimization import ClassificationLoss, gradients_are_finite
from agfl.metrics import classification_metrics
from .protocol import write_json


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
        logits = model(part, available)
        if logits.shape != (len(part), 4) or not bool(torch.isfinite(logits).all()):
            raise FloatingPointError('Invalid or nonfinite four-class logits')
        probabilities.append(logits.softmax(-1).cpu().numpy())
    return np.concatenate(probabilities)


def fit(model, train_x, train_y, val_x, val_y, train_masks, val_mask, options,
        *, seed, device, history_path, description, validation_masks=None):
    """train_masks(epoch) is independent of representation and attention identity."""
    model.to(device)
    counts = np.bincount(np.asarray(train_y), minlength=4)
    if np.any(counts == 0):
        raise ValueError('Training partition lacks a class')
    weights = (torch.tensor(counts.sum() / (4 * counts), dtype=torch.float32, device=device)
               if options['class_weights'] else None)
    criterion = ClassificationLoss(weights, options['loss'], options['focal_gamma'])
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=options['learning_rate'], weight_decay=options['weight_decay'])
    epochs, warmup = options['epochs'], options['warmup_epochs']
    def factor(epoch):
        if epoch < warmup:
            return 0.1 + 0.9 * epoch / max(1, warmup)
        return (1 + math.cos(math.pi * min(1., (epoch - warmup) / max(1, epochs - warmup)))) / 2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    history, best_key, best_state, best_epoch = [], None, None, None
    started = time.monotonic()
    train_y = torch.as_tensor(train_y, dtype=torch.long)
    with tqdm(range(epochs), desc=description, unit='epoch', dynamic_ncols=True,
              mininterval=5., disable=None) as progress:
        for epoch in progress:
            model.train()
            mask = torch.as_tensor(train_masks(epoch), dtype=torch.bool)
            order = np.random.default_rng(seed + 100003 * epoch).permutation(len(train_x))
            loss_sum, correct = 0., 0
            for start in range(0, len(order), options['batch_size']):
                indices = order[start:start + options['batch_size']]
                x, y = train_x[indices].to(device), train_y[indices].to(device)
                available = mask[indices].to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x, available)
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
                loss_sum += float(loss.detach()) * len(indices)
                correct += int((logits.argmax(-1) == y).sum())
            banks = validation_masks or [('full', val_mask)]
            grouped = defaultdict(list)
            for bank_name, bank_mask in banks:
                probabilities = predict(model, val_x, bank_mask, options['batch_size'], device)
                bank_score = metrics(np.asarray(val_y), probabilities)
                bank_score['log_loss'] = float(-np.log(np.clip(
                    probabilities[np.arange(len(val_y)), val_y], 1e-12, 1)).mean())
                grouped[bank_name].append(bank_score)
            # Repeats are averaged within each declared mask condition, then
            # conditions receive equal weight in the robustness score.
            evidence = {name: {key: float(np.mean([row[key] for row in rows]))
                               for key in ('balanced_accuracy', 'log_loss')}
                        for name, rows in grouped.items()}
            validation = (dict(evidence['full']) if len(evidence) == 1 else {
                'balanced_accuracy': float(np.mean([row['balanced_accuracy'] for row in evidence.values()])),
                'log_loss': float(np.mean([row['log_loss'] for row in evidence.values()]))})
            key = validation['balanced_accuracy'], -validation['log_loss']
            if best_key is None or key > best_key:
                best_key, best_epoch = key, epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            history.append({'epoch': epoch + 1, 'train_loss': loss_sum / len(train_x),
                            'train_accuracy': correct / len(train_x), 'validation': validation,
                            'validation_banks': evidence,
                            'learning_rate': optimizer.param_groups[0]['lr']})
            write_json(history_path, history)
            progress.set_postfix(loss=f'{loss_sum / len(train_x):.4f}', val_bacc=f"{validation['balanced_accuracy']:.1%}")
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
            'selection': ('validation_balanced_accuracy_then_log_loss' if not validation_masks else
                          'mean_validation_balanced_accuracy_then_mean_log_loss'),
            'elapsed_seconds': time.monotonic() - started}
