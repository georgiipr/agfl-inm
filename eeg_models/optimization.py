"""Supervised loss and gradient checks shared by the EEG study."""
import torch
from torch import nn
from torch.nn import functional as F


class ClassificationLoss(nn.Module):
    def __init__(self, weights=None, kind='cross_entropy', gamma=2.0):
        super().__init__()
        self.register_buffer('weights', weights)
        self.kind, self.gamma = kind, gamma

    def forward(self, logits, targets):
        # Class weights multiply focal loss; they must not change p_t.
        ce = F.cross_entropy(logits, targets, reduction='none')
        loss = ce if self.kind == 'cross_entropy' else (1 - torch.exp(-ce)).pow(self.gamma) * ce
        if self.weights is not None:
            loss = loss * self.weights[targets]
        return loss.mean()


def gradients_are_finite(parameters):
    """Check a single-device model, transferring only one final flag to Python.

    Converting each parameter's flag to bool separately synchronizes CUDA once
    per gradient tensor. Reduce the flags on the model's device first instead.
    Parameters without gradients (including frozen parameters) are ignored.
    """
    flags = [torch.isfinite(parameter.grad).all()
             for parameter in parameters if parameter.grad is not None]
    return not flags or bool(torch.stack(flags).all())


