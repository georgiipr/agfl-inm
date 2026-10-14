"""Recompute persisted predictions and validate the immutable fit seal."""
from .study import mode_audit

def audit(output, device="cpu", synthetic=False, assets=None):
    return mode_audit(output, device, synthetic, assets)
