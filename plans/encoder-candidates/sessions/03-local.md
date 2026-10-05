# Session 03 Implement matched local encoders

Bounded paths:

- `inm/encoder_candidates/local.py`
- `inm/encoder_candidates/models.py`
- `tests/encoder_candidates/test_local.py`

Implement LocalClassifier(kind) for local_control and local_power and the lazy
arm factory. Wrap the existing control and shared head without editing them.
Implement the four Conv1d log-variance branches exactly as contracted. Every
constructor value, variance convention, padding and feature order must survive
checkpoint reconstruction. Build identical shared head tensors using isolated
RNG initialization independent of encoder construction. Unsupported future model
imports may fail clearly until their sessions land; no fake successful models.

Expose forward_features [B,22,4,32], logits [B,4], constructor_settings,
clip_weights and an explicit freeze_features operation. Freeze must survive a
parent train() call until explicitly unfreezing for a NEW training operation.

Acceptance: compare log-variance against a direct reference, amplitude scaling
of a known filter response, finite gradients and variance epsilon on constant
inputs. Check observed-channel/window perturbation locality in eval mode,
hidden NaNs, mask rejection, full-mask/None equality, batch-size invariance,
state reload, and identical shared-head initialization. No real fits.
