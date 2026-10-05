# Clean selected-checkpoint replay

`inm.encoder_audit.checkpoints.audit_checkpoints(sources, aligned, cfg,
device="cpu")` restores the three saved neural baseline arms using each
checkpoint's constructor and strict state dictionary. It verifies historical
baseline replay-source identities, selected epoch, subject/seed/arm identity,
class order, channel order, and train-fitted raw normalization before scoring.
The canonical source is joined to persisted train and validation views by stable
sample IDs; labels are checked against those IDs. Test rows are not normalized,
scored, summarized, or included in returned objects.

Each arm has this stable result shape:

```json
{
  "selected_epoch": 12,
  "class_order": ["left_hand", "right_hand", "feet", "tongue"],
  "channel_order": ["Fz", "FC3", "..."],
  "metrics": {
    "train": {
      "balanced_accuracy": 0.8,
      "accuracy": 0.8,
      "f1": 0.8,
      "log_loss": 0.5,
      "confusion_matrix": [],
      "per_class": []
    },
    "validation": {}
  },
  "train_minus_validation_gaps": {
    "balanced_accuracy": 0.2,
    "accuracy": 0.2,
    "f1": 0.2,
    "log_loss": -0.1
  },
  "selected_history_max_abs_error": {},
  "reload_probability_max_abs_error": {},
  "model_state_unchanged": true,
  "input_artifact_sha256_unchanged": {},
  "optimization_time_metrics": {
    "masked_train_loss": 0.7,
    "masked_train_accuracy": 0.6
  },
  "optimization_time_label": "training_mode_with_epoch_masks_and_dropout"
}
```

`balanced_accuracy` is the mean of four class recalls. `f1` is macro-F1;
`per_class` includes recall, precision, support, and F1, and the confusion matrix
uses true classes as rows and predicted classes as columns. Gaps are
train-minus-validation. History clean metrics must reproduce within the
configured probability tolerance (including the scalar metrics) and confusion
counts must match exactly. Checkpoint/history file hashes and model parameter
and buffer values must remain unchanged. Reloaded predictions are independently
compared with the first restored instance.

The saved masked training loss and training-mode accuracy describe optimization
with masks and dropout enabled. They are not clean train metrics. The legacy
encoder selection score remains historical metadata: the saved legacy
calibration lacks the pretraining MHA classifier head, so its clean classifier
cannot be replayed. Session 05's linear probes are newly identified diagnostic
models and are not substitutes for that absent classifier.

## Checkpoint format policy

The artifact reader verifies recorded SHA-256 values before loading and uses
PyTorch's `weights_only=True` loader. The checkpoint must be the declared
`agfl-baseline-checkpoint-v1` format with the expected EEGNet model type,
primitive constructor settings, and a strict state dictionary. Unsupported,
legacy pickle-dependent, corrupt, or incompatible payloads fail explicitly;
there is no permissive fallback loader. This policy applies to trusted local
study artifacts and does not authorize loading arbitrary checkpoints.

Synthetic CPU coverage lives in `tests/encoder_audit/test_checkpoints.py` and
`test_alignment.py`. It exercises all three arms, frozen evaluation state,
reload equality, wrong epoch/class order, history disagreement, reordered or
mislabeled identities, nonfinite evaluated data, test-row poisoning, and
historical source mismatch. It does not establish readiness of the real saved
cohort checkpoints.
