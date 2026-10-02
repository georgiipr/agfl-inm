# Session 04 — A compact CNN with early spatial filtering

Read `inm/model.py` and `agfl/models/eegnet/backbone.py` to understand why the existing local path is not canonical EEGNet. Reference architecture: https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py (record any verified implementation differences).

Implement EEGNetClassifier in the new package using CONTRACT.md's concrete dimensions and operations. Standard and mask-conditioned variants share convolutional layers; the conditioned variant concatenates the original 88 Boolean mask flags to the flattened learned features before the classifier. Return logits, not softmax. Include clip_weights and serializable constructor settings.

Select out unavailable raw channel/window samples with torch.where BEFORE temporal convolution, spatial convolution, BatchNorm, or pooling. Validate shapes, dtype, devices, finite observed inputs and nonempty windows. All-observed mask and omitted mask must agree. Convolution may mix observed windows after selection. No cache of mixed full-input activations may enter degraded evaluation.

Use original EEGNet-style spatial depthwise kernel across all electrodes before ELU and pooling, pool 4 then 8, with multiple temporal positions retained. Compute feature shape without a training-mode dummy pass. Keep model small; no attention, Tucker, skip branches, temporal head or new framework.

Acceptance: CPU forward/backward, four logits, finite gradients, nonzero gradient through spatial filter; full mask equivalence; hidden NaN/random-value invariance under fixed RNG/eval; zero influence of hidden samples on gradients; state_dict roundtrip; invalid masks rejected. Confirm EEGNet spatial kernel is [22,1], not [1,1].

Handoff: model constructor and feature-sequence interface useful for optional temporal ablation. Explain original-reference deviations clearly.
