# Fixed class-information probes

`inm.encoder_audit.probes.fit_probes(sources, cfg, directory)` fits four
predeclared CPU logistic probes from the verified legacy cache. It reads only
`sources.legacy.train` and `.validation`, their labels and stable IDs, and the
saved Tucker factor state. It does not access test rows, the spatial EEGNet
activation cache, or the absent historical encoder/head predictions.

The three fixed views are the row-major flattening of `[22,4,32]`, the
window-axis mean flattened from `[22,32]`, and the frozen-factor full-input
Tucker core flattened from `[4,4,4]`. Each view gets its own training-fitted
`StandardScaler` and L2 multinomial logistic regression (`C=1`, L-BFGS,
`max_iter=1000`, `tol=1e-4`, `random_state=0`). There is no tuning, validation
refit, class weighting, or seed selection. The fourth fit uses the ordered
features with training labels permuted once by NumPy seed `seed + 700001`;
training and validation scores are both computed against the original labels.
Validation labels are never permuted.

Each `.npz` probe file contains numeric scaler/classifier state, class order,
the feature layout and task/input/split identities, convergence details, stable
train/validation IDs and labels, and per-sample probabilities. Files are loaded
with `allow_pickle=False`. `load_probe(path).predict_proba(x)` accepts a feature
matrix; the ordered probe also accepts normalized `[N,22,4,32]` features, which
lets the reconstruction audit apply the already fitted scaler and classifier
to completed features. It never fits on those completed values.

## Interpretation

Strong held-out performance supports the narrow claim that class information is
accessible to this fixed linear probe in that representation and split. A weak
probe does not show that all label information is absent: a nonlinear decoder,
another fixed feature map, or a different task sample could behave differently.
Window averaging also changes the input dimension (2816 to 704), and the core
has its own 64-dimensional geometry. Their scores are descriptive comparisons;
they do not isolate temporal order or prove that compression destroyed all
discriminative information.

The shuffled-label fit is a single sanity control. Its finite-sample score is
not a p-value or a required exact chance score. These are exploratory
train/validation findings on reused validation data, not independent
confirmation. The audit reports accuracy, balanced accuracy, log loss,
per-class recall, and confusion matrices without using validation or test
results to change a diagnostic decision.
