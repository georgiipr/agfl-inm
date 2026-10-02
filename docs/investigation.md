# Repository investigation — 2026-10-01

## Evidence and scope

Reviewed source revision: `84ae174dd2a225816ef02f07e994526bcc335b49` (`v0.13`).
The working tree was clean before documentation work. Reviewed the tracked
inventory, supplied configuration, proposal text, existing documentation, and
the loader/split/encoder/tensor/attention/training/study/reporting code.
This is a source review and documentation build, not an empirical replication.

The research question and supplied configuration agree on EEG channel-loss
robustness. The proposal and implementation use different features, splits,
attention domains, and method coverage. Existing documentation already discloses
these adaptations; the new onboarding and concept document make them easier
to understand together.

## What was checked locally

| Check | Finding |
|---|---|
| Git inventory | Both `inm/` and `agfl/`, including tracked dataset helpers, are present |
| Task planner | `python run.py --plan` succeeds: 27 tasks, 14 heads/task, 378 total |
| Interpreter | Python 3.13.13, satisfying the project's minimum version |
| Required package discovery | `torch`, `numpy`, `scipy`, `sklearn`, and `mne` absent; `tqdm` present |
| Optional plot package | `matplotlib` absent |
| Default data directory | `../ml` absent at review time |
| Saved results | No `results/` directory in this checkout at review time |
| Tests / launcher | No checked-in test suite or Slurm launch script found |
| Document toolchain | `pdflatex`, `latexmk`, TikZ, `pdfinfo`, and `pdftoppm` available |

The task planner does not import the numerical experiment modules, so its
success does not establish readiness for training or report rebuilding.
No dependencies or dataset were downloaded, and no classifier was trained.

## Verified design properties from source

- `inm/study.py::_calibrate` computes raw and feature statistics on training
  indices only, selects an encoder using full-input validation, freezes it,
  and fits one Tucker pair on full training features for reuse across arms.
- `inm/model.py` splits raw trials into separate channel/window instances before
  convolution. Frozen encoding disables dropout and BatchNorm updates.
- `inm/tensor_attention.py` selects observations before arithmetic, rejects
  empty windows, stores factors as buffers, and solves normalized ridge systems
  in float64. Positive ridge guarantees a unique conditional core, not recovery.
- `inm/availability.py` keys masks independently of attention/representation,
  allocates nonfull subsets to one data partition, and uses a dynamic A–B–A
  schedule. For four windows this is **A, B, A, A**.
- `inm/training.py` chooses epochs by full-input validation balanced accuracy,
  breaking ties with validation log loss. Test scores are not used for selection.
- `inm/reporting.py` validates coverage/provenance, checks paired mask hashes,
  averages repeats/seeds/participants hierarchically, and suppresses incomplete
  all-participant means. Paired intervals resample participants, not trials.

These properties were read in code; they have not been executed with numerical
inputs in this environment. Source syntax and document checks are recorded below.

## Interpretation issues to keep visible

| Issue | Consequence |
|---|---|
| Within-participant, within-T-session split | Does not establish unseen-participant or cross-session transfer |
| MHA-supervised shared encoder | Feature source may favor MHA in the attention-family comparison |
| Full-channel calibration | Does not test sensors permanently absent during calibration |
| Missing placeholders participate in baseline attention | Baseline uses mask-conditioned tokens rather than exact key-padding masks |
| Core/completion pooling and sizes differ | Controls and parameter counts are needed for attribution |
| No temporal positions; average windows | Dynamic loss tests availability changes, not learned temporal order |
| Unsupervised low-rank fitting | Reconstruction quality and label preservation can diverge |
| Small spatial sequences | Performer speed improvements cannot be assumed |
| Nine participants; many comparisons | Pointwise bootstrap intervals are exploratory, without multiplicity correction |
| Offline zero-phase filtering | Not a causal, online competition evaluation |
| Head weights not persisted | Saved metrics do not provide a deployable classifier checkpoint |

## Repository and workflow observations

The existing `.gitignore` contains a broad `datasets/` rule that also matches
tracked `agfl/datasets/`. Ordinary `rg --files` omits those helpers; use
`git ls-files` or an explicitly scoped `rg --files --no-ignore agfl/datasets`.
This is a discovery pitfall, not a missing-package defect. The same ignore file
ignores root `AGENTS.md`, so that orientation file is local unless explicitly shared.

The runner records source/package/configuration identity and refuses incompatible
reuse. Documentation changes are outside its Python-source digest, but future
code/configuration/dependency changes should use a new output directory.
The README refers to an external launcher; obtain the cluster-specific script
and allocation conventions from the project owner rather than inventing them.

## Suggested collaboration priorities

1. Agree on the scientific scope and success criteria with the supervisor.
2. Provision the declared environment and dataset on the experiment machine.
3. Add meaningful numerical checks for masking, ridge solves, completion, and
   aggregation before modifying those components.
4. Inspect one complete task and its provenance before running the full matrix.
5. Examine full-input costs and participant-level paired degraded gains together.
6. Only then design changes such as subject-disjoint evaluation, temporal modeling,
   fixed spectral features, or checkpoint export, each with its own protocol identity.

## Documentation verification

The task planner and a successful Python syntax parse over all 43 tracked `.py`
files are the local code checks. The 11-page concept PDF compiles with
`latexmk`/`pdflatex` without LaTeX warnings, unresolved references, or overfull
boxes. All pages were rendered for visual review; local Markdown links and
`git diff --check` pass. Full training, numerical correctness, CUDA behavior, and saved
study reporting remain unverified here because the scientific environment and
recordings are unavailable.

## Primary background sources

Dataset parameters were checked against the [official BCI IV 2a description](https://www.bbci.de/competition/iv/desc_2a.pdf).
Method background is linked to [EEGNet](https://arxiv.org/abs/1611.08024),
[Attention Is All You Need](https://arxiv.org/abs/1706.03762),
[Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794), and
[Kolda and Bader's tensor survey](https://www.kolda.net/publication/koba09/).
These sources explain the methods; they do not validate results for this checkout.
