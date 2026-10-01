"""Reproducible observation masks for the BCI IV 2a electrode universe.

Masks have shape [trial, original electrode, non-overlapping window].  They
never renumber surviving electrodes.  No labels or feature values are used.
"""

from __future__ import annotations

import hashlib
import json
from numbers import Integral
from typing import Iterable

import numpy as np


CHANNEL_IDS = (
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
    "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
)
# Fixed schematic 10-20 top-view locations: x left/right, y anterior/posterior.
# These are protocol coordinates, not measured locations or brain connectivity.
CHANNEL_COORDINATES = {
    "Fz": (0.0, 2.0),
    "FC3": (-2.0, 1.0), "FC1": (-1.0, 1.0), "FCz": (0.0, 1.0),
    "FC2": (1.0, 1.0), "FC4": (2.0, 1.0),
    "C5": (-3.0, 0.0), "C3": (-2.0, 0.0), "C1": (-1.0, 0.0),
    "Cz": (0.0, 0.0), "C2": (1.0, 0.0), "C4": (2.0, 0.0), "C6": (3.0, 0.0),
    "CP3": (-2.0, -1.0), "CP1": (-1.0, -1.0), "CPz": (0.0, -1.0),
    "CP2": (1.0, -1.0), "CP4": (2.0, -1.0),
    "P1": (-1.0, -2.0), "Pz": (0.0, -2.0), "P2": (1.0, -2.0),
    "POz": (0.0, -3.0),
}
RETAINED_COUNTS = (22, 16, 11, 6)
TRAINING_RETAINED_COUNTS = (22, 16, 6)
DEGRADED_PATTERNS = (
    "random_static", "spatial_static", "dynamic_random", "dynamic_spatial",
)
PARTITION_BUCKETS = {"train": 0, "validation": 1, "test": 2}
MASK_PROTOCOL_VERSION = "availability-v1"


def _partition_name(partition: str) -> str:
    name = {"val": "validation", "valid": "validation", "training": "train"}.get(
        partition, partition
    )
    if name not in PARTITION_BUCKETS:
        raise ValueError("partition must be train, validation, or test")
    return name


def _stable_seed(*parts: object) -> int:
    encoded = json.dumps(
        [MASK_PROTOCOL_VERSION, *parts], separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little")


def _channel_ids(channel_ids: Iterable[str]) -> tuple[str, ...]:
    ids = tuple(channel_ids)
    if len(ids) != len(CHANNEL_IDS) or set(ids) != set(CHANNEL_IDS):
        raise ValueError("channel_ids must contain each of the 22 BCI IV 2a EEG IDs once")
    return ids


def _sample_ids(n_trials: int, sample_ids: Iterable[object] | None) -> tuple[str, ...]:
    if not isinstance(n_trials, Integral) or isinstance(n_trials, bool) or n_trials < 0:
        raise ValueError("n_trials must be a nonnegative integer")
    ids = tuple(str(x) for x in (range(n_trials) if sample_ids is None else sample_ids))
    if len(ids) != n_trials:
        raise ValueError("sample_ids length must equal n_trials")
    if len(set(ids)) != len(ids):
        raise ValueError("sample_ids must identify distinct trials")
    return ids


def _validate_windows(n_windows: int, dynamic: bool = False) -> None:
    if not isinstance(n_windows, Integral) or isinstance(n_windows, bool) or n_windows < 1:
        raise ValueError("n_windows must be a positive integer")
    if dynamic and n_windows < 3:
        raise ValueError("dynamic availability needs at least 3 non-overlapping windows")


def subset_partition(observed: np.ndarray, channel_ids: Iterable[str] = CHANNEL_IDS) -> str:
    """Canonical allocation of a nonfull channel subset to one data partition.

    Hashing original IDs (rather than a trial/attention seed) guarantees that a
    particular subset cannot occur in two partitions, even across loss patterns.
    """
    ids = _channel_ids(channel_ids)
    observed = np.asarray(observed, dtype=bool)
    if observed.shape != (len(ids),):
        raise ValueError("observed must be one boolean per original electrode")
    if observed.all():
        return "shared_full"
    if not observed.any():
        raise ValueError("an all-missing window is forbidden")
    canonical_positions = {name: i for i, name in enumerate(CHANNEL_IDS)}
    bitset = sum(1 << canonical_positions[name] for name, keep in zip(ids, observed) if keep)
    bucket = _stable_seed("subset-partition", bitset) % 3
    return ("train", "validation", "test")[bucket]


def _draw_subset(
    rng: np.random.Generator,
    retained: int,
    spatial: bool,
    partition: str,
    channel_ids: tuple[str, ...],
    different_from: np.ndarray | None = None,
) -> np.ndarray:
    """Rejection sampling preserves counts while enforcing partition disjointness."""
    channels = len(channel_ids)
    coordinates = np.asarray([CHANNEL_COORDINATES[name] for name in channel_ids])
    for _ in range(4096):
        keep = np.ones(channels, dtype=bool)
        if spatial:
            # Drop the C-k electrodes nearest a single uniformly sampled center.
            # Extended bounds include peripheral losses. Tiny jitter only breaks
            # equal-distance ties in this schematic coordinate system.
            center = rng.uniform((-4.0, -4.0), (4.0, 3.0))
            distances = ((coordinates - center) ** 2).sum(axis=1)
            distances = distances + rng.uniform(0.0, 1e-10, size=channels)
            missing = np.argsort(distances, kind="stable")[:channels - retained]
            keep[missing] = False
        else:
            keep[:] = False
            keep[rng.choice(channels, size=retained, replace=False)] = True
        if different_from is not None and np.array_equal(keep, different_from):
            continue
        if subset_partition(keep, channel_ids) == partition:
            return keep
    raise RuntimeError(
        f"Unable to sample a distinct {retained}-channel "
        f"{'spatial' if spatial else 'random'} subset for {partition}; "
        "do not silently fall back to a different partition or pattern"
    )


def _trial_mask(
    n_windows: int,
    retained: int,
    pattern: str,
    rng: np.random.Generator,
    partition: str,
    channel_ids: tuple[str, ...],
) -> np.ndarray:
    if retained == 22:
        return np.ones((22, n_windows), dtype=bool)
    dynamic = pattern.startswith("dynamic_")
    spatial = pattern in ("spatial_static", "dynamic_spatial")
    first = _draw_subset(rng, retained, spatial, partition, channel_ids)
    mask = np.repeat(first[:, None], n_windows, axis=1)
    if dynamic:
        middle = _draw_subset(
            rng, retained, spatial, partition, channel_ids, different_from=first
        )
        # An A -> B -> A schedule gives disappearing AND returning channels.
        # A and B each contain exactly k observed electrodes; no averaging trick.
        start = n_windows // 3
        stop = (2 * n_windows) // 3
        mask[:, start:stop] = middle[:, None]
    return mask


def make_mask_bank(
    n_trials: int,
    n_windows: int,
    retained: int,
    pattern: str,
    seed: int = 0,
    partition: str = "test",
    subject: str = "",
    repeat: int = 0,
    sample_ids: Iterable[object] | None = None,
    channel_ids: Iterable[str] = CHANNEL_IDS,
) -> np.ndarray:
    """Create a deterministic evaluation/fit bank, shape [N,22,P], dtype bool.

    Seed keys deliberately exclude attention name and tensor representation.
    Supply stable original-trial IDs to remain invariant to trial reordering.
    Nonfull subsets are disjoint across train/validation/test partitions.
    """
    partition = _partition_name(partition)
    ids = _channel_ids(channel_ids)
    trials = _sample_ids(n_trials, sample_ids)
    if retained not in RETAINED_COUNTS:
        raise ValueError(f"retained must be one of {RETAINED_COUNTS}")
    if pattern not in ("full", *DEGRADED_PATTERNS):
        raise ValueError(f"unknown availability pattern: {pattern}")
    if (retained == 22) != (pattern == "full"):
        raise ValueError("22 channels uses pattern='full'; degraded patterns need 16, 11, or 6")
    _validate_windows(n_windows, pattern.startswith("dynamic_"))
    bank = np.empty((n_trials, 22, n_windows), dtype=bool)
    for row, trial in enumerate(trials):
        rng = np.random.default_rng(_stable_seed(
            int(seed), str(subject), partition, pattern, int(retained), int(repeat), trial
        ))
        bank[row] = _trial_mask(n_windows, retained, pattern, rng, partition, ids)
    return bank


def training_mask_bank(
    n_trials: int,
    n_windows: int,
    regime: str = "mixed",
    seed: int = 0,
    epoch: int = 0,
    subject: str = "",
    sample_ids: Iterable[object] | None = None,
    channel_ids: Iterable[str] = CHANNEL_IDS,
) -> np.ndarray:
    """Per-epoch masks shared by all comparison arms, with no 11-channel examples.

    Mixed: retain 22/16/6 uniformly per trial. At 16 or 6, choose random_static
    versus dynamic_random uniformly. Spatial patterns are evaluation-only.
    """
    if regime not in ("full", "mixed"):
        raise ValueError("training regime must be 'full' or 'mixed'")
    ids = _channel_ids(channel_ids)
    trials = _sample_ids(n_trials, sample_ids)
    _validate_windows(n_windows, dynamic=regime == "mixed")
    if regime == "full":
        return np.ones((n_trials, 22, n_windows), dtype=bool)
    bank = np.empty((n_trials, 22, n_windows), dtype=bool)
    for row, trial in enumerate(trials):
        rng = np.random.default_rng(_stable_seed(
            int(seed), str(subject), "train", "epoch", int(epoch), trial
        ))
        retained = TRAINING_RETAINED_COUNTS[int(rng.integers(3))]
        pattern = "full" if retained == 22 else (
            "random_static" if int(rng.integers(2)) == 0 else "dynamic_random"
        )
        bank[row] = _trial_mask(n_windows, retained, pattern, rng, "train", ids)
    return bank


def evaluation_scenarios() -> list[dict[str, object]]:
    """One full condition plus four patterns at each of 16/11/6 channels."""
    return [
        {"name": "full_22", "retained": 22, "pattern": "full"},
        *[
            {"name": f"{pattern}_{retained}", "retained": retained, "pattern": pattern}
            for retained in (16, 11, 6)
            for pattern in DEGRADED_PATTERNS
        ],
    ]


def mask_bank_digest(mask: np.ndarray) -> str:
    """Digest recorded alongside results to verify paired masks across arms."""
    mask = np.asarray(mask, dtype=bool)
    header = json.dumps({"shape": list(mask.shape), "protocol": MASK_PROTOCOL_VERSION})
    digest = hashlib.sha256(header.encode("utf-8"))
    digest.update(np.packbits(mask, bitorder="little").tobytes())
    return digest.hexdigest()


def availability_metadata() -> dict[str, object]:
    return {
        "protocol": MASK_PROTOCOL_VERSION,
        "channel_ids": list(CHANNEL_IDS),
        "coordinates": {name: list(CHANNEL_COORDINATES[name]) for name in CHANNEL_IDS},
        "coordinate_interpretation": "fixed schematic 10-20 projection, not measured anatomy",
        "retained_counts": list(RETAINED_COUNTS),
        "missing_percent": {str(k): 100.0 * (22 - k) / 22 for k in RETAINED_COUNTS},
        "training_retained_counts": list(TRAINING_RETAINED_COUNTS),
        "training_patterns": ["full", "random_static", "dynamic_random"],
        "evaluation_only": {"retained_count": 11, "patterns": ["spatial_static", "dynamic_spatial"]},
        "dynamic_schedule": "A for [0,P//3), B for [P//3,2*P//3), A thereafter",
        "dynamic_count": "exactly k observed electrodes in every window",
        "partition_disjointness": "canonical subset SHA256 modulo 3; full mask is shared exception",
        "scenarios": evaluation_scenarios(),
    }
