"""Deterministic whole-trial static electrode masks for the published study."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

import numpy as np

from .data import CHANNEL_IDS


STUDY_NAMESPACE = "published-checkpoint-covariance-v1"
MASK_PROTOCOL = "static-subset-sha256-pcg64-v1"
PARTITIONS = {"train": 0, "validation": 1, "E": 2}


def _key(namespace: str, subject: str, seed: int, trial_id: str,
         partition: str, retained: int, repeat: int, epoch: int) -> bytes:
    fields = [MASK_PROTOCOL, namespace, subject, int(seed), trial_id,
              partition, int(retained), int(repeat), int(epoch)]
    return json.dumps(fields, ensure_ascii=True, separators=(",", ":")).encode()


def _bucket(positions: np.ndarray) -> int:
    bits = sum(1 << int(i) for i in positions)
    payload = f"{MASK_PROTOCOL}:partition:{bits}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % 3


def static_masks(subject: str, seed: int, trial_ids: Iterable[object],
                 partition: str, retained: int, repeat: int = 0, epoch: int = 0,
                 *, namespace: str = STUDY_NAMESPACE) -> np.ndarray:
    """Return paired [N,22] masks with partition-disjoint nonfull subsets.

    A candidate subset belongs to one of three stable SHA256 buckets, mapped to
    train/validation/E. Rejection sampling is uniform within each bucket. The
    trial-specific PCG64 key includes every study, identity and schedule field.
    """
    if partition not in PARTITIONS:
        raise ValueError("partition must be train, validation, or E")
    if retained not in (6, 16, 22):
        raise ValueError("retained must be 6, 16, or 22")
    if not isinstance(subject, str) or not subject:
        raise ValueError("subject must be a nonempty string")
    for name, value in (("seed", seed), ("repeat", repeat), ("epoch", epoch)):
        if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    ids = tuple(str(x) for x in trial_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("trial_ids must be unique")
    if not namespace:
        raise ValueError("namespace must be nonempty")
    out = np.ones((len(ids), len(CHANNEL_IDS)), dtype=np.bool_)
    if retained == 22:
        return out
    target = PARTITIONS[partition]
    for row, trial_id in enumerate(ids):
        digest = hashlib.sha256(_key(namespace, subject, seed, trial_id, partition,
                                     retained, repeat, epoch)).digest()
        rng = np.random.Generator(np.random.PCG64(int.from_bytes(digest[:16], "little")))
        for _ in range(100_000):
            positions = np.sort(rng.choice(22, size=retained, replace=False))
            if _bucket(positions) == target:
                out[row] = False
                out[row, positions] = True
                break
        else:
            raise RuntimeError("could not draw a subset from the requested partition pool")
    return out
