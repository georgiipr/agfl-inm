"""Frozen native EEG-ATCNet run-1 checkpoints for the covariance study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import tensorflow as tf

from .atcnet_native import ATCNet_


_REPOSITORY = "https://github.com/Altaheri/EEG-ATCNet"
_REVISION = "65162fb359ea46a2f62c885a9987247ab491ee7a"
_SOURCE_HASHES = {
    "models.py": "d64ba0cadbad664f69342ef4b946465af472f6bcce2c231a59b5f4adcdf17b2c",
    "attention_models.py": "c118372098138f6503266ed9973b33b57fed90dcaa5c6730ce854201348452c7",
    "LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
}
_CHECKPOINT_HASHES = {
    "A01": "15a6a6cd7454d4d5b1b59adfe1d9636140fb0bd5a5e0f42adafadb4a36e9b1fd",
    "A02": "e9a26fca0e07a82b1374ffbdeedf3bf8d16ba67a39603d39160613a54590171d",
    "A03": "acec4e58ac81d0df72c89e1910aca444cda11a8c76465780b709c318925c2e9b",
    "A04": "87a3e5973b35d6e81500f6d134669a6dda16eeca41e24a90da7cbc757f457b92",
    "A05": "685da44928c8c0d57dcd9247016b0a48f09af02e041a923596ff5c5077c1aeee",
    "A06": "ddfffffa081e197b4e5117c24a186a056a74a358cc4f81f436864ae34b16be3d",
    "A07": "a9cb9da09f0b52f331fdcdd47899a7eec3affa4d4a3cfd674e2e8c4ac85f0866",
    "A08": "86ba692320a1d057b116b9a88f9b1de966ae7f93487bfc05d097fd87e18363c7",
    "A09": "46bc137b2b6e34e6ecdc092099918b2c2fd322830786bdbfc93d6f0ea7c97ba8",
}
_DEFAULT_ASSETS = (
    Path(__file__).resolve().parents[1]
    / ".session-runs/published-checkpoint-covariance/assets"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_asset(root: Path, relative: str) -> Path:
    """Resolve a regular, non-symlink asset strictly beneath the chosen root."""
    candidate = root / relative
    current = root
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"symlink is not an admitted asset path: {relative}")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ValueError(f"asset is missing or escapes the asset root: {relative}") from exc
    if not resolved.is_file():
        raise ValueError(f"asset is not a regular file: {relative}")
    return resolved


def _verified_checkpoint(subject: str, assets_root: str | Path) -> Path:
    root_arg = Path(assets_root).expanduser()
    if root_arg.is_symlink():
        raise ValueError("asset root must not be a symlink")
    try:
        root = root_arg.resolve(strict=True)
    except OSError as exc:
        raise ValueError("asset root does not exist") from exc
    if not root.is_dir():
        raise ValueError("asset root must be a directory")

    _safe_asset(root, "EEG-ATCNet/README.md")
    manifest_path = _safe_asset(root, "origin.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("invalid EEG-ATCNet origin manifest") from exc
    if manifest.get("repository") != _REPOSITORY or manifest.get("revision") != _REVISION:
        raise ValueError("origin manifest does not identify the audited EEG-ATCNet revision")
    declared = manifest.get("files")
    if not isinstance(declared, dict):
        raise ValueError("origin manifest has no file hash map")
    for relative, expected in _SOURCE_HASHES.items():
        path = _safe_asset(root, f"EEG-ATCNet/{relative}")
        if declared.get(relative) != expected or _sha256(path) != expected:
            raise ValueError(f"pinned source bytes differ: {relative}")

    checkpoint_relative = f"EEG-ATCNet/results/saved models/run-1/subject-{int(subject[1:])}.h5"
    checkpoint = _safe_asset(root, checkpoint_relative)
    expected_checkpoint = _CHECKPOINT_HASHES[subject]
    origin_relative = f"results/saved models/run-1/subject-{int(subject[1:])}.h5"
    if declared.get(origin_relative) != expected_checkpoint:
        raise ValueError(f"origin manifest checkpoint identity differs for {subject}")
    if _sha256(checkpoint) != expected_checkpoint:
        raise ValueError(f"checkpoint bytes differ from the audited {subject} run-1 asset")
    return checkpoint


class FrozenClassifier:
    """A verified, inference-only native model with differentiable inputs.

    Args:
        subject: One of ``A01`` through ``A09``.
        assets_root: Asset bundle containing ``origin.json`` and the pinned
            ``EEG-ATCNet`` source/checkpoint tree. Defaults to this study's
            audited bundle. All required bytes are checked before H5 loading.

    Calls accept finite values shaped ``[batch, 22, 1125]`` in native
    normalized channel/time order and return the original four-class softmax
    probabilities. Values are evaluated as float32, matching the H5 tensors.
    Gradients to the input remain enabled; all model weights and BatchNorm
    buffers stay frozen, and the graph always receives ``training=False``.
    """

    def __init__(self, subject: str, assets_root: str | Path = _DEFAULT_ASSETS):
        if not isinstance(subject, str) or subject not in _CHECKPOINT_HASHES:
            raise ValueError("subject must be one of A01 through A09")
        checkpoint = _verified_checkpoint(subject, assets_root)
        self.subject = subject
        self.checkpoint_sha256 = _CHECKPOINT_HASHES[subject]
        self.model = ATCNet_(n_classes=4)
        if tuple(self.model.input_shape[1:]) != (1, 22, 1125):
            raise RuntimeError(f"unexpected native input shape: {self.model.input_shape}")
        self.model.load_weights(str(checkpoint))
        if self.model.count_params() != 115172 or len(self.model.weights) != 195:
            raise RuntimeError("loaded native graph does not match the audited ATCNet signature")
        self.model.trainable = False
        if self.model.trainable_variables:
            raise RuntimeError("classifier weights did not become frozen")

    def __call__(self, values, training: bool = False):
        """Return native probabilities; caller training flags never unfreeze it."""
        if not isinstance(training, bool):
            raise ValueError("training must be a Boolean")
        tensor = tf.convert_to_tensor(values)
        if not tensor.dtype.is_floating:
            raise TypeError("classifier values must have a floating-point dtype")
        if tensor.shape.rank is not None and tensor.shape.rank != 3:
            raise ValueError("classifier values must have shape [batch, 22, 1125]")
        checks = [
            tf.debugging.assert_rank(tensor, 3, message="expected [batch, 22, 1125]"),
            tf.debugging.assert_positive(tf.shape(tensor)[0], message="batch cannot be empty"),
            tf.debugging.assert_equal(tf.shape(tensor)[1], 22, message="expected 22 channels"),
            tf.debugging.assert_equal(tf.shape(tensor)[2], 1125, message="expected 1125 samples"),
            tf.debugging.assert_all_finite(tensor, "classifier input contains NaN or infinity"),
        ]
        with tf.control_dependencies(checks):
            tensor = tf.cast(tf.identity(tensor), tf.float32)
        probabilities = self.model(tensor[:, None, :, :], training=False)
        tf.debugging.assert_equal(tf.shape(probabilities), [tf.shape(tensor)[0], 4])
        return probabilities
