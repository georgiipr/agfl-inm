"""Atomic, identity-bound persistence shared by the two student studies."""
from __future__ import annotations

import fcntl
import json
import os
from contextlib import contextmanager
from pathlib import Path

from .protocol import ROOT, sha256


def safe_path(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in [path, *path.parents]):
        raise ValueError("Refusing symlinked artifact path")
    return path


@contextmanager
def lock(path):
    path = safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def write_json(path, value, *, replace=False):
    path = safe_path(path)
    text = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if path.exists() and not replace:
        if path.read_text() != text:
            raise ValueError("Existing immutable artifact differs: " + str(path))
        return
    temporary = path.with_name(path.name + ".writing-" + str(os.getpid()))
    with temporary.open("x") as stream:
        stream.write(text)
    temporary.replace(path)


def write_npz(path, **arrays):
    import numpy as np
    path = safe_path(path)
    if path.exists():
        raise ValueError("Refusing to replace an array archive")
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def manifest(folder):
    folder = safe_path(folder)
    result = {}
    for path in sorted(folder.rglob("*")):
        safe_path(path)
        if path.is_file() and path != folder / "receipt.json":
            result[str(path.relative_to(folder))] = sha256(path)
    return result


def finish_task(folder, metadata):
    metadata = dict(metadata, files=manifest(folder))
    write_json(folder / "receipt.json", metadata)


def verify_task(folder, identity_hash):
    folder = safe_path(folder)
    value = json.loads((folder / "receipt.json").read_text())
    if value["identity_sha256"] != identity_hash or value["files"] != manifest(folder):
        raise ValueError("Task artifact identity or bytes changed: " + folder.name)
    return value


def output_path(config, output):
    out = safe_path(output or ROOT / config["output"])
    if out.parent != ROOT / "results" or not out.name.startswith(config["experiment"] + "-"):
        raise ValueError("Use a separate direct results/" + config["experiment"] + "-* output")
    return out


def open_study(config, output, expected_identity, *, create):
    out = output_path(config, output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with lock(out.parent / ("." + out.name + ".lock")):
        if out.exists():
            if not out.is_dir() or json.loads((out / "identity.json").read_text()) != expected_identity:
                raise ValueError("Experiment identity changed or output is foreign; use a fresh output suffix")
        elif create:
            out.mkdir()
            write_json(out / "identity.json", expected_identity)
        else:
            raise ValueError("Study has not been initialized")
        for name in ("fits", "evaluation", "admission"):
            safe_path(out / name).mkdir(exist_ok=True)
    return out
