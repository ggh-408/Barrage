"""Small, crash-safe helpers for reproducible training artifacts."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _replace_with_retry(source: Path, destination: Path) -> None:
    """Replace an artifact despite transient Windows reader/AV file locks."""
    attempts = 60
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(min(0.05 * (attempt + 1), 1.0))


def sha256_file(path: Path) -> str:
    """Reject legacy callers; provenance uses paths and direct content checks."""
    raise RuntimeError("Digest generation is disabled; compare contents directly")


def contents_equal(left: Any, right: Any) -> bool:
    """Compare saved model structures directly, without summary identifiers."""
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            contents_equal(value, right[key]) for key, value in left.items())
    if isinstance(left, (tuple, list)):
        return isinstance(right, (tuple, list)) and len(left) == len(right) and all(
            contents_equal(a, b) for a, b in zip(left, right))
    return left == right


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    _replace_with_retry(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, ensure_ascii=False))


def atomic_torch_save(value: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    # Loading the temporary file catches truncated writes before replacement.
    torch.load(temporary, map_location="cpu", weights_only=False)
    _replace_with_retry(temporary, path)


def atomic_save_npz(path: Path, **arrays: Any) -> None:
    """Write a NumPy archive atomically and verify it before replacement."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    with np.load(temporary, allow_pickle=False) as saved:
        if set(saved.files) != set(arrays):
            raise RuntimeError("temporary NumPy archive is incomplete")
    _replace_with_retry(temporary, path)


def atomic_copy(source: Path, destination: Path) -> None:
    """Copy a file without ever exposing a partially written destination."""
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copy2(source, temporary)
    _replace_with_retry(temporary, destination)


def prepare_new_output(path: Path) -> None:
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {path}. Use a new run directory."
        )
    path.mkdir(parents=True, exist_ok=True)


def git_revision(project_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "symbolic-ref", "--short", "HEAD"], cwd=project_root,
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
