"""Project paths, fingerprints, verified writes and report helpers.

Conventions (see CLAUDE.md): no absolute paths in outputs, JSON written with LF,
Parquet written to a temporary file, read back, compared exactly, then replaced.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa


def find_root(marker: str = "data/raw/events.csv", start: Path | None = None) -> Path:
    """Walk up from ``start`` (default: cwd) until ``marker`` exists."""
    start = (start or Path.cwd()).resolve()
    for path in [start, *start.parents]:
        if (path / marker).exists():
            return path
    raise FileNotFoundError(f"Không tìm thấy thư mục project chứa {marker}; mở notebook từ project hoặc notebooks/.")


def rel(path: Path, root: Path) -> str:
    return Path(path).resolve().relative_to(root.resolve()).as_posix()


def require(condition, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def fingerprint(path: Path) -> dict:
    """Byte-level size and SHA-256 of a file (identifies the exact file)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return {"bytes": Path(path).stat().st_size, "sha256": digest.hexdigest()}


def _list_columns(frame: pd.DataFrame) -> list:
    return [c for c in frame.columns if frame[c].dtype == object and len(frame)
            and isinstance(frame[c].iloc[0], (list, tuple, np.ndarray))]


def lists_to_python(frame: pd.DataFrame) -> pd.DataFrame:
    """Parquet list columns come back as numpy arrays; convert them to Python lists."""
    for column in _list_columns(frame):
        frame[column] = frame[column].map(list)
    return frame


def content_sha256(frame: pd.DataFrame, key) -> str:
    """Content SHA-256: sort by primary key, hash rows; independent of Parquet byte layout."""
    ordered = frame.sort_values(list(key), kind="stable").reset_index(drop=True)
    for column in _list_columns(ordered):  # list cells are hashed as "a|b|c"
        ordered[column] = ordered[column].map(lambda v: "|".join(map(str, v)))
    digest = hashlib.sha256(json.dumps([str(c) for c in ordered.columns]).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(ordered, index=False).to_numpy(dtype="<u8").tobytes())
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8", newline="\n")
        json.loads(temporary.read_text(encoding="utf-8"))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_verified(frame: pd.DataFrame, destination: Path, key, root: Path) -> dict:
    """Write Parquet via temp file, read back, assert exact equality, replace; return artifact info."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        frame.to_parquet(temporary, engine="pyarrow", index=False)
        restored = lists_to_python(pd.read_parquet(temporary, engine="pyarrow"))
        pd.testing.assert_frame_equal(frame.reset_index(drop=True), restored, check_exact=True)
        temporary.replace(destination)
        del restored
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": rel(destination, root), "rows": len(frame), **fingerprint(destination),
        "content_sha256": content_sha256(frame, key), "content_key": list(key),
        "parquet_roundtrip_verified": True,
    }


def verify_inputs(root: Path, expected: dict) -> dict:
    """Check each ``{relative_path: {"bytes", "sha256"}}`` entry against the file on disk."""
    actual = {}
    for path, info in expected.items():
        current = fingerprint(root / path)
        require(current["sha256"] == info["sha256"] and current["bytes"] == info["bytes"],
                f"{path} khác SHA-256 trong report bước trước; chạy lại bước trước hoặc dùng đúng phiên bản dữ liệu.")
        actual[path] = current
    return actual


def environment() -> dict:
    return {"python_version": sys.version.split()[0], "pandas_version": pd.__version__,
            "pyarrow_version": pa.__version__}


def utc_now() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()
