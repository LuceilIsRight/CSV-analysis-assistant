"""Load a CSV file with clear errors, encoding fallback and delimiter detection."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pandas as pd

ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")


class CSVLoadError(Exception):
    """Raised when the input file cannot be read as a usable CSV."""


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _detect_delimiter(path: Path, encoding: str) -> str:
    with open(path, "r", encoding=encoding, newline="") as f:
        sample = f.read(64 * 1024)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def load_csv(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Return (dataframe, load_info). Raises CSVLoadError with a readable message."""
    path = Path(path)
    if not path.exists():
        raise CSVLoadError(f"File not found: {path}")
    if not path.is_file():
        raise CSVLoadError(f"Not a file: {path}")
    if path.stat().st_size == 0:
        raise CSVLoadError(f"File is empty: {path}")

    last_err: Exception | None = None
    for enc in ENCODINGS:
        try:
            delim = _detect_delimiter(path, enc)
            df = pd.read_csv(path, encoding=enc, sep=delim)
        except UnicodeDecodeError as e:
            last_err = e
            continue
        except pd.errors.EmptyDataError:
            raise CSVLoadError(f"No columns could be parsed from {path}")
        except pd.errors.ParserError as e:
            raise CSVLoadError(f"Malformed CSV ({path.name}): {e}")
        if df.shape[1] == 0:
            raise CSVLoadError(f"No columns could be parsed from {path}")
        info = {
            "file": path.name,
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
            "encoding": enc,
            "delimiter": delim,
        }
        return df, info
    raise CSVLoadError(f"Could not decode {path.name} with any supported encoding: {last_err}")
