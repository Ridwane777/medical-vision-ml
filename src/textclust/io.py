"""Robust CSV loading: encoding, delimiter, header, ragged rows, text/label column detection."""
from __future__ import annotations

import csv
import io
import logging
import math
import re
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field

import pandas as pd

log = logging.getLogger(__name__)

ENCODINGS = ("utf-8-sig", "utf-8", "latin-1")
DELIMITERS = (",", ";", "\t", "|")
_NUMERIC = re.compile(r"^[+-]?(\d+([.,]\d*)?|\.\d+)([eE][+-]?\d+)?$")
_HEX = re.compile(r"^0x[0-9a-fA-F]+$")


@dataclass
class LoadedData:
    """A loaded table plus everything that was inferred about it."""

    frame: pd.DataFrame
    text_column: str
    label_column: str | None
    profile: dict = field(default_factory=dict)


def read_text(path) -> tuple[str, str]:
    """Decode a file with the first encoding of ENCODINGS that works."""
    raw = Path(path).read_bytes()
    for enc in ENCODINGS:
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    raise ValueError(f"cannot decode {path}")  # latin-1 never fails; kept for safety


def detect_delimiter(text: str) -> str:
    """csv.Sniffer first, then the candidate whose per-line count is most consistent."""
    sample = "\n".join(text.splitlines()[:50])
    try:
        return csv.Sniffer().sniff(sample, delimiters="".join(DELIMITERS)).delimiter
    except csv.Error:
        pass
    best, best_score = ",", -1.0
    lines = [ln for ln in sample.splitlines() if ln.strip()]
    for d in DELIMITERS:
        counts = [ln.count(d) for ln in lines]
        if not counts or max(counts) == 0:
            continue
        mode, freq = Counter(counts).most_common(1)[0]
        score = freq / len(counts) + (0.01 * mode if mode else 0)
        if mode > 0 and score > best_score:
            best, best_score = d, score
    return best


def _is_numberish(value: str) -> bool:
    v = value.strip()
    return bool(_NUMERIC.match(v) or _HEX.match(v))


def _is_label_like(value: str) -> bool:
    v = value.strip()
    return bool(v) and not _is_numberish(v) and len(v) <= 40 and len(v.split()) <= 4


def detect_header(rows: list[list[str]], sample_text: str) -> bool:
    """Header when the first row looks like column names and the body does not."""
    if len(rows) < 2:
        return False
    first, body = rows[0], rows[1:200]
    if not all(_is_label_like(v) for v in first):
        return False
    width = len(first)
    for j in range(width):
        col = [r[j] for r in body if len(r) > j and r[j].strip()]
        if not col:
            continue
        numeric_share = sum(_is_numberish(v) for v in col) / len(col)
        mean_len = sum(len(v) for v in col) / len(col)
        if numeric_share > 0.8 or mean_len > 3 * max(len(first[j]), 5):
            return True
    try:
        return csv.Sniffer().has_header(sample_text)
    except csv.Error:
        return False


def _text_score(values: list[str]) -> float:
    """Mean length weighted by the share of alphabetic characters."""
    vals = [v for v in values if v and v.strip()]
    if not vals:
        return 0.0
    joined = "".join(vals)
    alpha = sum(c.isalpha() for c in joined) / max(len(joined), 1)
    return (len(joined) / len(vals)) * alpha


def _resolve(spec, columns: list[str]) -> str:
    if isinstance(spec, int):
        return columns[spec]
    if spec in columns:
        return spec
    if isinstance(spec, str) and spec.isdigit() and int(spec) < len(columns):
        return columns[int(spec)]
    raise KeyError(f"column {spec!r} not found in {columns}")


def detect_label_column(df: pd.DataFrame, text_column: str) -> str | None:
    """A column is a label candidate if it is categorical: 2..max(2, sqrt(n)) distinct values,
    not an identifier, not constant, not mostly missing. Returns the first candidate or None."""
    n = len(df)
    max_classes = max(2, int(math.sqrt(n)))
    for col in df.columns:
        if col == text_column:
            continue
        s = df[col].dropna().astype(str).str.strip()
        s = s[s != ""]
        if len(s) < 0.9 * n:
            continue
        k = s.nunique()
        if k < 2 or k > max_classes or k > n / 2:
            continue
        if s.map(_is_numberish).mean() > 0.9 and pd.to_numeric(s, errors="coerce").is_monotonic_increasing:
            continue
        return col
    return None


def load_csv(path, text_column=None, label_column=None, delimiter=None, header=None) -> LoadedData:
    """Load any CSV of texts. Every argument except `path` is an optional override.

    `label_column=False` disables label detection."""
    text, encoding = read_text(path)
    delim = delimiter or detect_delimiter(text)
    rows = [r for r in csv.reader(io.StringIO(text), delimiter=delim) if any(c.strip() for c in r)]
    if not rows:
        raise ValueError(f"{path} contains no rows")
    sample = "\n".join(text.splitlines()[:50])
    has_header = detect_header(rows, sample) if header is None else bool(header)
    if has_header:
        columns, body = [c.strip() or f"col{i}" for i, c in enumerate(rows[0])], rows[1:]
    else:
        width = max(len(r) for r in rows)
        columns, body = [f"col{i}" for i in range(width)], rows
    width = len(columns)

    full = [r for r in body if len(r) == width] or body
    if text_column is not None:
        tcol = _resolve(text_column, columns)
    else:
        scores = [_text_score([r[j] for r in full if len(r) > j]) for j in range(width)]
        tcol = columns[max(range(width), key=lambda j: scores[j])]
    t = columns.index(tcol)

    repaired_short = repaired_long = 0
    fixed = []
    for r in body:
        if len(r) == width:
            fixed.append(r)
        elif len(r) < width:
            # Generic version of the old "shifted row" fix: the longest field is the text,
            # the remaining fields fill the other columns from left to right.
            repaired_short += 1
            i_long = max(range(len(r)), key=lambda i: len(r[i]))
            others = [v for i, v in enumerate(r) if i != i_long]
            row = [None] * width
            row[t] = r[i_long]
            slots = [j for j in range(width) if j != t]
            for j, v in zip(slots, others):
                row[j] = v
            fixed.append(row)
        else:
            # Unquoted delimiters inside the text: glue the surplus back into the text field.
            repaired_long += 1
            extra = len(r) - width
            fixed.append(r[:t] + [delim.join(r[t:t + extra + 1])] + r[t + extra + 1:])
    df = pd.DataFrame(fixed, columns=columns)

    if label_column is False:
        lcol = None
    elif label_column is not None:
        lcol = _resolve(label_column, columns)
    else:
        lcol = detect_label_column(df, tcol)

    profile = {
        "path": str(path),
        "encoding": encoding,
        "delimiter": delim,
        "header": has_header,
        "columns": columns,
        "rows": len(df),
        "ragged_rows_repaired": {"short": repaired_short, "long": repaired_long},
        "text_column": tcol,
        "text_column_source": "override" if text_column is not None else "auto",
        "label_column": lcol,
        "label_column_source": "override" if label_column not in (None, False) else "auto",
        "constant_columns": [c for c in columns if df[c].dropna().astype(str).nunique() <= 1],
    }
    log.info("loaded %s: %d rows, encoding=%s, delimiter=%r, header=%s, text=%s, label=%s",
             path, len(df), encoding, delim, has_header, tcol, lcol)
    return LoadedData(df, tcol, lcol, profile)
