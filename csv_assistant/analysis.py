"""Pure analysis functions. Nothing here modifies the input DataFrame."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# Values that very often mean "missing" but are read as real data.
SENTINEL_NUMBERS = (-999, -9999, 9999, 99999)
SENTINEL_STRINGS = ("", "na", "n/a", "nan", "null", "none", "-", "?", "missing", "unknown")
_ID_NAMES = {"id", "index", "idx", "uuid", "guid"}
# A real thousands-separated number: 1,234 or 12,345,678.90 (NOT 11158,11159).
_THOUSANDS = re.compile(r"^[+-]?\d{1,3}(,\d{3})+(\.\d+)?$")
# A delimited list such as 11158,11159 or a;b;c (single tokens joined by , ; or |).
_LIST = re.compile(r"^[\w.\-]+(\s*[,;|]\s*[\w.\-]+)+$")


def column_overview(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_null": df.notna().sum(),
        "unique": df.nunique(dropna=True),
    })


def missing_values(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    miss = df.isna().sum()
    out = pd.DataFrame({
        "missing": miss,
        "missing_pct": (miss / n * 100).round(2) if n else 0.0,
    })
    return out.sort_values("missing", ascending=False)


def duplicate_info(df: pd.DataFrame) -> dict:
    """Duplicate rows (first occurrence not counted).

    `first_duplicate_lines` are file line numbers (header = line 1), assuming one record per line.
    """
    dup_mask = df.duplicated(keep="first")
    idx = [int(i) for i in df.index[dup_mask][:10]]
    return {"duplicate_rows": int(dup_mask.sum()),
            "first_duplicate_indices": idx,
            "first_duplicate_lines": [i + 2 for i in idx]}


def key_duplicate_info(df: pd.DataFrame, key: list[str]) -> dict:
    """Rows repeating an earlier key, split into identical rows and conflicting rows
    (same key, different values). Conflicts are the dangerous kind: whole-row duplicate
    detection cannot see them."""
    key = list(key)
    key_dup = df.duplicated(subset=key, keep="first")
    full_dup = df.duplicated(keep="first")
    conflict = key_dup & ~full_dup
    idx = [int(i) for i in df.index[conflict][:10]]
    return {"key": key,
            "key_duplicates": int(key_dup.sum()),
            "identical": int((key_dup & full_dup).sum()),
            "conflicting": int(conflict.sum()),
            "first_conflicting_lines": [i + 2 for i in idx]}


def is_text(s: pd.Series) -> bool:
    """True for text columns under both pandas 2 (object) and pandas 3 (str) dtypes."""
    return s.dtype == object or pd.api.types.is_string_dtype(s)


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]


def _id_name(name) -> bool:
    n = str(name).strip()
    low = n.lower()
    return (low in _ID_NAMES or low.endswith(("_id", " id", "-id", "_index"))
            or (n.endswith("Id") and len(n) > 2))


def is_id_like(s: pd.Series, name=None) -> bool:
    """Numeric ID-like column (heuristic; `--id-cols` overrides it).

    True if the column is named like an ID, or if it is an integer column whose values
    form a (nearly) unique consecutive run, e.g. 1..N with a few repeats. A bare
    "all values unique" rule is NOT used: incomes or timestamps are often all unique.
    """
    v = s.dropna()
    if len(v) < 2 or not pd.api.types.is_numeric_dtype(v) or pd.api.types.is_bool_dtype(v):
        return False
    if name is not None and _id_name(name):
        return True
    if pd.api.types.is_integer_dtype(v):
        nu = v.nunique()
        return nu / len(v) >= 0.9 and (v.max() - v.min() + 1) == nu
    return False


def numeric_sentinels(s: pd.Series) -> dict:
    """Candidate missing-value sentinels. A value only counts if it sits at the minimum or
    maximum of the column: -999 below every real age is suspicious, but 9999 in the middle
    of a 1..15050 range is just a valid number."""
    if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return {}
    v = s.dropna()
    if pd.api.types.is_float_dtype(v):
        v = v[~np.isinf(v)]
    if v.empty:
        return {}
    lo, hi = v.min(), v.max()
    return {x: int((v == x).sum()) for x in SENTINEL_NUMBERS if x == lo or x == hi}


def _coerce_numeric(vals: pd.Series) -> pd.Series:
    """to_numeric that accepts real thousands separators but not comma-joined lists."""
    out = vals
    has = vals.str.contains(",", regex=False)
    if has.any():
        sub = vals[has]
        ok = sub.str.match(_THOUSANDS)
        out = vals.copy()
        out.loc[sub.index[ok]] = sub[ok].str.replace(",", "", regex=False)
    return pd.to_numeric(out, errors="coerce")


def text_kind(s: pd.Series) -> tuple[str, float, int] | None:
    """Detect text columns that really hold numbers or dates.

    Returns (kind, share, n_checked) or None. Numbers are checked on every value; dates on
    an evenly spaced sample of up to 1000 values (parsing dates is slow), so callers must
    say "sampled" when reporting a date result.
    """
    if not is_text(s):
        return None
    vals = s.dropna().astype(str).str.strip()
    vals = vals[~vals.str.lower().isin(SENTINEL_STRINGS)]
    if vals.empty:
        return None
    share = float(_coerce_numeric(vals).notna().mean())
    if share >= 0.9:
        return "numeric_text", share, len(vals)
    step = max(1, len(vals) // 1000)
    sample = vals.iloc[::step].head(1000)
    share = float(pd.to_datetime(sample, errors="coerce", format="mixed").notna().mean())
    if share >= 0.8:
        return "date_text", share, len(sample)
    return None


def infinite_counts(df: pd.DataFrame) -> dict:
    """Per-column count of +inf/-inf values (pandas parses 'inf' text as real floats)."""
    out = {}
    for c in numeric_columns(df):
        if pd.api.types.is_float_dtype(df[c]):
            n = int(np.isinf(df[c]).sum())
            if n:
                out[c] = n
    return out


def finite_view(df: pd.DataFrame) -> pd.DataFrame:
    """Copy with +/-inf replaced by NaN, for statistics. The original is never touched."""
    return df.replace([np.inf, -np.inf], np.nan)


def id_columns(df: pd.DataFrame, declared=()) -> list[str]:
    declared = set(declared)
    return [c for c in numeric_columns(df) if c in declared or is_id_like(df[c], c)]


def summary_statistics(df: pd.DataFrame, id_cols=()) -> pd.DataFrame:
    ids = set(id_columns(df, id_cols))
    cols = [c for c in numeric_columns(df) if c not in ids]
    if not cols:
        return pd.DataFrame()
    x = finite_view(df[cols])
    desc = x.describe().T
    desc["skew"] = x.skew()
    desc["zeros"] = (x == 0).sum()
    return desc


def data_quality_flags(df: pd.DataFrame, id_cols=(), outliers: dict | None = None) -> list[dict]:
    """Heuristic warnings, each with a machine-readable `kind`. Hints for a human, never fixes."""
    flags: list[dict] = []
    declared = set(id_cols)

    def add(col, kind, issue):
        flags.append({"column": col, "kind": kind, "issue": issue})

    for c in df.columns:
        s = df[c]
        if s.notna().sum() == 0:
            add(c, "all_missing", "all values missing")
            continue
        if c in declared:
            add(c, "id", "declared as an ID (--id-cols); excluded from plots, statistics and sentinel checks")
            continue
        if s.nunique(dropna=True) == 1:
            add(c, "constant", "constant column (one unique value)")
        if is_id_like(s, c):
            add(c, "id", "looks like an ID; excluded from plots and summary statistics "
                         "(use --id-cols to declare others)")
            continue
        inf_n = int(np.isinf(s).sum()) if pd.api.types.is_float_dtype(s) else 0
        if inf_n:
            add(c, "infinite", f"{inf_n} infinite value(s) (+inf/-inf); ignored in statistics and plots")
        hits = numeric_sentinels(s)
        if hits:
            add(c, "sentinel", f"possible numeric missing-value sentinel(s) at the column extreme: {hits}")
        if outliers and c in outliers:
            o = outliers[c]
            ex = ", ".join(f"{v:.3g}" for v in o["largest"])
            add(c, "outliers", f"{o['count']} value(s) beyond {o['k']:g}x IQR from the median "
                               f"(most extreme: {ex}); mean, std and skew are unreliable, use the median and quartiles")
        if pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
            nn, nu = int(s.notna().sum()), int(s.nunique())
            if nn >= 1000 and nu >= 20 and nu / nn < 0.01:
                add(c, "categorical_int", f"integer column with only {nu} distinct values in {nn} rows; "
                                          "may be a category or identifier (consider --id-cols)")
        if is_text(s):
            stripped = s.dropna().astype(str).str.strip()
            ph_mask = stripped.str.lower().isin(SENTINEL_STRINGS)
            total = int(ph_mask.sum())
            if total:
                top = stripped[ph_mask].value_counts().head(5).to_dict()
                add(c, "placeholder", f"{total} placeholder-like text value(s): {top}")
            tk = text_kind(s)
            if tk and tk[0] == "numeric_text":
                add(c, "numeric_text", f"stored as text but {tk[1]:.0%} of non-placeholder values parse as numbers")
            elif tk:
                add(c, "date_text", f"stored as text but {tk[1]:.0%} of {tk[2]} sampled values parse as dates")
            has_delim = stripped.str.contains(r"[,;|]", regex=True)
            if has_delim.any():
                cand = stripped[has_delim & ~ph_mask]
                cand = cand[~cand.str.match(_THOUSANDS)]
                n_real = int((~ph_mask).sum())
                if n_real and len(cand) / n_real >= 0.01 and cand.str.match(_LIST).mean() >= 0.8:
                    add(c, "delimited_list", f"{len(cand) / n_real:.1%} of non-placeholder values look like "
                                             f"delimited lists, e.g. {cand.iloc[0]!r}; the column is not single-valued")
            if (s.dropna().astype(str) != stripped).any():
                add(c, "whitespace", "leading/trailing whitespace in some values")
            if s.dropna().map(lambda v: type(v).__name__).nunique() > 1:
                add(c, "mixed_types", "mixed Python types in one column")
    return flags


def outlier_info(df: pd.DataFrame, id_cols=(), k: float = 1000.0) -> dict:
    """Columns with finite values further than k x IQR from the median.

    Quantile based, so a legitimately heavy tail (tens of IQRs) is not flagged, while
    corrupt values (orders of magnitude beyond the bulk) are. Returns
    {column: {count, share, k, median, iqr, lo, hi, largest}}.
    """
    out: dict = {}
    if k <= 0:
        return out
    ids = set(id_columns(df, id_cols))
    for c in numeric_columns(df):
        if c in ids:
            continue
        x = df[c].dropna()
        if pd.api.types.is_float_dtype(x):
            x = x[np.isfinite(x)]
        if len(x) < 10:
            continue
        q1, med, q3 = x.quantile([0.25, 0.5, 0.75])
        iqr = q3 - q1
        if not iqr > 0:
            continue
        lo, hi = med - k * iqr, med + k * iqr
        mask = (x < lo) | (x > hi)
        n = int(mask.sum())
        if n:
            ext = x[mask]
            order = (ext - med).abs().sort_values(ascending=False).index
            top = pd.unique(ext.loc[order])[:5]   # distinct values, most extreme first
            out[c] = {"count": n, "share": n / len(x), "k": float(k), "median": float(med),
                      "iqr": float(iqr), "lo": float(lo), "hi": float(hi),
                      "largest": [float(v) for v in top]}
    return out


def missingness_patterns(df: pd.DataFrame, max_cols: int = 12, top: int = 6) -> dict | None:
    """Which columns are missing together. None unless at least two columns have gaps."""
    miss_counts = df.isna().sum()
    cols = [c for c in miss_counts[miss_counts > 0].sort_values(ascending=False).index][:max_cols]
    if len(cols) < 2:
        return None
    counts = df[cols].isna().value_counts()
    patterns = []
    for key, n in counts.head(top).items():
        patterns.append({"missing": [c for c, flag in zip(cols, key) if flag],
                         "rows": int(n), "share": float(n) / len(df)})
    return {"columns": cols, "patterns": patterns, "n_patterns": int(len(counts))}


def empty_row_info(df: pd.DataFrame, id_cols=(), min_share: float = 0.001) -> tuple[dict | None, pd.Series | None]:
    """Rows where every informative numeric column is exactly 0 or missing.

    Such rows are often "no data" records (zeros standing in for missing totals, NaN for
    means). They distort medians, histograms and correlations. ID and constant columns are
    ignored. Returns (info, mask), or (None, None) when they are under 0.1% of rows.
    """
    ids = set(id_columns(df, id_cols))
    cols = [c for c in numeric_columns(df) if c not in ids and df[c].dropna().nunique() > 1]
    if len(cols) < 2 or len(df) == 0:
        return None, None
    sub = df[cols]
    mask = ((sub == 0) | sub.isna()).all(axis=1)
    n = int(mask.sum())
    if n == 0 or n / len(df) < min_share:
        return None, None
    idx = [int(i) for i in df.index[mask][:10]]
    return {"count": n, "share": n / len(df), "columns": cols,
            "first_lines": [i + 2 for i in idx]}, mask
