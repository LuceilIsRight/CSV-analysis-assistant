"""Detection only. Produces an item-level audit trail; never modifies the data.

Severity:
  error   - not a finite number, or breaks a constraint that follows from a documented definition
  warning - statistically unusual or suspicious; may well be a real measurement
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__, analysis

SEVERITY = {"infinite": "error", "rule_range": "error", "outlier_iqr": "warning",
            "sentinel": "warning", "duplicate_row": "warning", "blank_row": "warning"}
LOG_COLUMNS = ["check", "severity", "scope", "column", "rule_id", "row", "line", "value"]


@dataclass
class Detection:
    findings: pd.DataFrame
    summary: list[dict]
    outliers: dict
    rules_skipped: list[str] = field(default_factory=list)


def load_rules(path) -> dict:
    """Load and validate a rules file. Raises ValueError with a readable message."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"cannot read rules file {path}: {e}")
    rules = data.get("rules") if isinstance(data, dict) else None
    if not isinstance(rules, list) or not rules:
        raise ValueError("rules file must contain a non-empty 'rules' list")
    for r in rules:
        for key in ("id", "column", "description", "source"):
            if key not in r:
                raise ValueError(f"rule {r.get('id', r)} is missing '{key}' (every rule needs a source)")
        if "min" not in r and "max" not in r:
            raise ValueError(f"rule {r['id']} needs 'min' and/or 'max'")
    return {"name": data.get("name", "rules"), "rules": rules}


def run_detection(df: pd.DataFrame, id_cols=(), rules: dict | None = None,
                  outlier_k: float = 1000.0, cap: int = 100_000) -> Detection:
    ids = set(analysis.id_columns(df, id_cols))
    # Rows that look empty are reported on their own; they must not define what is 'normal'.
    _, empty_mask = analysis.empty_row_info(df, id_cols, min_share=0)
    baseline = df if empty_mask is None else df.loc[~empty_mask]
    outliers = analysis.outlier_info(baseline, id_cols, outlier_k)
    frames: list[pd.DataFrame] = []
    summary: list[dict] = []
    skipped: list[str] = []
    n = len(df)

    def add(check, scope, column, mask, values, reason, rule_id=""):
        total = int(mask.sum())
        if not total:
            return
        idx = np.flatnonzero(mask)[:cap]
        frames.append(pd.DataFrame({
            "check": check, "severity": SEVERITY[check], "scope": scope, "column": column,
            "rule_id": rule_id, "row": idx, "line": idx + 2,
            "value": values[idx] if values is not None else np.nan}))
        summary.append({"check": check, "severity": SEVERITY[check], "scope": scope, "column": column,
                        "rule_id": rule_id, "count": total, "written": int(len(idx)), "reason": reason})

    by_col: dict[str, list[dict]] = {}
    for r in (rules or {}).get("rules", []):
        if r["column"] in df.columns:
            by_col.setdefault(r["column"], []).append(r)
        else:
            skipped.append(f"{r['id']}: column {r['column']} not in file")

    for c in analysis.numeric_columns(df):
        if c in ids:
            continue
        x = df[c].to_numpy(dtype=float, na_value=np.nan)
        finite = np.isfinite(x)
        add("infinite", "cell", c, np.isinf(x), x, "value is +inf or -inf, not a physical measurement")
        for r in by_col.get(c, []):
            bad = np.zeros(n, dtype=bool)
            lo, hi = r.get("min"), r.get("max")
            if lo is not None:
                bad |= x < lo
            if hi is not None:
                bad |= x > hi
            span = f"{'-inf' if lo is None else lo} to {'+inf' if hi is None else hi}"
            add("rule_range", "cell", c, finite & bad, x,
                f"violates documented constraint {r['id']}: {r['description']} "
                f"(allowed {span}); basis: {r['basis'] if 'basis' in r else 'n/a'}; source: {r['source']}",
                r["id"])
        if c in outliers:
            o = outliers[c]
            add("outlier_iqr", "cell", c, finite & ((x < o["lo"]) | (x > o["hi"])), x,
                f"more than {o['k']:g} IQR from the median (median {o['median']:.3g}, IQR {o['iqr']:.3g}, "
                "computed on non-empty rows); unusual, but not necessarily wrong")
        for v in analysis.numeric_sentinels(df[c]):
            add("sentinel", "cell", c, x == v, x,
                f"equals {v}, a common missing-value code, and sits at the column's extreme")

    add("duplicate_row", "row", "", df.duplicated(keep="first").to_numpy(), None,
        "row is identical to an earlier row")
    if empty_mask is not None:
        add("blank_row", "row", "", empty_mask.to_numpy(), None,
            "every informative numeric column is exactly 0 or missing (a probable 'no data' record)")

    findings = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=LOG_COLUMNS)
    return Detection(findings, summary, outliers, skipped)


def write_audit(out_dir: Path, det: Detection, params: dict) -> None:
    """audit_log.csv holds one line per flagged cell/row; audit_summary.json holds the
    explanation (reason, rule, source) for every (check, column, rule) group."""
    out_dir.mkdir(parents=True, exist_ok=True)
    det.findings[LOG_COLUMNS].to_csv(out_dir / "audit_log.csv", index=False)
    payload = {"tool_version": __version__, "parameters": params,
               "rules_skipped": det.rules_skipped, "groups": det.summary}
    (out_dir / "audit_summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
