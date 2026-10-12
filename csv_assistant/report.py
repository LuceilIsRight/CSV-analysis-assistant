"""Write the Markdown report and CSV side outputs."""
from __future__ import annotations

import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from . import __version__


def _md_table(df: pd.DataFrame, index_name: str = "column") -> str:
    if df.empty:
        return "_None._"
    d = df.copy()
    d.index.name = index_name
    d = d.reset_index()
    int_cols = {"count", "non_null", "unique", "missing", "zeros"}

    def fmt(v, col):
        if isinstance(v, float):
            return str(int(v)) if col in int_cols and v.is_integer() else f"{v:.4g}"
        return str(v)

    d = pd.DataFrame({col: [fmt(v, col) for v in d[col]] for col in d.columns})
    header = "| " + " | ".join(d.columns) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    rows = ["| " + " | ".join(r) + " |" for r in d.values.tolist()]
    return "\n".join([header, sep, *rows])


def write_outputs(out_dir: Path, df: pd.DataFrame, info: dict, overview: pd.DataFrame,
                  missing: pd.DataFrame, dups: dict, stats: pd.DataFrame,
                  flags: list[dict], plots: list[str], skipped: list[str],
                  command: str, key_info: dict | None = None, miss_info: dict | None = None,
                  empty_info: dict | None = None, analysis_note: str | None = None,
                  audit_info: dict | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats.to_csv(out_dir / "summary_statistics.csv", index_label="column")
    missing.to_csv(out_dir / "missing_values.csv", index_label="column")

    dup_line = f"- Duplicate rows (excluding first occurrence): {dups['duplicate_rows']}"
    if dups["duplicate_rows"]:
        dup_line += (f" (file lines, header = line 1, first 10: {dups['first_duplicate_lines']}; "
                     "assumes one record per line)")
    has_missing = (missing["missing"] > 0).any()
    miss_lines: list[str] = []
    if miss_info:
        miss_lines = ["### Missingness patterns", "",
                      f"Which of the {len(miss_info['columns'])} columns with gaps are missing together "
                      f"({miss_info['n_patterns']} distinct patterns; top shown):", "",
                      "| rows | share | missing columns |", "|---|---|---|"]
        for p in miss_info["patterns"]:
            label = ", ".join(p["missing"]) if p["missing"] else "(none missing)"
            miss_lines.append(f"| {p['rows']} | {p['share']:.2%} | {label} |")
        miss_lines.append("")
    empty_lines: list[str] = []
    if empty_info:
        e = empty_info
        empty_lines = ["### Empty-looking rows", "",
                       f"- {e['count']} rows ({e['share']:.2%}) have every informative numeric column equal to "
                       f"exactly 0 or missing (columns checked: {', '.join(e['columns'])})",
                       f"- First file lines: {e['first_lines']}",
                       "- Such rows often mean \"no data\" records. They distort medians, quartiles, histograms "
                       "and correlations; `--exclude-empty` analyses without them (the file is never modified).", ""]
    ana_lines = [f"- {analysis_note}"] if analysis_note else []
    key_lines: list[str] = []
    if key_info:
        k = key_info
        conflict = f"- Same key but different values: {k['conflicting']}"
        if k["conflicting"]:
            conflict += f" (file lines, first 10: {k['first_conflicting_lines']})"
        key_lines = ["", f"## Key duplicates (key: {', '.join(k['key'])})", "",
                     f"- Rows repeating an earlier key: {k['key_duplicates']}",
                     f"- Identical to an earlier row: {k['identical']}", conflict]

    lines = [
        f"# Analysis report: {info['file']}", "",
        "## Reproducibility", "",
        f"- Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- Command: `{command}`",
        f"- Input SHA-256: `{info['sha256']}`",
        f"- Input size: {info['size_bytes']} bytes; encoding `{info['encoding']}`; delimiter `{info['delimiter']}`",
        f"- Tool v{__version__}; Python {platform.python_version()}; pandas {pd.__version__}; "
        f"NumPy {np.__version__}; Matplotlib {matplotlib.__version__}",
        "- The input file was read only and never modified.", "",
        "## Dataset size", "",
        f"- Rows: {len(df)}", f"- Columns: {df.shape[1]}", dup_line, *ana_lines, *key_lines, "",
        "## Columns", "", _md_table(overview), "",
        "## Missing values", "",
        _md_table(missing[missing["missing"] > 0]) if has_missing
        else "No missing values detected (as parsed).", "", *miss_lines, *empty_lines,
        "## Data-quality flags", "",
        "These are heuristics for a human to review; nothing was changed.", "",
    ]
    lines += [f"- **{f['column']}**: {f['issue']}" for f in flags] or ["No flags raised."]
    sent_cols = [f["column"] for f in flags if f["kind"] == "sentinel"]
    stat_note = ([f"**Caution:** statistics for {', '.join(sent_cols)} include suspected "
                  "missing-value sentinels and are likely distorted.", ""] if sent_cols else [])
    out_cols = [f["column"] for f in flags if f["kind"] == "outliers"]
    if out_cols:
        stat_note += [f"**Caution:** mean, std and skew of {', '.join(out_cols)} are dominated by extreme "
                      "values (see flags); use the median (50%) and quartiles instead.", ""]
    if analysis_note:
        stat_note += [f"{analysis_note}.", ""]
    lines += ["", "## Numerical summary", "",
              "ID-like columns are excluded.", ""] + stat_note + [
              _md_table(stats) if not stats.empty else "No numeric columns found.", "",
              "## Plots", ""]
    lines += [f"![{Path(p).stem}]({p})" for p in plots] or ["No plots generated."]
    if skipped:
        lines += ["", "### Plot notes", ""] + [f"- {s}" for s in skipped]
    audit_lines: list[str] = []
    if audit_info:
        rows = sorted(audit_info["summary"], key=lambda r: (r["severity"] != "error", -r["count"]))
        audit_lines = ["", "## Audit trail", "",
                       "Detection only: the data was not modified and nothing was corrected. "
                       f"Every flagged cell or row is listed in `audit_log.csv` (at most {audit_info['cap']} per "
                       "check and column) and explained in `audit_summary.json`.",
                       "**error** = not a finite number, or breaks a constraint that follows from a documented "
                       "definition. **warning** = statistically unusual or suspicious; it may be a real measurement.", ""]
        if audit_info.get("rules_name"):
            audit_lines += [f"Documented-constraint rules applied: {audit_info['rules_name']}.", ""]
        else:
            audit_lines += ["No documented-constraint rules were supplied (`--rules`), so no physical-consistency "
                            "checks ran; only statistical and structural checks.", ""]
        audit_lines += [f"- Rule not applied: {s}" for s in audit_info.get("rules_skipped", [])]
        if rows:
            audit_lines += ["| severity | check | column | rule | flagged | why |", "|---|---|---|---|---|---|"]
            for r in rows:
                why = r["reason"] if len(r["reason"]) <= 120 else r["reason"][:117] + "..."
                audit_lines.append(f"| {r['severity']} | {r['check']} | {r['column'] or '(row)'} | "
                                   f"{r['rule_id'] or '-'} | {r['count']} | {why} |")
        else:
            audit_lines += ["Nothing was flagged."]
    lines += audit_lines
    lines += ["", "## Limitations", "",
              "This is exploratory output. It does not decide whether outliers are errors "
              "or whether any statistical conclusion is justified.", ""]
    report = out_dir / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report
