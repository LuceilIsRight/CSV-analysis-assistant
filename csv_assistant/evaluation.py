"""Evaluation harness: inject labelled errors, run the real detectors, measure recall and precision.

What this can and cannot show
  * Labels come from construction, so recall is exact for the injected types, but the types are
    chosen by us. Recall on them says nothing about error kinds nobody thought to inject.
  * Valid-but-extreme measurements are generated from realistic heavy-tailed distributions inside the
    documented physical limits. Anything flagged there, or anywhere not injected, is a false alarm
    BY CONSTRUCTION (the base data is valid), so precision here is real for the synthetic base only.
  * Subtle in-range errors are injected on purpose: a statistical or range detector cannot see them,
    and the report must show that ceiling instead of hiding it.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from . import audit

NONNEG = ["USFLUX", "TOTUSJH", "SAVNCPP", "TOTPOT"]
SIGNED = ["MEANJZH", "MEANALP"]
ANGLE = ["MEANSHR", "MEANGAM"]
OTHER = ["MEANGBT"]
VALUE_COLS = NONNEG + SIGNED + ANGLE + OTHER
INPLANE = NONNEG + SIGNED + OTHER      # scaling these stays inside every documented range
ID_COLS = ["HARPNUM"]
CELL_TYPES = ["inf", "corrupt_magnitude", "sentinel", "negative_flip", "angle_out_of_range",
              "scale_moderate", "subtle_perturb"]
ROW_TYPES = ["duplicate_row", "blank_row"]
DESCRIPTION = {
    "inf": "value set to +/-inf",
    "corrupt_magnitude": "value x 1e8..1e25 (overflow-like corruption)",
    "sentinel": "value replaced by -999, -9999, 9999 or 99999",
    "negative_flip": "sign flipped in a documented non-negative column",
    "angle_out_of_range": "MEANGAM 95-170 deg (limit +/-90), MEANSHR 185-350 deg (limit 180)",
    "scale_moderate": "value x 3..30 (still plausible-looking)",
    "subtle_perturb": "value x 1.1..1.5 (small, in-range)",
    "duplicate_row": "exact copy of an earlier row appended",
    "blank_row": "all measurement columns set to 0",
}
DEFAULT_RULES = Path(__file__).resolve().parents[1] / "rules" / "sharp_hmi.json"


def make_base(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Valid SHARP-like data: heavy-tailed, signed where signed, inside every documented limit."""
    return pd.DataFrame({
        "T_REC": pd.date_range("2012-01-01", periods=n, freq="12min").strftime("%Y.%m.%d_%H:%M:%S_TAI"),
        "HARPNUM": rng.integers(1, 7000, n),
        "USFLUX": rng.lognormal(48.0, 1.5, n),
        "TOTUSJH": rng.lognormal(4.0, 1.6, n),
        "SAVNCPP": rng.lognormal(26.5, 1.6, n),
        "TOTPOT": rng.lognormal(48.5, 1.5, n),
        "MEANJZH": rng.standard_t(3, n) * 0.004,
        "MEANALP": rng.standard_t(3, n) * 0.012,
        "MEANSHR": np.clip(rng.normal(26, 8, n), 0.5, 179),
        "MEANGAM": np.clip(rng.normal(33, 9, n), 1, 89),
        "MEANGBT": rng.lognormal(4.7, 0.3, n),
    })


def inject(df: pd.DataFrame, rng: np.random.Generator, n_each: int):
    """Return (corrupted_df, cell_labels, row_labels, touched_rows). Rows never overlap between types."""
    n = len(df)
    out = df.copy()
    pool = rng.permutation(n)
    pos = 0

    def take(k):
        nonlocal pos
        r = pool[pos:pos + k]
        pos += k
        return [int(x) for x in r]

    cells: list[tuple] = []

    def put(rows, cols, fn, typ):
        for r in rows:
            c = cols[int(rng.integers(len(cols)))]
            out.at[r, c] = fn(c, float(df.at[r, c]))
            cells.append((r, c, typ))

    put(take(n_each), VALUE_COLS, lambda c, x: float(rng.choice([-np.inf, np.inf])), "inf")
    put(take(n_each), VALUE_COLS, lambda c, x: x * 10 ** rng.uniform(8, 25), "corrupt_magnitude")
    put(take(n_each), VALUE_COLS, lambda c, x: float(rng.choice([-999, -9999, 9999, 99999])), "sentinel")
    put(take(n_each), NONNEG, lambda c, x: -x, "negative_flip")
    put(take(n_each), ANGLE, lambda c, x: rng.uniform(95, 170) if c == "MEANGAM" else rng.uniform(185, 350),
        "angle_out_of_range")
    put(take(n_each), INPLANE, lambda c, x: x * rng.uniform(3, 30), "scale_moderate")
    put(take(n_each), INPLANE, lambda c, x: x * rng.uniform(1.1, 1.5), "subtle_perturb")
    dup_src, blank_rows = take(n_each), take(n_each)
    touched = {int(r) for r in pool[:pos]}
    for r in blank_rows:
        for c in VALUE_COLS:
            out.at[r, c] = 0.0
    out = pd.concat([out, out.iloc[dup_src]], ignore_index=True)
    rows = [(n + i, "duplicate_row") for i in range(len(dup_src))] + [(r, "blank_row") for r in blank_rows]
    return (out, pd.DataFrame(cells, columns=["row", "column", "type"]),
            pd.DataFrame(rows, columns=["row", "type"]), touched)


def valid_extremes(df: pd.DataFrame, touched: set, q: float = 0.9995) -> set:
    """Untouched, valid cells in the most extreme 0.05% of each column (robust distance from the median)."""
    keep = np.ones(len(df), dtype=bool)
    keep[list(touched)] = False
    ext = set()
    for c in VALUE_COLS:
        x = df[c].to_numpy(float)
        q1, med, q3 = np.percentile(x, [25, 50, 75])
        d = np.abs(x - med) / (q3 - q1)
        for r in np.flatnonzero((d >= np.quantile(d, q)) & keep):
            ext.add((int(r), c))
    return ext


def score(findings: pd.DataFrame, cells: pd.DataFrame, rows: pd.DataFrame, extremes: set) -> dict:
    fc = findings[findings["scope"] == "cell"]
    flagged = set(zip(fc["row"].tolist(), fc["column"].tolist()))
    errs = fc[fc["severity"] == "error"]
    err = set(zip(errs["row"].tolist(), errs["column"].tolist()))
    injected = {(r, c): t for r, c, t in zip(cells["row"], cells["column"], cells["type"])}
    inj = set(injected)
    res = {"n": Counter(), "tp": Counter(), "tp_err": Counter(), "fp_by_check": Counter(), "rows": {}}
    for key, t in injected.items():
        res["n"][t] += 1
        res["tp"][t] += key in flagged
        res["tp_err"][t] += key in err
    res.update(flagged_cells=len(flagged), tp_cells=len(flagged & inj),
               flagged_err=len(err), tp_err_cells=len(err & inj),
               extreme_total=len(extremes), extreme_flagged=len(extremes & flagged))
    for r, c, ch in zip(fc["row"], fc["column"], fc["check"]):
        if (r, c) not in inj:
            res["fp_by_check"][ch] += 1
    fr = findings[findings["scope"] == "row"]
    for check in ROW_TYPES:
        got = set(fr.loc[fr["check"] == check, "row"].tolist())
        truth = set(rows.loc[rows["type"] == check, "row"].tolist())
        res["rows"][check] = {"flagged": len(got), "tp": len(got & truth), "n": len(truth)}
    return res


def _merge(a: dict, b: dict) -> None:
    for k, v in b.items():
        if isinstance(v, dict):
            _merge(a.setdefault(k, type(v)()), v)
        else:
            a[k] = a.get(k, 0) + v


def evaluate_once(n, n_each, seed, rules, ks) -> dict:
    rng = np.random.default_rng(seed)
    base = make_base(n, rng)
    out, cells, rows, touched = inject(base, rng, n_each)
    ext = valid_extremes(base, touched)
    configs = {}
    for name, rl in (("stats_only", None), ("with_rules", rules)):
        det = audit.run_detection(out, ID_COLS, rl, 1000.0, cap=10 ** 9)
        configs[name] = score(det.findings, cells, rows, ext)
    sweep = {}
    for k in ks:
        if k == 1000:
            sweep[k] = configs["with_rules"]
            continue
        det = audit.run_detection(out, ID_COLS, rules, float(k), cap=10 ** 9)
        sweep[k] = score(det.findings, cells, rows, ext)
    return {"configs": configs, "sweep": sweep}


def run(n=200_000, n_each=300, seeds=3, rules_path=None, ks=(100, 300, 1000, 3000)) -> dict:
    rules = audit.load_rules(rules_path or DEFAULT_RULES)
    agg = {"configs": {}, "sweep": {}}
    for seed in range(seeds):
        one = evaluate_once(n, n_each, seed, rules, ks)
        for part in ("configs", "sweep"):
            for key, val in one[part].items():
                _merge(agg[part].setdefault(key, {}), val)
    agg["meta"] = {"n_rows": n, "n_each": n_each, "seeds": seeds, "outlier_k": 1000,
                   "rules": rules["name"], "ks": list(ks)}
    return agg


def _pct(a, b):
    return f"{a / b:.1%}" if b else "n/a"


def render(res: dict) -> str:
    m, cf, sw = res["meta"], res["configs"], res["sweep"]
    so, wr = cf["stats_only"], cf["with_rules"]
    L = ["# Detector evaluation (synthetic SHARP-like data, injected labelled errors)", "",
         f"{m['seeds']} seeds x {m['n_rows']} base rows x {m['n_each']} injections per error type. "
         f"Default outlier threshold {m['outlier_k']}x IQR. Rules: {m['rules']}.", "",
         "## 1. Recall by error type (a cell counts as caught if ANY check flagged it)", "",
         "| error type | injected | n | statistical checks only | + documented rules | caught as error-severity |",
         "|---|---|---|---|---|---|"]
    for t in CELL_TYPES:
        L.append(f"| {t} | {DESCRIPTION[t]} | {wr['n'][t]} | {_pct(so['tp'][t], so['n'][t])} | "
                 f"{_pct(wr['tp'][t], wr['n'][t])} | {_pct(wr['tp_err'][t], wr['n'][t])} |")
    for t in ROW_TYPES:
        a, b = so["rows"][t], wr["rows"][t]
        L.append(f"| {t} | {DESCRIPTION[t]} | {b['n']} | {_pct(a['tp'], a['n'])} | {_pct(b['tp'], b['n'])} | n/a (row check) |")
    tot_n = sum(wr["n"][t] for t in CELL_TYPES)
    L += ["", f"Overall cell recall: statistical only {_pct(sum(so['tp'][t] for t in CELL_TYPES), tot_n)}; "
              f"with documented rules {_pct(sum(wr['tp'][t] for t in CELL_TYPES), tot_n)}.", "",
          "## 2. Precision and false alarms", "",
          "| configuration | cells flagged | injected among them | cell precision | precision, error severity only |",
          "|---|---|---|---|---|"]
    for name, r in (("statistical only", so), ("with documented rules", wr)):
        L.append(f"| {name} | {r['flagged_cells']} | {r['tp_cells']} | {_pct(r['tp_cells'], r['flagged_cells'])} | "
                 f"{_pct(r['tp_err_cells'], r['flagged_err'])} |")
    fps = ", ".join(f"{k}: {v}" for k, v in sorted(wr["fp_by_check"].items())) or "none"
    L += ["", f"False alarms (with rules) by check: {fps}.",
          f"Valid-but-extreme cells (top 0.05% of each column, untouched): {wr['extreme_total']}; "
          f"flagged: {wr['extreme_flagged']} ({_pct(wr['extreme_flagged'], wr['extreme_total'])}).", ""]
    for t in ROW_TYPES:
        b = wr["rows"][t]
        L.append(f"- Row check `{t}`: flagged {b['flagged']}, correct {b['tp']} (precision {_pct(b['tp'], b['flagged'])}).")
    L += ["", "## 3. What the documented physical rules add", "",
          "| error type | statistical only | + documented rules |", "|---|---|---|"]
    for t in ("negative_flip", "angle_out_of_range", "sentinel"):
        L.append(f"| {t} | {_pct(so['tp'][t], so['n'][t])} | {_pct(wr['tp'][t], wr['n'][t])} |")
    L += ["", "Rules add recall only where a value is numerically ordinary but physically impossible; "
              "their false-alarm count is the 'rule_range' entry above (expected 0, because the base data respects "
              "every documented limit).", "",
          "## 4. Outlier-threshold sensitivity (documented rules on)", "",
          "| threshold (x IQR) | recall: corrupt_magnitude | recall: scale_moderate | recall: subtle_perturb "
          "| cell precision | valid extremes flagged |", "|---|---|---|---|---|---|"]
    for k in m["ks"]:
        r = sw[k]
        L.append(f"| {k} | {_pct(r['tp']['corrupt_magnitude'], r['n']['corrupt_magnitude'])} | "
                 f"{_pct(r['tp']['scale_moderate'], r['n']['scale_moderate'])} | "
                 f"{_pct(r['tp']['subtle_perturb'], r['n']['subtle_perturb'])} | "
                 f"{_pct(r['tp_cells'], r['flagged_cells'])} | {_pct(r['extreme_flagged'], r['extreme_total'])} |")
    L += ["", "## Limits of this evaluation", "",
          "- The base data is synthetic and the error types are our own, so recall is an upper bound for real data.",
          "- Errors that stay inside the plausible range (moderate scaling, small perturbations) are invisible to "
          "per-column statistics and range rules; catching them needs cross-field or temporal consistency checks, "
          "which this tool does not have.",
          "- Precision on real data cannot be measured without independent labels; use `--labels` for that.", ""]
    return "\n".join(L)


def score_against_labels(audit_log: str, labels: str) -> dict:
    """Score a real audit_log.csv against independently made labels.

    labels CSV: `line` (file line, header = 1) and optional `column` (blank = whole row).
    """
    log = pd.read_csv(audit_log)
    lab = pd.read_csv(labels)
    if "line" not in lab.columns:
        raise ValueError("labels file needs a 'line' column")
    lab["column"] = lab["column"].fillna("") if "column" in lab.columns else ""
    cell = {(int(l), c) for l, c in zip(lab["line"], lab["column"]) if c}
    rowl = {int(l) for l, c in zip(lab["line"], lab["column"]) if not c}
    flagged_cells = set(zip(log.loc[log["scope"] == "cell", "line"].astype(int), log.loc[log["scope"] == "cell", "column"]))
    flagged_rows = set(log["line"].astype(int))  # any finding on the line counts for a row label
    tp = len(cell & flagged_cells) + len(rowl & flagged_rows)
    total = len(cell) + len(rowl)
    flagged = len(flagged_cells) + len(set(log.loc[log["scope"] == "row", "line"].astype(int)))
    hit = len(flagged_cells & cell) + len(set(log.loc[log["scope"] == "row", "line"].astype(int)) & rowl)
    return {"labelled": total, "caught": tp, "recall": tp / total if total else None,
            "flagged_items": flagged, "flagged_and_labelled": hit,
            "precision_lower_bound": hit / flagged if flagged else None,
            "note": "unlabelled flags may be real errors nobody labelled, so this precision is a lower bound; "
                    "if audit_log.csv was capped (--audit-cap) recall is a lower bound too"}
