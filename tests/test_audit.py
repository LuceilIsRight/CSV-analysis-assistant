import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from csv_assistant import audit, evaluation  # noqa: E402
from main import main  # noqa: E402

RULES = {"name": "t", "rules": [{"id": "R-b-nonneg", "column": "b", "min": 0,
                                 "description": "b is a sum of moduli", "source": "test"}]}


def _frame():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(10, 1, 1000), "b": np.abs(rng.normal(5, 1, 1000)),
                       "c": rng.normal(0, 1, 1000), "d": rng.normal(3, 1, 1000)})
    df.loc[10, "a"] = np.inf
    df.loc[20, "b"] = -5.0
    df.loc[30:34, ["a", "b", "c", "d"]] = 0.0
    return pd.concat([df, df.iloc[[40]]], ignore_index=True)


def _rows(det, check):
    f = det.findings
    return sorted(f.loc[f["check"] == check, "row"].tolist())


def test_detection_finds_labelled_items_and_never_modifies():
    df = _frame()
    before = df.copy()
    det = audit.run_detection(df, rules=RULES)
    pd.testing.assert_frame_equal(df, before)
    assert _rows(det, "infinite") == [10]
    assert _rows(det, "rule_range") == [20]
    assert _rows(det, "blank_row") == [30, 31, 32, 33, 34]
    assert _rows(det, "duplicate_row") == [31, 32, 33, 34, 1000]  # rows 31-34 repeat the all-zero row 30
    f = det.findings
    assert f.loc[f["check"] == "rule_range", "line"].tolist() == [22]
    assert not ((f["check"] == "rule_range") & (f["column"] == "c")).any()


def test_signed_column_is_not_flagged_by_rules():
    det = audit.run_detection(_frame(), rules=RULES)
    assert (det.findings["column"] == "c").sum() == 0 or \
        not det.findings.loc[det.findings["column"] == "c", "check"].isin(["rule_range"]).any()


def test_rule_without_source_is_rejected(tmp_path):
    f = tmp_path / "r.json"
    f.write_text(json.dumps({"rules": [{"id": "x", "column": "b", "min": 0, "description": "d"}]}))
    with pytest.raises(ValueError, match="source"):
        audit.load_rules(f)


def test_cli_writes_audit_trail_and_explanations(tmp_path):
    out = tmp_path / "o"
    f = tmp_path / "d.csv"
    _frame().to_csv(f, index=False)
    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps(RULES))
    assert main([str(f), "-o", str(out), "--rules", str(rules)]) == 0
    log = pd.read_csv(out / "audit_log.csv")
    assert {"check", "severity", "scope", "column", "rule_id", "row", "line", "value"} <= set(log.columns)
    summ = json.loads((out / "audit_summary.json").read_text())
    groups = {(g["check"], g["column"]): g for g in summ["groups"]}
    assert "R-b-nonneg" in groups[("rule_range", "b")]["rule_id"]
    assert "source: test" in groups[("rule_range", "b")]["reason"]
    assert "Audit trail" in (out / "report.md").read_text()
    # the input file is untouched
    pd.testing.assert_frame_equal(pd.read_csv(f), pd.read_csv(f))


def test_audit_cap_limits_log_but_not_counts(tmp_path):
    df = pd.DataFrame({"a": np.arange(1000.0), "b": 1.0 + np.arange(1000.0) % 7})
    df["c"] = df["a"] % 5 + 1
    df.loc[:49, "a"] = np.inf
    det = audit.run_detection(df, cap=10)
    g = [s for s in det.summary if s["check"] == "infinite"][0]
    assert g["count"] == 50 and g["written"] == 10


def test_evaluation_smoke():
    res = evaluation.run(n=20000, n_each=60, seeds=1, ks=(1000,))
    wr, so = res["configs"]["with_rules"], res["configs"]["stats_only"]
    assert wr["tp"]["inf"] == wr["n"]["inf"]
    assert wr["tp"]["negative_flip"] >= 0.95 * wr["n"]["negative_flip"]
    assert so["tp"]["negative_flip"] <= 0.1 * so["n"]["negative_flip"]
    assert wr["fp_by_check"]["rule_range"] == 0
    assert wr["rows"]["duplicate_row"]["tp"] == wr["rows"]["duplicate_row"]["n"]
