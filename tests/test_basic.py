import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from csv_assistant import analysis  # noqa: E402
from csv_assistant.loader import CSVLoadError, load_csv  # noqa: E402
from main import main  # noqa: E402

SAMPLE = ROOT / "sample_data.csv"


def test_missing_file(tmp_path):
    with pytest.raises(CSVLoadError, match="not found"):
        load_csv(tmp_path / "nope.csv")


def test_empty_file(tmp_path):
    f = tmp_path / "e.csv"; f.write_text("")
    with pytest.raises(CSVLoadError, match="empty"):
        load_csv(f)


def test_header_only_loads_zero_rows(tmp_path):
    f = tmp_path / "h.csv"; f.write_text("a,b\n")
    df, _ = load_csv(f)
    assert df.shape == (0, 2)


def test_semicolon_and_latin1(tmp_path):
    f = tmp_path / "s.csv"; f.write_bytes("a;b\n1;caf\xe9\n2;x\n".encode("latin-1"))
    df, info = load_csv(f)
    assert df.shape == (2, 2) and info["delimiter"] == ";"


def test_duplicates_counted_and_input_untouched():
    df, _ = load_csv(SAMPLE)
    before = df.copy()
    analysis.data_quality_flags(df); analysis.summary_statistics(df)
    pd.testing.assert_frame_equal(df, before)


def test_exact_duplicate_detected(tmp_path):
    f = tmp_path / "d.csv"; f.write_text("a,b\n1,x\n2,y\n1,x\n")
    df, _ = load_csv(f)
    assert analysis.duplicate_info(df)["duplicate_rows"] == 1


def test_flags_catch_messy_data():
    df, _ = load_csv(SAMPLE)
    text = " ".join(f"{f['column']}:{f['issue']}" for f in analysis.data_quality_flags(df))
    assert "age:possible numeric missing-value sentinel" in text
    assert "income:stored as text but" in text
    assert "signup_date:stored as text but" in text and "sampled values parse as dates" in text
    assert "notes" in text and "placeholder" in text


def test_end_to_end_creates_outputs(tmp_path):
    out = tmp_path / "res"
    assert main([str(SAMPLE), "--output", str(out)]) == 0
    assert (out / "report.md").exists()
    assert (out / "summary_statistics.csv").exists()
    assert (out / "missing_values.csv").exists()
    assert any((out / "plots").glob("*.png"))


def test_cli_error_returns_nonzero(tmp_path, capsys):
    assert main([str(tmp_path / "missing.csv"), "-o", str(tmp_path / "o")]) == 1
    assert "Error" in capsys.readouterr().err


def test_income_like_column_is_not_an_id():
    s = pd.Series([52000, 48000, 61000, 57000, 45000])
    assert not analysis.is_id_like(s, "income")
    assert analysis.is_id_like(pd.Series([10, 20, 30]), "customer_id")
    assert analysis.is_id_like(pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 2]), "x")


def test_text_kind():
    assert analysis.text_kind(pd.Series(["1", "2,000", "3"]))[0] == "numeric_text"
    assert analysis.text_kind(pd.Series(["2024-01-02", "2024-02-03"]))[0] == "date_text"
    assert analysis.text_kind(pd.Series(["Pune", "Delhi"])) is None


def test_sample_plots_skip_misleading_columns(tmp_path):
    out = tmp_path / "o"
    assert main([str(SAMPLE), "-o", str(out)]) == 0
    names = {p.name for p in (out / "plots").glob("*.png")}
    assert names == {"hist_age.png", "hist_score.png", "bar_city.png", "bar_notes.png"}
    rpt = (out / "report.md").read_text()
    assert "Plot notes" in rpt and "sentinel" in rpt and "file lines" in rpt


def test_id_column_excluded_from_plots(tmp_path):
    f = tmp_path / "i.csv"; f.write_text("id,v\n1,2.5\n2,3.5\n3,1.0\n4,9.0\n")
    out = tmp_path / "o"
    assert main([str(f), "-o", str(out)]) == 0
    names = {p.name for p in (out / "plots").glob("*.png")}
    assert "hist_id.png" not in names and "hist_v.png" in names


def test_infinite_values_do_not_crash_or_warn(tmp_path):
    import warnings
    f = tmp_path / "inf.csv"
    f.write_text("a,b,c\n1.0,inf,x\n2.0,-inf,y\n3.5,4.0,x\n4.0,5.0,y\n5.0,6.0,x\n")
    out = tmp_path / "o"
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert main([str(f), "-o", str(out)]) == 0
    rpt = (out / "report.md").read_text()
    assert "infinite" in rpt
    assert (out / "plots" / "hist_b.png").exists()


def test_one_failing_column_does_not_abort(tmp_path, monkeypatch):
    from csv_assistant import plots
    f = tmp_path / "ok.csv"
    f.write_text("a,b\n1.0,2.0\n2.0,3.5\n3.5,1.0\n4.0,8.0\n")
    real = plots.plt.subplots
    calls = {"n": 0}
    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return real(*a, **k)
    monkeypatch.setattr(plots.plt, "subplots", flaky)
    out = tmp_path / "o"
    assert main([str(f), "-o", str(out)]) == 0
    assert "plot failed" in (out / "report.md").read_text()


def test_sentinel_only_flagged_at_extremes():
    assert analysis.numeric_sentinels(pd.Series([1, 5, 9999, 15000])) == {}
    assert analysis.numeric_sentinels(pd.Series([-999, 20, 30])) == {-999: 1}
    assert analysis.numeric_sentinels(pd.Series([10.0, 20.0, 9999.0])) == {9999: 1}


def test_comma_lists_are_not_numbers():
    s = pd.Series(["11158,11159"] * 5 + ["11160"] * 5)
    assert analysis.text_kind(s) is None
    assert analysis.text_kind(pd.Series(["45,000", "1,234.5", "7"]))[0] == "numeric_text"


def test_placeholder_values_are_named():
    df = pd.DataFrame({"x": ["MISSING"] * 3 + ["a", "b"]})
    issue = [f["issue"] for f in analysis.data_quality_flags(df) if f["kind"] == "placeholder"][0]
    assert "MISSING" in issue and "3" in issue


def test_counts_formatted_as_integers():
    from csv_assistant import report
    d = pd.DataFrame({"count": [5963121.0], "mean": [1234567.0]}, index=["a"])
    out = report._md_table(d)
    assert "5963121" in out and "e+" in out


def test_declared_id_cols_excluded(tmp_path):
    f = tmp_path / "h.csv"
    f.write_text("harp,x\n" + "".join(f"{i % 3 + 100},{i * 1.5}\n" for i in range(40)))
    out = tmp_path / "o"
    assert main([str(f), "-o", str(out), "--id-cols", "harp"]) == 0
    names = {p.name for p in (out / "plots").glob("*.png")}
    assert "hist_harp.png" not in names and "hist_x.png" in names
    assert "harp" not in pd.read_csv(out / "summary_statistics.csv")["column"].tolist()
    with pytest.raises(SystemExit):
        main([str(f), "-o", str(out), "--id-cols", "nope"])


def test_key_duplicates_split_identical_and_conflicting(tmp_path):
    df = pd.DataFrame({"t": [1, 1, 2, 2, 3], "h": [7] * 5, "v": [1.0, 1.0, 2.0, 9.0, 3.0]})
    k = analysis.key_duplicate_info(df, ["t", "h"])
    assert (k["key_duplicates"], k["identical"], k["conflicting"]) == (2, 1, 1)
    assert k["first_conflicting_lines"] == [5]
    f = tmp_path / "k.csv"; df.to_csv(f, index=False)
    out = tmp_path / "o"
    assert main([str(f), "-o", str(out), "--key", "t,h"]) == 0
    assert "Key duplicates (key: t, h)" in (out / "report.md").read_text()


def test_outlier_info_flags_corrupt_values_not_heavy_tails():
    rng = np.random.default_rng(0)
    x = rng.lognormal(3, 1, 5000)
    y = x.copy(); y[:3] = 1e30
    info = analysis.outlier_info(pd.DataFrame({"ok": x, "bad": y}))
    assert "bad" in info and "ok" not in info and info["bad"]["count"] == 3


def test_categorical_int_hint():
    df = pd.DataFrame({"g": np.repeat(np.arange(20), 150), "v": np.arange(3000) * 1.5})
    kinds = {(f["column"], f["kind"]) for f in analysis.data_quality_flags(df)}
    assert ("g", "categorical_int") in kinds


def test_delimited_list_hint_but_not_thousands():
    lists = pd.DataFrame({"a": ["11158,11159"] * 5 + ["11160"] * 95})
    assert "delimited_list" in {f["kind"] for f in analysis.data_quality_flags(lists)}
    money = pd.DataFrame({"a": ["45,000", "1,234", "7", "8", "9"]})
    assert "delimited_list" not in {f["kind"] for f in analysis.data_quality_flags(money)}


def test_missingness_patterns():
    df = pd.DataFrame({"a": [1, None, None, 4], "b": [1, None, None, 4], "c": [1, 2, None, 4]})
    m = analysis.missingness_patterns(df)
    assert m["patterns"][0]["rows"] == 2 and m["patterns"][0]["missing"] == []
    assert any(p["missing"] == ["a", "b"] for p in m["patterns"])
    assert analysis.missingness_patterns(pd.DataFrame({"a": [1, None]})) is None


def test_summary_has_zero_counts():
    assert analysis.summary_statistics(pd.DataFrame({"x": [0.0, 0.0, 1.0, 2.0]})).loc["x", "zeros"] == 2


def test_outlier_column_clipped_and_spearman(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=2000), "b": rng.normal(size=2000)})
    df.loc[:2, "b"] = 1e30
    f = tmp_path / "o.csv"; df.to_csv(f, index=False)
    out = tmp_path / "o"
    assert main([str(f), "-o", str(out)]) == 0
    rpt = (out / "report.md").read_text()
    assert "beyond 1000" in rpt and "clipped" in rpt and "Spearman" in rpt and "Caution" in rpt


def test_outlier_largest_values_are_distinct():
    x = np.r_[np.random.default_rng(0).normal(size=1000), [1e30] * 5]
    info = analysis.outlier_info(pd.DataFrame({"x": x}))
    assert info["x"]["count"] == 5 and info["x"]["largest"] == [1e30]


def test_log_scale_with_zero_block(tmp_path):
    rng = np.random.default_rng(2)
    x = rng.lognormal(10, 1.5, 3000); x[:700] = 0.0
    f = tmp_path / "z.csv"
    pd.DataFrame({"x": x, "y": rng.normal(size=3000)}).to_csv(f, index=False)
    out = tmp_path / "o"
    assert main([str(f), "-o", str(out)]) == 0
    rpt = (out / "report.md").read_text()
    assert "log10 scale" in rpt and "700 zero" in rpt


def _empty_frame(n=1000):
    rng = np.random.default_rng(1)
    a, b, c = rng.lognormal(3, 1, n), rng.normal(size=n), rng.lognormal(2, 1, n)
    empty = rng.random(n) < 0.2
    a[empty] = 0.0; b[empty] = np.nan; c[empty] = 0.0
    return pd.DataFrame({"a": a, "b": b, "c": c}), empty


def test_empty_rows_detected_and_warned(tmp_path):
    df, empty = _empty_frame()
    info, mask = analysis.empty_row_info(df)
    assert info["count"] == int(empty.sum()) and int(mask.sum()) == int(empty.sum())
    f = tmp_path / "e.csv"; df.to_csv(f, index=False)
    out = tmp_path / "o"
    assert main([str(f), "-o", str(out)]) == 0
    rpt = (out / "report.md").read_text()
    assert "Empty-looking rows" in rpt and "spurious correlations" in rpt


def test_exclude_empty_changes_analysis_view_only(tmp_path):
    df, empty = _empty_frame()
    f = tmp_path / "e.csv"; df.to_csv(f, index=False)
    before = f.read_bytes()
    out = tmp_path / "o"
    assert main([str(f), "-o", str(out), "--exclude-empty"]) == 0
    stats = pd.read_csv(out / "summary_statistics.csv", index_col=0)
    assert stats.loc["a", "zeros"] == 0 and stats.loc["a", "count"] == len(df) - int(empty.sum())
    assert f.read_bytes() == before
    assert "--exclude-empty" in (out / "report.md").read_text()
