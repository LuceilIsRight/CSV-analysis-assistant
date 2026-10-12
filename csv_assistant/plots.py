"""Plot selection rules (deliberately explicit) and rendering.

Rules:
  * numeric, non-ID, >= 2 unique finite values -> histogram (one PNG per column)
      - infinite values are excluded and counted in the plot notes
      - columns with extreme outliers are clipped to the 0.5-99.5 percentile range
      - mostly-positive columns (<5% negative) whose positive part is strongly right-skewed
        (skew > 2) are drawn on a log10 scale; zeros are not shown but are counted in the notes
  * text columns that are really numbers or dates -> skipped (they need cleaning first)
  * text/bool, 2..max_categories unique values, and not near-unique relative to row count
    (unique/non-null > 0.5 with > 5 unique) -> bar chart of counts, whitespace stripped
  * correlation matrix: plottable numeric columns without a suspected sentinel, if >= 2.
      Pearson, or Spearman when any column is skewed (|skew| > 2) or has extreme outliers.
  * everything else is skipped, and the reason is reported in the plot notes
  * a failure while plotting one column is reported as a note and never aborts the run
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from .analysis import (finite_view, is_id_like, is_text, numeric_columns,
                       numeric_sentinels, text_kind)


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(name))[:60]


def _plot_column(df, c, ctx):
    """Plot one column. Returns relative path or None. May raise; caller handles it."""
    notes, outliers = ctx["notes"], ctx["outliers"]
    s = df[c].dropna()
    if len(s) == 0:
        notes.append(f"{c}: no non-missing values")
        return None
    if c in ctx["num"]:
        if c in ctx["id_cols"] or is_id_like(df[c], c):
            notes.append(f"{c}: ID column")
            return None
        n_inf = int(np.isinf(s).sum()) if pd.api.types.is_float_dtype(s) else 0
        if n_inf:
            s = s[~np.isinf(s)]
            notes.append(f"{c}: {n_inf} infinite value(s) excluded from the plot")
        if len(s) == 0 or s.nunique() < 2:
            notes.append(f"{c}: constant or no finite values, not plotted")
            return None
        sent = numeric_sentinels(df[c])
        if sent:
            notes.append(f"{c}: histogram includes suspected sentinel value(s) {list(sent)}; "
                         "excluded from the correlation matrix")
        else:
            ctx["corr_cols"].append(c)
            ctx["skews"][c] = float(s.skew()) if len(s) > 2 else 0.0
        title_suffix, xlabel = "", str(c)
        if c in outliers:
            lo, hi = s.quantile([0.005, 0.995])
            clipped = s[(s >= lo) & (s <= hi)]
            if clipped.nunique() >= 2:
                notes.append(f"{c}: histogram clipped to the 0.5-99.5 percentile range "
                             f"({lo:.3g} to {hi:.3g}) because {outliers[c]['count']} value(s) are extreme")
                s, title_suffix = clipped, " (0.5-99.5 percentile)"
        pos = s[s > 0]
        if (len(pos) > 2 and len(pos) / len(s) >= 0.5 and (s < 0).mean() < 0.05
                and pos.skew() > 2):
            dropped = len(s) - len(pos)
            notes.append(f"{c}: histogram on a log10 scale (strongly right-skewed)"
                         + (f"; {dropped} zero/negative value(s) not shown" if dropped else ""))
            s, xlabel = np.log10(pos), f"log10({c})"
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(s, bins="auto", edgecolor="black")
        ax.set_title(f"Distribution of {c}{title_suffix}")
        ax.set_xlabel(xlabel); ax.set_ylabel("Count")
        fname = f"hist_{_safe(c)}.png"
    else:
        tk = text_kind(df[c])
        if tk:
            label = "numbers" if tk[0] == "numeric_text" else "dates"
            notes.append(f"{c}: text column that mostly holds {label}; needs cleaning before plotting")
            return None
        if is_text(df[c]):
            stripped = s.astype(str).str.strip()
            if (stripped != s.astype(str)).any():
                notes.append(f"{c}: leading/trailing whitespace stripped for counting")
            s = stripped
        n_unique = s.nunique()
        if n_unique < 2:
            notes.append(f"{c}: constant column")
            return None
        if n_unique > ctx["max_categories"]:
            notes.append(f"{c}: {n_unique} unique values (> {ctx['max_categories']}), "
                         "treated as high-cardinality")
            return None
        if n_unique > 5 and n_unique / len(s) > 0.5:
            notes.append(f"{c}: {n_unique} unique values in {len(s)} rows (near-unique), not plotted")
            return None
        fig, ax = plt.subplots(figsize=(6, 4))
        s.astype(str).value_counts().plot.bar(ax=ax, edgecolor="black")
        ax.set_title(f"Counts of {c}")
        ax.set_ylabel("Count")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fname = f"bar_{_safe(c)}.png"
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(ctx["plot_dir"] / fname, dpi=120)
    plt.close(fig)
    return f"plots/{fname}"


def make_plots(df: pd.DataFrame, out_dir: Path, max_categories: int = 20,
               max_plots: int = 30, id_cols=(), outliers: dict | None = None,
               corr_method: str = "auto", corr_warning: str | None = None) -> tuple[list[str], list[str]]:
    """Return (written_relative_paths, plot_notes)."""
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    ctx = {"num": set(numeric_columns(df)), "plot_dir": plot_dir, "max_categories": max_categories,
           "notes": [], "corr_cols": [], "skews": {}, "id_cols": set(id_cols),
           "outliers": outliers or {}}
    notes = ctx["notes"]

    for c in df.columns:
        if len(written) >= max_plots:
            notes.append(f"Plot limit ({max_plots}) reached; remaining columns not plotted.")
            break
        try:
            path = _plot_column(df, c, ctx)
        except Exception as e:  # one bad column must not abort the whole run
            plt.close("all")
            notes.append(f"{c}: plot failed ({type(e).__name__}: {e})")
            if c in ctx["corr_cols"]:
                ctx["corr_cols"].remove(c)
            continue
        if path:
            written.append(path)

    corr_cols = ctx["corr_cols"]
    if len(corr_cols) >= 2:
        try:
            skewed = [c for c in corr_cols if c in ctx["outliers"] or abs(ctx["skews"].get(c, 0)) > 2]
            method = corr_method
            if method == "auto":
                method = "spearman" if skewed else "pearson"
            data = finite_view(df[corr_cols])
            if method == "spearman":
                # ranks per column over all non-missing values, then Pearson on pairwise-complete rows
                data = data.rank()
                notes.append("correlation matrix: Spearman (rank) correlation"
                             + (f", chosen because {', '.join(skewed)} are heavy-tailed or have extreme values"
                                if corr_method == "auto" else "")
                             + "; ranks are computed per column over all its non-missing values")
            corr = data.corr()
            side = 1 + 0.8 * len(corr_cols)
            fig, ax = plt.subplots(figsize=(side, side))
            im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
            ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
            ax.set_xticklabels(corr.columns, rotation=45, ha="right")
            ax.set_yticklabels(corr.columns)
            fig.colorbar(im, ax=ax, label=f"{method.capitalize()} r")
            ax.set_title(f"Correlation matrix ({method}, {len(df)} rows)")
            fig.tight_layout()
            fig.savefig(plot_dir / "correlation_matrix.png", dpi=120)
            plt.close(fig)
            written.append("plots/correlation_matrix.png")
            if corr_warning:
                notes.append(corr_warning)
            if len(df) < 30:
                notes.append(f"correlation matrix: only {len(df)} rows; correlations are unreliable at this size")
        except Exception as e:
            plt.close("all")
            notes.append(f"correlation matrix failed ({type(e).__name__}: {e})")
    else:
        notes.append("correlation matrix: needs at least 2 numeric columns that are not IDs or suspected sentinels")
    return written, notes
