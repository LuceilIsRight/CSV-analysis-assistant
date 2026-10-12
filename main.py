#!/usr/bin/env python3
"""CSV Analysis Assistant: python main.py data.csv --output results/"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from csv_assistant import analysis, audit, plots, report
from csv_assistant.loader import CSVLoadError, load_csv


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="First-pass inspection of a CSV file.")
    p.add_argument("csv_file", help="Path to the input CSV file")
    p.add_argument("--output", "-o", default="results", help="Output directory (default: results/)")
    p.add_argument("--max-categories", type=int, default=20,
                   help="Max unique values for a categorical bar chart (default: 20)")
    p.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    p.add_argument("--id-cols", default="",
                   help="Comma-separated numeric columns to treat as identifiers "
                        "(no statistics, histograms, correlation or sentinel checks)")
    p.add_argument("--key", default="",
                   help="Comma-separated key columns; reports rows that repeat an earlier key, "
                        "split into identical and conflicting rows")
    p.add_argument("--outlier-iqr", type=float, default=1000.0,
                   help="Flag values further than this many IQRs from the median (default: 1000; 0 disables)")
    p.add_argument("--corr", choices=["auto", "pearson", "spearman"], default="auto",
                   help="Correlation method; auto uses Spearman for skewed or outlier-heavy columns")
    p.add_argument("--exclude-empty", action="store_true",
                   help="Exclude 'empty-looking' rows (every numeric column 0 or missing) from "
                        "statistics, plots and correlations; the input file is never modified")
    p.add_argument("--rules", default="",
                   help="JSON file of documented-constraint rules (e.g. rules/sharp_hmi.json)")
    p.add_argument("--audit-cap", type=int, default=100_000,
                   help="Max items written to audit_log.csv per check and column (counts are always exact)")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    out_dir = Path(args.output)
    try:
        df, info = load_csv(args.csv_file)
    except CSVLoadError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    def columns_from(spec: str, flag: str) -> list[str]:
        names = [x.strip() for x in spec.split(",") if x.strip()]
        unknown = [n for n in names if n not in df.columns]
        if unknown:
            parser.error(f"{flag}: unknown column(s) {unknown}. Available: {list(df.columns)}")
        return names

    id_cols = columns_from(args.id_cols, "--id-cols")
    key = columns_from(args.key, "--key")

    # Whole-file facts, always computed on every row.
    overview = analysis.column_overview(df)
    missing = analysis.missing_values(df)
    dups = analysis.duplicate_info(df)
    key_info = analysis.key_duplicate_info(df, key) if key else None
    miss_info = analysis.missingness_patterns(df)
    empty_info, empty_mask = analysis.empty_row_info(df, id_cols)
    rules = None
    if args.rules:
        try:
            rules = audit.load_rules(args.rules)
        except ValueError as e:
            parser.error(f"--rules: {e}")
    det = audit.run_detection(df, id_cols, rules, args.outlier_iqr, args.audit_cap)

    # The analysis view: optionally without empty-looking rows. The data file is untouched.
    adf, analysis_note = df, None
    if args.exclude_empty and empty_info:
        adf = df.loc[~empty_mask]
        analysis_note = (f"Statistics, plots and correlations use {len(adf)} rows, excluding "
                         f"{empty_info['count']} empty-looking rows (--exclude-empty)")
    corr_warning = None
    if empty_info and not args.exclude_empty:
        corr_warning = (f"correlation matrix includes {empty_info['count']} empty-looking rows "
                        "(every numeric column 0 or missing), which can create spurious correlations; "
                        "consider --exclude-empty")

    stats = analysis.summary_statistics(adf, id_cols)
    outliers = analysis.outlier_info(adf, id_cols, args.outlier_iqr)
    flags = analysis.data_quality_flags(adf, id_cols, outliers)

    written, skipped = ([], ["plots disabled by --no-plots"]) if args.no_plots else \
        plots.make_plots(adf, out_dir, max_categories=args.max_categories, id_cols=id_cols,
                         outliers=outliers, corr_method=args.corr, corr_warning=corr_warning)

    command = "python main.py " + " ".join(sys.argv[1:] if argv is None else argv)
    rpt = report.write_outputs(out_dir, df, info, overview, missing, dups, stats, flags,
                               written, skipped, command, key_info, miss_info,
                               empty_info, analysis_note,
                               {"summary": det.summary, "cap": args.audit_cap, "rules_skipped": det.rules_skipped,
                                "rules_name": rules["name"] if rules else None})
    audit.write_audit(out_dir, det, {"outlier_iqr": args.outlier_iqr, "audit_cap": args.audit_cap,
                                     "id_cols": id_cols, "rules_file": args.rules or None,
                                     "input_sha256": info["sha256"]})
    print(f"{len(df)} rows x {df.shape[1]} columns | duplicates: {dups['duplicate_rows']} | "
          f"empty-looking rows: {empty_info['count'] if empty_info else 0} | "
          f"flags: {len(flags)} | plots: {len(written)}")
    print(f"Report written to {rpt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
