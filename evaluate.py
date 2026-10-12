#!/usr/bin/env python3
"""Score the detectors.  python evaluate.py [--n 200000] [--n-each 300] [--seeds 3] [--out eval_results]
or, for real data with your own labels:  python evaluate.py --audit results/audit_log.csv --labels labels.csv"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from csv_assistant import evaluation


def main() -> int:
    p = argparse.ArgumentParser(description="Measure recall/precision of the detectors on labelled errors.")
    p.add_argument("--n", type=int, default=200_000, help="base rows per seed")
    p.add_argument("--n-each", type=int, default=300, help="injections per error type")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--out", default="eval_results")
    p.add_argument("--rules", default="", help="rules JSON (default: rules/sharp_hmi.json)")
    p.add_argument("--audit", default="", help="audit_log.csv from a real run (use with --labels)")
    p.add_argument("--labels", default="", help="CSV of independently labelled errors: line[,column]")
    a = p.parse_args()
    if a.labels:
        if not a.audit:
            p.error("--labels needs --audit")
        print(json.dumps(evaluation.score_against_labels(a.audit, a.labels), indent=2))
        return 0
    res = evaluation.run(a.n, a.n_each, a.seeds, a.rules or None)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    text = evaluation.render(res)
    (out / "evaluation_report.md").write_text(text, encoding="utf-8")
    (out / "evaluation_results.json").write_text(json.dumps(res, indent=2, default=dict), encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
