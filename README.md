# CSV Analysis Assistant 

A Python command-line tool for a first-pass inspection of CSV datasets. It reports the structure of a file, flags data-quality problems, draws basic plots, and writes a reproducible Markdown report together with an item-level audit trail. It **detects and explains; it never modifies or corrects your data.**

**Status: alpha (v0.2.0-dev).** The inspection, audit and evaluation features below are implemented and tested. Automatic cleaning is **not** implemented (see [Roadmap](#roadmap)).

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Alpha-orange?style=flat)
![Tests](https://img.shields.io/badge/Tests-37_passing-brightgreen?style=flat)

---

## Contents 

- [What it does](#what-it-does)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Command-line reference](#command-line-reference)
- [Outputs](#outputs)
- [What gets checked](#what-gets-checked)
- [Documented-constraint rules](#documented-constraint-rules)
- [Data handling principles](#data-handling-principles)
- [Evaluation](#evaluation)
- [Testing](#testing)
- [Project layout](#project-layout)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Licence](#licence)

---

## What it does

- Reads a CSV with encoding fallback (`utf-8-sig`, `utf-8`, `cp1252`, `latin-1`) and delimiter detection (`, ; tab |`), with readable errors for missing, empty or malformed files.
- Reports dimensions, column types, missing values, which columns are **missing together**, duplicate rows and (optionally) duplicate keys.
- Computes summary statistics, including exact-zero counts per column. Infinite values are ignored in statistics and counted separately.
- Raises heuristic **data-quality flags**: infinities, extreme outliers, suspected missing-value codes, placeholder text, numbers or dates stored as text, delimited lists, constant and ID-like columns, whitespace, mixed types.
- Draws histograms, bar charts and a correlation matrix using explicit, reported selection rules.
- Writes an **audit trail**: one line per flagged cell or row, with the check, severity, file line and an explanation.
- Optionally applies **documented physical constraints** from a rules file, so values that look numerically ordinary but are physically impossible are caught.
- Includes an **evaluation harness** that injects labelled errors and measures recall and precision of the detectors.

What it deliberately does **not** do: decide whether an unusual value is wrong, fix anything, or judge scientific validity. Statistical flags are warnings; only constraints from a documented rule are reported as errors.

## Installation

```bash
git clone https://github.com/LuceilIsRight/CSV-Analysis-Assistant.git
cd CSV-Analysis-Assistant
```

Create a virtual environment.

**Windows (PowerShell):**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies (`pytest` is only needed to run the tests):

```bash
python -m pip install --upgrade pip
python -m pip install pandas numpy matplotlib pytest
```

Tested with Python 3.12 / pandas 3.0.2 / NumPy 2.4 / Matplotlib 3.10, and with Python 3.14 / pandas 3.0.5 on a real 710 MB, 5.96M-row file. Other versions are untested. There is no `requirements.txt` yet.

## Quick start

```bash
python main.py sample_data.csv --output results/
```

Run it **from the project root**: `main.py` imports the `csv_assistant/` package that sits beside it.

A fuller example, for a time-series file with identifier columns, a duplicate key and documented constraints:

```bash
python main.py data.csv -o results/ \
    --id-cols HARPNUM,NOAA_AR \
    --key T_REC,HARPNUM \
    --rules rules/sharp_hmi.json
```

Then open `results/report.md`.

## Command-line reference

```text
python main.py CSV_FILE [options]
```

| Option | Meaning |
|---|---|
| `-o`, `--output DIR` | Output directory (default `results/`). |
| `--id-cols A,B` | Numeric columns to treat as identifiers: no statistics, histograms, correlation or sentinel checks. The tool also guesses IDs from names (`id`, `*_id`, `userId`, …) and from consecutive integer runs; this option overrides the guess. |
| `--key A,B` | Key columns. Reports rows repeating an earlier key, split into **identical** rows and **conflicting** rows (same key, different values). |
| `--rules FILE` | JSON rules file of documented constraints (see below). Without it, no physical-consistency checks run. |
| `--outlier-iqr K` | Flag values more than K×IQR from the median (default `1000`; `0` disables). |
| `--corr {auto,pearson,spearman}` | Correlation method. `auto` uses Spearman when a column is skewed (\|skew\| > 2) or has extreme values. |
| `--exclude-empty` | Compute statistics, plots and correlations without "empty-looking" rows (every informative numeric column is exactly 0 or missing). The file is never modified, and whole-file facts and the audit trail still cover every row. |
| `--max-categories N` | Largest number of distinct values for a bar chart (default `20`). |
| `--no-plots` | Skip plots. |
| `--audit-cap N` | Maximum items written to `audit_log.csv` per check and column (default `100000`). Counts in the report are always exact. |

Exit code `1` means the file could not be read; invalid options (for example an unknown column in `--id-cols`) stop with a usage error that lists the available columns.

## Outputs

```text
results/
├── report.md               human-readable report
├── summary_statistics.csv  numeric summary (finite values only)
├── missing_values.csv      missing counts per column
├── audit_log.csv           one line per flagged cell or row
├── audit_summary.json      explanation of every (check, column, rule) group
└── plots/
    ├── hist_<column>.png
    ├── bar_<column>.png
    └── correlation_matrix.png
```

Which plots exist depends on the data; every skipped column is listed with a reason under **Plot notes** in the report.

`report.md` contains: reproducibility details (command, SHA-256 of the input, encoding, delimiter, tool and library versions), dataset size and duplicates, key duplicates (if `--key`), column overview, missing values and missingness patterns, empty-looking rows, data-quality flags, numerical summary with cautions, plots with plot notes, the audit trail, and limitations.

### The audit trail

`audit_log.csv` columns: `check, severity, scope, column, rule_id, row, line, value`.

- `line` is the line number in the file (header = line 1; assumes one record per line). `row` is the 0-based position in the loaded table.
- `severity` is **error** (not a finite number, or breaks a constraint that follows from a documented definition) or **warning** (statistically unusual or suspicious; may be a real measurement).
- The reason, rule and source for each group are in `audit_summary.json` and in the report's *Audit trail* table.

## What gets checked

**Audit checks (item-level):**

| Check | Severity | Scope | Meaning |
|---|---|---|---|
| `infinite` | error | cell | value is `+inf` or `-inf` |
| `rule_range` | error | cell | breaks a constraint from the rules file |
| `outlier_iqr` | warning | cell | more than K×IQR from the median (bounds computed on non-empty rows) |
| `sentinel` | warning | cell | equals -999, -9999, 9999 or 99999 **and** sits at the column's minimum or maximum |
| `duplicate_row` | warning | row | identical to an earlier row |
| `blank_row` | warning | row | every informative numeric column is exactly 0 or missing |

**Column-level flags (in the report):** `all_missing`, `constant`, `id`, `infinite`, `outliers`, `sentinel`, `categorical_int` (integer column with very few distinct values, possibly a category or ID), `placeholder` (names the actual values, e.g. `MISSING`), `numeric_text`, `date_text`, `delimited_list`, `whitespace`, `mixed_types`.

**Plot selection rules** (all stated in the report):

- numeric, non-ID columns → histogram; columns with extreme outliers are clipped to the 0.5–99.5 percentile range, and strongly right-skewed positive columns (skew > 3) use a log₁₀ axis. Both are noted.
- text/boolean columns → bar chart when there are 2 to `--max-categories` distinct values and the column is not near-unique.
- text columns that really hold numbers or dates are skipped, because they need cleaning first.
- correlation matrix → needs two or more plottable numeric columns. Spearman here ranks each column once over its non-missing values, then correlates pairwise-complete rows (an approximation of exact pairwise Spearman, chosen so it scales).
- A failure while plotting one column becomes a note in the report and never aborts the run.

## Documented-constraint rules

A rules file lets you encode constraints that follow from the **published definition** of a quantity. A value that breaks one is an *error*; an unusual value that breaks none is only ever a *warning*.

```json
{
  "name": "My constraints",
  "rules": [
    {"id": "FLUX-nonneg", "column": "USFLUX", "min": 0,
     "basis": "definition: total UNSIGNED flux (sum of |Bz| dA)",
     "description": "unsigned flux cannot be negative",
     "source": "Bobra et al. 2014, arXiv:1404.1879"}
  ]
}
```

Every rule needs `id`, `column`, `description` and `source`, plus `min` and/or `max` (inclusive). `basis` and `unit` are optional. A rules file with a rule that has no source is rejected. Rules whose column is absent from the CSV are listed in the report as not applied.

`rules/sharp_hmi.json` covers the SDO/HMI SHARP keywords: `USFLUX`, `TOTUSJH`, `SAVNCPP` and `TOTPOT` are non-negative; `MEANSHR` lies in [0°, 180°]; `MEANGAM` lies within ±90°. `MEANJZH` and `MEANALP` are deliberately **not** constrained because they are signed quantities. No upper bounds are given for flux, helicity, current or energy keywords.

> The SHARP definitions were checked against formulas reproduced from Bobra et al. 2014 in arXiv:1603.03775 and arXiv:2207.07254, not against the primary table. Verify them before relying on them in a publication.

## Data handling principles

- The input file is only read. Its SHA-256 is recorded in the report.
- Nothing is removed, replaced or corrected. `--exclude-empty` changes only which rows the statistics and plots use, and the report says so.
- Detection and correction are separate by design: a future cleaning step should consume `audit_log.csv`, not re-derive its own opinion.
- Every flag carries its reason; heuristic flags say they are heuristics.
- Whole-file facts (size, missing values, duplicates, audit trail) always cover every row.

## Evaluation

`evaluate.py` measures the detectors on **synthetic SHARP-like data with deliberately injected, labelled errors** (3 seeds × 200,000 rows × 300 injections per error type), using the same detection code as `main.py`:

| Injected error | Recall: statistical checks only | Recall: with documented rules |
|---|---|---|
| `±inf` values | 100% | 100% |
| value ×10⁸ to ×10²⁵ (overflow-like) | 100% | 100% |
| duplicate rows / all-zero rows | 100% / 100% | 100% / 100% |
| sentinel codes (-999, 9999, …) | 42% | 72% |
| sign flip in a non-negative column | 0% | 100% |
| angle outside its documented range | 0% | 100% |
| value ×3 to ×30 (plausible-looking) | 0.2% | 0.2% |
| value ×1.1 to ×1.5 (small, in-range) | 0% | 0% |

Overall cell recall is 35% with statistics alone and 67% with the documented rules. Of 4,251 cells flagged, 2 were not injected (both warnings on valid heavy-tailed values), and 1 of 2,657 untouched valid-but-extreme cells was flagged. Lowering the outlier threshold from 1000 to 100×IQR raised the valid-extreme false-alarm rate to 17.5% while barely improving recall.

**Read these numbers with care.** The base data is synthetic and the error types were chosen by the author, so recall is an upper bound for real data. The synthetic heavy tails are lighter than those of some real SHARP columns, so the false-alarm rate is probably optimistic. Errors that stay inside the plausible range are invisible to this tool by construction.

```bash
python evaluate.py                       # synthetic benchmark (writes eval_results/evaluation_report.md)
python evaluate.py --n 100000 --seeds 5  # other sizes
python evaluate.py --audit results/audit_log.csv --labels my_labels.csv
```

The last form scores a real `audit_log.csv` against **your own independent labels**: a CSV with a `line` column (file line, header = 1) and an optional `column` (blank means the whole row). Precision from it is a lower bound, because unlabelled flags may be real errors nobody has labelled yet. If the audit log was capped by `--audit-cap`, recall is a lower bound too.

## Testing

```bash
python -m pytest -q
```

37 tests cover loading and error handling, flags and heuristics, plot selection, infinite values, duplicates and keys, missingness patterns, outliers, the audit trail (including that the input is never modified), rule validation, and a smoke test of the evaluation harness. Test results should be quoted only after actually running them.

## Project layout

```text
CSV-Analysis-Assistant/
├── main.py                  command-line entry point
├── evaluate.py              evaluation entry point
├── sample_data.csv          small deliberately messy demo file
├── rules/
│   └── sharp_hmi.json       documented constraints for SDO/HMI SHARP keywords
├── csv_assistant/
│   ├── loader.py            reading, encoding and delimiter detection
│   ├── analysis.py          statistics, flags, duplicates, outliers, missingness
│   ├── audit.py             item-level detection and audit-trail writing
│   ├── plots.py             plot selection and rendering
│   ├── report.py            Markdown report
│   └── evaluation.py        error injection, scoring, report rendering
└── tests/
    ├── test_basic.py
    └── test_audit.py
```

The modules use relative imports, so keep them inside `csv_assistant/`.

## Limitations

- **Exploratory only.** It does not determine whether outliers are errors or whether any statistical conclusion is justified. You remain responsible for interpreting the results.
- **No temporal or cross-field consistency checks.** Errors that stay inside the plausible range (a value scaled by 3 to 30, or perturbed by 10% to 50%) are effectively undetectable.
- **Sentinel detection is conservative.** A sentinel code is flagged only when it sits at a column's extreme, which avoids false alarms on valid numbers like a patch ID of 9999 but misses codes hidden inside wide-tailed columns.
- **Heuristics have thresholds** (outliers at 1000×IQR, ID detection, "numbers or dates stored as text" at 90% / 80%). Date detection parses an evenly spaced sample of up to 1,000 values, and the report says so.
- **The file is loaded fully into memory.** There is no chunked or sampled loading. It ran on a 710 MB, 5.96M-row file; memory use on that file was not profiled. The audit adds roughly 10 s per million rows in a synthetic test.
- **Line numbers assume one record per line** (no multi-line quoted fields).
- **Numbers or dates stored as text are not parsed.** They are flagged and kept out of plots and statistics until cleaned.
- **Physical rules are only as good as their sources.** Verify them against the primary documentation.
- Tested on a small number of environments (see [Installation](#installation)).

## Roadmap

Done:

- [x] Read a CSV; dimensions, types, missing values, duplicates, summary statistics
- [x] Plots with explicit selection rules; Markdown report with reproducibility details
- [x] Automated tests; configuration through command-line options; validation and error messages
- [x] Demonstration dataset and documented limitations
- [x] Infinite-value, outlier, sentinel, placeholder, delimited-list and empty-row detection
- [x] Missingness patterns, key duplicates, Spearman correlation, log and clipped histograms
- [x] Item-level audit trail and documented-constraint rules
- [x] Evaluation harness with labelled error injection

Next:

- [ ] Optional, explicit cleaning (`--clean`) that consumes the audit log, reports exactly what changed, and writes a separate cleaned file
- [ ] Parse dates and report time coverage and gaps for time-series files
- [ ] Temporal-consistency check per series (jump detection) to catch in-range errors
- [ ] Chunked or sampled loading for very large files
- [ ] Evaluation on independently labelled real data

Later:

- [ ] Optional HTML reports and more plot types
- [ ] `requirements.txt` / packaging
- [ ] Continuous integration

## Licence

No licence has been chosen yet. Choose one before inviting external reuse or contributions, and confirm that all code and sample data can be distributed under it.

## Author

[@LuceilIsRight](https://github.com/LuceilIsRight)
