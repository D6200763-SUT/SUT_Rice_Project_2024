#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
summarize_walk_forward.py
Aggregate Stage 1 + Stage 2 metrics across walk-forward CV folds.

Produces:
  cv_summary.csv    — one row per fold
  cv_report.md      — manuscript-ready table with Mean ± SD
  figures/cv_r2_barplot.png

Usage:
python scripts/summarize_walk_forward.py \
  --root results/walk_forward \
  --folds 2,3,4 \
  --out results/walk_forward/cv_summary.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARNING] Cannot parse {path}: {e}")
        return None


def collect_fold(root: Path, fold: int) -> Dict[str, Any]:
    """Gather Stage 1 + Stage 2 metrics for one fold."""
    fold_dir = root / f"fold{fold}"
    row: Dict[str, Any] = {"fold": fold}

    # ---- Stage 1 ----
    s1 = load_json(fold_dir / "stage1" / "metrics_stage1.json")
    if s1:
        row["recall_s1"]        = s1.get("recall")
        row["precision_s1"]     = s1.get("precision")
        row["f1_s1"]            = s1.get("f1")
        row["auc_s1"]           = s1.get("auc")
        row["best_threshold"]   = s1.get("best_threshold")
    else:
        print(f"[WARNING] Fold {fold}: metrics_stage1.json not found or invalid")
        for k in ["recall_s1", "precision_s1", "f1_s1", "auc_s1", "best_threshold"]:
            row[k] = None

    # ---- Stage 2 ----
    s2 = load_json(fold_dir / "stage2" / "metrics.json")
    if s2:
        tl = s2.get("test_log1p", {})
        tr = s2.get("test_raw",   {})
        row["r2_log1p_s2"]   = tl.get("r2")
        row["rmse_log1p_s2"] = tl.get("rmse")
        row["r2_raw_s2"]     = tr.get("r2")
        row["rmse_raw_s2"]   = tr.get("rmse")
    else:
        print(f"[WARNING] Fold {fold}: stage2/metrics.json not found or invalid")
        for k in ["r2_log1p_s2", "rmse_log1p_s2", "r2_raw_s2", "rmse_raw_s2"]:
            row[k] = None

    return row


def mean_sd(values: list) -> str:
    """Format mean±SD for non-None numeric values."""
    valid = [v for v in values if v is not None]
    if not valid:
        return "N/A"
    arr = np.array(valid, dtype=float)
    return f"{arr.mean():.3f}±{arr.std():.3f}"


def fmt(v) -> str:
    if v is None:
        return "N/A"
    return f"{float(v):.3f}"


def make_report(rows: list, folds: list) -> str:
    def col(key):
        return [r.get(key) for r in rows]

    header = "| Metric              | " + " | ".join(f"Fold {f}" for f in folds) + " | Mean ± SD |"
    sep    = "|---------------------|" + "|".join(["--------"] * len(folds)) + "|-----------|"

    metrics = [
        ("Stage1 Recall",    "recall_s1"),
        ("Stage1 Precision", "precision_s1"),
        ("Stage1 F1",        "f1_s1"),
        ("Stage1 AUC",       "auc_s1"),
        ("Best Threshold",   "best_threshold"),
        ("Stage2 R² (log1p)", "r2_log1p_s2"),
        ("Stage2 RMSE_log1p", "rmse_log1p_s2"),
        ("Stage2 R² (raw)",   "r2_raw_s2"),
        ("Stage2 RMSE_raw",   "rmse_raw_s2"),
    ]

    lines = [
        "## Walk-Forward Cross-Validation Results",
        "",
        header, sep,
    ]
    for label, key in metrics:
        vals = col(key)
        cells = " | ".join(fmt(v) for v in vals)
        lines.append(f"| {label:<19} | {cells} | {mean_sd(vals)} |")

    lines += [
        "",
        "**Note:** Fold 4 test period (2019) corresponds to the original single-split evaluation.",
        "Best threshold selected per fold to achieve Recall ≥ 0.80 on the validation set.",
        "",
        "**Baseline (single-stage CNN-LSTM, Fold 4 test period):** R²_log1p = 0.484",
    ]
    return "\n".join(lines)


def plot_r2_barplot(rows: list, folds: list, out_png: Path) -> None:
    import matplotlib.pyplot as plt

    r2_log1p = [r.get("r2_log1p_s2") for r in rows]
    r2_raw   = [r.get("r2_raw_s2")   for r in rows]

    x = np.arange(len(folds))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    bars1 = ax.bar(x - width / 2,
                   [v if v is not None else 0.0 for v in r2_log1p],
                   width, label="R² log1p (Stage 2)")
    bars2 = ax.bar(x + width / 2,
                   [v if v is not None else 0.0 for v in r2_raw],
                   width, label="R² raw (Stage 2)")

    # Annotate bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    # Baseline reference line
    ax.axhline(y=0.484, color="red", linestyle="--", lw=1.2, label="Baseline R²_log1p=0.484")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {f}" for f in folds])
    ax.set_ylabel("R²")
    ax.set_title("Walk-Forward CV: Stage 2 R² by Fold")
    ax.legend()
    ax.set_ylim(bottom=min(0, min((v for v in r2_log1p + r2_raw if v is not None), default=0)) - 0.05)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"[OK] Saved: {out_png}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Summarize walk-forward CV results")
    p.add_argument("--root",  required=True, help="Walk-forward results root (e.g. results/walk_forward)")
    p.add_argument("--folds", default="2,3,4")
    p.add_argument("--out",   required=True, help="Output CSV path")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = Path(args.root).expanduser().resolve()
    folds = [int(f.strip()) for f in args.folds.split(",")]
    out_csv = Path(args.out).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = [collect_fold(root, fold) for fold in folds]

    # ---- CSV ----
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved: {out_csv}")
    print(df.to_string(index=False))

    # ---- Markdown report ----
    report_md = make_report(rows, folds)
    out_md = out_csv.parent / "cv_report.md"
    out_md.write_text(report_md, encoding="utf-8")
    print(f"[OK] Saved: {out_md}")

    # ---- Bar plot ----
    out_png = root / "figures" / "cv_r2_barplot.png"
    try:
        plot_r2_barplot(rows, folds, out_png)
    except Exception as e:
        print(f"[WARNING] Could not generate bar plot: {e}")

    # ---- Console summary ----
    print("\n--- Summary ---")
    print(report_md)


if __name__ == "__main__":
    main()
