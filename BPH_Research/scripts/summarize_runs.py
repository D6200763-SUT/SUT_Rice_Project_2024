#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
summarize_runs.py
Collect metrics.json from multiple training output folders and create a comparison table.

Usage:
python code/summarize_runs.py --root out_train --out out_train/summary
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd

def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Root folder that contains multiple model run folders")
    p.add_argument("--out", required=True, help="Output folder for summary files")
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    root = Path(args.root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for m in root.rglob("metrics.json"):
        try:
            obj = json.loads(m.read_text(encoding="utf-8"))
        except Exception:
            continue
        model = obj.get("model", m.parent.name)
        npz = Path(obj.get("npz","")).name
        meta = obj.get("meta", {})
        set_name = meta.get("set_name","?")
        horizon = meta.get("horizon", meta.get("window", "?"))
        window = obj.get("window", meta.get("window","?"))
        n_features = obj.get("n_features", meta.get("n_features","?"))

        tlog = obj.get("test_log1p", {})
        traw = obj.get("test_raw", {})

        rows.append({
            "model": model,
            "run_dir": str(m.parent),
            "npz": npz,
            "set": set_name,
            "window": window,
            "horizon": meta.get("horizon", "?"),
            "n_features": n_features,
            "mae_log1p": tlog.get("mae", None),
            "rmse_log1p": tlog.get("rmse", None),
            "r2_log1p": tlog.get("r2", None),
            "smape_log1p": tlog.get("smape", None),
            "mae_raw": traw.get("mae", None),
            "rmse_raw": traw.get("rmse", None),
            "r2_raw": traw.get("r2", None),
            "smape_raw": traw.get("smape", None),
        })

    if not rows:
        raise SystemExit(f"No metrics.json found under: {root}")

    df = pd.DataFrame(rows)
    # Sort: best log1p RMSE then best raw RMSE (smaller is better)
    df = df.sort_values(["rmse_log1p", "rmse_raw"], ascending=True)

    df.to_csv(out / "comparison.csv", index=False, encoding="utf-8-sig")

    # Also write a simple markdown table
    md = df[[
        "model","set","window","horizon","n_features",
        "rmse_log1p","mae_log1p","r2_log1p",
        "rmse_raw","mae_raw","r2_raw"
    ]].to_markdown(index=False)
    (out / "comparison.md").write_text(md, encoding="utf-8")

    print("[OK] Wrote:", out / "comparison.csv")
    print("[OK] Wrote:", out / "comparison.md")

if __name__ == "__main__":
    main()
