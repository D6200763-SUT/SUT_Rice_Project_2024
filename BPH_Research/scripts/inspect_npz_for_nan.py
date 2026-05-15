#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_npz_for_nan.py
Usage:
python code/inspect_npz_for_nan.py --npz out_feature_sets/core/sequences_window30_h1.npz
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    return p.parse_args(argv)

def counts(a):
    a = np.asarray(a)
    if not np.issubdtype(a.dtype, np.floating):
        return 0, 0
    return int(np.isnan(a).sum()), int(np.isinf(a).sum())

def main(argv=None):
    args = parse_args(argv)
    d = np.load(Path(args.npz).expanduser().resolve(), allow_pickle=True)
    for k in sorted([k for k in d.files if k.startswith(("X_","y_"))]):
        nan, inf = counts(d[k])
        print(f"{k:20s} shape={d[k].shape} dtype={d[k].dtype} nan={nan} inf={inf}")
    if "meta" in d.files:
        print("meta present")

if __name__ == "__main__":
    main()
