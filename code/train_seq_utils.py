#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_seq_utils.py
Shared utilities for training models from sequences_window*_h*.npz.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, List

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass


def try_enable_tf_memory_growth() -> None:
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
    except Exception:
        pass


@dataclass
class SeqData:
    X_train: np.ndarray
    y_train: np.ndarray
    y_date_train: np.ndarray
    station_train: np.ndarray

    X_val: np.ndarray
    y_val: np.ndarray
    y_date_val: np.ndarray
    station_val: np.ndarray

    X_test: np.ndarray
    y_test: np.ndarray
    y_date_test: np.ndarray
    station_test: np.ndarray

    meta: Dict[str, Any]


def load_sequences_npz(npz_path: str | Path) -> SeqData:
    npz_path = Path(npz_path).expanduser().resolve()
    data = np.load(npz_path, allow_pickle=True)

    required = ["X_train","y_train","X_val","y_val","X_test","y_test"]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise RuntimeError(f"NPZ missing keys: {missing}. Found keys: {data.files}")

    meta = {}
    if "meta" in data.files:
        try:
            meta_str = data["meta"][0]
            meta = json.loads(str(meta_str))
        except Exception:
            meta = {"raw_meta": str(data["meta"][:1])}

    def _get(name, default):
        return data[name] if name in data.files else default

    return SeqData(
        X_train=data["X_train"],
        y_train=data["y_train"],
        y_date_train=_get("y_date_train", np.array([], dtype="datetime64[ns]")),
        station_train=_get("station_train", np.array([], dtype=object)),

        X_val=data["X_val"],
        y_val=data["y_val"],
        y_date_val=_get("y_date_val", np.array([], dtype="datetime64[ns]")),
        station_val=_get("station_val", np.array([], dtype=object)),

        X_test=data["X_test"],
        y_test=data["y_test"],
        y_date_test=_get("y_date_test", np.array([], dtype="datetime64[ns]")),
        station_test=_get("station_test", np.array([], dtype=object)),

        meta=meta,
    )


def make_out_dir(out_dir: str | Path) -> Path:
    p = Path(out_dir).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    (p / "figures").mkdir(parents=True, exist_ok=True)
    return p


def inverse_log1p(x: np.ndarray) -> np.ndarray:
    return np.expm1(np.asarray(x))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def compute_smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    denom = np.abs(y_true) + np.abs(y_pred) + eps
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def save_predictions_csv(
    out_dir: Path,
    split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_date: Optional[np.ndarray] = None,
    station: Optional[np.ndarray] = None,
    also_raw: bool = True
) -> Path:
    df = pd.DataFrame({"y_true": np.asarray(y_true).reshape(-1), "y_pred": np.asarray(y_pred).reshape(-1)})
    if y_date is not None and len(y_date) == len(df):
        df["date"] = pd.to_datetime(y_date)
    if station is not None and len(station) == len(df):
        df["station_id"] = station.astype(str)
    if also_raw:
        df["y_true_raw"] = inverse_log1p(df["y_true"].values)
        df["y_pred_raw"] = inverse_log1p(df["y_pred"].values)
    if "date" in df.columns:
        df = df.sort_values(["station_id","date"] if "station_id" in df.columns else ["date"])
    path = out_dir / f"predictions_{split}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def pick_station_for_plot(stations: np.ndarray) -> Optional[str]:
    if stations is None or len(stations) == 0:
        return None
    s = pd.Series(stations.astype(str))
    return str(s.value_counts().index[0])


def subset_by_station(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_date: np.ndarray,
    station: np.ndarray,
    station_id: str
) -> pd.DataFrame:
    df = pd.DataFrame({
        "station_id": station.astype(str),
        "date": pd.to_datetime(y_date),
        "y_true": np.asarray(y_true).reshape(-1),
        "y_pred": np.asarray(y_pred).reshape(-1),
    })
    df = df[df["station_id"] == str(station_id)].sort_values("date")
    return df


def write_text_report(out_path: Path, lines: List[str]) -> None:
    out_path.write_text("\n".join(lines), encoding="utf-8")


def save_json(out_path: Path, obj: Dict[str, Any]) -> None:
    out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_training_history(history: Dict[str, Any], out_png: Path) -> None:
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)
    if "loss" in history:
        ax.plot(history["loss"], label="loss")
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="val_loss")
    ax.set_title("Training loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_scatter_true_pred(y_true: np.ndarray, y_pred: np.ndarray, out_png: Path, title: str) -> None:
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(np.asarray(y_true).reshape(-1), np.asarray(y_pred).reshape(-1), s=8, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("True")
    ax.set_ylabel("Pred")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_residual_hist(y_true: np.ndarray, y_pred: np.ndarray, out_png: Path, title: str) -> None:
    import matplotlib.pyplot as plt
    resid = np.asarray(y_pred).reshape(-1) - np.asarray(y_true).reshape(-1)
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(resid, bins=40)
    ax.set_title(title)
    ax.set_xlabel("Residual (pred-true)")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_timeseries_one_station(df_station: pd.DataFrame, out_png: Path, title: str) -> None:
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(df_station["date"], df_station["y_true"], label="true")
    ax.plot(df_station["date"], df_station["y_pred"], label="pred")
    ax.set_title(title)
    ax.set_xlabel("date")
    ax.set_ylabel("target (log1p)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
