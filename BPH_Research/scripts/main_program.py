"""Utility script for preparing the rice dataset and plotting selected columns."""
import sys
print(sys.executable)

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

from make_model_ready_v3_stream_with_plots import (
    add_station_derivatives,
    add_targets,
    add_time_features,
    add_wind_uv,
    read_csv_any,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare dataframe and plot selected columns")
    project_root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--raw_csv",
        type=Path,
        default=project_root / "data" / "env_daily_with_rice_monthly_raw.csv",
        help="Path to the raw daily CSV",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=project_root / "data" / "out_model_ready_v1",
        help="Directory to store outputs",
    )
    parser.add_argument(
        "--plot_columns",
        nargs="+",
        default=None,
        help="One or more column names to plot against date",
    )
    parser.add_argument(
        "--station_id",
        type=str,
        default="",
        help="Optional station_id to filter before plotting",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=2000,
        help="Limit plot to the last N records after sorting by date (use <=0 for all)",
    )
    return parser.parse_args()


def prepare_dataframe(raw_csv: Path) -> pd.DataFrame:
    if not raw_csv.exists():
        raise FileNotFoundError(f"RAW CSV not found: {raw_csv}")

    df = read_csv_any(raw_csv)
    if "date" not in df.columns:
        raise ValueError("RAW ต้องมีคอลัมน์ date")
    if "station_id" not in df.columns:
        raise ValueError("RAW ต้องมีคอลัมน์ station_id")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    df = add_targets(df)
    df = add_time_features(df)
    df = add_wind_uv(df)
    df = add_station_derivatives(df, roll_days=7)
    return df

def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    columns = [c.strip() for c in columns if c.strip()]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in dataframe: {missing}")
    return columns

def plot_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
    out_dir: Path,
    station_id: str = "",
    tail: int = 2000,
) -> Path:
    if "date" not in df.columns:
        raise ValueError("Dataframe must contain a date column for plotting")

    cols = _ensure_columns(df, columns)

    dplot = df.copy()
    if station_id:
        dplot = dplot[dplot["station_id"].astype(str) == station_id]
        if dplot.empty:
            raise ValueError(f"No rows found for station_id={station_id}")

    dplot = dplot.sort_values("date")
    if tail > 0 and len(dplot) > tail:
        dplot = dplot.tail(tail)

    fig, axes = plt.subplots(len(cols), 1, figsize=(12, max(4, 3 * len(cols))), sharex=True)
    if not isinstance(axes, (list, tuple, pd.Series)):
        axes = [axes]

    for ax, col in zip(axes, cols):
        series = pd.to_numeric(dplot[col], errors="coerce")
        ax.plot(dplot["date"], series, label=col)
        ax.set_ylabel(col)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("date")

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "selected_columns_plot.png"
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    return fig_path

def main() -> None:
    args = parse_args()
    df = prepare_dataframe(args.raw_csv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("RAW_CSV =", args.raw_csv)
    print("OUT_DIR =", args.out_dir)
    print("DATAFRAME SHAPE =", df.shape)

    if args.plot_columns:
        fig_path = plot_columns(
            df,
            args.plot_columns,
            args.out_dir / "figures",
            station_id=args.station_id,
            tail=args.tail,
        )
        print(f"Saved plot to {fig_path}")
    else:
        print("No columns specified for plotting (use --plot_columns)")

if __name__ == "__main__":
    main()
