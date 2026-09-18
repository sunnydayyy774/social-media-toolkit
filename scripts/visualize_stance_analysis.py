"""Visualize post-level stance classification results."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATPLOTLIB_CONFIG_DIR = PROJECT_ROOT / "data" / "tmp" / "matplotlib"
MATPLOTLIB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CONFIG_DIR))

import matplotlib.pyplot as plt


DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "stance_analysis" / "posts_stance.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "stance_analysis" / "visualizations"

STANCE_ORDER = ["oppose_revision", "support_revision", "neutral_descriptive", "unclear"]
STANCE_COLORS = {
    "oppose_revision": "#D64F4F",
    "support_revision": "#4C78A8",
    "neutral_descriptive": "#72B7B2",
    "unclear": "#B8B8B8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize stance classification outputs.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--time-unit", choices=["year", "month"], default="year")
    return parser.parse_args()


def check_dependencies() -> None:
    missing = [
        package
        for module, package in {"pandas": "pandas", "matplotlib": "matplotlib"}.items()
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        raise SystemExit("Missing required dependencies: " + ", ".join(missing))


def set_plot_style() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"


def parse_create_time(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.where(numeric < 10_000_000_000, numeric / 1000)
    return pd.to_datetime(numeric, unit="s", errors="coerce")


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def read_stance(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing stance result file: {path}")
    df = pd.read_csv(path, dtype=str).fillna("")
    required = {"aweme_id", "create_time", "stance_label", "classification_status"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Stance file is missing required columns: " + ", ".join(sorted(missing)))
    df = df[df["classification_status"].eq("success")].copy()
    df = df[df["stance_label"].isin(STANCE_ORDER)].copy()
    df["published_at"] = parse_create_time(df["create_time"])
    df = df.dropna(subset=["published_at"])
    return df


def plot_stance_distribution(df: pd.DataFrame, output_dir: Path) -> Path:
    counts = df["stance_label"].value_counts().reindex(STANCE_ORDER, fill_value=0)
    total = max(int(counts.sum()), 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(counts.index, counts.values, color=[STANCE_COLORS[label] for label in counts.index])
    ax.set_title("Post Stance Distribution", fontsize=16, fontweight="bold")
    ax.set_xlabel("stance_label")
    ax.set_ylabel("post_count")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, counts.values):
        pct = value / total * 100
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}\n{pct:.1f}%", ha="center", va="bottom", fontsize=9)
    path = output_dir / "stance_distribution.png"
    save_fig(path)
    return path


def build_time_table(df: pd.DataFrame, time_unit: str) -> pd.DataFrame:
    data = df.copy()
    if time_unit == "month":
        data["period"] = data["published_at"].dt.to_period("M").astype(str)
    else:
        data["period"] = data["published_at"].dt.year.astype(str)
    table = (
        data.groupby(["period", "stance_label"], dropna=False)
        .size()
        .reset_index(name="count")
        .pivot(index="period", columns="stance_label", values="count")
        .fillna(0)
        .sort_index()
    )
    return table.reindex(columns=STANCE_ORDER, fill_value=0)


def plot_stance_over_time_count(df: pd.DataFrame, output_dir: Path, time_unit: str) -> Path:
    table = build_time_table(df, time_unit)
    fig, ax = plt.subplots(figsize=(12, 6))
    table.plot.area(ax=ax, color=[STANCE_COLORS[label] for label in table.columns], alpha=0.85, linewidth=0.5)
    ax.set_title("Post Stance Over Time", fontsize=16, fontweight="bold")
    ax.set_xlabel(f"publish_{time_unit}")
    ax.set_ylabel("post_count")
    ax.grid(alpha=0.25)
    ax.legend(title="stance", frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    path = output_dir / f"stance_over_time_{time_unit}_count.png"
    save_fig(path)
    return path


def plot_stance_over_time_ratio(df: pd.DataFrame, output_dir: Path, time_unit: str) -> Path:
    table = build_time_table(df, time_unit)
    ratio = table.div(table.sum(axis=1).replace(0, pd.NA), axis=0).fillna(0)
    fig, ax = plt.subplots(figsize=(12, 6))
    ratio.plot(ax=ax, marker="o", color=[STANCE_COLORS[label] for label in ratio.columns])
    ax.set_title("Post Stance Ratio Over Time", fontsize=16, fontweight="bold")
    ax.set_xlabel(f"publish_{time_unit}")
    ax.set_ylabel("stance_ratio")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(title="stance", frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    path = output_dir / f"stance_over_time_{time_unit}_ratio.png"
    save_fig(path)
    return path


def write_summary(output_dir: Path, paths: list[Path], df: pd.DataFrame) -> Path:
    path = output_dir / "stance_visualization_summary.md"
    counts = df["stance_label"].value_counts().reindex(STANCE_ORDER, fill_value=0)
    lines = [
        "# Stance Analysis Visualizations",
        "",
        f"Successful classified posts used: {len(df):,}",
        "",
        "## Stance Counts",
        "",
    ]
    for label, count in counts.items():
        lines.append(f"- {label}: {int(count):,}")
    lines.extend(["", "## Generated Figures", ""])
    for output_path in paths:
        lines.append(f"- {output_path.name}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    check_dependencies()
    set_plot_style()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_stance(args.input.resolve())
    paths = [
        plot_stance_distribution(df, output_dir),
        plot_stance_over_time_count(df, output_dir, args.time_unit),
        plot_stance_over_time_ratio(df, output_dir, args.time_unit),
    ]
    summary_path = write_summary(output_dir, paths, df)

    print(f"Stance rows used: {len(df):,}")
    for path in paths:
        print(f"Generated: {path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
