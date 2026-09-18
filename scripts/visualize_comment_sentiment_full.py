"""Visualize full comment sentiment analysis results."""

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


DEFAULT_INPUT = (
    PROJECT_ROOT / "data" / "processed" / "comment_sentiment_analysis_full" / "comments_sentiment_full.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "processed" / "comment_sentiment_analysis_full" / "visualizations"
)

SENTIMENT_ORDER = ["negative", "neutral", "positive"]
SENTIMENT_COLORS = {
    "negative": "#D64F4F",
    "neutral": "#72B7B2",
    "positive": "#4C78A8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize full comment sentiment analysis.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-comments-for-topic-year", type=int, default=100)
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


def read_comments(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing comment sentiment file: {path}")
    df = pd.read_csv(path, dtype=str).fillna("")
    required = {
        "comment_id",
        "aweme_id",
        "user_id",
        "create_time",
        "topic",
        "post_sentiment_label",
        "comment_sentiment_label",
        "sentiment_status",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Input is missing required columns: " + ", ".join(sorted(missing)))
    df = df[df["sentiment_status"].eq("success")].copy()
    df = df[df["comment_sentiment_label"].isin(SENTIMENT_ORDER)].copy()
    df = df[df["post_sentiment_label"].isin(SENTIMENT_ORDER)].copy()
    df["published_at"] = parse_create_time(df["create_time"])
    df = df.dropna(subset=["published_at"])
    df["publish_year"] = df["published_at"].dt.year.astype(str)
    df["topic"] = df["topic"].replace("", "blank").astype(str)
    return df


def plot_comment_sentiment_distribution(df: pd.DataFrame, output_dir: Path) -> Path:
    counts = df["comment_sentiment_label"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)
    total = max(int(counts.sum()), 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values, color=[SENTIMENT_COLORS[label] for label in counts.index])
    ax.set_title("Comment Sentiment Distribution", fontsize=16, fontweight="bold")
    ax.set_xlabel("sentiment_label")
    ax.set_ylabel("comment_count")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, counts.values):
        pct = value / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{int(value):,}\n{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    path = output_dir / "comment_sentiment_distribution_full.png"
    save_fig(path)
    return path


def plot_comment_sentiment_over_time_count(df: pd.DataFrame, output_dir: Path) -> Path:
    table = (
        df.groupby(["publish_year", "comment_sentiment_label"])
        .size()
        .reset_index(name="count")
        .pivot(index="publish_year", columns="comment_sentiment_label", values="count")
        .fillna(0)
        .sort_index()
        .reindex(columns=SENTIMENT_ORDER, fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    table.plot.area(ax=ax, color=[SENTIMENT_COLORS[label] for label in table.columns], alpha=0.85, linewidth=0.5)
    ax.set_title("Comment Sentiment Over Time", fontsize=16, fontweight="bold")
    ax.set_xlabel("comment_year")
    ax.set_ylabel("comment_count")
    ax.grid(alpha=0.25)
    ax.legend(title="sentiment", frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    path = output_dir / "comment_sentiment_over_time_year_count_full.png"
    save_fig(path)
    return path


def plot_post_comment_sentiment_matrix(df: pd.DataFrame, output_dir: Path) -> Path:
    table = (
        df.groupby(["comment_sentiment_label", "post_sentiment_label"])
        .size()
        .reset_index(name="count")
        .pivot(index="comment_sentiment_label", columns="post_sentiment_label", values="count")
        .fillna(0)
        .reindex(index=SENTIMENT_ORDER, columns=SENTIMENT_ORDER, fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(table.values, aspect="auto", cmap="Blues")
    ax.set_title("Post Sentiment x Comment Sentiment | Full Comments", fontsize=15, fontweight="bold")
    ax.set_xlabel("post_sentiment")
    ax.set_ylabel("comment_sentiment")
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(table.columns)
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index)
    max_value = table.values.max() if table.size else 0
    for row_idx, row_label in enumerate(table.index):
        for col_idx, col_label in enumerate(table.columns):
            value = int(table.loc[row_label, col_label])
            color = "white" if max_value and value >= max_value * 0.55 else "black"
            ax.text(col_idx, row_idx, f"{value:,}", ha="center", va="center", color=color, fontsize=10)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("comment_count")
    path = output_dir / "post_comment_sentiment_matrix_full.png"
    save_fig(path)
    return path


def plot_comment_negative_ratio_by_post_sentiment(df: pd.DataFrame, output_dir: Path) -> Path:
    grouped = (
        df.assign(is_negative=df["comment_sentiment_label"].eq("negative").astype(int))
        .groupby("post_sentiment_label")
        .agg(negative_ratio=("is_negative", "mean"), comment_count=("comment_id", "count"))
        .reindex(SENTIMENT_ORDER)
        .fillna(0)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(grouped.index, grouped["negative_ratio"], color=[SENTIMENT_COLORS[label] for label in grouped.index])
    ax.set_title("Comment Negative Ratio by Post Sentiment | Full Comments", fontsize=15, fontweight="bold")
    ax.set_xlabel("post_sentiment")
    ax.set_ylabel("negative_comment_ratio")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    for bar, (_, row) in zip(bars, grouped.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            row["negative_ratio"],
            f"{row['negative_ratio']:.1%}\nn={int(row['comment_count']):,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    path = output_dir / "comment_negative_ratio_by_post_sentiment_full.png"
    save_fig(path)
    return path


def plot_topic_year_negative_ratio_heatmap(df: pd.DataFrame, output_dir: Path, min_comments: int) -> Path:
    grouped = (
        df.assign(is_negative=df["comment_sentiment_label"].eq("negative").astype(int))
        .groupby(["topic", "publish_year"])
        .agg(negative_ratio=("is_negative", "mean"), comment_count=("comment_id", "count"))
        .reset_index()
    )
    grouped.loc[grouped["comment_count"] < min_comments, "negative_ratio"] = pd.NA
    topic_order = df.groupby("topic").size().sort_values(ascending=False).index.astype(str).tolist()
    table = (
        grouped.pivot(index="topic", columns="publish_year", values="negative_ratio")
        .reindex(index=topic_order)
        .sort_index(axis=1)
    )
    fig, ax = plt.subplots(figsize=(max(8, len(table.columns) * 0.85), max(5, len(table.index) * 0.34)))
    image = ax.imshow(table.fillna(0).values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_title("Topic-Year Negative Comment Ratio | Full Comments", fontsize=15, fontweight="bold")
    ax.set_xlabel("comment_year")
    ax.set_ylabel("topic")
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(table.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index.astype(str))
    for row_idx, topic in enumerate(table.index):
        for col_idx, year in enumerate(table.columns):
            value = table.loc[topic, year]
            label = "" if pd.isna(value) else f"{value:.0%}"
            if label:
                ax.text(col_idx, row_idx, label, ha="center", va="center", fontsize=8)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("negative_comment_ratio")
    path = output_dir / "topic_year_negative_ratio_heatmap_full.png"
    save_fig(path)
    return path


def plot_user_activity_sentiment_scatter(df: pd.DataFrame, output_dir: Path) -> Path:
    counts = (
        df.groupby(["user_id", "comment_sentiment_label"])
        .size()
        .reset_index(name="sentiment_count")
    )
    dominant = counts.sort_values(["user_id", "sentiment_count"], ascending=[True, False]).drop_duplicates("user_id")
    activity = (
        df.groupby("user_id")
        .agg(total_comments=("comment_id", "count"), commented_video_count=("aweme_id", "nunique"))
        .reset_index()
        .merge(dominant[["user_id", "comment_sentiment_label"]], on="user_id", how="left")
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    for label in SENTIMENT_ORDER:
        part = activity[activity["comment_sentiment_label"].eq(label)]
        if part.empty:
            continue
        sizes = part["total_comments"].clip(lower=1, upper=80) * 3
        ax.scatter(
            part["commented_video_count"],
            part["total_comments"],
            s=sizes,
            alpha=0.22,
            label=label,
            color=SENTIMENT_COLORS[label],
            edgecolors="none",
        )
    ax.set_title("User Activity x Dominant Comment Sentiment | Full Comments", fontsize=15, fontweight="bold")
    ax.set_xlabel("commented_video_count")
    ax.set_ylabel("total_comments")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("symlog", linthresh=1)
    ax.grid(alpha=0.25)
    ax.legend(title="dominant_comment_sentiment", frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    path = output_dir / "user_activity_sentiment_scatter_full.png"
    save_fig(path)
    return path


def write_summary(output_dir: Path, paths: list[Path], df: pd.DataFrame) -> Path:
    path = output_dir / "comment_sentiment_full_visualization_summary.md"
    counts = df["comment_sentiment_label"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)
    lines = [
        "# Full Comment Sentiment Visualizations",
        "",
        f"Successful comment sentiment rows used: {len(df):,}",
        f"Videos covered: {df['aweme_id'].nunique():,}",
        f"Comment users covered: {df['user_id'].nunique():,}",
        "",
        "## Comment Sentiment Counts",
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

    df = read_comments(args.input.resolve())
    paths = [
        plot_comment_sentiment_distribution(df, output_dir),
        plot_comment_sentiment_over_time_count(df, output_dir),
        plot_post_comment_sentiment_matrix(df, output_dir),
        plot_comment_negative_ratio_by_post_sentiment(df, output_dir),
        plot_topic_year_negative_ratio_heatmap(df, output_dir, args.min_comments_for_topic_year),
        plot_user_activity_sentiment_scatter(df, output_dir),
    ]
    summary_path = write_summary(output_dir, paths, df)

    print(f"Comment sentiment rows used: {len(df):,}")
    for path in paths:
        print(f"Generated: {path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
