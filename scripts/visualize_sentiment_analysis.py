"""Visualize post-level sentiment analysis results."""

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


DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "sentiment_analysis" / "posts_sentiment.csv"
DEFAULT_POSTS_PATH = PROJECT_ROOT / "data" / "processed" / "analysis_inputs" / "posts_for_analysis_topic_labeled.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "sentiment_analysis" / "visualizations"
DEFAULT_MANUAL_SAMPLE = PROJECT_ROOT / "data" / "processed" / "sentiment_analysis" / "manual_check_sample.csv"

SENTIMENT_ORDER = ["negative", "neutral", "positive"]
SENTIMENT_COLORS = {
    "negative": "#D64F4F",
    "neutral": "#72B7B2",
    "positive": "#4C78A8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize sentiment analysis outputs.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--posts-path", type=Path, default=DEFAULT_POSTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manual-sample-output", type=Path, default=DEFAULT_MANUAL_SAMPLE)
    parser.add_argument("--time-unit", choices=["year", "month"], default="year")
    parser.add_argument("--top-topics", type=int, default=15)
    parser.add_argument("--top-interaction-n", type=int, default=100)
    parser.add_argument("--manual-sample-per-label", type=int, default=30)
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


def as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0)


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def read_sentiment(path: Path, posts_path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing sentiment result file: {path}")
    df = pd.read_csv(path, dtype=str).fillna("")
    required = {"aweme_id", "create_time", "topic", "sentiment_label", "sentiment_status"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Sentiment file is missing required columns: " + ", ".join(sorted(missing)))
    df = df[df["sentiment_status"].eq("success")].copy()
    df = df[df["sentiment_label"].isin(SENTIMENT_ORDER)].copy()
    df["published_at"] = parse_create_time(df["create_time"])
    df = df.dropna(subset=["published_at"])

    if posts_path.exists() and ("comment_count" not in df.columns or "likes" not in df.columns):
        posts = pd.read_csv(posts_path, dtype=str).fillna("")
        merge_cols = [col for col in ["aweme_id", "comment_count", "likes"] if col in posts.columns]
        if {"aweme_id", "comment_count", "likes"}.issubset(merge_cols):
            df = df.merge(posts[merge_cols].drop_duplicates("aweme_id"), on="aweme_id", how="left")

    df["comment_count_num"] = as_numeric(df["comment_count"] if "comment_count" in df.columns else pd.Series([0] * len(df)))
    df["likes_num"] = as_numeric(df["likes"] if "likes" in df.columns else pd.Series([0] * len(df)))
    return df


def plot_sentiment_distribution(df: pd.DataFrame, output_dir: Path) -> Path:
    counts = df["sentiment_label"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)
    total = max(int(counts.sum()), 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values, color=[SENTIMENT_COLORS[label] for label in counts.index])
    ax.set_title("Post Sentiment Distribution", fontsize=16, fontweight="bold")
    ax.set_xlabel("sentiment_label")
    ax.set_ylabel("post_count")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, counts.values):
        pct = value / total * 100
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}\n{pct:.1f}%", ha="center", va="bottom", fontsize=9)
    path = output_dir / "sentiment_distribution.png"
    save_fig(path)
    return path


def build_time_table(df: pd.DataFrame, time_unit: str) -> pd.DataFrame:
    data = df.copy()
    if time_unit == "month":
        data["period"] = data["published_at"].dt.to_period("M").astype(str)
    else:
        data["period"] = data["published_at"].dt.year.astype(str)
    table = (
        data.groupby(["period", "sentiment_label"], dropna=False)
        .size()
        .reset_index(name="count")
        .pivot(index="period", columns="sentiment_label", values="count")
        .fillna(0)
        .sort_index()
    )
    return table.reindex(columns=SENTIMENT_ORDER, fill_value=0)


def plot_sentiment_over_time_count(df: pd.DataFrame, output_dir: Path, time_unit: str) -> Path:
    table = build_time_table(df, time_unit)
    fig, ax = plt.subplots(figsize=(12, 6))
    table.plot.area(ax=ax, color=[SENTIMENT_COLORS[label] for label in table.columns], alpha=0.85, linewidth=0.5)
    ax.set_title("Post Sentiment Over Time", fontsize=16, fontweight="bold")
    ax.set_xlabel(f"publish_{time_unit}")
    ax.set_ylabel("post_count")
    ax.grid(alpha=0.25)
    ax.legend(title="sentiment", frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    path = output_dir / f"sentiment_over_time_{time_unit}_count.png"
    save_fig(path)
    return path


def plot_sentiment_over_time_ratio(df: pd.DataFrame, output_dir: Path, time_unit: str) -> Path:
    table = build_time_table(df, time_unit)
    ratio = table.div(table.sum(axis=1).replace(0, pd.NA), axis=0).fillna(0)
    fig, ax = plt.subplots(figsize=(12, 6))
    ratio.plot(ax=ax, marker="o", color=[SENTIMENT_COLORS[label] for label in ratio.columns])
    ax.set_title("Post Sentiment Ratio Over Time", fontsize=16, fontweight="bold")
    ax.set_xlabel(f"publish_{time_unit}")
    ax.set_ylabel("sentiment_ratio")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(title="sentiment", frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    path = output_dir / f"sentiment_over_time_{time_unit}_ratio.png"
    save_fig(path)
    return path


def plot_topic_sentiment_stacked_bar(df: pd.DataFrame, output_dir: Path, top_topics: int) -> Path:
    data = df.copy()
    data["topic"] = data["topic"].replace("", "blank").astype(str)
    top = data.groupby("topic").size().sort_values(ascending=False).head(top_topics).index
    table = (
        data[data["topic"].isin(top)]
        .groupby(["topic", "sentiment_label"])
        .size()
        .reset_index(name="count")
        .pivot(index="topic", columns="sentiment_label", values="count")
        .fillna(0)
        .reindex(columns=SENTIMENT_ORDER, fill_value=0)
    )
    table["total"] = table.sum(axis=1)
    table = table.sort_values("total", ascending=False).drop(columns=["total"])
    fig, ax = plt.subplots(figsize=(12, 7))
    table.plot(kind="bar", stacked=True, ax=ax, color=[SENTIMENT_COLORS[label] for label in table.columns])
    ax.set_title(f"Topic Sentiment Distribution | Top {top_topics} Topics", fontsize=16, fontweight="bold")
    ax.set_xlabel("topic")
    ax.set_ylabel("post_count")
    ax.tick_params(axis="x", rotation=60)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="sentiment", frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    path = output_dir / "topic_sentiment_stacked_bar.png"
    save_fig(path)
    return path


def plot_topic_sentiment_heatmap_ratio(df: pd.DataFrame, output_dir: Path, top_topics: int) -> Path:
    data = df.copy()
    data["topic"] = data["topic"].replace("", "blank").astype(str)
    top = data.groupby("topic").size().sort_values(ascending=False).head(top_topics).index
    counts = (
        data[data["topic"].isin(top)]
        .groupby(["topic", "sentiment_label"])
        .size()
        .reset_index(name="count")
        .pivot(index="topic", columns="sentiment_label", values="count")
        .fillna(0)
        .reindex(columns=SENTIMENT_ORDER, fill_value=0)
    )
    counts["total"] = counts.sum(axis=1)
    counts = counts.sort_values("total", ascending=False)
    ratio = counts[SENTIMENT_ORDER].div(counts["total"].replace(0, pd.NA), axis=0).fillna(0)

    fig, ax = plt.subplots(figsize=(9, max(5, len(ratio) * 0.38)))
    image = ax.imshow(ratio.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_title(f"Topic x Sentiment Heatmap | Top {top_topics} Topics", fontsize=16, fontweight="bold")
    ax.set_xlabel("sentiment")
    ax.set_ylabel("topic")
    ax.set_xticks(range(len(ratio.columns)))
    ax.set_xticklabels(ratio.columns)
    ax.set_yticks(range(len(ratio.index)))
    ax.set_yticklabels(ratio.index.astype(str))

    for row_idx, topic in enumerate(ratio.index):
        for col_idx, label in enumerate(ratio.columns):
            value = ratio.loc[topic, label]
            text_color = "white" if value >= 0.55 else "black"
            ax.text(col_idx, row_idx, f"{value:.1%}", ha="center", va="center", color=text_color, fontsize=9)

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("sentiment_ratio")
    path = output_dir / "topic_sentiment_heatmap_ratio.png"
    save_fig(path)
    return path


def plot_top_interaction_sentiment(df: pd.DataFrame, output_dir: Path, top_n: int) -> Path:
    metric = "comment_count_num" if df["comment_count_num"].sum() > 0 else "likes_num"
    top = df.sort_values(metric, ascending=False).head(top_n)
    counts = top["sentiment_label"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)
    total = max(int(counts.sum()), 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values, color=[SENTIMENT_COLORS[label] for label in counts.index])
    ax.set_title(f"Top {top_n} High-Interaction Posts Sentiment", fontsize=16, fontweight="bold")
    ax.set_xlabel("sentiment_label")
    ax.set_ylabel("post_count")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, counts.values):
        pct = value / total * 100
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}\n{pct:.1f}%", ha="center", va="bottom", fontsize=9)
    path = output_dir / "top_interaction_sentiment.png"
    save_fig(path)
    return path


def write_manual_sample(df: pd.DataFrame, output_path: Path, per_label: int) -> Path:
    samples = []
    for label in SENTIMENT_ORDER:
        group = df[df["sentiment_label"].eq(label)]
        if group.empty:
            continue
        samples.append(group.sample(n=min(per_label, len(group)), random_state=42))
    if samples:
        sample = pd.concat(samples, ignore_index=True)
    else:
        sample = df.head(0).copy()
    columns = [
        "aweme_id",
        "create_time",
        "topic",
        "text",
        "text_for_sentiment",
        "sentiment_label",
        "model_label",
        "model_score",
        "matched_keywords",
    ]
    for column in columns:
        if column not in sample.columns:
            sample[column] = ""
    sample = sample[columns].copy()
    sample["manual_label"] = ""
    sample["note"] = ""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def write_summary(output_dir: Path, paths: list[Path], df: pd.DataFrame, manual_path: Path) -> Path:
    path = output_dir / "sentiment_visualization_summary.md"
    counts = df["sentiment_label"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)
    lines = [
        "# Sentiment Analysis Visualizations",
        "",
        f"Successful sentiment rows used: {len(df):,}",
        "",
        "## Sentiment Counts",
        "",
    ]
    for label, count in counts.items():
        lines.append(f"- {label}: {int(count):,}")
    lines.extend(["", "## Generated Figures", ""])
    for output_path in paths:
        lines.append(f"- {output_path.name}")
    lines.extend(["", "## Manual Check Sample", "", f"- {manual_path.name}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    check_dependencies()
    set_plot_style()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_sentiment(args.input.resolve(), args.posts_path.resolve())
    paths = [
        plot_sentiment_distribution(df, output_dir),
        plot_sentiment_over_time_count(df, output_dir, args.time_unit),
        plot_sentiment_over_time_ratio(df, output_dir, args.time_unit),
        plot_topic_sentiment_stacked_bar(df, output_dir, args.top_topics),
        plot_topic_sentiment_heatmap_ratio(df, output_dir, args.top_topics),
        plot_top_interaction_sentiment(df, output_dir, args.top_interaction_n),
    ]
    manual_path = write_manual_sample(df, args.manual_sample_output.resolve(), args.manual_sample_per_label)
    summary_path = write_summary(output_dir, paths, df, manual_path)

    print(f"Sentiment rows used: {len(df):,}")
    for path in paths:
        print(f"Generated: {path}")
    print(f"Manual check sample: {manual_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
