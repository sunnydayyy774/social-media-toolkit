"""Create visualizations for the final BERTopic post model."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import os
from math import ceil
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATPLOTLIB_CONFIG_DIR = PROJECT_ROOT / "data" / "tmp" / "matplotlib"
MATPLOTLIB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CONFIG_DIR))

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_RESULT_DIR = (
    PROJECT_ROOT / "data" / "processed" / "bertopic_posts" / "final_mcs15_ms15_nn15_nc5"
)

REQUIRED_PACKAGES = {
    "bertopic": "bertopic",
    "matplotlib": "matplotlib",
    "pandas": "pandas",
    "plotly": "plotly",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize the final BERTopic post model.")
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR, help="Final BERTopic result directory.")
    parser.add_argument("--top-n-words", type=int, default=8, help="Top words to show for each topic.")
    parser.add_argument("--topics-per-page", type=int, default=12, help="Number of topics per word-score image.")
    return parser.parse_args()


def check_dependencies() -> None:
    missing = [
        package_name
        for module_name, package_name in REQUIRED_PACKAGES.items()
        if importlib.util.find_spec(module_name) is None
    ]
    if missing:
        raise SystemExit("Missing required dependencies: " + ", ".join(missing))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def read_run_config_path(result_dir: Path, key: str, default_path: Path) -> Path:
    config_path = result_dir / "run_config.json"
    if not config_path.exists():
        return default_path
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default_path
    configured_path = config.get(key)
    if not configured_path:
        return default_path
    path = Path(configured_path)
    return path if path.exists() else default_path


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


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def plot_topic_word_scores(
    topic_model: Any,
    topic_counts: dict[int, int],
    output_dir: Path,
    top_n_words: int,
    topics_per_page: int,
) -> list[Path]:
    topics = sorted(topic for topic in topic_counts if topic != -1)
    if not topics:
        return []

    output_paths: list[Path] = []
    cols = 4
    rows_per_page = ceil(topics_per_page / cols)
    colors = plt.cm.tab20.colors

    for page_index in range(ceil(len(topics) / topics_per_page)):
        page_topics = topics[page_index * topics_per_page : (page_index + 1) * topics_per_page]
        fig, axes = plt.subplots(rows_per_page, cols, figsize=(18, 4.2 * rows_per_page))
        axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]

        for ax, topic in zip(axes_list, page_topics):
            words = topic_model.get_topic(topic) or []
            words = words[:top_n_words]
            labels = [word for word, _ in words][::-1]
            scores = [score for _, score in words][::-1]

            ax.barh(labels, scores, color=colors[topic % len(colors)])
            ax.set_title(f"Topic {topic} | n={topic_counts.get(topic, 0)}", fontsize=12, pad=8)
            ax.tick_params(axis="y", labelsize=9)
            ax.tick_params(axis="x", labelsize=8)
            ax.grid(axis="x", alpha=0.25)
            ax.set_axisbelow(True)

        for ax in axes_list[len(page_topics) :]:
            ax.axis("off")

        first_topic = page_topics[0]
        last_topic = page_topics[-1]
        fig.suptitle(f"Topic Word Scores | Topic {first_topic}-{last_topic}", fontsize=24, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.95))

        output_path = output_dir / f"topic_word_scores_{first_topic}_{last_topic}.png"
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths


def topic_display_name(topic: int, topic_names: dict[int, str]) -> str:
    raw_name = topic_names.get(topic, "")
    if not raw_name:
        return f"Topic {topic}"
    parts = [part for part in raw_name.split("_") if part and part != str(topic)]
    if not parts:
        return f"Topic {topic}"
    return f"{topic}_" + "_".join(parts[:4])


def parse_create_time_to_year(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid_values = values.dropna()
    if valid_values.empty:
        return pd.Series([pd.NA] * len(series), index=series.index)

    median_value = valid_values.median()
    unit = "ms" if median_value > 10_000_000_000 else "s"
    datetimes = pd.to_datetime(values, unit=unit, errors="coerce")
    return datetimes.dt.year


def plot_topics_by_year(
    topic_results: list[dict[str, str]],
    topic_names: dict[int, str],
    output_dir: Path,
) -> Path | None:
    df = pd.DataFrame(topic_results)
    if df.empty or "topic" not in df.columns or "create_time" not in df.columns:
        return None

    df["topic"] = pd.to_numeric(df["topic"], errors="coerce")
    df["year"] = parse_create_time_to_year(df["create_time"])
    df = df.dropna(subset=["topic", "year"]).copy()
    df["topic"] = df["topic"].astype(int)
    df["year"] = df["year"].astype(int)
    df = df[df["topic"] != -1]
    if df.empty:
        return None

    min_year = int(df["year"].min())
    max_year = int(df["year"].max())
    years = list(range(min_year, max_year + 1))
    topics = sorted(df["topic"].unique())

    counts = (
        df.groupby(["year", "topic"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=years, columns=topics, fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(18, 10))
    colors = plt.cm.tab20.colors
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    for index, topic in enumerate(topics):
        ax.plot(
            counts.index,
            counts[topic],
            marker=markers[index % len(markers)],
            linewidth=1.8,
            markersize=4,
            color=colors[index % len(colors)],
            label=topic_display_name(topic, topic_names),
        )

    ax.set_title("BERTopic Topics by Year", fontsize=22, fontweight="bold", pad=18)
    ax.set_xlabel("year", fontsize=12)
    ax.set_ylabel("video_count", fontsize=12)
    ax.set_xticks(years)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.25)
    ax.legend(
        title="topic_name",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=8,
        title_fontsize=9,
        frameon=False,
    )
    fig.tight_layout()

    output_path = output_dir / "topics_by_year_all_non_noise.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_visualization_summary(
    path: Path,
    topic_results_count: int,
    non_noise_topic_count: int,
    noise_count: int,
    word_score_paths: list[Path],
    distance_html_path: Path,
    topics_by_year_path: Path | None,
) -> None:
    noise_ratio = noise_count / topic_results_count if topic_results_count else 0
    lines = [
        "# BERTopic 最终模型可视化说明",
        "",
        f"- 总记录数：{topic_results_count}",
        f"- 非噪声主题数：{non_noise_topic_count}",
        f"- topic=-1 噪声数：{noise_count}",
        f"- 噪声比例：{noise_ratio:.1%}",
        "",
        "## Topic Word Scores",
        "",
        "这些图展示每个主题里权重最高的关键词。横轴越长，说明这个词越能代表该主题。它适合用来判断主题是否可解释，以及关键词里是否还有噪声。",
        "",
    ]
    for output_path in word_score_paths:
        lines.append(f"- {output_path.name}")

    lines.extend(
        [
            "",
            "## Intertopic Distance Map",
            "",
            "这个 HTML 图展示主题之间的距离。距离越近，说明两个主题越相似；圆越大，说明该主题包含的视频越多。它适合用来检查主题是否过碎、是否有多个主题其实在讲相近内容。",
            "",
            f"- {distance_html_path.name}",
        ]
    )
    lines.extend(
        [
            "",
            "## Topics by Year",
            "",
            "This chart uses create_time as the video publish time, converts it to year, and plots all non-noise topics.",
        ]
    )
    if topics_by_year_path is not None:
        lines.append(f"- {topics_by_year_path.name}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    check_dependencies()

    from bertopic import BERTopic

    result_dir = args.result_dir.resolve()
    topic_model_path = result_dir / "topic_model.pkl"
    topic_results_path = read_run_config_path(result_dir, "topic_results_path", result_dir / "topic_results.csv")
    topic_summary_path = read_run_config_path(result_dir, "topic_summary_path", result_dir / "topic_summary.csv")
    output_dir = result_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not topic_model_path.exists():
        raise FileNotFoundError(f"Topic model not found. Re-run run_bertopic_posts.py first: {topic_model_path}")
    if not topic_results_path.exists():
        raise FileNotFoundError(f"topic_results.csv not found: {topic_results_path}")
    if not topic_summary_path.exists():
        raise FileNotFoundError(f"topic_summary.csv not found: {topic_summary_path}")

    set_plot_style()
    topic_model = BERTopic.load(str(topic_model_path))
    topic_results = read_csv_rows(topic_results_path)
    topic_summary = read_csv_rows(topic_summary_path)

    topic_counts = {
        parse_int(row.get("topic")): parse_int(row.get("count"))
        for row in topic_summary
        if row.get("topic") not in ("", None)
    }
    topic_names = {
        parse_int(row.get("topic")): row.get("topic_name", "")
        for row in topic_summary
        if row.get("topic") not in ("", None)
    }
    noise_count = sum(1 for row in topic_results if row.get("topic") == "-1")
    non_noise_topic_count = sum(1 for topic in topic_counts if topic != -1)

    word_score_paths = plot_topic_word_scores(
        topic_model=topic_model,
        topic_counts=topic_counts,
        output_dir=output_dir,
        top_n_words=args.top_n_words,
        topics_per_page=args.topics_per_page,
    )

    distance_html_path = output_dir / "intertopic_distance_map.html"
    fig = topic_model.visualize_topics()
    fig.write_html(str(distance_html_path), include_plotlyjs="cdn")
    topics_by_year_path = plot_topics_by_year(
        topic_results=topic_results,
        topic_names=topic_names,
        output_dir=output_dir,
    )

    summary_path = output_dir / "visualization_summary.md"
    write_visualization_summary(
        path=summary_path,
        topic_results_count=len(topic_results),
        non_noise_topic_count=non_noise_topic_count,
        noise_count=noise_count,
        word_score_paths=word_score_paths,
        distance_html_path=distance_html_path,
        topics_by_year_path=topics_by_year_path,
    )

    logging.info("Word score images: %s", ", ".join(str(path) for path in word_score_paths))
    logging.info("Intertopic distance HTML: %s", distance_html_path)
    if topics_by_year_path is not None:
        logging.info("Topics by year image: %s", topics_by_year_path)
    logging.info("Visualization summary: %s", summary_path)


if __name__ == "__main__":
    main()
