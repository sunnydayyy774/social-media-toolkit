"""Run BERTopic stage-1 HDBSCAN parameter evaluation.

This script fixes input text, embedding, UMAP, vectorizer, and nr_topics, then
varies HDBSCAN min_cluster_size and min_samples. It writes each run into its own
directory and creates a compact metrics table plus heatmaps.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "data" / "tmp" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from hdbscan.validity import validity_index
import jieba
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import davies_bouldin_score, silhouette_score
from umap import UMAP


INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "posts_for_bertopic_with_asr.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "bertopic_evaluation" / "stage1"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
TEXT_COLUMN = "text_clean_bertopic_with_asr"

MIN_CLUSTER_SIZES = [5, 8, 10, 15, 20, 30, 40]
MIN_SAMPLES = [1, 5, 10, 15]

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
ASCII_LETTER_RE = re.compile(r"[A-Za-z]")
PURE_NUMBER_RE = re.compile(r"^[\d①②③④⑤⑥⑦⑧⑨⑩⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽]+$")

RESULT_FIELDS = [
    "id",
    "aweme_id",
    "text",
    "text_clean_bertopic",
    "text_clean_bertopic_with_asr",
    "topic",
    "topic_probability",
    "is_noise",
    "create_time",
    "likes",
    "comment_count",
    "share_count",
    "collect_count",
    "play_count",
    "author_id",
    "author_nickname",
    "search_keyword",
    "hashtag_names_csv",
    "updated_at",
]

SUMMARY_FIELDS = [
    "topic",
    "count",
    "topic_name",
    "top_words",
    "example_text_1",
    "example_text_2",
    "example_text_3",
]

STAGE1_FIELDS = [
    "min_cluster_size",
    "min_samples",
    "topic_count",
    "outlier_rate",
    "dbcv",
    "silhouette",
    "davies_bouldin",
]


def load_input_rows(path: Path, text_column: str) -> tuple[list[dict[str, str]], list[str], list[int]]:
    rows: list[dict[str, str]] = []
    docs: list[str] = []
    doc_row_indices: list[int] = []

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV has no header: {path}")
        if text_column not in reader.fieldnames:
            raise ValueError(f"Input CSV must contain text column: {text_column}")

        for row in reader:
            rows.append(row)
            text = (row.get(text_column) or "").strip()
            if text:
                docs.append(text)
                doc_row_indices.append(len(rows) - 1)

    return rows, docs, doc_row_indices


def probability_at(probabilities: Any, index: int) -> str:
    if probabilities is None:
        return ""
    try:
        value = probabilities[index]
        if hasattr(value, "max"):
            value = value.max()
        return f"{float(value):.6f}"
    except Exception:
        return ""


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> int:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)
    return len(rows)


def write_csv_with_fallback(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> Path:
    try:
        write_csv(path, fieldnames, rows)
        return path
    except PermissionError as exc:
        fallback_path = path.with_name(f"{path.stem}_unlocked{path.suffix}")
        logging.warning("Could not update %s, writing fallback %s: %s", path, fallback_path, exc)
        write_csv(fallback_path, fieldnames, rows)
        return fallback_path


def is_valid_topic_token(token: str) -> bool:
    token = token.strip().lower()
    if not token:
        return False
    if PURE_NUMBER_RE.fullmatch(token):
        return False
    has_cjk = bool(CJK_RE.search(token))
    has_ascii_letter = bool(ASCII_LETTER_RE.search(token))
    if has_cjk:
        return len(token) >= 2
    if has_ascii_letter:
        if len(token) < 3:
            return False
        digit_count = sum(char.isdigit() for char in token)
        if digit_count and digit_count / len(token) > 0.5:
            return False
        if len(token) > 20:
            return False
        return True
    return False


def jieba_tokenizer(text: str) -> list[str]:
    return [token.strip().lower() for token in jieba.lcut(text) if is_valid_topic_token(token)]


def pretokenize_for_vectorizer(text: str) -> str:
    return " ".join(jieba_tokenizer(text))


def build_topic_summary(topic_model: BERTopic, result_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[int, int] = {}
    examples: dict[int, list[str]] = {}

    for row in result_rows:
        topic_value = row.get("topic")
        if topic_value in ("", None):
            continue
        topic = int(topic_value)
        counts[topic] = counts.get(topic, 0) + 1
        examples.setdefault(topic, [])
        if len(examples[topic]) < 3:
            examples[topic].append(row.get("text", ""))

    summary_rows: list[dict[str, Any]] = []
    for topic in sorted(counts):
        bertopic_words = topic_model.get_topic(topic) or []
        words = [word for word, _ in bertopic_words[:10]]
        top_words = " ".join(words[:10])
        topic_name_words = words[:4]
        example_texts = examples.get(topic, [])
        summary_rows.append(
            {
                "topic": topic,
                "count": counts[topic],
                "topic_name": f"{topic}_{'_'.join(topic_name_words)}" if topic_name_words else str(topic),
                "top_words": top_words,
                "example_text_1": example_texts[0] if len(example_texts) > 0 else "",
                "example_text_2": example_texts[1] if len(example_texts) > 1 else "",
                "example_text_3": example_texts[2] if len(example_texts) > 2 else "",
            }
        )
    return summary_rows


def safe_metric(value: float | None) -> str:
    if value is None:
        return ""
    try:
        if math.isnan(float(value)) or math.isinf(float(value)):
            return ""
    except Exception:
        return ""
    return f"{float(value):.6f}"


def calculate_metrics(reduced_embeddings: np.ndarray, labels: np.ndarray) -> dict[str, str]:
    non_noise_mask = labels != -1
    non_noise_labels = labels[non_noise_mask]
    topic_count = len(set(int(label) for label in non_noise_labels))
    outlier_rate = float(np.mean(labels == -1)) if len(labels) else 0.0

    silhouette: float | None = None
    davies_bouldin: float | None = None
    dbcv: float | None = None

    if topic_count >= 2 and len(non_noise_labels) > topic_count:
        metric_embeddings = reduced_embeddings[non_noise_mask]
        try:
            silhouette = float(silhouette_score(metric_embeddings, non_noise_labels))
        except Exception as exc:
            logging.warning("Silhouette failed: %s", exc)
        try:
            davies_bouldin = float(davies_bouldin_score(metric_embeddings, non_noise_labels))
        except Exception as exc:
            logging.warning("Davies-Bouldin failed: %s", exc)
        try:
            dbcv = float(validity_index(metric_embeddings.astype(np.float64), non_noise_labels.astype(np.int64)))
        except Exception as exc:
            logging.warning("DBCV failed: %s", exc)

    return {
        "topic_count": str(topic_count),
        "outlier_rate": safe_metric(outlier_rate),
        "dbcv": safe_metric(dbcv),
        "silhouette": safe_metric(silhouette),
        "davies_bouldin": safe_metric(davies_bouldin),
    }


def luminance(rgba: tuple[float, float, float, float]) -> float:
    red, green, blue, _ = rgba
    return 0.299 * red + 0.587 * green + 0.114 * blue


def plot_heatmap(
    results: pd.DataFrame,
    value_column: str,
    output_path: Path,
    title: str,
    cmap: str,
    percent: bool = False,
) -> None:
    pivot = results.pivot(index="min_cluster_size", columns="min_samples", values=value_column)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    values = pivot.to_numpy(dtype=float)
    masked_values = np.ma.masked_invalid(values)
    image = ax.imshow(masked_values, aspect="auto", cmap=cmap)

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels([str(value) for value in pivot.columns], fontsize=9)
    ax.set_yticklabels([str(value) for value in pivot.index], fontsize=9)
    ax.set_xlabel("min_samples", fontsize=10)
    ax.set_ylabel("min_cluster_size", fontsize=10)
    ax.set_title(title, fontsize=14, fontweight="bold", color="#12395a")

    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
        spine.set_color("#222222")

    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            if np.isnan(value):
                label = "NA"
                color = "#1b2733"
            elif percent:
                label = f"{value * 100:.1f}%"
                color = "white" if luminance(image.cmap(image.norm(value))) < 0.46 else "#1b2733"
            elif value_column == "topic_count":
                label = f"{value:.0f}"
                color = "white" if luminance(image.cmap(image.norm(value))) < 0.46 else "#1b2733"
            else:
                label = f"{value:.3f}"
                color = "white" if luminance(image.cmap(image.norm(value))) < 0.46 else "#1b2733"
            ax.text(col_idx, row_idx, label, ha="center", va="center", color=color, fontsize=8)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def load_existing_results(path: Path) -> list[dict[str, str]]:
    fallback_path = path.with_name(f"{path.stem}_unlocked{path.suffix}")
    if fallback_path.exists() and (
        not path.exists() or fallback_path.stat().st_mtime > path.stat().st_mtime
    ):
        path = fallback_path
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            return []
        return [row for row in reader]


def run_one(
    rows: list[dict[str, str]],
    docs: list[str],
    vectorizer_docs: list[str],
    doc_row_indices: list[int],
    embeddings: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    stage_runs_dir: Path,
) -> dict[str, str]:
    run_name = f"mcs{min_cluster_size}_ms{min_samples}"
    run_dir = stage_runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Running %s", run_name)
    started = time.time()

    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer_model = CountVectorizer(
        tokenizer=str.split,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
    )
    topic_model = BERTopic(
        embedding_model=None,
        vectorizer_model=vectorizer_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        nr_topics=None,
        calculate_probabilities=True,
        language="multilingual",
        verbose=True,
    )

    topics, probabilities = topic_model.fit_transform(vectorizer_docs, embeddings)
    reduced_embeddings = np.asarray(topic_model.umap_model.embedding_, dtype=np.float64)
    labels = np.asarray(topics, dtype=np.int64)
    metrics = calculate_metrics(reduced_embeddings, labels)

    result_rows: list[dict[str, Any]] = []
    for row in rows:
        result_row = dict(row)
        result_row["topic"] = ""
        result_row["topic_probability"] = ""
        result_row["is_noise"] = ""
        result_rows.append(result_row)

    for doc_index, row_index in enumerate(doc_row_indices):
        topic = int(topics[doc_index])
        result_rows[row_index]["topic"] = topic
        result_rows[row_index]["topic_probability"] = probability_at(probabilities, doc_index)
        result_rows[row_index]["is_noise"] = "true" if topic == -1 else "false"

    summary_rows = build_topic_summary(topic_model, result_rows)
    write_csv(run_dir / "topic_results.csv", RESULT_FIELDS, result_rows)
    write_csv(run_dir / "topic_summary.csv", SUMMARY_FIELDS, summary_rows)

    config = {
        "input": str(INPUT_CSV),
        "text_column": TEXT_COLUMN,
        "embedding_model": MODEL_NAME,
        "ngram_range": [1, 2],
        "vectorizer": {
            "tokenizer": "jieba_pretokenized_space_separated",
            "filter_pure_number_token": True,
            "filter_english_token": True,
            "filter_short_garbage_token": True,
        },
        "nr_topics": None,
        "language": "multilingual",
        "umap": {
            "n_neighbors": 15,
            "n_components": 5,
            "min_dist": 0.0,
            "metric": "cosine",
            "random_state": 42,
        },
        "hdbscan": {
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "metric": "euclidean",
            "cluster_selection_method": "eom",
            "prediction_data": True,
        },
        "runtime_seconds": round(time.time() - started, 3),
    }
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)

    return {
        "min_cluster_size": str(min_cluster_size),
        "min_samples": str(min_samples),
        **metrics,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    runs_dir = OUTPUT_DIR / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "stage1_results.csv"

    rows, docs, doc_row_indices = load_input_rows(INPUT_CSV, TEXT_COLUMN)
    logging.info("Input rows: %s", len(rows))
    logging.info("Input texts used for BERTopic: %s", len(docs))
    logging.info("Pretokenizing texts for CountVectorizer topic words")
    vectorizer_docs = [pretokenize_for_vectorizer(doc) for doc in docs]
    empty_vectorizer_docs = sum(1 for doc in vectorizer_docs if not doc.strip())
    logging.info("Pretokenized empty texts: %s", empty_vectorizer_docs)
    embeddings_path = OUTPUT_DIR / "stage1_embeddings.npy"
    if embeddings_path.exists():
        logging.info("Loading cached embeddings: %s", embeddings_path)
        embeddings = np.load(embeddings_path)
    else:
        logging.info("Loading embedding model: %s", MODEL_NAME)
        embedding_model = SentenceTransformer(MODEL_NAME, local_files_only=True)
        logging.info("Encoding documents once for all stage-1 runs")
        embeddings = embedding_model.encode(docs, show_progress_bar=True)
        np.save(embeddings_path, embeddings)

    results: list[dict[str, str]] = load_existing_results(results_path)
    completed = {
        (int(row["min_cluster_size"]), int(row["min_samples"]))
        for row in results
        if row.get("min_cluster_size") and row.get("min_samples")
    }
    if completed:
        logging.info("Existing completed parameter combinations: %s", len(completed))

    for min_cluster_size in MIN_CLUSTER_SIZES:
        for min_samples in MIN_SAMPLES:
            if (min_cluster_size, min_samples) in completed:
                logging.info("Skipping completed mcs%s_ms%s", min_cluster_size, min_samples)
                continue
            results.append(
                run_one(
                    rows=rows,
                    docs=docs,
                    vectorizer_docs=vectorizer_docs,
                    doc_row_indices=doc_row_indices,
                    embeddings=embeddings,
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    stage_runs_dir=runs_dir,
                )
            )
            completed.add((min_cluster_size, min_samples))
            write_csv_with_fallback(results_path, STAGE1_FIELDS, results)

    results_df = pd.DataFrame(results)
    for column in STAGE1_FIELDS:
        if column not in {"min_cluster_size", "min_samples"}:
            results_df[column] = pd.to_numeric(results_df[column], errors="coerce")
    results_df["min_cluster_size"] = pd.to_numeric(results_df["min_cluster_size"], errors="coerce")
    results_df["min_samples"] = pd.to_numeric(results_df["min_samples"], errors="coerce")

    plot_heatmap(
        results_df,
        "outlier_rate",
        OUTPUT_DIR / "stage1_outlier_heatmap.png",
        "Outlier Rate",
        cmap="YlOrRd",
        percent=True,
    )
    plot_heatmap(results_df, "dbcv", OUTPUT_DIR / "stage1_dbcv_heatmap.png", "DBCV", cmap="Blues")
    plot_heatmap(
        results_df,
        "silhouette",
        OUTPUT_DIR / "stage1_silhouette_heatmap.png",
        "Silhouette",
        cmap="Blues",
    )
    logging.info("Stage 1 complete: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
