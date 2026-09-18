"""Run BERTopic stage-2 UMAP parameter evaluation.

This script fixes HDBSCAN at min_cluster_size=15 and min_samples=15,
then evaluates UMAP n_neighbors and n_components combinations. It writes
per-run BERTopic outputs, a compact metrics table, a candidate table, and
a scatter plot for selecting the next manual-review candidates.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
import string
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
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "bertopic_evaluation" / "stage2"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
TEXT_COLUMN = "text_clean_bertopic_with_asr"

N_NEIGHBORS = [10, 15, 30, 50]
N_COMPONENTS = [5, 10, 15]
UMAP_MIN_DIST = 0.0
UMAP_METRIC = "cosine"
RANDOM_STATE = 42
HDBSCAN_MIN_CLUSTER_SIZE = 15
HDBSCAN_MIN_SAMPLES = 15

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

STAGE2_FIELDS = [
    "run_id",
    "candidate_id",
    "umap_n_neighbors",
    "umap_n_components",
    "umap_min_dist",
    "umap_metric",
    "hdbscan_min_cluster_size",
    "hdbscan_min_samples",
    "topic_count",
    "outlier_rate",
    "dbcv",
    "silhouette",
    "davies_bouldin",
    "is_pareto",
    "is_candidate",
]

CANDIDATE_FIELDS = STAGE2_FIELDS + ["selection_reason"]


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
        return False
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
    vectorizer_docs: list[str],
    doc_row_indices: list[int],
    embeddings: np.ndarray,
    run_id: str,
    run_dir: Path,
    n_neighbors: int,
    n_components: int,
) -> dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Running %s: n_neighbors=%s, n_components=%s", run_id, n_neighbors, n_components)
    started = time.time()

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=RANDOM_STATE,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
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
        "run_id": run_id,
        "input": str(INPUT_CSV),
        "text_column": TEXT_COLUMN,
        "embedding_model": MODEL_NAME,
        "ngram_range": [1, 2],
        "vectorizer": {
            "tokenizer": "jieba_pretokenized_space_separated",
            "filter_pure_number_token": True,
            "filter_english_token": True,
        },
        "nr_topics": None,
        "language": "multilingual",
        "umap": {
            "n_neighbors": n_neighbors,
            "n_components": n_components,
            "min_dist": UMAP_MIN_DIST,
            "metric": UMAP_METRIC,
            "random_state": RANDOM_STATE,
        },
        "hdbscan": {
            "min_cluster_size": HDBSCAN_MIN_CLUSTER_SIZE,
            "min_samples": HDBSCAN_MIN_SAMPLES,
            "metric": "euclidean",
            "cluster_selection_method": "eom",
            "prediction_data": True,
        },
        "runtime_seconds": round(time.time() - started, 3),
    }
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)

    return {
        "run_id": run_id,
        "candidate_id": "",
        "umap_n_neighbors": str(n_neighbors),
        "umap_n_components": str(n_components),
        "umap_min_dist": safe_metric(UMAP_MIN_DIST),
        "umap_metric": UMAP_METRIC,
        "hdbscan_min_cluster_size": str(HDBSCAN_MIN_CLUSTER_SIZE),
        "hdbscan_min_samples": str(HDBSCAN_MIN_SAMPLES),
        **metrics,
        "is_pareto": "false",
        "is_candidate": "false",
    }


def is_pareto_efficient(results: pd.DataFrame) -> pd.Series:
    pareto = []
    for _, row in results.iterrows():
        dominated = False
        for _, other in results.iterrows():
            if other.name == row.name:
                continue
            other_better_or_equal = (
                other["dbcv"] >= row["dbcv"]
                and other["outlier_rate"] <= row["outlier_rate"]
            )
            other_strictly_better = (
                other["dbcv"] > row["dbcv"]
                or other["outlier_rate"] < row["outlier_rate"]
            )
            if other_better_or_equal and other_strictly_better:
                dominated = True
                break
        pareto.append(not dominated)
    return pd.Series(pareto, index=results.index)


def select_candidates(results: pd.DataFrame, max_candidates: int = 6) -> pd.DataFrame:
    scored = results.copy()
    scored["is_pareto"] = is_pareto_efficient(scored)
    dbcv_min, dbcv_max = scored["dbcv"].min(), scored["dbcv"].max()
    out_min, out_max = scored["outlier_rate"].min(), scored["outlier_rate"].max()
    scored["dbcv_norm"] = 0.0 if dbcv_max == dbcv_min else (scored["dbcv"] - dbcv_min) / (dbcv_max - dbcv_min)
    scored["outlier_norm"] = 0.0 if out_max == out_min else (out_max - scored["outlier_rate"]) / (out_max - out_min)
    scored["balanced_score"] = (scored["dbcv_norm"] + scored["outlier_norm"]) / 2

    chosen_indices: list[int] = []
    reasons: dict[int, str] = {}

    pareto_sorted = scored[scored["is_pareto"]].sort_values(
        ["balanced_score", "dbcv", "outlier_rate"], ascending=[False, False, True]
    )
    for idx, _ in pareto_sorted.head(max_candidates).iterrows():
        chosen_indices.append(idx)
        reasons[idx] = "pareto_balanced"

    best_dbcv_idx = scored["dbcv"].idxmax()
    if best_dbcv_idx not in chosen_indices:
        chosen_indices.append(best_dbcv_idx)
        reasons[best_dbcv_idx] = "best_dbcv"

    lowest_outlier_idx = scored["outlier_rate"].idxmin()
    if lowest_outlier_idx not in chosen_indices:
        chosen_indices.append(lowest_outlier_idx)
        reasons[lowest_outlier_idx] = "lowest_outlier"

    chosen = scored.loc[chosen_indices].drop_duplicates(subset=["run_id"]).head(max_candidates).copy()
    chosen = chosen.sort_values(["balanced_score", "dbcv"], ascending=[False, False]).reset_index(drop=True)
    letters = list(string.ascii_uppercase)
    chosen["candidate_id"] = [letters[index] for index in range(len(chosen))]
    chosen["selection_reason"] = [reasons.get(index, "candidate") for index in chosen_indices[: len(chosen)]]

    return chosen


def plot_candidate_scatter(results: pd.DataFrame, candidates: pd.DataFrame, output_path: Path) -> None:
    fig = plt.figure(figsize=(11.5, 6.4))
    grid = fig.add_gridspec(1, 2, width_ratios=[2.1, 1.15], wspace=0.05)
    ax = fig.add_subplot(grid[0, 0])
    table_ax = fig.add_subplot(grid[0, 1])

    ax.scatter(
        results["outlier_rate"] * 100,
        results["dbcv"],
        s=48,
        color="#c8d0d8",
        edgecolor="white",
        linewidth=0.8,
        label="All 12 runs",
        zorder=2,
    )
    ax.scatter(
        candidates["outlier_rate"] * 100,
        candidates["dbcv"],
        s=82,
        color="#1f77b4",
        edgecolor="#12395a",
        linewidth=0.8,
        label="Candidates",
        zorder=3,
    )

    for _, row in candidates.iterrows():
        ax.annotate(
            row["candidate_id"],
            (row["outlier_rate"] * 100, row["dbcv"]),
            xytext=(5, 6),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            color="#12395a",
        )

    ax.set_xlabel("Outlier rate (%)", fontsize=11)
    ax.set_ylabel("DBCV", fontsize=11)
    ax.set_title("Stage 2 UMAP Candidates", fontsize=15, fontweight="bold", color="#12395a")
    ax.grid(alpha=0.22)
    ax.legend(loc="best", fontsize=9, frameon=False)

    table_ax.axis("off")
    table_rows = [
        [
            row["candidate_id"],
            f"{int(row['umap_n_neighbors'])}/{int(row['umap_n_components'])}",
            "15/15",
            f"{row['outlier_rate'] * 100:.1f}%",
            f"{row['dbcv']:.3f}",
        ]
        for _, row in candidates.iterrows()
    ]
    table = table_ax.table(
        cellText=table_rows,
        colLabels=["ID", "UMAP nn/nc", "HDBSCAN", "Outlier", "DBCV"],
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.08, 1.45)
    for (row_idx, _), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#12395a")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#eef5f9" if row_idx % 2 else "#ffffff")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    runs_dir = OUTPUT_DIR / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "stage2_results.csv"

    rows, docs, doc_row_indices = load_input_rows(INPUT_CSV, TEXT_COLUMN)
    logging.info("Input rows: %s", len(rows))
    logging.info("Input texts used for BERTopic: %s", len(docs))
    if not docs:
        raise ValueError(f"No non-empty {TEXT_COLUMN} values found.")

    logging.info("Pretokenizing texts for CountVectorizer topic words")
    vectorizer_docs = [pretokenize_for_vectorizer(doc) for doc in docs]
    empty_vectorizer_docs = sum(1 for doc in vectorizer_docs if not doc.strip())
    logging.info("Pretokenized empty texts: %s", empty_vectorizer_docs)

    embeddings_path = OUTPUT_DIR / "stage2_embeddings.npy"
    if embeddings_path.exists():
        logging.info("Loading cached embeddings: %s", embeddings_path)
        embeddings = np.load(embeddings_path)
    else:
        logging.info("Loading embedding model: %s", MODEL_NAME)
        embedding_model = SentenceTransformer(MODEL_NAME, local_files_only=True)
        logging.info("Encoding documents once for all stage-2 runs")
        embeddings = embedding_model.encode(docs, show_progress_bar=True)
        np.save(embeddings_path, embeddings)

    results: list[dict[str, str]] = load_existing_results(results_path)
    completed = {row["run_id"] for row in results if row.get("run_id")}
    if completed:
        logging.info("Existing completed runs: %s", len(completed))

    run_specs: list[tuple[str, Path, int, int]] = []
    run_number = 1
    for n_neighbors in N_NEIGHBORS:
        for n_components in N_COMPONENTS:
            run_id = f"run_{run_number:03d}_nn{n_neighbors}_nc{n_components}_mcs15_ms15"
            run_specs.append((run_id, runs_dir / run_id, n_neighbors, n_components))
            run_number += 1

    for run_id, run_dir, n_neighbors, n_components in run_specs:
        if run_id in completed:
            logging.info("Skipping completed %s", run_id)
            continue
        results.append(
            run_one(
                rows=rows,
                vectorizer_docs=vectorizer_docs,
                doc_row_indices=doc_row_indices,
                embeddings=embeddings,
                run_id=run_id,
                run_dir=run_dir,
                n_neighbors=n_neighbors,
                n_components=n_components,
            )
        )
        completed.add(run_id)
        write_csv_with_fallback(results_path, STAGE2_FIELDS, results)

    results_df = pd.DataFrame(results)
    for column in [
        "umap_n_neighbors",
        "umap_n_components",
        "umap_min_dist",
        "hdbscan_min_cluster_size",
        "hdbscan_min_samples",
        "topic_count",
        "outlier_rate",
        "dbcv",
        "silhouette",
        "davies_bouldin",
    ]:
        results_df[column] = pd.to_numeric(results_df[column], errors="coerce")

    candidates_df = select_candidates(results_df)
    results_df["is_pareto"] = is_pareto_efficient(results_df).map(lambda value: "true" if value else "false")
    results_df["is_candidate"] = "false"
    results_df["candidate_id"] = ""
    for _, candidate in candidates_df.iterrows():
        match = results_df["run_id"] == candidate["run_id"]
        results_df.loc[match, "is_candidate"] = "true"
        results_df.loc[match, "candidate_id"] = candidate["candidate_id"]

    write_csv(results_path, STAGE2_FIELDS, results_df[STAGE2_FIELDS].to_dict("records"))
    write_csv(
        OUTPUT_DIR / "stage2_candidates.csv",
        CANDIDATE_FIELDS,
        candidates_df[CANDIDATE_FIELDS].to_dict("records"),
    )
    plot_candidate_scatter(results_df, candidates_df, OUTPUT_DIR / "stage2_candidate_scatter.png")
    logging.info("Stage 2 complete: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
