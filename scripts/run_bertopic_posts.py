"""Run one final BERTopic model for cleaned Douyin posts with ASR text.

The script reads data/processed/posts_for_bertopic_with_asr.csv and writes
topic modeling outputs to data/processed/bertopic_posts/final_mcs15_ms15_nn15_nc5/.
It does not modify the source CSV files or the raw database.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import jieba


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "posts_for_bertopic_with_asr.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "bertopic_posts" / "final_mcs15_ms15_nn15_nc5"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_TEXT_COLUMN = "text_clean_bertopic_with_asr"

REQUIRED_PACKAGES = {
    "bertopic": "bertopic",
    "jieba": "jieba",
    "sentence_transformers": "sentence-transformers",
    "umap": "umap-learn",
    "hdbscan": "hdbscan",
}

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
ASCII_LETTER_RE = re.compile(r"[A-Za-z]")
PURE_NUMBER_RE = re.compile(r"^\d+$")

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final BERTopic on posts_for_bertopic_with_asr.csv.")
    parser.add_argument("--input", type=Path, default=INPUT_CSV, help="Input posts CSV.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--model-name", default=MODEL_NAME, help="SentenceTransformer model name.")
    parser.add_argument("--ngram-min", type=int, default=1, help="Minimum ngram size for topic representation.")
    parser.add_argument("--ngram-max", type=int, default=2, help="Maximum ngram size for topic representation.")
    parser.add_argument("--nr-topics", default="none", help='BERTopic nr_topics, e.g. "none", "auto", or 30.')
    parser.add_argument("--umap-n-neighbors", type=int, default=15, help="UMAP n_neighbors.")
    parser.add_argument("--umap-n-components", type=int, default=5, help="UMAP n_components.")
    parser.add_argument("--umap-min-dist", type=float, default=0.0, help="UMAP min_dist.")
    parser.add_argument("--umap-metric", default="cosine", help="UMAP distance metric.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for UMAP.")
    parser.add_argument(
        "--hdbscan-min-cluster-size",
        type=int,
        default=15,
        help="HDBSCAN min_cluster_size.",
    )
    parser.add_argument("--hdbscan-min-samples", type=int, default=15, help="HDBSCAN min_samples.")
    parser.add_argument("--hdbscan-metric", default="euclidean", help="HDBSCAN distance metric.")
    parser.add_argument(
        "--cluster-selection-method",
        default="eom",
        choices=["eom", "leaf"],
        help="HDBSCAN cluster_selection_method.",
    )
    parser.add_argument(
        "--text-column",
        default=DEFAULT_TEXT_COLUMN,
        help="Input CSV column to use as BERTopic text.",
    )
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow downloading the embedding model if it is not already cached locally.",
    )
    return parser.parse_args()


def check_dependencies() -> None:
    missing = [
        package_name
        for module_name, package_name in REQUIRED_PACKAGES.items()
        if importlib.util.find_spec(module_name) is None
    ]
    if missing:
        print("Missing required dependencies:", file=sys.stderr)
        for package_name in missing:
            print(f"- {package_name}", file=sys.stderr)
        print("Please install the missing dependencies before running this script.", file=sys.stderr)
        raise SystemExit(1)


def parse_nr_topics(value: str) -> str | int | None:
    normalized = value.strip().lower()
    if normalized in {"none", "null"}:
        return None
    if normalized == "auto":
        return value
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('--nr-topics must be "none", "auto", or an integer.') from exc


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


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> Path:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    try:
        temp_path.replace(path)
        return path
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_path = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
        temp_path.replace(fallback_path)
        logging.warning("Could not overwrite %s. Saved fallback CSV to %s", path, fallback_path)
        return fallback_path


def is_valid_topic_token(token: str) -> bool:
    token = token.strip().lower()
    if not token:
        return False
    if PURE_NUMBER_RE.fullmatch(token):
        return False
    if ASCII_LETTER_RE.search(token):
        return False
    if CJK_RE.search(token):
        return len(token) >= 2
    return False


def pretokenize_for_vectorizer(text: str) -> str:
    tokens = [token.strip() for token in jieba.cut(text)]
    return " ".join(token for token in tokens if is_valid_topic_token(token))


def build_topic_summary(topic_model: Any, result_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
        words = topic_model.get_topic(topic) or []
        top_words = " ".join(word for word, _ in words[:10])
        example_texts = examples.get(topic, [])
        summary_rows.append(
            {
                "topic": topic,
                "count": counts[topic],
                "topic_name": topic_model.topic_labels_.get(topic, str(topic)),
                "top_words": top_words,
                "example_text_1": example_texts[0] if len(example_texts) > 0 else "",
                "example_text_2": example_texts[1] if len(example_texts) > 1 else "",
                "example_text_3": example_texts[2] if len(example_texts) > 2 else "",
            }
        )
    return summary_rows


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    check_dependencies()
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sentence_transformers import SentenceTransformer
    from umap import UMAP

    input_csv = args.input.resolve()
    output_dir = args.output_dir.resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)
    topic_results_path = output_dir / "topic_results.csv"
    topic_summary_path = output_dir / "topic_summary.csv"
    run_config_path = output_dir / "run_config.json"
    topic_model_path = output_dir / "topic_model.pkl"

    started_at = time.time()
    rows, docs, doc_row_indices = load_input_rows(input_csv, args.text_column)
    logging.info("Input rows: %s", len(rows))
    logging.info("Text column: %s", args.text_column)
    logging.info("Input texts used for BERTopic: %s", len(docs))
    if not docs:
        raise ValueError(f"No non-empty {args.text_column} values found.")
    if args.ngram_min <= 0 or args.ngram_max <= 0:
        raise ValueError("--ngram-min and --ngram-max must be greater than 0.")
    if args.ngram_min > args.ngram_max:
        raise ValueError("--ngram-min cannot be greater than --ngram-max.")

    nr_topics = parse_nr_topics(str(args.nr_topics))
    logging.info("Embedding model: %s", args.model_name)
    logging.info("Topic representation: jieba pretokenized Chinese text, CountVectorizer split on spaces")
    logging.info("BERTopic parameters: nr_topics=%s, calculate_probabilities=True", nr_topics)
    logging.info("CountVectorizer parameters: ngram_range=(%s, %s)", args.ngram_min, args.ngram_max)
    logging.info(
        "UMAP parameters: n_neighbors=%s, n_components=%s, min_dist=%s, metric=%s, random_state=%s",
        args.umap_n_neighbors,
        args.umap_n_components,
        args.umap_min_dist,
        args.umap_metric,
        args.random_state,
    )
    logging.info(
        "HDBSCAN parameters: min_cluster_size=%s, min_samples=%s, metric=%s, cluster_selection_method=%s",
        args.hdbscan_min_cluster_size,
        args.hdbscan_min_samples,
        args.hdbscan_metric,
        args.cluster_selection_method,
    )

    embedding_model = SentenceTransformer(args.model_name, local_files_only=not args.allow_model_download)
    from sklearn.feature_extraction.text import CountVectorizer

    logging.info("Generating embeddings from original cleaned texts.")
    embeddings = embedding_model.encode(docs, show_progress_bar=True)
    vectorizer_docs = [pretokenize_for_vectorizer(doc) for doc in docs]
    empty_representation_count = sum(1 for doc in vectorizer_docs if not doc.strip())
    if empty_representation_count:
        logging.warning("Texts with empty topic-representation tokens: %s", empty_representation_count)

    vectorizer_model = CountVectorizer(
        tokenizer=str.split,
        token_pattern=None,
        lowercase=False,
        ngram_range=(args.ngram_min, args.ngram_max),
    )
    umap_model = UMAP(
        n_neighbors=args.umap_n_neighbors,
        n_components=args.umap_n_components,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        random_state=args.random_state,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=args.hdbscan_min_cluster_size,
        min_samples=args.hdbscan_min_samples,
        metric=args.hdbscan_metric,
        cluster_selection_method=args.cluster_selection_method,
        prediction_data=True,
    )
    topic_model = BERTopic(
        embedding_model=None,
        vectorizer_model=vectorizer_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        nr_topics=nr_topics,
        calculate_probabilities=True,
        language="multilingual",
        verbose=True,
    )

    topics, probabilities = topic_model.fit_transform(vectorizer_docs, embeddings)

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
    non_noise_topics = {int(row["topic"]) for row in summary_rows if int(row["topic"]) != -1}
    noise_count = sum(1 for row in result_rows if row.get("topic") == -1)

    actual_topic_results_path = write_csv(topic_results_path, RESULT_FIELDS, result_rows)
    actual_topic_summary_path = write_csv(topic_summary_path, SUMMARY_FIELDS, summary_rows)
    topic_model.save(str(topic_model_path), serialization="pickle")

    run_config = {
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "topic_model_path": str(topic_model_path),
        "topic_results_path": str(actual_topic_results_path),
        "topic_summary_path": str(actual_topic_summary_path),
        "text_column": args.text_column,
        "model_name": args.model_name,
        "embedding_source": "original cleaned text",
        "topic_representation_source": "jieba pretokenized cleaned text",
        "ngram_range": [args.ngram_min, args.ngram_max],
        "nr_topics": nr_topics,
        "calculate_probabilities": True,
        "language": "multilingual",
        "umap": {
            "n_neighbors": args.umap_n_neighbors,
            "n_components": args.umap_n_components,
            "min_dist": args.umap_min_dist,
            "metric": args.umap_metric,
            "random_state": args.random_state,
        },
        "hdbscan": {
            "min_cluster_size": args.hdbscan_min_cluster_size,
            "min_samples": args.hdbscan_min_samples,
            "metric": args.hdbscan_metric,
            "cluster_selection_method": args.cluster_selection_method,
            "prediction_data": True,
        },
        "runtime_seconds": round(time.time() - started_at, 3),
    }
    run_config_path.write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

    logging.info("Topics generated excluding topic=-1: %s", len(non_noise_topics))
    logging.info("topic=-1 count: %s", noise_count)
    logging.info("topic_results.csv: %s", actual_topic_results_path)
    logging.info("topic_summary.csv: %s", actual_topic_summary_path)
    logging.info("topic_model.pkl: %s", topic_model_path)
    logging.info("run_config.json: %s", run_config_path)


if __name__ == "__main__":
    main()
