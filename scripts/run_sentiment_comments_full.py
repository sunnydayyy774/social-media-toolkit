"""Run sentiment analysis for all comments kept in the analysis set."""

from __future__ import annotations

import argparse
import importlib.util
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POSTS = PROJECT_ROOT / "data" / "processed" / "analysis_inputs" / "posts_for_analysis_topic_labeled.csv"
DEFAULT_COMMENTS = PROJECT_ROOT / "data" / "processed" / "analysis_inputs" / "comments_for_analysis_topic_labeled.csv"
DEFAULT_POST_SENTIMENT = PROJECT_ROOT / "data" / "processed" / "sentiment_analysis" / "posts_sentiment.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "comment_sentiment_analysis_full"
DEFAULT_MODEL = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"

SENTIMENT_ORDER = ["negative", "neutral", "positive"]
COMMENT_INPUT_COLUMNS = [
    "comment_id",
    "aweme_id",
    "user_id",
    "create_time",
    "topic",
    "text",
    "post_sentiment_label",
    "include_in_analysis",
]
OUTPUT_COLUMNS = [
    "comment_id",
    "aweme_id",
    "user_id",
    "create_time",
    "topic",
    "text",
    "text_for_sentiment",
    "post_sentiment_label",
    "comment_sentiment_label",
    "comment_sentiment_score",
    "model_label",
    "model_score",
    "sentiment_status",
    "include_in_analysis",
    "error",
    "updated_at",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full comment sentiment analysis.")
    parser.add_argument("--posts", type=Path, default=DEFAULT_POSTS)
    parser.add_argument("--comments", type=Path, default=DEFAULT_COMMENTS)
    parser.add_argument("--post-sentiment", type=Path, default=DEFAULT_POST_SENTIMENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--max-chars", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--prepare-only", action="store_true", help="Only write comments_for_sentiment_full.csv.")
    parser.add_argument("--dry-run", action="store_true", help="Print counts without writing sentiment rows.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing comments_sentiment_full.csv.")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)


def check_dependencies(load_model: bool) -> None:
    required = {"pandas": "pandas"}
    if load_model:
        required.update({"torch": "torch", "transformers": "transformers"})
    missing = [package for module, package in required.items() if importlib.util.find_spec(module) is None]
    if missing:
        raise SystemExit("Missing required dependencies: " + ", ".join(missing))


def clean_text_for_sentiment(value: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text[:max_chars]


def normalize_label(label: str) -> str:
    label = str(label).lower().strip()
    if label in SENTIMENT_ORDER:
        return label
    raise ValueError(f"Unsupported model label: {label}")


def as_bool_true(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def load_sentiment_pipeline(model_name: str, device: str):
    import torch
    from transformers import pipeline

    if device == "cuda":
        device_id = 0
    elif device == "cpu":
        device_id = -1
    else:
        device_id = 0 if torch.cuda.is_available() else -1

    logging.info("Loading sentiment model: %s", model_name)
    logging.info("Pipeline device: %s", "cuda:0" if device_id == 0 else "cpu")
    return pipeline("sentiment-analysis", model=model_name, tokenizer=model_name, device=device_id)


def read_post_sentiment(path: Path) -> pd.DataFrame:
    post_sentiment = pd.read_csv(path, dtype=str).fillna("")
    required = {"aweme_id", "sentiment_label", "sentiment_status"}
    missing = required - set(post_sentiment.columns)
    if missing:
        raise ValueError("Post sentiment file is missing columns: " + ", ".join(sorted(missing)))
    post_sentiment = post_sentiment[post_sentiment["sentiment_status"].eq("success")].copy()
    return post_sentiment.rename(columns={"sentiment_label": "post_sentiment_label"})[
        ["aweme_id", "post_sentiment_label"]
    ]


def build_input(args: argparse.Namespace) -> pd.DataFrame:
    posts = pd.read_csv(args.posts, dtype=str).fillna("")
    comments = pd.read_csv(args.comments, dtype=str).fillna("")
    post_sentiment = read_post_sentiment(args.post_sentiment)

    post_required = {"aweme_id", "topic", "include_in_analysis"}
    comment_required = {"comment_id", "aweme_id", "user_uid", "create_time", "text", "include_in_analysis"}
    missing_posts = post_required - set(posts.columns)
    missing_comments = comment_required - set(comments.columns)
    if missing_posts:
        raise ValueError("Posts file is missing columns: " + ", ".join(sorted(missing_posts)))
    if missing_comments:
        raise ValueError("Comments file is missing columns: " + ", ".join(sorted(missing_comments)))

    posts = posts[as_bool_true(posts["include_in_analysis"])][["aweme_id", "topic"]].drop_duplicates("aweme_id")
    selected = comments[as_bool_true(comments["include_in_analysis"])].copy()
    selected = selected.rename(columns={"user_uid": "user_id"})
    selected = selected.merge(posts, on="aweme_id", how="left")
    selected = selected.merge(post_sentiment, on="aweme_id", how="left")
    selected["topic"] = selected["topic"].fillna("missing")
    selected["post_sentiment_label"] = selected["post_sentiment_label"].fillna("missing")
    return selected[COMMENT_INPUT_COLUMNS].copy()


def load_existing_success(output_path: Path, overwrite: bool) -> set[str]:
    if overwrite or not output_path.exists():
        return set()
    existing = pd.read_csv(output_path, dtype=str).fillna("")
    if "comment_id" not in existing.columns or "sentiment_status" not in existing.columns:
        return set()
    return set(existing.loc[existing["sentiment_status"].eq("success"), "comment_id"].astype(str))


def build_output_row(row: dict[str, Any], status: str, error: str = "") -> dict[str, Any]:
    return {
        "comment_id": row.get("comment_id", ""),
        "aweme_id": row.get("aweme_id", ""),
        "user_id": row.get("user_id", ""),
        "create_time": row.get("create_time", ""),
        "topic": row.get("topic", ""),
        "text": row.get("text", ""),
        "text_for_sentiment": row.get("text_for_sentiment", ""),
        "post_sentiment_label": row.get("post_sentiment_label", ""),
        "comment_sentiment_label": "",
        "comment_sentiment_score": "",
        "model_label": "",
        "model_score": "",
        "sentiment_status": status,
        "include_in_analysis": row.get("include_in_analysis", ""),
        "error": error,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def append_rows(output_path: Path, rows: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    write_header = not output_path.exists()
    df.to_csv(output_path, mode="a", header=write_header, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    setup_logging()
    check_dependencies(load_model=not (args.prepare_only or args.dry_run))

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / "comments_for_sentiment_full.csv"
    output_path = output_dir / "comments_sentiment_full.csv"

    selected_comments = build_input(args)
    selected_comments.to_csv(input_path, index=False, encoding="utf-8-sig")
    if args.overwrite and output_path.exists() and not args.dry_run:
        output_path.unlink()

    selected_comments["text_for_sentiment"] = selected_comments["text"].map(
        lambda value: clean_text_for_sentiment(value, args.max_chars)
    )
    completed = load_existing_success(output_path, args.overwrite)
    if completed:
        selected_comments = selected_comments[~selected_comments["comment_id"].astype(str).isin(completed)].copy()
    if args.max_items is not None:
        selected_comments = selected_comments.head(args.max_items).copy()

    logging.info("Full comments input: %s", input_path)
    logging.info("Already completed success comments skipped: %s", len(completed))
    logging.info("Comment rows to process in this run: %s", len(selected_comments))
    logging.info("Output CSV: %s", output_path)
    if args.prepare_only or args.dry_run:
        return

    classifier = load_sentiment_pipeline(args.model_name, args.device)
    records = selected_comments.to_dict(orient="records")
    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        rows_buffer: list[dict[str, Any]] = []
        valid_rows = [row for row in batch if str(row.get("text_for_sentiment", "")).strip()]
        skipped_rows = [row for row in batch if not str(row.get("text_for_sentiment", "")).strip()]

        for row in skipped_rows:
            rows_buffer.append(build_output_row(row, "skipped_empty_text"))

        if valid_rows:
            texts = [row["text_for_sentiment"] for row in valid_rows]
            try:
                predictions = classifier(texts, truncation=True)
                for row, prediction in zip(valid_rows, predictions):
                    model_label = normalize_label(prediction.get("label", ""))
                    model_score = float(prediction.get("score", 0))
                    output_row = build_output_row(row, "success")
                    output_row.update(
                        {
                            "comment_sentiment_label": model_label,
                            "comment_sentiment_score": model_score,
                            "model_label": model_label,
                            "model_score": model_score,
                        }
                    )
                    rows_buffer.append(output_row)
            except Exception as exc:  # noqa: BLE001
                for row in valid_rows:
                    rows_buffer.append(build_output_row(row, "failed", str(exc)))

        append_rows(output_path, rows_buffer)
        processed = min(start + args.batch_size, len(records))
        if processed % max(args.batch_size * 20, 1) == 0 or processed == len(records):
            logging.info("Processed %s/%s comments", processed, len(records))

    logging.info("Full comment sentiment complete: %s", output_path)


if __name__ == "__main__":
    main()
