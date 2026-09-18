"""Run post-level sentiment analysis for Douyin analysis posts.

First version: model decides the final sentiment label. Keyword rules are only
recorded as auxiliary evidence and do not change model labels.
"""

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
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "analysis_inputs" / "posts_for_analysis_topic_labeled.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "sentiment_analysis" / "posts_sentiment.csv"
DEFAULT_MODEL = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"

SENTIMENT_ORDER = ["negative", "neutral", "positive"]
OUTPUT_COLUMNS = [
    "aweme_id",
    "create_time",
    "topic",
    "text",
    "text_for_sentiment",
    "sentiment_label",
    "sentiment_score",
    "model_label",
    "model_score",
    "sentiment_method",
    "matched_keywords",
    "include_in_analysis",
    "sentiment_status",
    "error",
    "updated_at",
]

EMOTION_KEYWORDS = [
    "危险",
    "威胁",
    "警惕",
    "反对",
    "愤怒",
    "恐惧",
    "担忧",
    "军国主义",
    "战争",
    "火药桶",
    "侵略",
    "挑衅",
    "扩军",
    "再军事化",
    "右翼",
    "不安",
    "危机",
    "反战",
    "抗议",
    "和平",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run post-level sentiment analysis.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--text-column", default="text_clean_bertopic_with_asr")
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dry-run", action="store_true", help="Print input counts without loading the model.")
    parser.add_argument("--overwrite", action="store_true", help="Ignore existing success rows and append a fresh run.")
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


def find_keywords(text: str) -> str:
    matched = [keyword for keyword in EMOTION_KEYWORDS if keyword in text]
    return ";".join(dict.fromkeys(matched))


def load_existing_success(output_path: Path, overwrite: bool) -> set[str]:
    if overwrite or not output_path.exists():
        return set()
    existing = pd.read_csv(output_path, dtype=str).fillna("")
    if "aweme_id" not in existing.columns or "sentiment_status" not in existing.columns:
        return set()
    return set(existing.loc[existing["sentiment_status"].eq("success"), "aweme_id"].astype(str))


def append_rows(output_path: Path, rows: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    write_header = not output_path.exists()
    df.to_csv(output_path, mode="a", header=write_header, index=False, encoding="utf-8-sig")


def normalize_label(label: str) -> str:
    label = str(label).lower().strip()
    if label in SENTIMENT_ORDER:
        return label
    # Some pipelines return LABEL_0 style labels; keep failures explicit rather
    # than silently guessing a wrong sentiment.
    raise ValueError(f"Unsupported model label: {label}")


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


def build_output_row(row: dict[str, Any], model_name: str, status: str, error: str = "") -> dict[str, Any]:
    now = datetime.now().isoformat(timespec="seconds")
    return {
        "aweme_id": row.get("aweme_id", ""),
        "create_time": row.get("create_time", ""),
        "topic": row.get("topic", ""),
        "text": row.get("text", ""),
        "text_for_sentiment": row.get("text_for_sentiment", ""),
        "sentiment_label": "",
        "sentiment_score": "",
        "model_label": "",
        "model_score": "",
        "sentiment_method": "model_only",
        "matched_keywords": row.get("matched_keywords", ""),
        "include_in_analysis": row.get("include_in_analysis", ""),
        "sentiment_status": status,
        "error": error,
        "updated_at": now,
    }


def main() -> None:
    args = parse_args()
    setup_logging()
    check_dependencies(load_model=not args.dry_run)

    posts = pd.read_csv(args.input, dtype=str).fillna("")
    required = {"aweme_id", "create_time", "topic", "text", args.text_column, "include_in_analysis"}
    missing = required - set(posts.columns)
    if missing:
        raise SystemExit("Input is missing required columns: " + ", ".join(sorted(missing)))

    selected = posts[posts["include_in_analysis"].str.lower().eq("true")].copy()
    selected["text_for_sentiment"] = selected[args.text_column].map(
        lambda value: clean_text_for_sentiment(value, args.max_chars)
    )
    selected["matched_keywords"] = selected["text_for_sentiment"].map(find_keywords)
    eligible_count = len(selected[selected["text_for_sentiment"].str.len() > 0])

    completed = load_existing_success(args.output, args.overwrite)
    if completed:
        selected = selected[~selected["aweme_id"].astype(str).isin(completed)].copy()
    if args.max_items is not None:
        selected = selected.head(args.max_items).copy()

    logging.info("Input posts: %s", len(posts))
    logging.info("Rows with include_in_analysis=true and non-empty text: %s", eligible_count)
    logging.info("Already completed success rows skipped: %s", len(completed))
    logging.info("Rows to process in this run: %s", len(selected))
    logging.info("Output CSV: %s", args.output.resolve())
    if args.dry_run:
        return

    classifier = load_sentiment_pipeline(args.model_name, args.device)
    rows_buffer: list[dict[str, Any]] = []

    records = selected.to_dict(orient="records")
    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        valid_rows = [row for row in batch if str(row.get("text_for_sentiment", "")).strip()]
        skipped_rows = [row for row in batch if not str(row.get("text_for_sentiment", "")).strip()]

        for row in skipped_rows:
            rows_buffer.append(build_output_row(row, args.model_name, "skipped_empty_text"))

        if valid_rows:
            texts = [row["text_for_sentiment"] for row in valid_rows]
            try:
                predictions = classifier(texts, truncation=True)
                for row, prediction in zip(valid_rows, predictions):
                    output_row = build_output_row(row, args.model_name, "success")
                    model_label = normalize_label(prediction.get("label", ""))
                    model_score = float(prediction.get("score", 0))
                    output_row.update(
                        {
                            "sentiment_label": model_label,
                            "sentiment_score": model_score,
                            "model_label": model_label,
                            "model_score": model_score,
                        }
                    )
                    rows_buffer.append(output_row)
            except Exception as exc:  # noqa: BLE001 - keep row-level results resumable.
                for row in valid_rows:
                    rows_buffer.append(build_output_row(row, args.model_name, "failed", str(exc)))

        append_rows(args.output, rows_buffer)
        rows_buffer = []
        processed = min(start + args.batch_size, len(records))
        logging.info("Processed %s/%s rows", processed, len(records))

    logging.info("Sentiment analysis complete: %s", args.output.resolve())


if __name__ == "__main__":
    main()
