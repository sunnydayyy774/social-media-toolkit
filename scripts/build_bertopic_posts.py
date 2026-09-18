"""Create a BERTopic-ready posts CSV without changing posts_clean.csv."""

from __future__ import annotations

import argparse
import csv
import logging
import re
import unicodedata
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INPUT_CSV = PROCESSED_DIR / "posts_clean.csv"
OUTPUT_CSV = PROCESSED_DIR / "posts_for_bertopic.csv"

URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
MENTION_RE = re.compile(r"@\S+")
WHITESPACE_RE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create data/processed/posts_for_bertopic.csv.")
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_CSV,
        help="Input posts_clean.csv path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_CSV,
        help="Output posts_for_bertopic.csv path.",
    )
    return parser.parse_args()


def keep_for_topic_modeling(char: str) -> bool:
    """Keep letters, numbers, CJK characters, and spaces."""
    if char.isspace():
        return True
    category = unicodedata.category(char)
    if category.startswith(("L", "N")):
        return True
    return False


def clean_for_bertopic(text: str) -> str:
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = text.replace("#", " ")
    text = "".join(char if keep_for_topic_modeling(char) else " " for char in text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    input_csv = args.input.resolve()
    output_csv = args.output.resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output_csv.with_suffix(output_csv.suffix + ".tmp")

    total_rows = 0
    empty_clean_rows = 0
    with input_csv.open("r", newline="", encoding="utf-8-sig") as infile:
        reader = csv.DictReader(infile)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV has no header: {input_csv}")

        output_fields = list(reader.fieldnames)
        if "text_clean_bertopic" not in output_fields:
            output_fields.append("text_clean_bertopic")

        with temp_output.open("w", newline="", encoding="utf-8-sig") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=output_fields)
            writer.writeheader()
            for row in reader:
                total_rows += 1
                cleaned = clean_for_bertopic(row.get("text", ""))
                if not cleaned:
                    empty_clean_rows += 1
                row["text_clean_bertopic"] = cleaned
                writer.writerow(row)

    temp_output.replace(output_csv)
    logging.info("Input: %s", input_csv)
    logging.info("Output: %s", output_csv)
    logging.info("Rows processed: %s", total_rows)
    logging.info("Rows with empty BERTopic text: %s", empty_clean_rows)


if __name__ == "__main__":
    main()
