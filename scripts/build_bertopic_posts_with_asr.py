"""Create a BERTopic input CSV that combines post text with ASR transcripts.

This script reads processed CSV files only. It does not modify raw data,
posts_clean.csv, posts_for_bertopic.csv, or media_transcripts.csv.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import unicodedata
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
POSTS_INPUT = PROCESSED_DIR / "posts_for_bertopic.csv"
TRANSCRIPTS_INPUT = PROCESSED_DIR / "media_transcripts.csv"
OUTPUT_CSV = PROCESSED_DIR / "posts_for_bertopic_with_asr.csv"

URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
MENTION_RE = re.compile(r"@\S+")
WHITESPACE_RE = re.compile(r"\s+")
ASCII_LETTER_RE = re.compile(r"[A-Za-z]")
NOISE_PHRASES = [
    "点赞",
    "关注",
    "订阅",
    "谢谢观看",
    "感谢观看",
    "请不吝点赞",
    "点个关注",
    "记得关注",
    "评论区",
    "转发",
    "收藏",
    "中文字幕",
    "字幕",
    "欢迎收看",
    "下期再见",
]
MAX_TRANSCRIPT_CHARS = 1500

ADDED_FIELDS = [
    "text_clean_bertopic_original",
    "transcript_clean",
    "transcript_status",
    "transcript_char_count",
    "text_clean_bertopic_with_asr",
    "asr_join_status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build posts_for_bertopic_with_asr.csv.")
    parser.add_argument("--posts-input", type=Path, default=POSTS_INPUT, help="Input posts_for_bertopic.csv.")
    parser.add_argument("--transcripts-input", type=Path, default=TRANSCRIPTS_INPUT, help="Input media_transcripts.csv.")
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV, help="Output CSV path.")
    return parser.parse_args()


def keep_for_topic_modeling(char: str) -> bool:
    if char.isspace():
        return True
    if char.isdigit():
        return False
    if ASCII_LETTER_RE.fullmatch(char):
        return False
    category = unicodedata.category(char)
    if category.startswith("L"):
        return True
    return False


def filter_token(token: str) -> str:
    token = token.strip()
    if not token:
        return ""
    if any(char.isdigit() for char in token):
        return ""
    if ASCII_LETTER_RE.search(token):
        cjk_letters = sum("\u4e00" <= char <= "\u9fff" for char in token)
        if cjk_letters == 0:
            return ""
        token = ASCII_LETTER_RE.sub("", token)
    return token


def clean_for_bertopic(text: str, max_chars: int | None = None) -> str:
    if max_chars is not None:
        text = (text or "")[:max_chars]
    text = URL_RE.sub(" ", text or "")
    text = MENTION_RE.sub(" ", text)
    text = text.replace("#", " ")
    for phrase in NOISE_PHRASES:
        text = text.replace(phrase, " ")
    text = "".join(char if keep_for_topic_modeling(char) else " " for char in text)
    text = WHITESPACE_RE.sub(" ", text)
    tokens = [filter_token(token) for token in text.split()]
    return " ".join(token for token in tokens if token)


def load_success_transcripts(path: Path) -> dict[str, dict[str, str]]:
    transcripts: dict[str, dict[str, str]] = {}
    duplicate_success_rows = 0

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV has no header: {path}")
        required = {"aweme_id", "transcript_clean", "transcript_status", "transcript_char_count"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing transcript columns in {path}: {sorted(missing)}")

        for row in reader:
            aweme_id = (row.get("aweme_id") or "").strip()
            if not aweme_id:
                continue
            if (row.get("transcript_status") or "").strip() != "success":
                continue
            if aweme_id in transcripts:
                duplicate_success_rows += 1
            transcripts[aweme_id] = row

    if duplicate_success_rows:
        logging.warning("Duplicate success transcript rows overwritten by later rows: %s", duplicate_success_rows)
    return transcripts


def join_text(post_text: str, transcript_text: str) -> str:
    parts = [part for part in [post_text.strip(), transcript_text.strip()] if part]
    return " ".join(parts)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    posts_input = args.posts_input.resolve()
    transcripts_input = args.transcripts_input.resolve()
    output_csv = args.output.resolve()

    if not posts_input.exists():
        raise FileNotFoundError(f"Posts input not found: {posts_input}")
    if not transcripts_input.exists():
        raise FileNotFoundError(f"Transcripts input not found: {transcripts_input}")

    transcripts = load_success_transcripts(transcripts_input)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output_csv.with_suffix(output_csv.suffix + ".tmp")

    total_rows = 0
    matched_success_rows = 0
    rows_with_asr_text = 0
    rows_with_combined_text = 0
    rows_with_empty_combined_text = 0

    with posts_input.open("r", newline="", encoding="utf-8-sig") as infile:
        reader = csv.DictReader(infile)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV has no header: {posts_input}")
        if "aweme_id" not in reader.fieldnames or "text_clean_bertopic" not in reader.fieldnames:
            raise ValueError("Posts input must contain aweme_id and text_clean_bertopic.")

        output_fields = list(reader.fieldnames)
        for field in ADDED_FIELDS:
            if field not in output_fields:
                output_fields.append(field)

        with temp_output.open("w", newline="", encoding="utf-8-sig") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=output_fields, extrasaction="ignore")
            writer.writeheader()

            for row in reader:
                total_rows += 1
                aweme_id = (row.get("aweme_id") or "").strip()
                post_text = clean_for_bertopic(row.get("text_clean_bertopic") or row.get("text", ""))
                transcript_row = transcripts.get(aweme_id)

                row["text_clean_bertopic_original"] = post_text
                row["transcript_clean"] = ""
                row["transcript_status"] = ""
                row["transcript_char_count"] = ""
                row["asr_join_status"] = "no_success_transcript"

                if transcript_row is not None:
                    matched_success_rows += 1
                    row["transcript_status"] = transcript_row.get("transcript_status", "")
                    row["transcript_char_count"] = transcript_row.get("transcript_char_count", "")
                    transcript_text = clean_for_bertopic(
                        transcript_row.get("transcript_clean", ""),
                        max_chars=MAX_TRANSCRIPT_CHARS,
                    )
                    row["transcript_clean"] = transcript_text
                    row["asr_join_status"] = "matched_success_empty_text"
                    if transcript_text:
                        rows_with_asr_text += 1
                        row["asr_join_status"] = "matched_success_with_text"
                else:
                    transcript_text = ""

                combined_text = join_text(post_text, transcript_text)
                row["text_clean_bertopic_with_asr"] = combined_text
                if combined_text:
                    rows_with_combined_text += 1
                else:
                    rows_with_empty_combined_text += 1

                writer.writerow(row)

    temp_output.replace(output_csv)
    logging.info("Posts input: %s", posts_input)
    logging.info("Transcripts input: %s", transcripts_input)
    logging.info("Output: %s", output_csv)
    logging.info("Post rows processed: %s", total_rows)
    logging.info("Rows matched to successful ASR: %s", matched_success_rows)
    logging.info("Rows with non-empty ASR text: %s", rows_with_asr_text)
    logging.info("Rows with non-empty combined BERTopic text: %s", rows_with_combined_text)
    logging.info("Rows with empty combined BERTopic text: %s", rows_with_empty_combined_text)


if __name__ == "__main__":
    main()
