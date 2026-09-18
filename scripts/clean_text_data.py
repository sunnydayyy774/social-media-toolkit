"""Export conservative clean text datasets from the working DuckDB database."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKING_DB = PROJECT_ROOT / "data" / "working" / "douyin_work.duckdb"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

POSTS_OUTPUT = PROCESSED_DIR / "posts_clean.csv"
COMMENTS_OUTPUT = PROCESSED_DIR / "comments_clean.csv"
TEXT_OUTPUT = PROCESSED_DIR / "text_for_analysis.csv"

WHITESPACE_RE = re.compile(r"\s+")
CONTROL_CATEGORY_PREFIXES = {"C"}
KEEP_CONTROLS = {"\n", "\r", "\t"}

POST_FIELDS = [
    "id",
    "aweme_id",
    "text",
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

COMMENT_FIELDS = [
    "id",
    "comment_id",
    "aweme_id",
    "parent_comment_id",
    "text",
    "create_time",
    "likes",
    "reply_comment_total",
    "user_uid",
    "user_nickname",
    "user_region",
    "has_media",
    "image_count",
    "video_count",
    "audio_count",
    "updated_at",
]

TEXT_FIELDS = ["id", "type", "text", "create_time", "likes", "aweme_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create processed clean text CSV files.")
    parser.add_argument(
        "--db",
        type=Path,
        default=WORKING_DB,
        help="Working DuckDB path. Defaults to data/working/douyin_work.duckdb.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Processed output directory. Defaults to data/processed.",
    )
    return parser.parse_args()


def remove_control_chars(text: str) -> str:
    chars: list[str] = []
    for char in text:
        if char in KEEP_CONTROLS:
            chars.append(char)
            continue
        category = unicodedata.category(char)
        if category and category[0] in CONTROL_CATEGORY_PREFIXES:
            continue
        chars.append(char)
    return "".join(chars)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = remove_control_chars(text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def safe_json_loads(raw: str, table: str, row_id: str) -> dict[str, Any] | None:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        logging.warning("Skipping invalid JSON in %s id=%s", table, row_id)
        return None
    if not isinstance(value, dict):
        logging.warning("Skipping non-object JSON in %s id=%s", table, row_id)
        return None
    return value


def scalar(value: Any, default: Any = "") -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def iter_table(con: duckdb.DuckDBPyConnection, table: str) -> Iterable[tuple[str, str, Any]]:
    query = f'select id, data, updated_at from "{table}"'
    yield from con.execute(query).fetchall()


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> int:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temp_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    temp_path.replace(path)
    return count


def build_post_rows(con: duckdb.DuckDBPyConnection) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    post_rows: list[dict[str, Any]] = []
    text_rows: list[dict[str, Any]] = []

    for row_id, raw_data, updated_at in iter_table(con, "douyin_posts"):
        obj = safe_json_loads(raw_data, "douyin_posts", row_id)
        if obj is None:
            continue
        text = clean_text(obj.get("desc"))

        aweme_id = scalar(obj.get("aweme_id") or obj.get("id") or row_id)
        likes = scalar(obj.get("digg_count"), 0)
        post = {
            "id": row_id,
            "aweme_id": aweme_id,
            "text": text,
            "create_time": scalar(obj.get("create_time")),
            "likes": likes,
            "comment_count": scalar(obj.get("comment_count"), 0),
            "share_count": scalar(obj.get("share_count"), 0),
            "collect_count": scalar(obj.get("collect_count"), 0),
            "play_count": scalar(obj.get("play_count"), 0),
            "author_id": scalar(obj.get("author_uid") or obj.get("author_id")),
            "author_nickname": scalar(obj.get("author_nickname")),
            "search_keyword": scalar(obj.get("search_keyword")),
            "hashtag_names_csv": scalar(obj.get("hashtag_names_csv")),
            "updated_at": updated_at,
        }
        post_rows.append(post)
        if text:
            text_rows.append(
                {
                    "id": aweme_id,
                    "type": "post",
                    "text": text,
                    "create_time": post["create_time"],
                    "likes": likes,
                    "aweme_id": aweme_id,
                }
            )

    return post_rows, text_rows


def build_comment_rows(con: duckdb.DuckDBPyConnection) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    comment_rows: list[dict[str, Any]] = []
    text_rows: list[dict[str, Any]] = []
    seen_comment_ids: set[str] = set()

    for row_id, raw_data, updated_at in iter_table(con, "douyin_comments"):
        obj = safe_json_loads(raw_data, "douyin_comments", row_id)
        if obj is None:
            continue
        text = clean_text(obj.get("text"))
        if not text:
            continue

        comment_id = str(scalar(obj.get("comment_id") or obj.get("id") or row_id))
        if comment_id in seen_comment_ids:
            continue
        seen_comment_ids.add(comment_id)

        aweme_id = scalar(obj.get("aweme_id"))
        likes = scalar(obj.get("digg_count"), 0)
        comment = {
            "id": row_id,
            "comment_id": comment_id,
            "aweme_id": aweme_id,
            "parent_comment_id": scalar(obj.get("parent_comment_id")),
            "text": text,
            "create_time": scalar(obj.get("create_time")),
            "likes": likes,
            "reply_comment_total": scalar(obj.get("reply_comment_total"), 0),
            "user_uid": scalar(obj.get("user_uid")),
            "user_nickname": scalar(obj.get("user_nickname")),
            "user_region": scalar(obj.get("user_region")),
            "has_media": scalar(obj.get("has_media"), False),
            "image_count": scalar(obj.get("image_count"), 0),
            "video_count": scalar(obj.get("video_count"), 0),
            "audio_count": scalar(obj.get("audio_count"), 0),
            "updated_at": updated_at,
        }
        comment_rows.append(comment)
        text_rows.append(
            {
                "id": comment_id,
                "type": "comment",
                "text": text,
                "create_time": comment["create_time"],
                "likes": likes,
                "aweme_id": aweme_id,
            }
        )

    return comment_rows, text_rows


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    db_path = args.db.resolve()
    output_dir = args.output_dir.resolve()
    if not db_path.exists():
        raise FileNotFoundError(f"Working database not found: {db_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    posts_output = output_dir / POSTS_OUTPUT.name
    comments_output = output_dir / COMMENTS_OUTPUT.name
    text_output = output_dir / TEXT_OUTPUT.name

    logging.info("Opening working database read-only: %s", db_path)
    con = duckdb.connect(str(db_path), read_only=True)

    logging.info("Building post rows.")
    post_rows, post_text_rows = build_post_rows(con)
    logging.info("Building comment rows.")
    comment_rows, comment_text_rows = build_comment_rows(con)

    text_rows = post_text_rows + comment_text_rows
    logging.info("Writing posts: %s", posts_output)
    post_count = write_csv(posts_output, POST_FIELDS, post_rows)
    logging.info("Writing comments: %s", comments_output)
    comment_count = write_csv(comments_output, COMMENT_FIELDS, comment_rows)
    logging.info("Writing unified text dataset: %s", text_output)
    text_count = write_csv(text_output, TEXT_FIELDS, text_rows)

    logging.info("Done.")
    logging.info("posts_clean rows=%s", post_count)
    logging.info("comments_clean rows=%s", comment_count)
    logging.info("text_for_analysis rows=%s", text_count)


if __name__ == "__main__":
    main()
