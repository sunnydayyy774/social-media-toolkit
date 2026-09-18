"""Build unified post/comment inputs for downstream analysis from topic labels."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOPIC_RESULTS = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bertopic_posts"
    / "final_mcs15_ms15_nn15_nc5"
    / "topic_results.csv"
)
DEFAULT_COMMENTS = PROJECT_ROOT / "data" / "processed" / "comments_clean.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "analysis_inputs"
EXCLUDED_TOPICS = {"8", "22", "27", "28"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build post/comment analysis inputs from BERTopic topic labels.")
    parser.add_argument("--topic-results", type=Path, default=DEFAULT_TOPIC_RESULTS)
    parser.add_argument("--comments", type=Path, default=DEFAULT_COMMENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--exclude-topics",
        nargs="*",
        default=sorted(EXCLUDED_TOPICS, key=int),
        help="Topic ids whose posts and comments should be marked include_in_analysis=false.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return reader.fieldnames, list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)
    return len(rows)


def with_include_field(fieldnames: list[str]) -> list[str]:
    output_fieldnames = list(fieldnames)
    if "include_in_analysis" not in output_fieldnames:
        output_fieldnames.append("include_in_analysis")
    return output_fieldnames


def normalize_id(value: str | None) -> str:
    return (value or "").strip()


def build_posts(
    topic_results_path: Path,
    output_path: Path,
    excluded_topics: set[str],
) -> tuple[int, int, int, dict[str, str]]:
    fieldnames, rows = read_csv(topic_results_path)
    if "topic" not in fieldnames:
        raise ValueError(f"topic_results.csv must contain topic column: {topic_results_path}")
    if "aweme_id" not in fieldnames:
        raise ValueError(f"topic_results.csv must contain aweme_id column: {topic_results_path}")

    include_by_aweme_id: dict[str, str] = {}
    true_count = 0
    false_count = 0

    for row in rows:
        topic = normalize_id(row.get("topic"))
        include = topic not in excluded_topics
        include_text = "true" if include else "false"
        row["include_in_analysis"] = include_text
        aweme_id = normalize_id(row.get("aweme_id"))
        if aweme_id:
            include_by_aweme_id[aweme_id] = include_text
        if include:
            true_count += 1
        else:
            false_count += 1

    written = write_csv(output_path, with_include_field(fieldnames), rows)
    return written, true_count, false_count, include_by_aweme_id


def build_comments(
    comments_path: Path,
    output_path: Path,
    include_by_aweme_id: dict[str, str],
) -> tuple[int, int, int, int]:
    fieldnames, rows = read_csv(comments_path)
    if "aweme_id" not in fieldnames:
        raise ValueError(f"comments_clean.csv must contain aweme_id column: {comments_path}")

    true_count = 0
    false_count = 0
    unmatched_count = 0

    for row in rows:
        aweme_id = normalize_id(row.get("aweme_id"))
        include_text = include_by_aweme_id.get(aweme_id)
        if include_text is None:
            include_text = "false"
            unmatched_count += 1
        row["include_in_analysis"] = include_text
        if include_text == "true":
            true_count += 1
        else:
            false_count += 1

    written = write_csv(output_path, with_include_field(fieldnames), rows)
    return written, true_count, false_count, unmatched_count


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    excluded_topics = {str(topic).strip() for topic in args.exclude_topics}

    posts_output = output_dir / "posts_for_analysis_topic_labeled.csv"
    comments_output = output_dir / "comments_for_analysis_topic_labeled.csv"

    post_rows, post_true, post_false, include_by_aweme_id = build_posts(
        topic_results_path=args.topic_results.resolve(),
        output_path=posts_output,
        excluded_topics=excluded_topics,
    )
    comment_rows, comment_true, comment_false, comment_unmatched = build_comments(
        comments_path=args.comments.resolve(),
        output_path=comments_output,
        include_by_aweme_id=include_by_aweme_id,
    )

    print(f"Excluded topics: {', '.join(sorted(excluded_topics, key=lambda value: int(value)))}")
    print(f"Posts output: {posts_output}")
    print(f"Post rows: {post_rows}")
    print(f"Post include_in_analysis=true: {post_true}")
    print(f"Post include_in_analysis=false: {post_false}")
    print(f"Comments output: {comments_output}")
    print(f"Comment rows: {comment_rows}")
    print(f"Comment include_in_analysis=true: {comment_true}")
    print(f"Comment include_in_analysis=false: {comment_false}")
    print(f"Comment rows without matching kept/excluded post aweme_id: {comment_unmatched}")


if __name__ == "__main__":
    main()
