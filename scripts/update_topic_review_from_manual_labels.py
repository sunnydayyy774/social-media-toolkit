"""Update topic-level review stats from an existing manual label sample."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = PROJECT_ROOT / "data" / "processed" / "bertopic_posts" / "final_mcs15_ms15_nn15_nc5"

FORCE_MANUAL_TOPICS = {-1, 0, 1, 2, 3}

REVIEW_FIELDS = [
    "topic",
    "total_count",
    "sample_count",
    "suggested_A_count",
    "suggested_B_count",
    "suggested_C_count",
    "suggested_blank_count",
    "manual_A_count",
    "manual_B_count",
    "manual_C_count",
    "manual_blank_count",
    "manual_c_ratio_in_sample",
    "manual_c_ratio_in_labeled",
    "review_decision",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update topic_level_review_manual_updated.csv from manual labels.")
    parser.add_argument("--run-dir", type=Path, default=FINAL_DIR, help="BERTopic run directory.")
    parser.add_argument("--topic-results", type=Path, default=None, help="Input topic_results.csv.")
    parser.add_argument("--manual-sample", type=Path, default=None, help="Input manual_label_sample.csv.")
    parser.add_argument("--review-output", type=Path, default=None, help="Output manual-updated topic review CSV.")
    return parser.parse_args()


def topic_sort_key(topic: str) -> tuple[int, int | str]:
    try:
        return (0, int(topic))
    except ValueError:
        return (1, topic)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def normalize_label(value: str) -> str:
    label = (value or "").strip().upper()
    return label if label in {"A", "B", "C"} else ""


def review_decision(topic: str, manual_c_ratio_in_labeled: float, labeled_count: int) -> str:
    try:
        topic_value = int(topic)
    except ValueError:
        topic_value = None

    if topic_value in FORCE_MANUAL_TOPICS:
        return "manual_review"
    if labeled_count == 0:
        return "manual_review"
    if manual_c_ratio_in_labeled >= 0.8:
        return "candidate_delete"
    if manual_c_ratio_in_labeled >= 0.4:
        return "manual_review"
    return "candidate_keep"


def write_csv(path: Path, rows: list[dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=REVIEW_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)
    return len(rows)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    topic_results = args.topic_results or (run_dir / "topic_results.csv")
    manual_sample = args.manual_sample or (run_dir / "manual_label_sample.csv")
    review_output = args.review_output or (run_dir / "topic_level_review_manual_updated.csv")

    topic_rows = read_csv(topic_results)
    sample_rows = read_csv(manual_sample)

    total_counts = Counter((row.get("topic") or "").strip() for row in topic_rows if (row.get("topic") or "").strip())
    sample_by_topic: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in sample_rows:
        topic = (row.get("topic") or "").strip()
        if topic:
            sample_by_topic[topic].append(row)

    review_rows: list[dict[str, object]] = []
    for topic in sorted(total_counts, key=topic_sort_key):
        samples = sample_by_topic.get(topic, [])
        suggested_counts = Counter(normalize_label(row.get("suggested_label", "")) for row in samples)
        manual_counts = Counter(normalize_label(row.get("manual_label", "")) for row in samples)

        sample_count = len(samples)
        manual_c_ratio_in_sample = manual_counts["C"] / sample_count if sample_count else 0.0
        labeled_count = manual_counts["A"] + manual_counts["B"] + manual_counts["C"]
        manual_c_ratio_in_labeled = manual_counts["C"] / labeled_count if labeled_count else 0.0

        review_rows.append(
            {
                "topic": topic,
                "total_count": total_counts[topic],
                "sample_count": sample_count,
                "suggested_A_count": suggested_counts["A"],
                "suggested_B_count": suggested_counts["B"],
                "suggested_C_count": suggested_counts["C"],
                "suggested_blank_count": suggested_counts[""],
                "manual_A_count": manual_counts["A"],
                "manual_B_count": manual_counts["B"],
                "manual_C_count": manual_counts["C"],
                "manual_blank_count": manual_counts[""],
                "manual_c_ratio_in_sample": f"{manual_c_ratio_in_sample:.4f}",
                "manual_c_ratio_in_labeled": f"{manual_c_ratio_in_labeled:.4f}",
                "review_decision": review_decision(topic, manual_c_ratio_in_labeled, labeled_count),
            }
        )

    count = write_csv(review_output, review_rows)
    print(f"Run dir: {run_dir}")
    print(f"Topic results rows: {len(topic_rows)}")
    print(f"Manual sample rows: {len(sample_rows)}")
    print(f"Topic level review rows written: {count}")
    print(f"Review output: {review_output}")


if __name__ == "__main__":
    main()
