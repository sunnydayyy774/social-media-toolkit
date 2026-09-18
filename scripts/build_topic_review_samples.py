"""Build manual review samples for BERTopic post-cleaning decisions."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = PROJECT_ROOT / "data" / "processed" / "bertopic_posts" / "final_mcs15_ms15_nn15_nc5"

SAMPLE_SIZE = 20
RANDOM_SEED = 20260802

A_TERMS = [
    "修宪",
    "改宪",
    "宪法第九条",
    "第九条",
    "第9条",
    "和平宪法",
    "自卫队入宪",
    "修宪公投",
    "修宪势力",
]

B_TERMS = [
    "再军事化",
    "扩军",
    "自卫队",
    "军事正常化",
    "军国主义",
    "台海",
    "台湾有事",
    "存亡危机事态",
    "高市早苗",
    "岸田",
    "安倍",
    "自民党",
    "公明党",
]

C_TERMS = [
    "旅游",
    "留学",
    "日语",
    "动漫",
    "美食",
    "足球",
    "财经",
    "经济",
    "日元",
    "游戏",
    "音乐",
    "电影",
    "明星",
    "法律科普",
    "正当防卫",
]

SAMPLE_FIELDS = [
    "topic",
    "aweme_id",
    "text",
    "text_clean_bertopic",
    "text_clean_bertopic_with_asr",
    "search_keyword",
    "hashtag_names_csv",
    "topic_probability",
    "suggested_label",
    "manual_label",
    "note",
]

REVIEW_FIELDS = [
    "topic",
    "total_count",
    "sample_count",
    "suggested_A_count",
    "suggested_B_count",
    "suggested_C_count",
    "suggested_blank_count",
    "c_ratio_in_sample",
    "review_decision",
]

FORCE_MANUAL_TOPICS = {-1, 0, 1, 2, 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BERTopic manual review sample CSVs.")
    parser.add_argument("--run-dir", type=Path, default=FINAL_DIR, help="BERTopic run directory.")
    parser.add_argument("--input", type=Path, default=None, help="Input topic_results.csv.")
    parser.add_argument("--sample-output", type=Path, default=None, help="Manual label sample CSV.")
    parser.add_argument("--review-output", type=Path, default=None, help="Topic level review CSV.")
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE, help="Rows sampled per topic.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for reproducible sampling.")
    return parser.parse_args()


def topic_sort_key(topic: str) -> tuple[int, int | str]:
    try:
        return (0, int(topic))
    except ValueError:
        return (1, topic)


def contains_any(text: str, terms: Iterable[str]) -> bool:
    return any(term in text for term in terms)


def suggested_label(row: dict[str, str]) -> str:
    combined = " ".join(
        [
            row.get("text", ""),
            row.get("text_clean_bertopic", ""),
            row.get("text_clean_bertopic_with_asr", ""),
            row.get("search_keyword", ""),
            row.get("hashtag_names_csv", ""),
        ]
    )
    hit_a = contains_any(combined, A_TERMS)
    hit_b = contains_any(combined, B_TERMS)
    hit_c = contains_any(combined, C_TERMS)
    if hit_a:
        return "A"
    if hit_b:
        return "B"
    if hit_c:
        return "C"
    return ""


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV has no header: {path}")
        required = {
            "topic",
            "aweme_id",
            "text",
            "search_keyword",
            "hashtag_names_csv",
            "topic_probability",
        }
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Input CSV missing required fields: {sorted(missing)}")
        return list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)
    return len(rows)


def review_decision(topic: str, c_ratio: float) -> str:
    try:
        topic_value = int(topic)
    except ValueError:
        topic_value = None
    if topic_value in FORCE_MANUAL_TOPICS:
        return "manual_review"
    if c_ratio >= 0.8:
        return "candidate_delete"
    if c_ratio >= 0.4:
        return "manual_review"
    return "candidate_keep"


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    input_csv = args.input or (run_dir / "topic_results.csv")
    sample_output = args.sample_output or (run_dir / "manual_label_sample.csv")
    review_output = args.review_output or (run_dir / "topic_level_review.csv")

    rows = read_rows(input_csv)

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        topic = (row.get("topic") or "").strip()
        if not topic:
            continue
        grouped[topic].append(row)

    rng = random.Random(args.seed)
    sample_rows: list[dict[str, object]] = []
    review_rows: list[dict[str, object]] = []

    for topic in sorted(grouped, key=topic_sort_key):
        topic_rows = grouped[topic]
        sampled = list(topic_rows) if len(topic_rows) <= args.sample_size else rng.sample(topic_rows, args.sample_size)
        sampled.sort(key=lambda row: row.get("aweme_id", ""))

        label_counts = {"A": 0, "B": 0, "C": 0, "": 0}
        for row in sampled:
            label = suggested_label(row)
            label_counts[label] += 1
            sample_rows.append(
                {
                    "topic": topic,
                    "aweme_id": row.get("aweme_id", ""),
                    "text": row.get("text", ""),
                    "text_clean_bertopic": row.get("text_clean_bertopic", ""),
                    "text_clean_bertopic_with_asr": row.get("text_clean_bertopic_with_asr", ""),
                    "search_keyword": row.get("search_keyword", ""),
                    "hashtag_names_csv": row.get("hashtag_names_csv", ""),
                    "topic_probability": row.get("topic_probability", ""),
                    "suggested_label": label,
                    "manual_label": "",
                    "note": "",
                }
            )

        sample_count = len(sampled)
        c_ratio = label_counts["C"] / sample_count if sample_count else 0.0
        review_rows.append(
            {
                "topic": topic,
                "total_count": len(topic_rows),
                "sample_count": sample_count,
                "suggested_A_count": label_counts["A"],
                "suggested_B_count": label_counts["B"],
                "suggested_C_count": label_counts["C"],
                "suggested_blank_count": label_counts[""],
                "c_ratio_in_sample": f"{c_ratio:.4f}",
                "review_decision": review_decision(topic, c_ratio),
            }
        )

    sample_count = write_csv(sample_output, SAMPLE_FIELDS, sample_rows)
    review_count = write_csv(review_output, REVIEW_FIELDS, review_rows)

    print(f"Run dir: {run_dir}")
    print(f"Input rows: {len(rows)}")
    print(f"Topics: {len(grouped)}")
    print(f"Manual label sample rows: {sample_count}")
    print(f"Topic level review rows: {review_count}")
    print(f"Sample output: {sample_output}")
    print(f"Review output: {review_output}")


if __name__ == "__main__":
    main()
