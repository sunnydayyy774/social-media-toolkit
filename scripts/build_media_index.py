"""Build a CSV index for Douyin media files without copying media assets."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEDIA_ROOT = PROJECT_ROOT / "data" / "douyin-media"
WORKING_DIR = PROJECT_ROOT / "data" / "working"
OUTPUT_CSV = WORKING_DIR / "media_index.csv"

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".bmp", ".tiff"}
GIF_EXTENSIONS = {".gif"}

FIELDS = [
    "file_path",
    "file_name",
    "file_type",
    "file_size",
    "related_aweme_id",
    "related_comment_id",
    "has_ocr_text",
    "has_audio_transcript",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create data/working/media_index.csv.")
    parser.add_argument(
        "--media-root",
        type=Path,
        default=MEDIA_ROOT,
        help="Media root to scan. Defaults to data/douyin-media.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_CSV,
        help="Output CSV path. Defaults to data/working/media_index.csv.",
    )
    return parser.parse_args()


def classify_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    if suffix in GIF_EXTENSIONS:
        return "gif"
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    return "other"


def parse_relation(media_root: Path, path: Path) -> tuple[str, str]:
    try:
        parts = path.relative_to(media_root).parts
    except ValueError:
        return "", ""

    if len(parts) >= 2 and parts[0] == "videos":
        return parts[1], ""
    if len(parts) >= 3 and parts[0] == "comments":
        return parts[1], parts[2]
    return "", ""


def iter_media_rows(media_root: Path):
    for path in sorted(media_root.rglob("*")):
        if not path.is_file():
            continue
        aweme_id, comment_id = parse_relation(media_root, path)
        yield {
            "file_path": str(path.relative_to(PROJECT_ROOT)),
            "file_name": path.name,
            "file_type": classify_file(path),
            "file_size": path.stat().st_size,
            "related_aweme_id": aweme_id,
            "related_comment_id": comment_id,
            "has_ocr_text": "false",
            "has_audio_transcript": "false",
        }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    media_root = args.media_root.resolve()
    output = args.output.resolve()
    if not media_root.exists():
        raise FileNotFoundError(f"Media root not found: {media_root}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output.with_suffix(output.suffix + ".tmp")

    count = 0
    type_counts: dict[str, int] = {}
    logging.info("Scanning media root: %s", media_root)
    with temp_output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for row in iter_media_rows(media_root):
            writer.writerow(row)
            count += 1
            type_counts[row["file_type"]] = type_counts.get(row["file_type"], 0) + 1

    temp_output.replace(output)
    logging.info("Media index written: %s", output)
    logging.info("Indexed files: %s", count)
    logging.info("Type counts: %s", type_counts)


if __name__ == "__main__":
    main()
