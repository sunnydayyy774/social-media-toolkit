"""Transcribe Douyin video audio with local faster-whisper.

The script reads:
- data/working/media_index.csv
- data/working/douyin_work.duckdb by default, or data/processed/posts_clean.csv

It writes:
- data/processed/media_transcripts.csv

Temporary audio is extracted to data/tmp/asr_audio/ and deleted after a
successful transcription by default.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEDIA_INDEX_CSV = PROJECT_ROOT / "data" / "working" / "media_index.csv"
WORKING_DB = PROJECT_ROOT / "data" / "working" / "douyin_work.duckdb"
POSTS_CLEAN_CSV = PROJECT_ROOT / "data" / "processed" / "posts_clean.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "media_transcripts.csv"
TMP_AUDIO_DIR = PROJECT_ROOT / "data" / "tmp" / "asr_audio"
DEFAULT_MODEL_SIZE = "large-v3"
DURATION_FIELD_CANDIDATES = [
    "duration_ms",
    "video_duration_ms",
    "duration",
    "video_duration",
    "duration_seconds",
    "duration_sec",
]

FIELDNAMES = [
    "aweme_id",
    "file_path",
    "file_type",
    "audio_path",
    "duration_seconds",
    "transcript_text",
    "transcript_clean",
    "transcript_char_count",
    "asr_model",
    "language",
    "transcript_status",
    "error",
    "updated_at",
    "note",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local faster-whisper ASR on Douyin video media.")
    parser.add_argument("--media-index", type=Path, default=MEDIA_INDEX_CSV)
    parser.add_argument("--working-db", type=Path, default=WORKING_DB)
    parser.add_argument("--posts-clean", type=Path, default=POSTS_CLEAN_CSV)
    parser.add_argument(
        "--align-source",
        choices=["raw-db", "posts-clean"],
        default="raw-db",
        help="Choose which post universe video aweme_id must align to.",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--tmp-audio-dir", type=Path, default=TMP_AUDIO_DIR)
    parser.add_argument("--model-size", default=DEFAULT_MODEL_SIZE, help="faster-whisper model size/name.")
    parser.add_argument("--device", default="auto", help='Inference device, e.g. "cuda", "cpu", or "auto".')
    parser.add_argument(
        "--compute-type",
        default="default",
        help='CTranslate2 compute type, e.g. "float16" on GPU or "int8" on CPU.',
    )
    parser.add_argument("--model-cache-dir", type=Path, default=None, help="Optional local model cache directory.")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not download model files; use only an existing local cache.",
    )
    parser.add_argument("--language", default="zh", help="Audio language hint.")
    parser.add_argument("--batch-size", type=int, default=10, help="Progress log interval.")
    parser.add_argument("--max-items", type=int, default=None, help="Maximum videos to process in this run.")
    parser.add_argument("--max-duration-seconds", type=float, default=None)
    parser.add_argument("--min-duration-seconds", type=float, default=None)
    parser.add_argument("--keep-audio", action="store_true", help="Keep all extracted temporary audio files.")
    parser.add_argument(
        "--keep-failed-audio",
        action="store_true",
        help="Keep audio files only when transcription fails.",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg executable path/name.")
    parser.add_argument("--ffprobe", default="ffprobe", help="ffprobe executable path/name.")
    parser.add_argument("--no-vad-filter", action="store_true", help="Disable faster-whisper VAD filtering.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only check inputs and report eligible video count; do not extract or transcribe.",
    )
    return parser.parse_args()


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def resolve_project_path(path_text: str) -> Path:
    raw = Path(path_text)
    if raw.is_absolute():
        return raw
    return PROJECT_ROOT / path_text.replace("\\", os.sep)


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def clean_transcript(text: str) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def require_executable(name_or_path: str, label: str) -> None:
    candidate = Path(name_or_path)
    if candidate.exists():
        return
    if shutil.which(name_or_path):
        return
    raise RuntimeError(f"Missing {label}: {name_or_path}. Please install it or pass --{label} PATH.")


def check_runtime(args: argparse.Namespace) -> None:
    if not args.media_index.exists():
        raise FileNotFoundError(f"media_index.csv not found: {args.media_index}")
    if args.align_source == "posts-clean" and not args.posts_clean.exists():
        raise FileNotFoundError(f"posts_clean.csv not found: {args.posts_clean}")
    if args.align_source == "raw-db" and not args.working_db.exists():
        raise FileNotFoundError(f"working database not found: {args.working_db}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0.")
    if args.max_items is not None and args.max_items <= 0:
        raise ValueError("--max-items must be greater than 0.")
    if args.min_duration_seconds is not None and args.min_duration_seconds < 0:
        raise ValueError("--min-duration-seconds cannot be negative.")
    if args.max_duration_seconds is not None and args.max_duration_seconds < 0:
        raise ValueError("--max-duration-seconds cannot be negative.")
    if (
        args.min_duration_seconds is not None
        and args.max_duration_seconds is not None
        and args.min_duration_seconds > args.max_duration_seconds
    ):
        raise ValueError("--min-duration-seconds cannot be greater than --max-duration-seconds.")

    if args.dry_run:
        if args.min_duration_seconds is not None or args.max_duration_seconds is not None:
            require_executable(args.ffprobe, "ffprobe")
        return

    try:
        import faster_whisper  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Missing Python dependency: faster-whisper.") from exc

    require_executable(args.ffmpeg, "ffmpeg")
    if args.min_duration_seconds is not None or args.max_duration_seconds is not None:
        require_executable(args.ffprobe, "ffprobe")


def load_clean_post_aweme_ids(path: Path) -> set[str]:
    aweme_ids: set[str] = set()
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or "aweme_id" not in reader.fieldnames:
            raise ValueError("posts_clean.csv must contain aweme_id.")
        for row in reader:
            aweme_id = (row.get("aweme_id") or "").strip()
            if aweme_id:
                aweme_ids.add(aweme_id)
    return aweme_ids


def load_raw_post_aweme_ids(db_path: Path) -> set[str]:
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("Missing Python dependency: duckdb.") from exc

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute(
            """
            SELECT DISTINCT
                COALESCE(
                    json_extract_string(data, '$.aweme_id'),
                    json_extract_string(data, '$.aweme_id_str'),
                    id,
                    post_id
                ) AS aweme_id
            FROM douyin_posts
            """
        ).fetchall()
    finally:
        con.close()

    return {str(row[0]).strip() for row in rows if row[0] is not None and str(row[0]).strip()}


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def nested_value(obj: dict[str, Any], path: str) -> Any:
    value: Any = obj
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def duration_to_seconds(field_name: str, value: float) -> float:
    normalized = field_name.lower()
    if normalized.endswith("_ms") or normalized == "duration_ms" or value > 10000:
        return value / 1000
    return value


def detect_duration_field(samples: list[dict[str, Any]]) -> str | None:
    candidate_paths = DURATION_FIELD_CANDIDATES + [f"video.{name}" for name in DURATION_FIELD_CANDIDATES]
    best_field = None
    best_count = 0
    for field_name in candidate_paths:
        count = sum(1 for obj in samples if as_float(nested_value(obj, field_name)) is not None)
        if count > best_count:
            best_field = field_name
            best_count = count
    return best_field


def load_db_duration_seconds(db_path: Path) -> tuple[dict[str, float], str | None]:
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("Missing Python dependency: duckdb.") from exc

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute("SELECT id, post_id, data FROM douyin_posts").fetchall()
    finally:
        con.close()

    parsed_rows: list[tuple[str, str, dict[str, Any]]] = []
    samples: list[dict[str, Any]] = []
    for row_id, post_id, raw_data in rows:
        try:
            obj = __import__("json").loads(raw_data)
        except Exception:
            continue
        aweme_id = str(obj.get("aweme_id") or obj.get("aweme_id_str") or row_id or post_id).strip()
        if aweme_id:
            parsed_rows.append((aweme_id, str(post_id or "").strip(), obj))
            samples.append(obj)

    duration_field = detect_duration_field(samples)
    if not duration_field:
        return {}, None

    durations: dict[str, float] = {}
    for aweme_id, post_id, obj in parsed_rows:
        raw_value = as_float(nested_value(obj, duration_field))
        if raw_value is None:
            continue
        duration_seconds = duration_to_seconds(duration_field, raw_value)
        durations[aweme_id] = duration_seconds
        if post_id:
            durations[post_id] = duration_seconds

    return durations, duration_field


def load_video_rows(media_index_path: Path, post_aweme_ids: set[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with media_index_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"file_path", "file_type", "related_aweme_id"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"media_index.csv missing required columns: {', '.join(sorted(missing))}")

        for row in reader:
            aweme_id = (row.get("related_aweme_id") or "").strip()
            file_type = (row.get("file_type") or "").strip().lower()
            if file_type == "video" and aweme_id in post_aweme_ids:
                rows.append(row)
    return rows


def load_completed_keys(output_path: Path) -> set[tuple[str, str]]:
    if not output_path.exists():
        return set()
    keys: set[tuple[str, str]] = set()
    completed_statuses = {"success", "skipped_duration_filter"}
    with output_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if (row.get("transcript_status") or "").strip().lower() in completed_statuses:
                aweme_id = (row.get("aweme_id") or "").strip()
                file_path = (row.get("file_path") or "").strip()
                if aweme_id and file_path:
                    keys.add((aweme_id, file_path))
    return keys


def append_result(output_path: Path, row: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exists = output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        file.flush()


def temp_audio_path(tmp_dir: Path, aweme_id: str, file_path: str) -> Path:
    digest = hashlib.sha1(file_path.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return tmp_dir / f"{aweme_id}_{digest}.wav"


def get_duration_seconds(ffprobe: str, video_path: Path) -> str:
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return f"{float(result.stdout.strip()):.3f}"


def duration_filter_reason(args: argparse.Namespace, duration_seconds: float) -> str:
    reasons: list[str] = []
    if args.min_duration_seconds is not None and duration_seconds < args.min_duration_seconds:
        reasons.append(f"duration_seconds={duration_seconds:.3f} < min_duration_seconds={args.min_duration_seconds}")
    if args.max_duration_seconds is not None and duration_seconds > args.max_duration_seconds:
        reasons.append(f"duration_seconds={duration_seconds:.3f} > max_duration_seconds={args.max_duration_seconds}")
    return "; ".join(reasons)


def get_row_duration_seconds(
    args: argparse.Namespace,
    row: dict[str, str],
    db_durations: dict[str, float],
    allow_ffprobe: bool,
) -> float | None:
    aweme_id = (row.get("related_aweme_id") or "").strip()
    if aweme_id in db_durations:
        return db_durations[aweme_id]
    if not allow_ffprobe:
        return None

    file_path_text = (row.get("file_path") or "").strip()
    video_path = resolve_project_path(file_path_text)
    if not video_path.exists():
        return None
    if not (Path(args.ffprobe).exists() or shutil.which(args.ffprobe)):
        return None
    try:
        return float(get_duration_seconds(args.ffprobe, video_path))
    except Exception as exc:
        logging.warning("ffprobe duration fallback failed for aweme_id=%s: %s", aweme_id, exc)
        return None


def summarize_duration_filter(
    args: argparse.Namespace,
    rows: list[dict[str, str]],
    db_durations: dict[str, float],
    allow_ffprobe: bool,
) -> tuple[int, int, int]:
    duration_available_count = 0
    skipped_duration_count = 0
    over_30min_count = 0
    for row in rows:
        duration_seconds = get_row_duration_seconds(args, row, db_durations, allow_ffprobe)
        if duration_seconds is None:
            continue
        duration_available_count += 1
        if duration_seconds > 1800:
            over_30min_count += 1
        if duration_filter_reason(args, duration_seconds):
            skipped_duration_count += 1
    return duration_available_count, skipped_duration_count, over_30min_count


def extract_audio(ffmpeg: str, video_path: Path, audio_path: Path) -> None:
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        str(audio_path),
    ]
    subprocess.run(command, capture_output=True, text=True, check=True)


def load_whisper_model(args: argparse.Namespace) -> Any:
    from faster_whisper import WhisperModel

    download_root = str(args.model_cache_dir.resolve()) if args.model_cache_dir else None
    logging.info(
        "Loading faster-whisper model: model_size=%s, device=%s, compute_type=%s",
        args.model_size,
        args.device,
        args.compute_type,
    )
    return WhisperModel(
        args.model_size,
        device=args.device,
        compute_type=args.compute_type,
        download_root=download_root,
        local_files_only=args.local_files_only,
    )


def transcribe_audio(whisper_model: Any, audio_path: Path, language: str, vad_filter: bool) -> str:
    segments, _info = whisper_model.transcribe(
        str(audio_path),
        language=language,
        vad_filter=vad_filter,
    )
    return "".join(segment.text for segment in segments).strip()


def delete_audio_if_needed(audio_path: Path, should_delete: bool) -> None:
    if should_delete and audio_path.exists():
        audio_path.unlink()


def process_one(
    args: argparse.Namespace,
    row: dict[str, str],
    whisper_model: Any,
    db_durations: dict[str, float],
) -> dict[str, Any]:
    aweme_id = (row.get("related_aweme_id") or "").strip()
    file_path_text = (row.get("file_path") or "").strip()
    video_path = resolve_project_path(file_path_text)
    audio_path = temp_audio_path(args.tmp_audio_dir, aweme_id, file_path_text)
    duration_seconds = ""

    base_result: dict[str, Any] = {
        "aweme_id": aweme_id,
        "file_path": file_path_text,
        "file_type": "video",
        "audio_path": "",
        "duration_seconds": "",
        "transcript_text": "",
        "transcript_clean": "",
        "transcript_char_count": 0,
        "asr_model": f"faster-whisper-{args.model_size}",
        "language": args.language,
        "transcript_status": "failed",
        "error": "",
        "updated_at": now_text(),
        "note": "",
    }

    try:
        if not video_path.exists():
            raise FileNotFoundError(f"video file not found: {video_path}")
        duration_value = get_row_duration_seconds(args, row, db_durations, allow_ffprobe=True)
        duration_seconds = f"{duration_value:.3f}" if duration_value is not None else ""
        base_result["duration_seconds"] = duration_seconds
        filter_reason = duration_filter_reason(args, duration_value) if duration_value is not None else ""
        if filter_reason:
            base_result.update(
                {
                    "audio_path": "",
                    "transcript_status": "skipped_duration_filter",
                    "error": "",
                    "note": filter_reason,
                }
            )
            return base_result

        extract_audio(args.ffmpeg, video_path, audio_path)
        transcript_text = transcribe_audio(
            whisper_model=whisper_model,
            audio_path=audio_path,
            language=args.language,
            vad_filter=not args.no_vad_filter,
        )
        transcript_clean = clean_transcript(transcript_text)

        if args.keep_audio:
            stored_audio_path = display_path(audio_path)
            note = "temporary audio retained by --keep-audio"
        else:
            delete_audio_if_needed(audio_path, should_delete=True)
            stored_audio_path = "deleted_after_success"
            note = "temporary audio deleted after successful transcription"

        base_result.update(
            {
                "audio_path": stored_audio_path,
                "transcript_text": transcript_text,
                "transcript_clean": transcript_clean,
                "transcript_char_count": len(transcript_clean),
                "transcript_status": "success",
                "error": "",
                "note": note,
            }
        )
    except Exception as exc:
        keep_failed = args.keep_audio or args.keep_failed_audio
        delete_audio_if_needed(audio_path, should_delete=not keep_failed)
        if audio_path.exists():
            stored_audio_path = display_path(audio_path)
            note = "temporary audio retained for failed transcription"
        else:
            stored_audio_path = "deleted_after_failure" if audio_path.name else ""
            note = "temporary audio deleted after failed transcription"
        base_result.update(
            {
                "audio_path": stored_audio_path,
                "duration_seconds": duration_seconds,
                "transcript_status": "failed",
                "error": str(exc),
                "note": note,
            }
        )

    return base_result


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args.media_index = args.media_index.resolve()
    args.working_db = args.working_db.resolve()
    args.posts_clean = args.posts_clean.resolve()
    args.output = args.output.resolve()
    args.tmp_audio_dir = args.tmp_audio_dir.resolve()

    check_runtime(args)

    if args.align_source == "raw-db":
        post_aweme_ids = load_raw_post_aweme_ids(args.working_db)
        post_source_label = "working douyin_posts"
    else:
        post_aweme_ids = load_clean_post_aweme_ids(args.posts_clean)
        post_source_label = "posts_clean.csv"

    db_durations, duration_field = load_db_duration_seconds(args.working_db)
    if duration_field:
        logging.info("Detected database duration field: douyin_posts.data.%s", duration_field)
    else:
        logging.info("No database duration field detected; duration checks will fallback to ffprobe when needed.")

    video_rows = load_video_rows(args.media_index, post_aweme_ids)
    completed_keys = load_completed_keys(args.output)
    pending_rows = [
        row
        for row in video_rows
        if ((row.get("related_aweme_id") or "").strip(), (row.get("file_path") or "").strip()) not in completed_keys
    ]
    if args.max_items is not None:
        pending_rows = pending_rows[: args.max_items]

    logging.info("Post universe source: %s", post_source_label)
    logging.info("Posts in selected universe: %s", len(post_aweme_ids))
    logging.info("Video rows aligned to selected universe: %s", len(video_rows))
    logging.info("Already completed aweme_id + file_path rows skipped: %s", len(completed_keys))
    logging.info("Input videos to process in this run: %s", len(pending_rows))
    logging.info("Output CSV: %s", args.output)
    logging.info("Temporary audio directory: %s", args.tmp_audio_dir)

    allow_ffprobe_for_report = not duration_field and shutil.which(args.ffprobe) is not None
    duration_available_count, skipped_duration_count, over_30min_count = summarize_duration_filter(
        args,
        pending_rows,
        db_durations,
        allow_ffprobe=allow_ffprobe_for_report,
    )
    logging.info("Total video rows before success skip: %s", len(video_rows))
    logging.info("Transcribable pending video rows: %s", len(pending_rows))
    logging.info("Duration-filtered transcribable rows: %s", len(pending_rows) - skipped_duration_count)
    logging.info("Duration available rows: %s", duration_available_count)
    logging.info("Skipped by duration filter in this run: %s", skipped_duration_count)
    logging.info(">30min video rows: %s", over_30min_count)

    if args.dry_run:
        logging.info("Dry run complete. No audio extraction or transcription was performed.")
        return

    whisper_model = load_whisper_model(args)

    processed = 0
    for row in pending_rows:
        processed += 1
        aweme_id = (row.get("related_aweme_id") or "").strip()
        logging.info("Processing %s/%s aweme_id=%s", processed, len(pending_rows), aweme_id)
        result = process_one(args, row, whisper_model, db_durations)
        append_result(args.output, result)
        if processed % args.batch_size == 0:
            logging.info("Checkpoint: %s rows processed and saved.", processed)

    logging.info("ASR run complete. Processed rows: %s", processed)
    logging.info("Results saved to: %s", args.output)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
