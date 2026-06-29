from __future__ import annotations

import asyncio
import hashlib
import mimetypes
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

from loguru import logger

from storage import DuckDBDatabase, Record
from .crawler import DouyinCrawler, DouyinCrawlerConfig, search_aweme_sec_user_id
from .storage import DouyinStore
from .utils import video_url, value_to_str


DEFAULT_MEDIA_TYPES = {
    "video",
    "cover",
    "comment-image",
    "comment-sticker",
    "comment-video",
    "danmaku-sticker",
}

IMAGE_KEYS = {"image", "image_list", "origin_url", "static_url", "url"}
COMMENT_STICKER_KEYS = {"animated_image", "emoji", "sticker"}
COMMENT_VIDEO_KEYS = {"aweme_video", "comment_video", "video", "video_list"}
DANMAKU_STICKER_KEYS = {"danmaku_logos", "emoji", "icon", "image", "sticker"}


@dataclass(frozen=True, slots=True)
class MediaCandidate:
    asset_type: str
    source_url: str
    aweme_id: str | None = None
    comment_id: str | None = None
    danmaku_id: str | None = None
    task_id: str | None = None

    @property
    def id(self) -> str:
        digest = hashlib.sha1(
            "|".join(
                [
                    self.asset_type,
                    self.aweme_id or "",
                    self.comment_id or "",
                    self.danmaku_id or "",
                    self.source_url,
                ]
            ).encode("utf-8")
        ).hexdigest()
        return f"{self.asset_type}:{digest}"


@dataclass(slots=True)
class DownloadResult:
    discovered: int = 0
    downloaded: int = 0
    skipped: int = 0
    failed: int = 0


class DouyinMediaDownloader:
    """Download media files from Douyin records already saved in DuckDB."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        output_dir: str | Path = "data/douyin-media",
        media_types: Iterable[str] | None = None,
        only_aweme_ids: Iterable[str] | None = None,
        from_task_id: str | None = None,
        limit: int | None = None,
        retry_failed: bool = False,
        overwrite: bool = False,
        dry_run: bool = False,
        refresh_video_urls: bool = True,
        headless: bool = False,
        user_data_dir: str | Path | None = None,
        timeout_seconds: int = 30,
    ) -> None:
        self.db = DuckDBDatabase(db_path)
        self.store = DouyinStore(self.db)
        self.output_dir = Path(output_dir)
        self.media_types = set(media_types or DEFAULT_MEDIA_TYPES)
        self.only_aweme_ids = set(only_aweme_ids or [])
        self.from_task_id = from_task_id
        self.limit = limit
        self.retry_failed = retry_failed
        self.overwrite = overwrite
        self.dry_run = dry_run
        self.refresh_video_urls = refresh_video_urls
        self.headless = headless
        self.user_data_dir = Path(user_data_dir) if user_data_dir is not None else Path("data/douyin-browser-profile")
        self.timeout_seconds = timeout_seconds

    def close(self) -> None:
        self.db.close()

    def run(self) -> DownloadResult:
        result = DownloadResult()
        seen: set[str] = set()
        for candidate in self.iter_candidates():
            if candidate.id in seen:
                continue
            seen.add(candidate.id)
            if self.limit is not None and result.discovered >= self.limit:
                break
            result.discovered += 1
            status = self.process_candidate(candidate)
            if status == "DOWNLOADED":
                result.downloaded += 1
            elif status == "SKIPPED":
                result.skipped += 1
            elif status == "FAILED":
                result.failed += 1
        return result

    def iter_candidates(self) -> Iterator[MediaCandidate]:
        if "video" in self.media_types or "cover" in self.media_types:
            yield from self.iter_video_candidates()
        if any(item in self.media_types for item in {"comment-image", "comment-sticker", "comment-video"}):
            yield from self.iter_comment_candidates()
        if "danmaku-sticker" in self.media_types:
            yield from self.iter_danmaku_candidates()

    def iter_video_candidates(self) -> Iterator[MediaCandidate]:
        for record in self.store.list_videos():
            if not self.record_matches(record):
                continue
            aweme_id = value_to_str(record.get("aweme_id") or record.get("id"))
            video_json = record.get("video_json")
            if not aweme_id or not isinstance(video_json, dict):
                continue
            if "video" in self.media_types:
                url = first_url_from_paths(
                    video_json,
                    [
                        ("video", "play_addr"),
                        ("video", "download_addr"),
                        ("video", "play_addr_h264"),
                        ("video", "bit_rate"),
                    ],
                )
                if url:
                    yield MediaCandidate("video", url, aweme_id=aweme_id, task_id=value_to_str(record.get("task_id")))
            if "cover" in self.media_types:
                url = first_url_from_paths(
                    video_json,
                    [
                        ("cover",),
                        ("origin_cover",),
                        ("dynamic_cover",),
                        ("video", "cover"),
                        ("video", "origin_cover"),
                        ("video", "dynamic_cover"),
                    ],
                )
                if url:
                    yield MediaCandidate("cover", url, aweme_id=aweme_id, task_id=value_to_str(record.get("task_id")))

    def iter_comment_candidates(self) -> Iterator[MediaCandidate]:
        for record in self.store.list_comments():
            if not self.record_matches(record):
                continue
            aweme_id = value_to_str(record.get("aweme_id"))
            comment_id = value_to_str(record.get("comment_id") or record.get("id"))
            if not aweme_id or not comment_id:
                continue
            task_id = value_to_str(record.get("task_id"))
            data = record.get("data")
            if "comment-image" in self.media_types:
                if isinstance(data, dict):
                    image_urls = preferred_urls_from_media_list(data.get("image_list"))
                    if not image_urls:
                        image_urls = preferred_urls_from_raw_key(data.get("raw_comment_json"), "image_list")
                    if not image_urls:
                        image_urls = first_url_group(csv_urls(record.get("image_urls_csv")))
                    for url in image_urls:
                        yield MediaCandidate("comment-image", url, aweme_id=aweme_id, comment_id=comment_id, task_id=task_id)
            if "comment-sticker" in self.media_types:
                if isinstance(data, dict):
                    sticker_urls = preferred_urls_from_raw_keys(data.get("raw_comment_json"), COMMENT_STICKER_KEYS)
                    if not sticker_urls:
                        sticker_urls = preferred_urls_from_media_list(data.get("video_list"))
                    if not sticker_urls:
                        sticker_urls = first_url_group(csv_urls(record.get("video_urls_csv")))
                    for url in sticker_urls:
                        yield MediaCandidate("comment-sticker", url, aweme_id=aweme_id, comment_id=comment_id, task_id=task_id)
            if "comment-video" in self.media_types:
                if isinstance(data, dict):
                    comment_video_urls = preferred_urls_from_raw_keys(data.get("raw_comment_json"), COMMENT_VIDEO_KEYS)
                    if not comment_video_urls:
                        comment_video_urls = preferred_urls_from_media_list(data.get("video_list"))
                    for url in comment_video_urls:
                        yield MediaCandidate("comment-video", url, aweme_id=aweme_id, comment_id=comment_id, task_id=task_id)

    def iter_danmaku_candidates(self) -> Iterator[MediaCandidate]:
        for record in self.store.list_danmaku():
            if not self.record_matches(record):
                continue
            aweme_id = value_to_str(record.get("aweme_id"))
            danmaku_id = value_to_str(record.get("danmaku_id") or record.get("id"))
            data = record.get("data")
            if not aweme_id or not danmaku_id or not isinstance(data, dict):
                continue
            for url in preferred_urls_from_raw_keys(data, DANMAKU_STICKER_KEYS):
                yield MediaCandidate(
                    "danmaku-sticker",
                    url,
                    aweme_id=aweme_id,
                    danmaku_id=danmaku_id,
                    task_id=value_to_str(record.get("task_id")),
                )

    def record_matches(self, record: Record) -> bool:
        aweme_id = value_to_str(record.get("aweme_id") or record.get("id"))
        if self.only_aweme_ids and aweme_id not in self.only_aweme_ids:
            return False
        if self.from_task_id is not None and record.get("task_id") != self.from_task_id:
            return False
        return True

    def process_candidate(self, candidate: MediaCandidate) -> str:
        existing = self.store.db.read("douyin_media_assets", candidate.id)
        local_path = self.local_path_for(candidate)
        if existing is not None and existing.get("download_status") == "DONE" and not self.overwrite:
            if not self.retry_failed:
                return "SKIPPED"
            if existing.get("local_path") and Path(str(existing["local_path"])).exists():
                return "SKIPPED"
        if existing is not None and existing.get("download_status") == "FAILED" and not self.retry_failed:
            return "SKIPPED"

        if self.dry_run:
            logger.info("Would download {} -> {}", candidate.source_url, local_path)
            return "SKIPPED"

        self.store.save_media_asset(
            candidate.id,
            asset_type=candidate.asset_type,
            source_url=candidate.source_url,
            aweme_id=candidate.aweme_id,
            comment_id=candidate.comment_id,
            danmaku_id=candidate.danmaku_id,
            local_path=str(local_path),
            download_status="PENDING",
            task_id=candidate.task_id,
        )

        try:
            path, content_type = self.download(candidate.source_url, local_path)
            file_size = path.stat().st_size if path.exists() else None
            self.store.save_media_asset(
                candidate.id,
                asset_type=candidate.asset_type,
                source_url=candidate.source_url,
                aweme_id=candidate.aweme_id,
                comment_id=candidate.comment_id,
                danmaku_id=candidate.danmaku_id,
                local_path=str(path),
                download_status="DONE",
                file_size=file_size,
                content_type=content_type,
                task_id=candidate.task_id,
            )
            logger.info("Downloaded {} -> {}", candidate.asset_type, path)
            return "DOWNLOADED"
        except HTTPError as exc:
            if candidate.asset_type == "video" and exc.code == 403 and self.refresh_video_urls:
                refreshed = self.refresh_video_candidate(candidate)
                if refreshed is not None and refreshed.source_url != candidate.source_url:
                    logger.info("Retrying Douyin video download with refreshed URL for {}", candidate.aweme_id)
                    return self.process_candidate(refreshed)
            self.save_failed_candidate(candidate, local_path, exc)
            return "FAILED"
        except Exception as exc:
            self.save_failed_candidate(candidate, local_path, exc)
            return "FAILED"

    def save_failed_candidate(self, candidate: MediaCandidate, local_path: Path, exc: Exception) -> None:
        self.store.save_media_asset(
            candidate.id,
            asset_type=candidate.asset_type,
            source_url=candidate.source_url,
            aweme_id=candidate.aweme_id,
            comment_id=candidate.comment_id,
            danmaku_id=candidate.danmaku_id,
            local_path=str(local_path),
            download_status="FAILED",
            error=str(exc)[:1000],
            task_id=candidate.task_id,
        )
        logger.warning("Failed downloading {}: {}", candidate.source_url, exc)

    def refresh_video_candidate(self, candidate: MediaCandidate) -> MediaCandidate | None:
        if not candidate.aweme_id:
            return None
        try:
            return asyncio.run(self.refresh_video_candidate_async(candidate))
        except Exception as exc:
            logger.warning("Failed refreshing Douyin video URL for {}: {}", candidate.aweme_id, exc)
            return None

    async def refresh_video_candidate_async(self, candidate: MediaCandidate) -> MediaCandidate | None:
        aweme_id = candidate.aweme_id
        if aweme_id is None:
            return None
        async with DouyinCrawler(
            DouyinCrawlerConfig(
                db_path=self.db.path,
                headless=self.headless,
                user_data_dir=self.user_data_dir,
            )
        ) as crawler:
            page = await crawler.new_page()
            await page.goto(video_url(aweme_id), wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(5000)
            detail = await crawler._fetch_aweme_detail(page, aweme_id)
            aweme = detail.get("aweme_detail") if isinstance(detail, dict) else None
            if not isinstance(aweme, dict):
                return None
            crawler.store.save_video_raw(
                aweme_id,
                search_aweme_sec_user_id(aweme) or f"aweme:{aweme_id}",
                aweme,
                task_id=candidate.task_id,
            )
            url = first_url_from_paths(
                aweme,
                [
                    ("video", "play_addr"),
                    ("video", "download_addr"),
                    ("video", "play_addr_h264"),
                    ("video", "bit_rate"),
                ],
            )
            if not url:
                return None
            return MediaCandidate(
                "video",
                url,
                aweme_id=aweme_id,
                comment_id=candidate.comment_id,
                danmaku_id=candidate.danmaku_id,
                task_id=candidate.task_id,
            )

    def local_path_for(self, candidate: MediaCandidate) -> Path:
        parsed = urlparse(candidate.source_url)
        ext = extension_from_url(parsed.path)
        digest = hashlib.sha1(candidate.source_url.encode("utf-8")).hexdigest()[:12]
        if candidate.asset_type in {"video", "cover"}:
            folder = self.output_dir / "videos" / safe_part(candidate.aweme_id or "unknown")
            stem = candidate.asset_type
        elif candidate.asset_type.startswith("comment-"):
            folder = (
                self.output_dir
                / "comments"
                / safe_part(candidate.aweme_id or "unknown")
                / safe_part(candidate.comment_id or "unknown")
            )
            stem = candidate.asset_type
        else:
            folder = (
                self.output_dir
                / "danmaku"
                / safe_part(candidate.aweme_id or "unknown")
                / safe_part(candidate.danmaku_id or "unknown")
            )
            stem = candidate.asset_type
        return folder / f"{stem}-{digest}{ext}"

    def download(self, url: str, path: Path) -> tuple[Path, str | None]:
        path.parent.mkdir(parents=True, exist_ok=True)
        request = Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
                ),
                "Referer": "https://www.douyin.com/",
                "Accept": "*/*",
            },
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            content_type = response.headers.get("Content-Type")
            target = path_with_content_type(path, content_type)
            if target.exists() and not self.overwrite:
                return target, content_type
            with target.open("wb") as file:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    file.write(chunk)
            return target, content_type


def best_urls_from_paths(value: dict[str, Any], paths: list[tuple[str, ...]]) -> list[str]:
    urls: list[str] = []
    for path in paths:
        node: Any = value
        for part in path:
            if isinstance(node, dict):
                node = node.get(part)
            else:
                node = None
                break
        urls.extend(collect_url_values(node))
    return dedupe_urls(urls)


def first_url_from_paths(value: dict[str, Any], paths: list[tuple[str, ...]]) -> str | None:
    for url in best_urls_from_paths(value, paths):
        return url
    return None


def collect_urls_under_keys(value: Any, key_names: set[str]) -> list[str]:
    urls: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in key_names:
                urls.extend(collect_url_values(child))
            elif isinstance(child, dict | list):
                urls.extend(collect_urls_under_keys(child, key_names))
    elif isinstance(value, list):
        for child in value:
            urls.extend(collect_urls_under_keys(child, key_names))
    return dedupe_urls(urls)


def preferred_urls_from_media_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    urls: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        url = preferred_url_for_media_item(item)
        if url:
            urls.append(url)
    return dedupe_urls(urls)


def preferred_urls_from_raw_key(value: Any, key_name: str) -> list[str]:
    urls: list[str] = []
    for item in iter_values_for_key(value, key_name):
        if isinstance(item, list):
            for child in item:
                if isinstance(child, dict):
                    url = preferred_url_for_media_item(child)
                    if url:
                        urls.append(url)
                elif isinstance(child, str) and is_http_url(child):
                    urls.append(child)
        elif isinstance(item, dict):
            url = preferred_url_for_media_item(item)
            if url:
                urls.append(url)
        elif isinstance(item, str) and is_http_url(item):
            urls.append(item)
    return dedupe_urls(urls)


def preferred_urls_from_raw_keys(value: Any, key_names: set[str]) -> list[str]:
    urls: list[str] = []
    for key_name in key_names:
        urls.extend(preferred_urls_from_raw_key(value, key_name))
    return dedupe_urls(urls)


def preferred_url_for_media_item(item: dict[str, Any]) -> str | None:
    candidates = collect_url_values(item)
    if not candidates:
        return None
    return sorted(candidates, key=url_preference_score)[0]


def first_url_group(urls: list[str]) -> list[str]:
    return urls[:1]


def iter_values_for_key(value: Any, key_name: str) -> Iterator[Any]:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == key_name:
                yield child
            if isinstance(child, dict | list):
                yield from iter_values_for_key(child, key_name)
    elif isinstance(value, list):
        for child in value:
            yield from iter_values_for_key(child, key_name)


def url_preference_score(url: str) -> tuple[int, int, int, str]:
    parsed = urlparse(url)
    text = url.lower()
    query = parse_qs(parsed.query)
    path = parsed.path.lower()
    score = 0
    if "thumb" in text or query.get("sc") == ["thumb"]:
        score += 40
    if "watermark" in text or query.get("sc") == ["watermark"]:
        score += 30
    if ".heic" in path:
        score += 20
    if ".jpeg" in path or ".jpg" in path:
        score += 5
    if ".webp" in path or ".image" in path:
        score -= 5
    if "origin" in text or query.get("sc") == ["image"]:
        score -= 10
    return (score, len(url), stable_host_rank(parsed.netloc), url)


def stable_host_rank(host: str) -> int:
    if host.startswith("p3-"):
        return 0
    if host.startswith("p11-"):
        return 1
    if host.startswith("p26-"):
        return 2
    return 3


def collect_url_values(value: Any) -> list[str]:
    urls: list[str] = []
    if isinstance(value, str):
        if is_http_url(value):
            urls.append(value)
    elif isinstance(value, dict):
        for key, child in value.items():
            if key in {"url", "uri", "src", "download_url", "main_url"}:
                urls.extend(collect_url_values(child))
            elif key in {"url_list", "url_lists", "urls", "download_url_list"}:
                urls.extend(collect_url_values(child))
            elif isinstance(child, dict | list):
                urls.extend(collect_url_values(child))
    elif isinstance(value, list):
        for child in value:
            urls.extend(collect_url_values(child))
    return dedupe_urls(urls)


def is_http_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def csv_urls(value: Any) -> list[str]:
    if not isinstance(value, str) or not value:
        return []
    return dedupe_urls(part.strip() for part in value.split(",") if part.strip())


def dedupe_urls(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    urls: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        urls.append(value)
    return urls


def extension_from_url(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext and len(ext) <= 8:
        return ext
    return ".bin"


def path_with_content_type(path: Path, content_type: str | None) -> Path:
    if path.suffix != ".bin":
        return path
    if not content_type:
        return path
    media_type = content_type.split(";", 1)[0].strip().lower()
    ext = mimetypes.guess_extension(media_type)
    if ext:
        return path.with_suffix(ext)
    return path


def safe_part(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
    return safe[:120] or "unknown"
