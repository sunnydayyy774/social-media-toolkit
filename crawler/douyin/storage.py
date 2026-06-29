from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from storage import DuckDBDatabase, DuplicateRecordError, Record

from .utils import csv_from_array_key, value_to_bool, value_to_int, value_to_str


AUTHOR_COLLECTION = "douyin_authors"
VIDEO_RAW_COLLECTION = "douyin_videos_raw"
COMMENT_RAW_COLLECTION = "douyin_comments_raw"
DANMAKU_RAW_COLLECTION = "douyin_danmaku_raw"
MEDIA_ASSET_COLLECTION = "douyin_media_assets"

COMMENTS_FAILED_PANEL = "COMMENTS_FAILED_PANEL"
COMMENTS_PARTIAL = "COMMENTS_PARTIAL"
DANMAKU_DONE = "DANMAKU_DONE"
DANMAKU_PARTIAL = "DANMAKU_PARTIAL"
DANMAKU_ERROR = "DANMAKU_ERROR"
DANMAKU_UNAVAILABLE = "DANMAKU_UNAVAILABLE"
DANMAKU_RETRIEVED = "DANMAKU_RETRIEVED"


class DouyinVideoStatus(StrEnum):
    ID_ONLY = "ID_ONLY"
    RETRIEVED = "RETRIEVED"
    COMMENTS_DONE = "COMMENTS_DONE"
    ERROR = "ERROR"
    COMMENTS_FAILED_PANEL = COMMENTS_FAILED_PANEL
    COMMENTS_PARTIAL = COMMENTS_PARTIAL


class DouyinStore:
    """Douyin storage adapter backed by DuckDB."""

    def __init__(self, db: DuckDBDatabase) -> None:
        self.db = db

    def ensure_author(self, sec_user_id: str) -> Record:
        existing = self.db.read(AUTHOR_COLLECTION, sec_user_id)
        if existing is not None:
            return existing
        return self.db.create(
            AUTHOR_COLLECTION,
            {
                "id": sec_user_id,
                "sec_user_id": sec_user_id,
                "updated_at": now_iso(),
            },
            sec_user_id,
        )

    def save_author_profile(self, sec_user_id: str, profile: dict[str, Any]) -> Record:
        projected = project_author(profile)
        record: Record = {
            "id": sec_user_id,
            "sec_user_id": sec_user_id,
            "profile_json": profile,
            "updated_at": now_iso(),
        }
        record.update(projected)

        existing = self.db.read(AUTHOR_COLLECTION, sec_user_id)
        if existing is None:
            return self.db.create(AUTHOR_COLLECTION, record, sec_user_id)
        merged = existing.copy()
        merged.update(record)
        return self.db.replace(AUTHOR_COLLECTION, sec_user_id, merged)

    def save_video_raw(
        self,
        aweme_id: str,
        sec_user_id: str,
        video_json: dict[str, Any] | None,
        *,
        task_id: str | None = None,
        search_keyword: str | None = None,
        search_page: int | None = None,
        search_position: int | None = None,
        source: str | None = None,
    ) -> Record:
        self.ensure_author(sec_user_id)
        existing = self.db.read(VIDEO_RAW_COLLECTION, aweme_id)
        status = DouyinVideoStatus.RETRIEVED.value if video_json is not None else DouyinVideoStatus.ID_ONLY.value
        record: Record = {
            "id": aweme_id,
            "aweme_id": aweme_id,
            "sec_user_id": sec_user_id,
            "video_json": video_json,
            "updated_at": now_iso(),
            "status": status,
            "comment_cursor": 0,
            "danmaku_cursor_ms": 0,
            "task_id": task_id,
        }
        if search_keyword is not None:
            record["search_keyword"] = search_keyword
        if search_page is not None:
            record["search_page"] = search_page
        if search_position is not None:
            record["search_position"] = search_position
        if source is not None:
            record["source"] = source
        if video_json is not None:
            record.update(project_video(video_json))

        if existing is None:
            try:
                return self.db.create(VIDEO_RAW_COLLECTION, record, aweme_id)
            except DuplicateRecordError:
                existing = self.db.read(VIDEO_RAW_COLLECTION, aweme_id)

        merged = existing.copy() if existing is not None else {}
        if merged.get("status") == DouyinVideoStatus.COMMENTS_DONE.value and video_json is not None:
            status = DouyinVideoStatus.COMMENTS_DONE.value
        merged.update(record)
        merged["status"] = status
        if existing is not None and existing.get("comment_cursor") and not record.get("comment_cursor"):
            merged["comment_cursor"] = existing["comment_cursor"]
        if existing is not None and existing.get("danmaku_cursor_ms") and not record.get("danmaku_cursor_ms"):
            merged["danmaku_cursor_ms"] = existing["danmaku_cursor_ms"]
        if existing is not None and existing.get("danmaku_status"):
            merged["danmaku_status"] = existing["danmaku_status"]
        if existing is not None and task_id is None and existing.get("task_id"):
            merged["task_id"] = existing["task_id"]
        return self.db.replace(VIDEO_RAW_COLLECTION, aweme_id, merged)

    def save_danmaku(
        self,
        aweme_id: str,
        danmaku: dict[str, Any],
        *,
        task_id: str | None = None,
        search_keyword: str | None = None,
    ) -> bool:
        danmaku_id = value_to_str(danmaku.get("danmaku_id")) or stable_danmaku_id(aweme_id, danmaku)
        record: Record = {
            "id": danmaku_id,
            "danmaku_id": danmaku_id,
            "aweme_id": aweme_id,
            "data": danmaku,
            "updated_at": now_iso(),
        }
        if task_id is not None:
            record["task_id"] = task_id
        if search_keyword is not None:
            record["search_keyword"] = search_keyword
        record.update(project_danmaku(danmaku))

        existing = self.db.read(DANMAKU_RAW_COLLECTION, danmaku_id)
        if existing is None:
            self.db.create(DANMAKU_RAW_COLLECTION, record, danmaku_id)
        else:
            merged = existing.copy()
            merged.update(record)
            self.db.replace(DANMAKU_RAW_COLLECTION, danmaku_id, merged)
        return True

    def save_media_asset(
        self,
        asset_id: str,
        *,
        asset_type: str,
        source_url: str,
        aweme_id: str | None = None,
        comment_id: str | None = None,
        danmaku_id: str | None = None,
        local_path: str | None = None,
        download_status: str = "PENDING",
        file_size: int | None = None,
        content_type: str | None = None,
        error: str | None = None,
        task_id: str | None = None,
    ) -> Record:
        record: Record = {
            "id": asset_id,
            "asset_id": asset_id,
            "asset_type": asset_type,
            "source_url": source_url,
            "url": source_url,
            "download_status": download_status,
            "status": download_status,
            "updated_at": now_iso(),
        }
        optional = {
            "aweme_id": aweme_id,
            "comment_id": comment_id,
            "danmaku_id": danmaku_id,
            "local_path": local_path,
            "file_size": file_size,
            "content_type": content_type,
            "error": error,
            "task_id": task_id,
        }
        record.update({key: value for key, value in optional.items() if value is not None})

        existing = self.db.read(MEDIA_ASSET_COLLECTION, asset_id)
        if existing is None:
            return self.db.create(MEDIA_ASSET_COLLECTION, record, asset_id)
        merged = existing.copy()
        merged.update(record)
        return self.db.replace(MEDIA_ASSET_COLLECTION, asset_id, merged)

    def save_comment_raw(
        self,
        comment_id: str,
        aweme_id: str,
        parent_comment_id: str | None,
        data: dict[str, Any],
        *,
        task_id: str | None = None,
    ) -> Record:
        projected = project_comment(data)
        record: Record = {
            "id": comment_id,
            "comment_id": comment_id,
            "aweme_id": aweme_id,
            "parent_comment_id": parent_comment_id,
            "data": data,
            "updated_at": now_iso(),
        }
        if task_id is not None:
            record["task_id"] = task_id
        record.update(projected)

        existing = self.db.read(COMMENT_RAW_COLLECTION, comment_id)
        if existing is None:
            return self.db.create(COMMENT_RAW_COLLECTION, record, comment_id)
        merged = existing.copy()
        merged.update(record)
        return self.db.replace(COMMENT_RAW_COLLECTION, comment_id, merged)

    def save_comment(
        self,
        aweme_id: str,
        comment: dict[str, Any],
        parent_comment_id: str | None = None,
        *,
        task_id: str | None = None,
    ) -> bool:
        comment_id = value_to_str(comment.get("cid"))
        if not comment_id:
            return False
        parent_id = parent_comment_id or value_to_str(comment.get("reply_id"))
        if parent_id == "0":
            parent_id = None
        self.save_comment_raw(
            comment_id,
            aweme_id,
            parent_id,
            extract_comment_data(comment),
            task_id=task_id,
        )
        return True

    def mark_video_status(self, aweme_id: str, status: str) -> Record | None:
        existing = self.db.read(VIDEO_RAW_COLLECTION, aweme_id)
        if existing is None:
            return None
        updated = existing.copy()
        updated["status"] = status
        updated["updated_at"] = now_iso()
        return self.db.replace(VIDEO_RAW_COLLECTION, aweme_id, updated)

    def mark_video_comments_done(self, aweme_id: str) -> Record | None:
        return self.mark_video_status(aweme_id, DouyinVideoStatus.COMMENTS_DONE.value)

    def mark_video_error(self, aweme_id: str) -> Record | None:
        return self.mark_video_status(aweme_id, DouyinVideoStatus.ERROR.value)

    def mark_video_comments_failed_panel(self, aweme_id: str) -> Record | None:
        return self.mark_video_status(aweme_id, COMMENTS_FAILED_PANEL)

    def mark_video_comments_partial(self, aweme_id: str) -> Record | None:
        return self.mark_video_status(aweme_id, COMMENTS_PARTIAL)

    def is_video_comments_done(self, aweme_id: str) -> bool:
        record = self.db.read(VIDEO_RAW_COLLECTION, aweme_id)
        return bool(record and record.get("status") == DouyinVideoStatus.COMMENTS_DONE.value)

    def count_saved_comments(self, aweme_id: str) -> int:
        return sum(1 for item in self.db.list(COMMENT_RAW_COLLECTION) if item.get("aweme_id") == aweme_id)

    def get_unfinished_video_ids(self, sec_user_id: str) -> list[str]:
        unfinished_statuses = {
            DouyinVideoStatus.ID_ONLY.value,
            DouyinVideoStatus.RETRIEVED.value,
            COMMENTS_PARTIAL,
            COMMENTS_FAILED_PANEL,
        }
        videos = self.db.list(VIDEO_RAW_COLLECTION)
        return [
            str(video["aweme_id"])
            for video in videos
            if video.get("sec_user_id") == sec_user_id and video.get("status") in unfinished_statuses
        ]

    def get_unfinished_keyword_video_ids(self, keyword: str) -> list[str]:
        unfinished_statuses = {
            DouyinVideoStatus.ID_ONLY.value,
            DouyinVideoStatus.RETRIEVED.value,
            COMMENTS_PARTIAL,
            COMMENTS_FAILED_PANEL,
        }
        videos = self.db.list(VIDEO_RAW_COLLECTION)
        return [
            str(video["aweme_id"])
            for video in videos
            if video.get("search_keyword") == keyword and video.get("status") in unfinished_statuses
        ]

    def count_video_statuses(self, sec_user_id: str) -> tuple[int, int, int]:
        videos = [
            video
            for video in self.db.list(VIDEO_RAW_COLLECTION)
            if video.get("sec_user_id") == sec_user_id
        ]
        completed = sum(1 for video in videos if video.get("status") == DouyinVideoStatus.COMMENTS_DONE.value)
        partial = sum(1 for video in videos if video.get("status") in {COMMENTS_PARTIAL, COMMENTS_FAILED_PANEL})
        failed = sum(1 for video in videos if video.get("status") == DouyinVideoStatus.ERROR.value)
        return completed, partial, failed

    def count_keyword_video_statuses(self, keyword: str) -> tuple[int, int, int]:
        videos = [
            video
            for video in self.db.list(VIDEO_RAW_COLLECTION)
            if video.get("search_keyword") == keyword
        ]
        completed = sum(1 for video in videos if video.get("status") == DouyinVideoStatus.COMMENTS_DONE.value)
        partial = sum(1 for video in videos if video.get("status") in {COMMENTS_PARTIAL, COMMENTS_FAILED_PANEL})
        failed = sum(1 for video in videos if video.get("status") == DouyinVideoStatus.ERROR.value)
        return completed, partial, failed

    def update_video_comment_cursor(self, aweme_id: str, cursor: int) -> Record | None:
        existing = self.db.read(VIDEO_RAW_COLLECTION, aweme_id)
        if existing is None:
            return None
        updated = existing.copy()
        updated["comment_cursor"] = cursor
        updated["updated_at"] = now_iso()
        return self.db.replace(VIDEO_RAW_COLLECTION, aweme_id, updated)

    def get_video_comment_cursor(self, aweme_id: str) -> int:
        existing = self.db.read(VIDEO_RAW_COLLECTION, aweme_id)
        return value_to_int(existing.get("comment_cursor")) if existing else 0

    def get_video_expected_comment_count(self, aweme_id: str) -> int | None:
        existing = self.db.read(VIDEO_RAW_COLLECTION, aweme_id)
        return value_to_int(existing.get("comment_count")) if existing else None

    def get_video_record(self, aweme_id: str) -> Record | None:
        return self.db.read(VIDEO_RAW_COLLECTION, aweme_id)

    def update_video_danmaku_cursor(self, aweme_id: str, cursor_ms: int) -> Record | None:
        existing = self.db.read(VIDEO_RAW_COLLECTION, aweme_id)
        if existing is None:
            return None
        updated = existing.copy()
        updated["danmaku_cursor_ms"] = cursor_ms
        updated["danmaku_status"] = DANMAKU_RETRIEVED
        updated["updated_at"] = now_iso()
        return self.db.replace(VIDEO_RAW_COLLECTION, aweme_id, updated)

    def get_video_danmaku_cursor(self, aweme_id: str) -> int:
        existing = self.db.read(VIDEO_RAW_COLLECTION, aweme_id)
        return value_to_int(existing.get("danmaku_cursor_ms")) if existing else 0

    def mark_video_danmaku_status(self, aweme_id: str, status: str) -> Record | None:
        existing = self.db.read(VIDEO_RAW_COLLECTION, aweme_id)
        if existing is None:
            return None
        updated = existing.copy()
        updated["danmaku_status"] = status
        updated["updated_at"] = now_iso()
        return self.db.replace(VIDEO_RAW_COLLECTION, aweme_id, updated)

    def mark_video_danmaku_done(self, aweme_id: str) -> Record | None:
        return self.mark_video_danmaku_status(aweme_id, DANMAKU_DONE)

    def mark_video_danmaku_partial(self, aweme_id: str) -> Record | None:
        return self.mark_video_danmaku_status(aweme_id, DANMAKU_PARTIAL)

    def mark_video_danmaku_error(self, aweme_id: str) -> Record | None:
        return self.mark_video_danmaku_status(aweme_id, DANMAKU_ERROR)

    def mark_video_danmaku_unavailable(self, aweme_id: str) -> Record | None:
        return self.mark_video_danmaku_status(aweme_id, DANMAKU_UNAVAILABLE)

    def get_unfinished_danmaku_video_ids(self, sec_user_id: str) -> list[str]:
        videos = self.db.list(VIDEO_RAW_COLLECTION)
        return [
            str(video["aweme_id"])
            for video in videos
            if video.get("sec_user_id") == sec_user_id and not is_danmaku_terminal(video.get("danmaku_status"))
        ]

    def get_unfinished_keyword_danmaku_video_ids(self, keyword: str) -> list[str]:
        videos = self.db.list(VIDEO_RAW_COLLECTION)
        return [
            str(video["aweme_id"])
            for video in videos
            if video.get("search_keyword") == keyword and not is_danmaku_terminal(video.get("danmaku_status"))
        ]

    def list_authors(self) -> list[Record]:
        return self.db.list(AUTHOR_COLLECTION)

    def list_videos(self) -> list[Record]:
        return self.db.list(VIDEO_RAW_COLLECTION)

    def list_comments(self) -> list[Record]:
        return self.db.list(COMMENT_RAW_COLLECTION)

    def list_danmaku(self) -> list[Record]:
        return self.db.list(DANMAKU_RAW_COLLECTION)

    def list_media_assets(self) -> list[Record]:
        return self.db.list(MEDIA_ASSET_COLLECTION)


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def project_author(profile: dict[str, Any]) -> Record:
    user = profile.get("user") if isinstance(profile.get("user"), dict) else profile
    return {
        "uid": value_to_str(user.get("uid")),
        "short_id": value_to_str(user.get("short_id")),
        "unique_id": value_to_str(user.get("unique_id")),
        "nickname": value_to_str(user.get("nickname")),
        "signature": value_to_str(user.get("signature")),
        "gender": value_to_int(user.get("gender")),
        "ip_location": value_to_str(user.get("ip_location")),
        "verification_type": value_to_int(user.get("verification_type")),
        "custom_verify": value_to_str(user.get("custom_verify")),
        "enterprise_verify_reason": value_to_str(user.get("enterprise_verify_reason")),
        "is_star": value_to_bool(user.get("is_star")),
        "aweme_count": value_to_int(user.get("aweme_count")),
        "favoriting_count": value_to_int(user.get("favoriting_count")),
        "follower_count": value_to_int(user.get("follower_count")),
        "following_count": value_to_int(user.get("following_count")),
        "total_favorited": value_to_int(user.get("total_favorited")),
        "mplatform_followers_count": value_to_int(user.get("mplatform_followers_count")),
        "max_follower_count": value_to_int(user.get("max_follower_count")),
    }


def project_video(video: dict[str, Any]) -> Record:
    author = video.get("author") if isinstance(video.get("author"), dict) else {}
    statistics = video.get("statistics") if isinstance(video.get("statistics"), dict) else {}
    music = video.get("music") if isinstance(video.get("music"), dict) else {}
    aweme_control = video.get("aweme_control") if isinstance(video.get("aweme_control"), dict) else {}
    video_control = video.get("video_control") if isinstance(video.get("video_control"), dict) else {}
    image_list = video.get("image_list") if isinstance(video.get("image_list"), list) else []

    return {
        "author_uid": value_to_str(author.get("uid")),
        "author_sec_uid": value_to_str(author.get("sec_uid")),
        "author_nickname": value_to_str(author.get("nickname")),
        "desc": value_to_str(video.get("desc")),
        "create_time": value_to_int(video.get("create_time")),
        "aweme_type": value_to_int(video.get("aweme_type")),
        "media_type": value_to_int(video.get("media_type")),
        "duration_ms": value_to_int(video.get("duration")),
        "region": value_to_str(video.get("region")),
        "is_top": value_to_int(video.get("is_top")),
        "is_ads": value_to_bool(video.get("is_ads")),
        "is_image_album": bool(image_list),
        "image_count": len(image_list),
        "digg_count": value_to_int(statistics.get("digg_count")),
        "comment_count": value_to_int(statistics.get("comment_count")),
        "share_count": value_to_int(statistics.get("share_count")),
        "collect_count": value_to_int(statistics.get("collect_count")),
        "play_count": value_to_int(statistics.get("play_count")),
        "recommend_count": value_to_int(statistics.get("recommend_count")),
        "admire_count": value_to_int(statistics.get("admire_count")),
        "music_id": value_to_str(music.get("id_str")) or value_to_str(music.get("id")),
        "music_title": value_to_str(music.get("title")),
        "music_author": value_to_str(music.get("author")),
        "hashtag_names_csv": csv_from_array_key(video, "text_extra", "hashtag_name"),
        "can_comment": value_to_bool(aweme_control.get("can_comment")),
        "allow_share": value_to_bool(aweme_control.get("can_share")),
        "allow_download": value_to_bool(video_control.get("allow_download")),
    }


def extract_comment_data(raw: dict[str, Any]) -> Record:
    data: Record = {"raw_comment_json": raw}
    for key in [
        "text",
        "create_time",
        "digg_count",
        "reply_comment_total",
        "ip_label",
        "level",
        "is_hot",
        "content_type",
        "is_folded",
        "reply_id",
    ]:
        if key in raw:
            data[key] = raw[key]

    user = raw.get("user")
    if isinstance(user, dict):
        data["user"] = {
            key: user[key]
            for key in ["uid", "nickname", "sec_uid", "region"]
            if key in user
        }

    image_list = raw.get("image_list")
    if isinstance(image_list, list):
        data["image_list"] = normalize_media_list(image_list, media_type="image")

    audio_list = collect_media_by_keys(
        raw,
        {
            "audio",
            "audio_comment",
            "comment_audio",
            "sound",
            "voice",
            "voice_comment",
        },
    )
    if audio_list:
        data["audio_list"] = normalize_media_list(audio_list, media_type="audio")

    video_list = collect_media_by_keys(
        raw,
        {
            "animated_image",
            "aweme_video",
            "comment_video",
            "sticker",
            "video",
            "video_list",
        },
    )
    if video_list:
        data["video_list"] = normalize_media_list(video_list, media_type="video")

    return data


def project_comment(data: dict[str, Any]) -> Record:
    user = data.get("user") if isinstance(data.get("user"), dict) else {}
    image_list = data.get("image_list") if isinstance(data.get("image_list"), list) else []
    audio_list = data.get("audio_list") if isinstance(data.get("audio_list"), list) else []
    video_list = data.get("video_list") if isinstance(data.get("video_list"), list) else []
    media_types = []
    if image_list:
        media_types.append("image")
    if audio_list:
        media_types.append("audio")
    if video_list:
        media_types.append("video")
    return {
        "text": value_to_str(data.get("text")),
        "content_type": value_to_int(data.get("content_type")),
        "create_time": value_to_int(data.get("create_time")),
        "digg_count": value_to_int(data.get("digg_count")),
        "reply_comment_total": value_to_int(data.get("reply_comment_total")),
        "ip_label": value_to_str(data.get("ip_label")),
        "level": value_to_int(data.get("level")),
        "is_hot": value_to_bool(data.get("is_hot")),
        "is_folded": value_to_bool(data.get("is_folded")),
        "user_uid": value_to_str(user.get("uid")),
        "user_sec_uid": value_to_str(user.get("sec_uid")),
        "user_nickname": value_to_str(user.get("nickname")),
        "user_region": value_to_str(user.get("region")),
        "image_uris_csv": csv_from_array_key(data, "image_list", "uri"),
        "image_urls_csv": csv_from_media_urls(image_list),
        "audio_urls_csv": csv_from_media_urls(audio_list),
        "video_urls_csv": csv_from_media_urls(video_list),
        "image_count": len(image_list),
        "audio_count": len(audio_list),
        "video_count": len(video_list),
        "has_media": bool(media_types),
        "media_types_csv": ",".join(media_types) if media_types else None,
        "reply_id": value_to_str(data.get("reply_id")),
    }


def project_danmaku(data: dict[str, Any]) -> Record:
    offset_time = value_to_int(data.get("offset_time"))
    item_id = value_to_str(data.get("item_id"))
    projected: Record = {
        "text": value_to_str(data.get("text")),
        "offset_time": offset_time,
        "offset_seconds": offset_time / 1000 if offset_time is not None else None,
        "user_id": value_to_str(data.get("user_id")),
        "digg_count": value_to_int(data.get("digg_count")),
        "danmaku_type": value_to_int(data.get("danmaku_type")),
        "status": value_to_int(data.get("status")),
        "score": data.get("score") if isinstance(data.get("score"), int | float) else None,
        "has_emoji": value_to_bool(data.get("has_emoji")),
        "is_ad": value_to_bool(data.get("is_ad")),
    }
    if item_id:
        projected["aweme_id"] = item_id
    return projected


def stable_danmaku_id(aweme_id: str, danmaku: dict[str, Any]) -> str:
    source = "|".join(
        [
            aweme_id,
            value_to_str(danmaku.get("item_id")) or "",
            value_to_str(danmaku.get("user_id")) or "",
            value_to_str(danmaku.get("offset_time")) or "",
            value_to_str(danmaku.get("text")) or "",
        ]
    )
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()
    return f"{aweme_id}:{digest}"


def is_danmaku_terminal(status: Any) -> bool:
    return status in {DANMAKU_DONE, DANMAKU_UNAVAILABLE}


def collect_media_by_keys(value: Any, key_names: set[str]) -> list[dict[str, Any]]:
    media: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in key_names:
                if isinstance(child, dict):
                    media.append(child)
                elif isinstance(child, list):
                    media.extend(item for item in child if isinstance(item, dict))
                continue
            if isinstance(child, dict | list):
                media.extend(collect_media_by_keys(child, key_names))
    elif isinstance(value, list):
        for child in value:
            media.extend(collect_media_by_keys(child, key_names))
    return dedupe_media(media)


def dedupe_media(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in items:
        marker = (
            value_to_str(item.get("uri"))
            or value_to_str(item.get("url"))
            or ",".join(extract_url_list(item))
            or str(id(item))
        )
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(item)
    return deduped


def normalize_media_list(items: list[Any], *, media_type: str) -> list[Record]:
    normalized: list[Record] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        media: Record = {
            "type": media_type,
            "uri": first_text_value(item, ["uri", "id", "id_str", "tos_key"]),
            "url": first_text_value(item, ["url", "src"]),
            "url_list": extract_url_list(item),
            "width": value_to_int(item.get("width")),
            "height": value_to_int(item.get("height")),
            "duration": value_to_int(item.get("duration") or item.get("duration_ms")),
            "format": first_text_value(item, ["format", "mime_type", "file_type"]),
            "raw": item,
        }

        origin_url = item.get("origin_url")
        if isinstance(origin_url, dict):
            media["origin_uri"] = value_to_str(origin_url.get("uri"))
            origin_urls = extract_url_list(origin_url)
            if origin_urls and not media["url_list"]:
                media["url_list"] = origin_urls

        play_addr = item.get("play_addr")
        if isinstance(play_addr, dict):
            media["play_uri"] = value_to_str(play_addr.get("uri"))
            play_urls = extract_url_list(play_addr)
            if play_urls and not media["url_list"]:
                media["url_list"] = play_urls

        download_addr = item.get("download_addr")
        if isinstance(download_addr, dict):
            media["download_uri"] = value_to_str(download_addr.get("uri"))
            media["download_url_list"] = extract_url_list(download_addr)

        normalized.append({key: value for key, value in media.items() if value not in (None, [], "")})
    return normalized


def first_text_value(value: dict[str, Any], keys: list[str]) -> str | None:
    for key in keys:
        text = value_to_str(value.get(key))
        if text:
            return text
    return None


def extract_url_list(value: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    for key in ["url_list", "url_lists", "urls"]:
        raw_urls = value.get(key)
        if isinstance(raw_urls, list):
            urls.extend(str(item) for item in raw_urls if isinstance(item, str) and item)
    for key in ["url", "src", "download_url"]:
        url = value_to_str(value.get(key))
        if url:
            urls.append(url)
    return list(dict.fromkeys(urls))


def csv_from_media_urls(items: list[Any]) -> str | None:
    urls: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        url = value_to_str(item.get("url"))
        if url:
            urls.append(url)
        url_list = item.get("url_list")
        if isinstance(url_list, list):
            urls.extend(str(value) for value in url_list if isinstance(value, str) and value)
    deduped = list(dict.fromkeys(urls))
    return ",".join(deduped) if deduped else None
