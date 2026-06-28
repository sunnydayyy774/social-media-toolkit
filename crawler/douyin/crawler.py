from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlencode

from loguru import logger

from crawler.base import BrowserCrawler, BrowserCrawlerConfig
from storage import DuckDBDatabase
from .storage import DouyinStore
from .utils import (
    DOUYIN_BASE_URL,
    browser_fetch_json,
    get_page_debug_info,
    navigate,
    smart_sleep,
    user_profile_url,
    value_to_bool,
    value_to_int,
    value_to_str,
    video_url,
    wait_for_login,
)

DOUYIN_SEARCH_API_PATH = "/aweme/v1/web/general/search/single/"
DOUYIN_DANMAKU_API_PATH = "/aweme/v1/web/danmaku/get_v2/"


@dataclass(slots=True)
class DouyinCrawlerConfig(BrowserCrawlerConfig):
    login_timeout_ms: int = 3000
    max_empty_pages: int = 5
    max_video_pages: int | None = None
    max_search_pages: int | None = None
    max_comment_pages: int | None = None
    max_reply_pages: int | None = None
    max_danmaku_windows: int | None = None
    collect_comments: bool = True
    collect_danmaku: bool = False
    danmaku_only: bool = False
    danmaku_window_ms: int = 32000
    request_count: int = 20
    search_channel: str = "aweme_video_web"
    search_api_path: str = DOUYIN_SEARCH_API_PATH
    danmaku_api_path: str = DOUYIN_DANMAKU_API_PATH


class DouyinCrawler(BrowserCrawler[DouyinCrawlerConfig, DouyinStore]):
    """Douyin crawler using cloakbrowser and browser-side authenticated fetch calls."""

    db_cls = DuckDBDatabase
    store_cls = DouyinStore

    async def by_keyword(
        self,
        keyword: str,
        *,
        id_only: bool = False,
        collect_comments: bool | None = None,
        collect_danmaku: bool | None = None,
        danmaku_only: bool | None = None,
        restrict_to_aweme_ids: Iterable[str] | None = None,
        skip_aweme_ids: Iterable[str] | None = None,
        use_local_index: bool = False,
        task_id: str | None = None,
        page: Any | None = None,
    ) -> None:
        page = page or await self._new_page()
        collect_comments = self.config.collect_comments if collect_comments is None else collect_comments
        collect_danmaku = self.config.collect_danmaku if collect_danmaku is None else collect_danmaku
        danmaku_only = self.config.danmaku_only if danmaku_only is None else danmaku_only

        if not use_local_index:
            await self._prepare_search_page(page, keyword)
            discovered_videos = await self._collect_keyword_video_list(page, keyword, task_id=task_id)
            logger.info("Douyin keyword discovery saved {} videos", len(discovered_videos))

        restrict_set = set(restrict_to_aweme_ids or [])
        skip_set = set(skip_aweme_ids or [])
        if id_only or not collect_comments:
            logger.info(
                "Douyin keyword scrape stopped before comment collection (id_only={}, collect_comments={})",
                id_only,
                collect_comments,
            )
            return

        if use_local_index:
            await self._prepare_search_page(page, keyword)

        if collect_comments and not danmaku_only:
            pending_videos = filter_aweme_ids(
                self.store.get_unfinished_keyword_video_ids(keyword),
                restrict_set=restrict_set,
                skip_set=skip_set,
            )
            logger.info("Douyin pending keyword videos from local store: {}", len(pending_videos))
            await self._collect_comments_for_videos(page, keyword, pending_videos, task_id=task_id)

        if collect_danmaku or danmaku_only:
            pending_danmaku_videos = filter_aweme_ids(
                self.store.get_unfinished_keyword_danmaku_video_ids(keyword),
                restrict_set=restrict_set,
                skip_set=skip_set,
            )
            logger.info("Douyin pending keyword videos for danmaku: {}", len(pending_danmaku_videos))
            await self._collect_danmaku_for_videos(
                page,
                pending_danmaku_videos,
                task_id=task_id,
                search_keyword=keyword,
            )
        completed, partial, failed = self.store.count_keyword_video_statuses(keyword)
        logger.info(
            "Douyin keyword scrape complete for {} (completed={}, partial={}, failed={})",
            keyword,
            completed,
            partial,
            failed,
        )

    async def by_author(
        self,
        sec_user_id: str,
        *,
        id_only: bool = False,
        collect_comments: bool | None = None,
        collect_danmaku: bool | None = None,
        danmaku_only: bool | None = None,
        restrict_to_aweme_ids: Iterable[str] | None = None,
        skip_aweme_ids: Iterable[str] | None = None,
        use_local_index: bool = False,
        task_id: str | None = None,
        page: Any | None = None,
    ) -> None:
        page = page or await self._new_page()
        collect_comments = self.config.collect_comments if collect_comments is None else collect_comments
        collect_danmaku = self.config.collect_danmaku if collect_danmaku is None else collect_danmaku
        danmaku_only = self.config.danmaku_only if danmaku_only is None else danmaku_only

        if not use_local_index:
            await self._prepare_profile_page(page, sec_user_id)
            profile = await self._fetch_author_profile(page, sec_user_id)
            if profile is not None:
                self.store.save_author_profile(sec_user_id, profile)
                logger.info("Saved Douyin author profile {}", sec_user_id)

            discovered_videos = await self._collect_video_list(page, sec_user_id, task_id=task_id)
            logger.info("Douyin discovery saved {} videos", len(discovered_videos))

        restrict_set = set(restrict_to_aweme_ids or [])
        skip_set = set(skip_aweme_ids or [])
        if id_only or not collect_comments:
            logger.info(
                "Douyin scrape stopped before comment collection (id_only={}, collect_comments={})",
                id_only,
                collect_comments,
            )
            return

        if use_local_index:
            await self._prepare_profile_page(page, sec_user_id)

        if collect_comments and not danmaku_only:
            pending_videos = filter_aweme_ids(
                self.store.get_unfinished_video_ids(sec_user_id),
                restrict_set=restrict_set,
                skip_set=skip_set,
            )
            logger.info("Douyin pending videos from local store: {}", len(pending_videos))
            await self._collect_comments_for_videos(
                page,
                sec_user_id,
                pending_videos,
                task_id=task_id,
            )

        if collect_danmaku or danmaku_only:
            pending_danmaku_videos = filter_aweme_ids(
                self.store.get_unfinished_danmaku_video_ids(sec_user_id),
                restrict_set=restrict_set,
                skip_set=skip_set,
            )
            logger.info("Douyin pending videos for danmaku: {}", len(pending_danmaku_videos))
            await self._collect_danmaku_for_videos(page, pending_danmaku_videos, task_id=task_id)
        completed, partial, failed = self.store.count_video_statuses(sec_user_id)
        logger.info(
            "Douyin scrape complete for {} (completed={}, partial={}, failed={})",
            sec_user_id,
            completed,
            partial,
            failed,
        )

    async def scrape_author_posts(self, sec_user_id: str, **kwargs: Any) -> None:
        await self.by_author(sec_user_id, **kwargs)

    async def scrape_author_info(
        self,
        sec_user_id: str,
        *,
        page: Any | None = None,
    ) -> None:
        page = page or await self._new_page()
        await self._prepare_profile_page(page, sec_user_id)
        profile = await self._fetch_author_profile(page, sec_user_id)
        if profile is None:
            raise RuntimeError(f"No Douyin profile response for {sec_user_id}")
        self.store.save_author_profile(sec_user_id, profile)

    async def _prepare_profile_page(self, page: Any, sec_user_id: str) -> None:
        url = user_profile_url(sec_user_id)
        logger.info("Navigating to Douyin profile {}", url)
        await navigate(page, url)
        needed_login = await wait_for_login(
            page,
            timeout_ms=self.config.login_timeout_ms,
            prompt=self.prompt,
        )
        if needed_login:
            logger.info("Reloading Douyin profile after login")
            await navigate(page, url)
        logger.info("Douyin page: {}", await get_page_debug_info(page))

    async def _prepare_search_page(self, page: Any, keyword: str) -> None:
        url = keyword_search_url(keyword)
        logger.info("Navigating to Douyin keyword search {}", url)
        await navigate(page, url)
        needed_login = await wait_for_login(
            page,
            timeout_ms=self.config.login_timeout_ms,
            prompt=self.prompt,
        )
        if needed_login:
            logger.info("Reloading Douyin search after login")
            await navigate(page, url)
        logger.info("Douyin page: {}", await get_page_debug_info(page))

    async def _fetch_author_profile(self, page: Any, sec_user_id: str) -> dict[str, Any] | None:
        url = f"{DOUYIN_BASE_URL}/aweme/v1/web/user/profile/?{urlencode(base_params(sec_user_id=sec_user_id))}"
        logger.info("Fetching Douyin profile API {}", url)
        try:
            response = await browser_fetch_json(page, url)
        except RuntimeError as exc:
            logger.warning("Douyin profile API failed: {}", exc)
            return None
        return response if isinstance(response, dict) else None

    async def _collect_video_list(
        self,
        page: Any,
        sec_user_id: str,
        *,
        task_id: str | None,
    ) -> list[str]:
        logger.info("Phase A: collecting Douyin video list for {}", sec_user_id)
        seen: set[str] = set()
        collected: list[str] = []
        max_cursor = 0
        empty_pages = 0
        page_number = 0

        while True:
            if self.config.max_video_pages is not None and page_number >= self.config.max_video_pages:
                logger.info("Reached max_video_pages={}", self.config.max_video_pages)
                break

            page_number += 1
            response = await self._fetch_video_page(page, sec_user_id, max_cursor)
            aweme_list = response.get("aweme_list") if isinstance(response, dict) else None
            if not isinstance(aweme_list, list) or not aweme_list:
                empty_pages += 1
                logger.info(
                    "No Douyin videos on page {} (empty {}/{})",
                    page_number,
                    empty_pages,
                    self.config.max_empty_pages,
                )
                if empty_pages >= self.config.max_empty_pages:
                    break
                await smart_sleep()
                continue

            empty_pages = 0
            new_count = 0
            for aweme in aweme_list:
                if not isinstance(aweme, dict):
                    continue
                aweme_id = value_to_str(aweme.get("aweme_id"))
                if not aweme_id or aweme_id in seen:
                    continue
                seen.add(aweme_id)
                collected.append(aweme_id)
                self.store.save_video_raw(aweme_id, sec_user_id, aweme, task_id=task_id)
                new_count += 1

            has_more = value_to_bool(response.get("has_more"))
            next_cursor = value_to_int(response.get("max_cursor"))
            logger.info(
                "Douyin video page {} saved {} new videos (total={}, has_more={}, max_cursor={})",
                page_number,
                new_count,
                len(collected),
                has_more,
                next_cursor,
            )
            if not has_more:
                break
            if next_cursor is None or next_cursor == max_cursor:
                empty_pages += 1
                if empty_pages >= self.config.max_empty_pages:
                    break
            else:
                max_cursor = next_cursor
            await smart_sleep()

        logger.info("Phase A complete. Total Douyin videos collected: {}", len(collected))
        return collected

    async def _collect_keyword_video_list(
        self,
        page: Any,
        keyword: str,
        *,
        task_id: str | None,
    ) -> list[str]:
        logger.info("Phase A: collecting Douyin keyword video list for {}", keyword)
        seen: set[str] = set()
        collected: list[str] = []
        cursor = 0
        empty_pages = 0
        page_number = 0

        while True:
            if self.config.max_search_pages is not None and page_number >= self.config.max_search_pages:
                logger.info("Reached max_search_pages={}", self.config.max_search_pages)
                break

            page_number += 1
            response = await self._fetch_search_page(page, keyword, cursor)
            aweme_list = extract_search_awemes(response)
            if not aweme_list:
                empty_pages += 1
                logger.info(
                    "No Douyin keyword videos on page {} (empty {}/{})",
                    page_number,
                    empty_pages,
                    self.config.max_empty_pages,
                )
                if empty_pages >= self.config.max_empty_pages:
                    break
                await smart_sleep()
                continue

            empty_pages = 0
            new_count = 0
            for index, aweme in enumerate(aweme_list, start=1):
                aweme_id = value_to_str(aweme.get("aweme_id"))
                if not aweme_id or aweme_id in seen:
                    continue
                seen.add(aweme_id)
                collected.append(aweme_id)
                sec_user_id = search_aweme_sec_user_id(aweme) or f"keyword:{keyword}"
                self.store.save_video_raw(
                    aweme_id,
                    sec_user_id,
                    aweme,
                    task_id=task_id,
                    search_keyword=keyword,
                    search_page=page_number,
                    search_position=index,
                    source="douyin_keyword_search",
                )
                new_count += 1

            has_more = value_to_bool(response.get("has_more"))
            next_cursor = value_to_int(response.get("cursor"))
            if next_cursor is None:
                next_cursor = value_to_int(response.get("next_cursor"))
            logger.info(
                "Douyin keyword page {} saved {} new videos (total={}, has_more={}, cursor={})",
                page_number,
                new_count,
                len(collected),
                has_more,
                next_cursor,
            )
            if not has_more:
                break
            if next_cursor is None or next_cursor == cursor:
                empty_pages += 1
                if empty_pages >= self.config.max_empty_pages:
                    break
            else:
                cursor = next_cursor
            await smart_sleep()

        logger.info("Phase A complete. Total Douyin keyword videos collected: {}", len(collected))
        return collected

    async def _fetch_video_page(
        self,
        page: Any,
        sec_user_id: str,
        max_cursor: int,
    ) -> dict[str, Any]:
        params = base_params(
            sec_user_id=sec_user_id,
            max_cursor=max_cursor,
            count=self.config.request_count,
            locate_query=False,
            show_live_replay_strategy=1,
            need_time_list=1,
        )
        url = f"{DOUYIN_BASE_URL}/aweme/v1/web/aweme/post/?{urlencode(params)}"
        logger.info("Fetching Douyin video page cursor={}", max_cursor)
        response = await browser_fetch_json(page, url)
        return response if isinstance(response, dict) else {}

    async def _fetch_search_page(
        self,
        page: Any,
        keyword: str,
        cursor: int,
    ) -> dict[str, Any]:
        params = base_params(
            keyword=keyword,
            search_channel=self.config.search_channel,
            cursor=cursor,
            count=self.config.request_count,
        )
        api_path = self.config.search_api_path
        if not api_path.startswith("/"):
            api_path = f"/{api_path}"
        url = f"{DOUYIN_BASE_URL}{api_path}?{urlencode(params)}"
        logger.info("Fetching Douyin keyword page cursor={}", cursor)
        response = await browser_fetch_json(page, url)
        return response if isinstance(response, dict) else {}

    async def _collect_comments_for_videos(
        self,
        page: Any,
        sec_user_id: str,
        aweme_ids: list[str],
        *,
        task_id: str | None,
    ) -> None:
        unfinished = list(aweme_ids)
        logger.info("Phase B: collecting comments for {} Douyin videos", len(unfinished))
        for aweme_id in unfinished:
            try:
                await self._collect_comments_for_video(page, aweme_id, task_id=task_id)
            except Exception as exc:
                logger.exception("Failed collecting Douyin comments for {}: {}", aweme_id, exc)
                self.store.mark_video_error(aweme_id)

    async def _collect_comments_for_video(
        self,
        page: Any,
        aweme_id: str,
        *,
        task_id: str | None,
    ) -> None:
        logger.info("Collecting Douyin comments for {}", aweme_id)
        await navigate(page, video_url(aweme_id), wait_ms=2500)

        total_saved = 0
        cursor = self.store.get_video_comment_cursor(aweme_id)
        expected_comments = self.store.get_video_expected_comment_count(aweme_id) or 0
        empty_pages = 0
        page_number = 0

        while True:
            if self.config.max_comment_pages is not None and page_number >= self.config.max_comment_pages:
                self.store.mark_video_comments_partial(aweme_id)
                logger.info("Reached max_comment_pages={} for {}", self.config.max_comment_pages, aweme_id)
                return

            response = await self._fetch_comment_page(page, aweme_id, cursor)
            comments = response.get("comments") if isinstance(response, dict) else None
            if not isinstance(comments, list) or not comments:
                empty_pages += 1
                saved_comments = self.store.count_saved_comments(aweme_id)
                logger.info(
                    "No Douyin comments for {} at cursor={} (empty {}/{}, saved={}, expected={})",
                    aweme_id,
                    cursor,
                    empty_pages,
                    self.config.max_empty_pages,
                    saved_comments,
                    expected_comments,
                )
                if expected_comments and saved_comments < expected_comments and empty_pages < self.config.max_empty_pages:
                    await smart_sleep()
                    continue
                if expected_comments and saved_comments < expected_comments:
                    self.store.mark_video_comments_partial(aweme_id)
                    logger.warning(
                        "Marked Douyin video {} comments partial after empty pages (saved={}, expected={})",
                        aweme_id,
                        saved_comments,
                        expected_comments,
                    )
                    return
                break

            empty_pages = 0
            page_number += 1
            for comment in comments:
                if not isinstance(comment, dict):
                    continue
                if self.store.save_comment(aweme_id, comment, task_id=task_id):
                    total_saved += 1
                parent_cid = value_to_str(comment.get("cid"))
                replies = comment.get("reply_comment")
                if isinstance(replies, list):
                    for reply in replies:
                        if isinstance(reply, dict) and self.store.save_comment(
                            aweme_id,
                            reply,
                            parent_cid,
                            task_id=task_id,
                        ):
                            total_saved += 1
                reply_total = value_to_int(comment.get("reply_comment_total")) or 0
                if reply_total > (len(replies) if isinstance(replies, list) else 0) and parent_cid:
                    total_saved += await self._collect_replies_for_comment(
                        page,
                        aweme_id,
                        parent_cid,
                        task_id=task_id,
                    )

            has_more = value_to_bool(response.get("has_more"))
            next_cursor = value_to_int(response.get("cursor"))
            if next_cursor is None:
                next_cursor = value_to_int(response.get("next_cursor"))
            logger.info(
                "Douyin comments {} page {} saved_total={} has_more={} cursor={}",
                aweme_id,
                page_number,
                total_saved,
                has_more,
                next_cursor,
            )
            if has_more and next_cursor is None:
                self.store.mark_video_comments_partial(aweme_id)
                logger.warning("Marked Douyin video {} comments partial because next cursor is missing", aweme_id)
                return
            if has_more and next_cursor == cursor:
                self.store.mark_video_comments_partial(aweme_id)
                logger.warning("Marked Douyin video {} comments partial because cursor did not advance", aweme_id)
                return
            if next_cursor is not None:
                self.store.update_video_comment_cursor(aweme_id, next_cursor)
                cursor = next_cursor
            if not has_more:
                break
            await smart_sleep()

        saved_comments = self.store.count_saved_comments(aweme_id)
        if expected_comments and saved_comments < expected_comments:
            self.store.mark_video_comments_partial(aweme_id)
            logger.warning(
                "Marked Douyin video {} comments partial (saved={}, expected={})",
                aweme_id,
                saved_comments,
                expected_comments,
            )
            return
        self.store.mark_video_comments_done(aweme_id)
        logger.info("Marked Douyin video {} comments done (new_saved={}, total_saved={})", aweme_id, total_saved, saved_comments)

    async def _fetch_comment_page(
        self,
        page: Any,
        aweme_id: str,
        cursor: int,
    ) -> dict[str, Any]:
        params = base_params(
            aweme_id=aweme_id,
            cursor=cursor,
            count=self.config.request_count,
            item_type=0,
        )
        url = f"{DOUYIN_BASE_URL}/aweme/v1/web/comment/list/?{urlencode(params)}"
        response = await browser_fetch_json(page, url)
        return response if isinstance(response, dict) else {}

    async def _collect_replies_for_comment(
        self,
        page: Any,
        aweme_id: str,
        comment_id: str,
        *,
        task_id: str | None,
    ) -> int:
        saved = 0
        cursor = 0
        page_number = 0
        while True:
            if self.config.max_reply_pages is not None and page_number >= self.config.max_reply_pages:
                return saved
            response = await self._fetch_reply_page(page, aweme_id, comment_id, cursor)
            comments = response.get("comments") if isinstance(response, dict) else None
            if not isinstance(comments, list) or not comments:
                return saved
            page_number += 1
            for comment in comments:
                if isinstance(comment, dict) and self.store.save_comment(
                    aweme_id,
                    comment,
                    comment_id,
                    task_id=task_id,
                ):
                    saved += 1
            has_more = value_to_bool(response.get("has_more"))
            next_cursor = value_to_int(response.get("cursor"))
            if next_cursor is None:
                next_cursor = value_to_int(response.get("next_cursor"))
            if not has_more or next_cursor is None or next_cursor == cursor:
                return saved
            cursor = next_cursor
            await smart_sleep(0.5, 1.5)

    async def _fetch_reply_page(
        self,
        page: Any,
        aweme_id: str,
        comment_id: str,
        cursor: int,
    ) -> dict[str, Any]:
        params = base_params(
            item_id=aweme_id,
            aweme_id=aweme_id,
            comment_id=comment_id,
            cursor=cursor,
            count=self.config.request_count,
            item_type=0,
        )
        url = f"{DOUYIN_BASE_URL}/aweme/v1/web/comment/list/reply/?{urlencode(params)}"
        response = await browser_fetch_json(page, url)
        return response if isinstance(response, dict) else {}

    async def _collect_danmaku_for_videos(
        self,
        page: Any,
        aweme_ids: list[str],
        *,
        task_id: str | None,
        search_keyword: str | None = None,
    ) -> None:
        logger.info("Phase C: collecting danmaku for {} Douyin videos", len(aweme_ids))
        for aweme_id in aweme_ids:
            try:
                await self._collect_danmaku_for_video(
                    page,
                    aweme_id,
                    task_id=task_id,
                    search_keyword=search_keyword,
                )
            except Exception as exc:
                logger.exception("Failed collecting Douyin danmaku for {}: {}", aweme_id, exc)
                self.store.mark_video_danmaku_error(aweme_id)

    async def _collect_danmaku_for_video(
        self,
        page: Any,
        aweme_id: str,
        *,
        task_id: str | None,
        search_keyword: str | None = None,
    ) -> None:
        logger.info("Collecting Douyin danmaku for {}", aweme_id)
        await navigate(page, video_url(aweme_id), wait_ms=2500)

        detail = await self._video_detail_for_danmaku(page, aweme_id)
        if detail is None:
            logger.warning("No Douyin video detail available for danmaku {}", aweme_id)
            self.store.mark_video_danmaku_error(aweme_id)
            return

        control = detail.get("danmaku_control") if isinstance(detail.get("danmaku_control"), dict) else {}
        if value_to_bool(control.get("enable_danmaku")) is False:
            logger.info("Douyin danmaku disabled for {}", aweme_id)
            self.store.mark_video_danmaku_unavailable(aweme_id)
            return

        duration_ms = value_to_int(detail.get("duration"))
        if duration_ms is None:
            duration_ms = self._video_duration_from_record(aweme_id)
        authentication_token = value_to_str(detail.get("authentication_token"))
        if not duration_ms or not authentication_token:
            logger.warning(
                "Missing Douyin danmaku parameters for {} (duration_ms={}, token={})",
                aweme_id,
                duration_ms,
                bool(authentication_token),
            )
            self.store.mark_video_danmaku_unavailable(aweme_id)
            return

        start_time = max(0, self.store.get_video_danmaku_cursor(aweme_id))
        window_ms = max(1000, self.config.danmaku_window_ms)
        windows = 0

        while start_time < duration_ms:
            if self.config.max_danmaku_windows is not None and windows >= self.config.max_danmaku_windows:
                self.store.mark_video_danmaku_partial(aweme_id)
                logger.info("Reached max_danmaku_windows={} for {}", self.config.max_danmaku_windows, aweme_id)
                return

            end_time = min(start_time + window_ms, duration_ms)
            response = await self._fetch_danmaku_window(
                page,
                aweme_id,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                authentication_token=authentication_token,
            )
            danmaku_list = response.get("danmaku_list") if isinstance(response, dict) else None
            saved = 0
            if isinstance(danmaku_list, list):
                for item in danmaku_list:
                    if isinstance(item, dict) and self.store.save_danmaku(
                        aweme_id,
                        item,
                        task_id=task_id,
                        search_keyword=search_keyword,
                    ):
                        saved += 1

            self.store.update_video_danmaku_cursor(aweme_id, end_time)
            logger.info(
                "Douyin danmaku {} window {}-{} saved={}",
                aweme_id,
                start_time,
                end_time,
                saved,
            )
            start_time = end_time
            windows += 1
            await smart_sleep(0.5, 1.5)

        self.store.mark_video_danmaku_done(aweme_id)
        logger.info("Marked Douyin video {} danmaku done", aweme_id)

    async def _fetch_danmaku_window(
        self,
        page: Any,
        aweme_id: str,
        *,
        start_time: int,
        end_time: int,
        duration_ms: int,
        authentication_token: str,
    ) -> dict[str, Any]:
        params = base_params(
            group_id=aweme_id,
            item_id=aweme_id,
            start_time=start_time,
            end_time=end_time,
            duration=duration_ms,
            authentication_token=authentication_token,
        )
        api_path = self.config.danmaku_api_path
        if not api_path.startswith("/"):
            api_path = f"/{api_path}"
        url = f"{DOUYIN_BASE_URL}{api_path}?{urlencode(params)}"
        response = await browser_fetch_json(page, url)
        return response if isinstance(response, dict) else {}

    async def _video_detail_for_danmaku(self, page: Any, aweme_id: str) -> dict[str, Any] | None:
        record = self.store.get_video_record(aweme_id)
        video_json = record.get("video_json") if isinstance(record, dict) else None
        if isinstance(video_json, dict) and value_to_str(video_json.get("authentication_token")):
            return video_json

        response = await self._fetch_aweme_detail(page, aweme_id)
        detail = response.get("aweme_detail") if isinstance(response, dict) else None
        if isinstance(detail, dict):
            sec_user_id = search_aweme_sec_user_id(detail)
            if sec_user_id is None and isinstance(record, dict):
                sec_user_id = value_to_str(record.get("sec_user_id"))
            self.store.save_video_raw(aweme_id, sec_user_id or f"aweme:{aweme_id}", detail)
            return detail
        return video_json if isinstance(video_json, dict) else None

    async def _fetch_aweme_detail(self, page: Any, aweme_id: str) -> dict[str, Any]:
        params = base_params(aweme_id=aweme_id)
        url = f"{DOUYIN_BASE_URL}/aweme/v1/web/aweme/detail/?{urlencode(params)}"
        response = await browser_fetch_json(page, url)
        return response if isinstance(response, dict) else {}

    def _video_duration_from_record(self, aweme_id: str) -> int | None:
        record = self.store.get_video_record(aweme_id)
        if not isinstance(record, dict):
            return None
        duration = value_to_int(record.get("duration_ms"))
        if duration is not None:
            return duration
        video_json = record.get("video_json")
        if isinstance(video_json, dict):
            return value_to_int(video_json.get("duration"))
        return None

def keyword_search_url(keyword: str) -> str:
    return f"{DOUYIN_BASE_URL}/search/{quote(keyword)}?type=video"


def extract_search_awemes(response: dict[str, Any]) -> list[dict[str, Any]]:
    awemes: list[dict[str, Any]] = []
    seen: set[str] = set()

    for item in iter_dicts(response):
        aweme = item.get("aweme_info")
        if not isinstance(aweme, dict):
            aweme = item.get("aweme")
        if not isinstance(aweme, dict) and item.get("aweme_id") is not None:
            aweme = item
        if not isinstance(aweme, dict):
            continue
        aweme_id = value_to_str(aweme.get("aweme_id"))
        if not aweme_id or aweme_id in seen:
            continue
        seen.add(aweme_id)
        awemes.append(aweme)

    return awemes


def iter_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from iter_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_dicts(child)


def search_aweme_sec_user_id(aweme: dict[str, Any]) -> str | None:
    author = aweme.get("author")
    if not isinstance(author, dict):
        return None
    return value_to_str(author.get("sec_uid")) or value_to_str(author.get("sec_user_id"))


def filter_aweme_ids(
    aweme_ids: list[str],
    *,
    restrict_set: set[str],
    skip_set: set[str],
) -> list[str]:
    filtered = aweme_ids
    if restrict_set:
        filtered = [aweme_id for aweme_id in filtered if aweme_id in restrict_set]
    if skip_set:
        filtered = [aweme_id for aweme_id in filtered if aweme_id not in skip_set]
    return filtered


def base_params(**overrides: Any) -> dict[str, Any]:
    params: dict[str, Any] = {
        "device_platform": "webapp",
        "aid": "6383",
        "channel": "channel_pc_web",
        "pc_client_type": "1",
        "version_code": "190500",
        "version_name": "19.5.0",
        "cookie_enabled": "true",
        "screen_width": "1440",
        "screen_height": "900",
        "browser_language": "zh-CN",
        "browser_platform": "MacIntel",
        "browser_name": "Chrome",
        "browser_version": "120.0.0.0",
    }
    params.update({key: value for key, value in overrides.items() if value is not None})
    return params
