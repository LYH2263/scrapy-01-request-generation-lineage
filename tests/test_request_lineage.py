from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

from scrapy import Request, Spider, signals
from scrapy.core.engine import ExecutionEngine
from scrapy.downloadermiddlewares.redirect import (
    MetaRefreshMiddleware,
    RedirectMiddleware,
)
from scrapy.downloadermiddlewares.retry import RetryMiddleware
from scrapy.dupefilters import RFPDupeFilter
from scrapy.exceptions import IgnoreRequest
from scrapy.http import (
    HtmlResponse,
    LineageAdmission,
    LineageBodyAction,
    LineageGenerator,
    RequestLineage,
    Response,
)
from scrapy.utils.defer import maybe_deferred_to_future
from scrapy.utils.misc import build_from_crawler
from scrapy.utils.spider import DefaultSpider
from scrapy.utils.test import get_crawler
from tests.utils.decorators import coroutine_test

if TYPE_CHECKING:
    from scrapy.crawler import Crawler

START = LineageGenerator.START
CALLBACK = LineageGenerator.CALLBACK
RETRY = LineageGenerator.RETRY
PRESERVE = LineageGenerator.REDIRECT_PRESERVE
TO_GET = LineageGenerator.REDIRECT_TO_GET
META_REFRESH = LineageGenerator.META_REFRESH

SCHEDULER = LineageAdmission.SCHEDULER
READMIT = LineageAdmission.SCHEDULER_READMIT
INLINE = LineageAdmission.INLINE_FOLLOW

PRESERVED = LineageBodyAction.PRESERVED
FORCED_GET = LineageBodyAction.FORCED_GET


def lineage_of(request: Request) -> RequestLineage:
    assert request.lineage is not None
    return request.lineage


def redirect_response(request: Request, location: str, status: int = 302) -> Response:
    return Response(
        request.url,
        status=status,
        headers={"Location": location},
        request=request,
    )


def make_redirect_middleware() -> tuple[RedirectMiddleware, Crawler]:
    crawler = get_crawler(DefaultSpider)
    crawler.spider = crawler._create_spider()
    return build_from_crawler(RedirectMiddleware, crawler), crawler


class TestRequestLineageObject:
    def test_default_lineage_is_none(self):
        assert Request("http://example.com").lineage is None

    def test_lineage_is_read_only(self):
        lineage = RequestLineage.spawn(None, START, admission=SCHEDULER)
        with pytest.raises(AttributeError):
            lineage.generator = RETRY  # type: ignore[misc]
        with pytest.raises(AttributeError):
            lineage.admission = INLINE  # type: ignore[misc]
        with pytest.raises(AttributeError):
            lineage.body_action = FORCED_GET  # type: ignore[misc]
        with pytest.raises(AttributeError):
            lineage.parent = lineage  # type: ignore[misc]

    def test_lineage_setter_validates_type(self):
        request = Request("http://example.com")
        request.lineage = RequestLineage.spawn(None, START)
        request.lineage = None
        with pytest.raises(TypeError):
            request.lineage = "not a lineage"  # type: ignore[assignment]

    def test_values_compare_as_plain_strings(self):
        lineage = RequestLineage.spawn(None, TO_GET, body_action=FORCED_GET)
        assert lineage.generator == "redirect_to_get"
        assert lineage.body_action == "forced_get"
        assert lineage.with_admission(READMIT).admission == "scheduler_readmit"

    def test_copy_and_replace_do_not_share_lineage(self):
        parent_lineage = RequestLineage.spawn(None, START, admission=SCHEDULER)
        request = Request("http://example.com")
        request.lineage = RequestLineage.spawn(parent_lineage, PRESERVE)

        copied = request.copy()
        replaced = request.replace(url="http://example.com/other")

        assert copied.lineage is not request.lineage
        assert replaced.lineage is not request.lineage
        # data is preserved ...
        assert lineage_of(copied).generator == PRESERVE
        assert lineage_of(replaced).generator == PRESERVE
        # ... and the direct parent chain stays linked to the same object
        assert lineage_of(copied).parent is parent_lineage
        assert lineage_of(replaced).parent is parent_lineage

        # changing the visible state of the copy cannot affect the original
        copied.lineage = lineage_of(copied).with_admission(INLINE)
        assert lineage_of(copied).admission == INLINE
        assert lineage_of(request).admission is None
        assert lineage_of(replaced).admission is None

    def test_with_admission_returns_new_object(self):
        lineage = RequestLineage.spawn(None, RETRY)
        admitted = lineage.with_admission(READMIT)
        assert admitted is not lineage
        assert lineage.admission is None
        assert admitted.admission == READMIT
        assert admitted.generator == RETRY

    def test_for_admission_without_existing_lineage(self):
        lineage = RequestLineage.for_admission(None, INLINE)
        assert lineage.generator is None
        assert lineage.body_action is None
        assert lineage.admission == INLINE
        assert lineage.parent is None


class TestRedirectLineage:
    @pytest.mark.parametrize(
        (
            "status",
            "method",
            "body",
            "expected_generator",
            "expected_method",
            "expected_body_action",
        ),
        [
            (301, "POST", b"payload", TO_GET, "GET", FORCED_GET),
            (302, "POST", b"payload", TO_GET, "GET", FORCED_GET),
            (303, "POST", b"payload", TO_GET, "GET", FORCED_GET),
            (303, "PUT", b"payload", TO_GET, "GET", FORCED_GET),
            (303, "DELETE", b"", TO_GET, "GET", FORCED_GET),
            (301, "GET", b"", PRESERVE, "GET", PRESERVED),
            (302, "GET", b"", PRESERVE, "GET", PRESERVED),
            (303, "GET", b"", PRESERVE, "GET", PRESERVED),
            (303, "HEAD", b"", PRESERVE, "HEAD", PRESERVED),
            (302, "HEAD", b"", PRESERVE, "HEAD", PRESERVED),
            (307, "POST", b"payload", PRESERVE, "POST", PRESERVED),
            (308, "POST", b"payload", PRESERVE, "POST", PRESERVED),
            (308, "PUT", b"payload", PRESERVE, "PUT", PRESERVED),
        ],
    )
    def test_redirect_generation(
        self,
        status,
        method,
        body,
        expected_generator,
        expected_method,
        expected_body_action,
    ):
        mw, _ = make_redirect_middleware()
        request = Request(
            "http://example.com/page",
            method=method,
            body=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        request.lineage = RequestLineage.spawn(None, START, admission=SCHEDULER)

        child = mw.process_response(
            request,
            redirect_response(request, "http://example.com/next", status=status),
        )

        assert isinstance(child, Request)
        assert child.method == expected_method
        lineage = lineage_of(child)
        assert lineage.generator == expected_generator
        assert lineage.body_action == expected_body_action
        # stamped at production time, not yet admitted anywhere
        assert lineage.admission is None
        # the direct parent is the request that produced the redirect response
        assert lineage.parent is request.lineage
        if expected_body_action == FORCED_GET:
            assert child.body == b""
            assert "Content-Type" not in child.headers
        else:
            assert child.body == body
            assert child.headers["Content-Type"] == b"application/x-www-form-urlencoded"

    def test_dont_filter_is_inherited_not_forced(self):
        mw, _ = make_redirect_middleware()
        request = Request("http://example.com/page", dont_filter=True)
        child = mw.process_response(
            request, redirect_response(request, "http://example.com/next")
        )
        assert isinstance(child, Request)
        assert child.dont_filter is True

        request = Request("http://example.com/page", dont_filter=False)
        child = mw.process_response(
            request, redirect_response(request, "http://example.com/next")
        )
        assert isinstance(child, Request)
        assert child.dont_filter is False

    def test_redirect_chain_parent_is_direct_previous_generation(self):
        mw, _ = make_redirect_middleware()
        first = Request("http://example.com/a", method="POST", body=b"payload")
        first.lineage = RequestLineage.spawn(None, START, admission=SCHEDULER)

        second = mw.process_response(
            first, redirect_response(first, "http://example.com/b", status=302)
        )
        assert isinstance(second, Request)
        assert lineage_of(second).generator == TO_GET
        assert lineage_of(second).parent is first.lineage

        third_response = Response(
            second.url,
            status=303,
            headers={"Location": "http://example.com/c"},
            request=second,
        )
        third = mw.process_response(second, third_response)
        assert isinstance(third, Request)
        assert lineage_of(third).generator == PRESERVE
        assert lineage_of(third).parent is lineage_of(second)
        assert lineage_of(third).parent is not first.lineage
        # existing redirect ledgers stay intact
        assert third.meta["redirect_times"] == 2
        assert third.meta["redirect_urls"] == [
            "http://example.com/a",
            "http://example.com/b",
        ]

    def test_post_parent_and_forced_get_child_have_different_fingerprints(self):
        mw, _ = make_redirect_middleware()
        parent = Request("http://example.com/same", method="POST", body=b"payload")
        child = mw.process_response(
            parent, redirect_response(parent, "http://example.com/same")
        )
        assert isinstance(child, Request)
        assert child.method == "GET"
        dupefilter = RFPDupeFilter()
        assert dupefilter.request_fingerprint(parent) != dupefilter.request_fingerprint(
            child
        )

    def test_too_many_redirects_raises_ignore_request(self):
        mw, _ = make_redirect_middleware()
        request = Request(
            "http://example.com/page",
            meta={"redirect_times": 30, "redirect_urls": ["http://example.com/page"]},
        )
        request.lineage = RequestLineage.spawn(None, TO_GET, admission=READMIT)
        with pytest.raises(IgnoreRequest):
            mw.process_response(
                request,
                redirect_response(request, "http://example.com/next"),
            )

    def test_prefilled_redirect_times_meta_does_not_create_lineage_alone(self):
        # a plain request pre-filled with redirect ledgers has no lineage
        request = Request(
            "http://example.com/page",
            meta={"redirect_times": 4, "redirect_urls": ["http://example.com/x"]},
        )
        assert request.lineage is None
        assert request.method == "GET"


class TestMetaRefreshLineage:
    def setup_method(self):
        crawler = get_crawler(DefaultSpider)
        crawler.spider = crawler._create_spider()
        self.crawler = crawler
        self.mw = build_from_crawler(MetaRefreshMiddleware, crawler)

    def _response(self, request: Request, interval: int = 0) -> HtmlResponse:
        html = (
            f'<html><head><meta http-equiv="refresh" '
            f'content="{interval};url=http://example.com/new"/></head></html>'
        ).encode()
        return HtmlResponse(request.url, body=html, encoding="utf-8")

    def test_meta_refresh_generation_forces_get(self):
        request = Request(
            "http://example.com/old",
            method="POST",
            body=b"payload",
            headers={"Content-Type": "text/plain", "Content-Length": "7"},
        )
        request.lineage = RequestLineage.spawn(None, START, admission=SCHEDULER)

        child = self.mw.process_response(request, self._response(request))

        assert isinstance(child, Request)
        assert child.method == "GET"
        assert child.body == b""
        lineage = lineage_of(child)
        assert lineage.generator == META_REFRESH
        assert lineage.body_action == FORCED_GET
        assert lineage.admission is None
        assert lineage.parent is request.lineage
        assert "Content-Type" not in child.headers
        assert "Content-Length" not in child.headers

    def test_high_interval_does_not_produce_child(self):
        request = Request("http://example.com/old")
        response = self._response(request, interval=1000)
        assert self.mw.process_response(request, response) is response


class TestRetryLineage:
    def setup_method(self):
        crawler = get_crawler(DefaultSpider)
        crawler.spider = crawler._create_spider()
        self.crawler = crawler
        self.mw = build_from_crawler(RetryMiddleware, crawler)

    def test_retry_generation(self):
        request = Request("http://example.com/page", method="POST", body=b"payload")
        request.lineage = RequestLineage.spawn(None, PRESERVE, admission=READMIT)

        retry = self.mw._retry(request, "unit test")

        assert isinstance(retry, Request)
        assert retry is not request
        lineage = lineage_of(retry)
        assert lineage.generator == RETRY
        assert lineage.body_action == PRESERVED
        assert retry.body == b"payload"
        assert lineage.admission is None
        assert lineage.parent is request.lineage
        assert retry.dont_filter is True
        assert retry.meta["retry_times"] == 1

    def test_retry_of_redirect_child_keeps_redirect_ledger(self):
        redirect_mw = build_from_crawler(RedirectMiddleware, self.crawler)
        post = Request("http://example.com/page", method="POST", body=b"payload")
        post.lineage = RequestLineage.spawn(None, START, admission=SCHEDULER)
        redirected = redirect_mw.process_response(
            post, redirect_response(post, "http://example.com/page", status=302)
        )
        assert isinstance(redirected, Request)
        # the engine re-admits the redirect child before it is downloaded
        redirected.lineage = lineage_of(redirected).with_admission(READMIT)

        retry = self.mw._retry(redirected, "connection lost")

        assert isinstance(retry, Request)
        lineage = lineage_of(retry)
        assert lineage.generator == RETRY
        # the direct parent is the retried redirect child, not the original POST
        assert lineage.parent is lineage_of(redirected)
        assert lineage.parent is not post.lineage
        assert retry.dont_filter is True
        assert retry.meta["retry_times"] == 1
        assert retry.meta["redirect_times"] == redirected.meta["redirect_times"]
        assert retry.meta["redirect_urls"] == redirected.meta["redirect_urls"]

    def test_retry_without_previous_lineage_has_no_parent(self):
        request = Request("http://example.com/page")
        retry = self.mw._retry(request, "unit test")
        assert isinstance(retry, Request)
        assert lineage_of(retry).generator == RETRY
        assert lineage_of(retry).parent is None

    def test_exhausted_retries_return_none(self):
        request = Request("http://example.com/page", meta={"retry_times": 5})
        assert self.mw._retry(request, "unit test") is None

    def test_prefilled_retry_times_meta_is_not_a_retry_generation(self):
        # pre-filling meta alone never marks a request as a retry
        request = Request("http://example.com/page", meta={"retry_times": 3})
        assert request.lineage is None


def _start_lineaged_request(url: str, **kwargs: Any) -> Request:
    request = Request(url, **kwargs)
    request.lineage = RequestLineage.spawn(None, START, admission=SCHEDULER)
    return request


async def _open_engine(
    settings_dict: dict[str, Any] | None = None,
) -> tuple[Crawler, ExecutionEngine]:
    crawler = get_crawler(DefaultSpider, settings_dict or {})
    engine = ExecutionEngine(crawler, lambda _: None)
    # register the engine the same way Crawler.crawl_async() does, so that
    # components accessing crawler.engine (e.g. the priority queue) work
    crawler.engine = engine
    crawler.spider = crawler._create_spider()
    await engine.open_spider_async()
    return crawler, engine


class TestSchedulerAdmission:
    @coroutine_test
    async def test_readmit_goes_through_dupefilter(self):
        crawler, engine = await _open_engine()
        try:
            redirect_mw = build_from_crawler(RedirectMiddleware, crawler)
            post = _start_lineaged_request(
                "http://example.com/r", method="POST", body=b"abc"
            )
            response = redirect_response(post, "http://example.com/r", status=302)
            child = redirect_mw.process_response(post, response)

            # simulate the POST parent fingerprint already being known
            assert engine._slot is not None
            scheduler = cast("Any", engine._slot.scheduler)
            assert scheduler.df.request_seen(post) is False

            dropped: list[Request] = []

            def collect_dropped(request, **kwargs):
                dropped.append(request)

            crawler.signals.connect(collect_dropped, signals.request_dropped)

            await maybe_deferred_to_future(
                engine._handle_downloader_output(child, post)
            )

            # the forced-GET child has a different fingerprint and is admitted
            # through the scheduler despite the parent fingerprint being known
            popped = scheduler.next_request()
            assert popped is child
            assert isinstance(child, Request)
            lineage = lineage_of(child)
            assert lineage.admission == READMIT
            assert lineage.generator == TO_GET
            assert lineage.body_action == FORCED_GET
            assert lineage.parent is post.lineage
            assert dropped == []

            # an identical child is now seen by the dupefilter and dropped ...
            child2 = redirect_mw.process_response(post, response)
            await maybe_deferred_to_future(
                engine._handle_downloader_output(child2, post)
            )
            assert scheduler.next_request() is None
            assert dropped == [child2]
            assert isinstance(child2, Request)
            # ... but the dropped object still carries its full lineage
            lineage2 = lineage_of(child2)
            assert lineage2.admission == READMIT
            assert lineage2.generator == TO_GET
            assert lineage2.parent is not None
        finally:
            await engine.close_async()


class InlineFollowMiddleware:
    """Turns the response of one data: request into another data: request
    carrying a redirect-like lineage, as a status-code redirect middleware
    would."""

    emitted: list[Request] = []

    def process_response(self, request, response, spider=None):
        if request.url == "data:text/html,hop0":
            subrequest = Request("data:text/html,hop1")
            subrequest.lineage = RequestLineage.spawn(request.lineage, PRESERVE)
            type(self).emitted.append(subrequest)
            return subrequest
        return response


class RawInlineFollowMiddleware:
    """Like InlineFollowMiddleware but emits a request with no lineage."""

    emitted: list[Request] = []

    def process_response(self, request, response, spider=None):
        if request.url == "data:text/html,hop0":
            subrequest = Request("data:text/html,hop1")
            type(self).emitted.append(subrequest)
            return subrequest
        return response


class TestInlineFollowAdmission:
    @coroutine_test
    async def test_inline_follow_does_not_schedule_and_ignores_dupefilter(self):
        InlineFollowMiddleware.emitted = []
        crawler, engine = await _open_engine(
            {
                "DOWNLOADER_MIDDLEWARES": {
                    InlineFollowMiddleware: 650,
                }
            }
        )
        try:
            assert engine._slot is not None
            scheduler = cast("Any", engine._slot.scheduler)
            # the spider dupefilter already knows the follow-up URL
            assert scheduler.df.request_seen(Request("data:text/html,hop1")) is False
            enqueued_before = crawler.stats.get_value("scheduler/enqueued", 0)

            response = await engine.download_async(
                _start_lineaged_request("data:text/html,hop0")
            )

            assert "hop1" in response.url
            assert len(InlineFollowMiddleware.emitted) == 1
            follow = InlineFollowMiddleware.emitted[0]
            lineage = lineage_of(follow)
            assert lineage.admission == INLINE
            # production information from the (redirect-like) middleware stays
            assert lineage.generator == PRESERVE
            assert lineage.body_action == PRESERVED
            # nothing entered the scheduler
            assert crawler.stats.get_value("scheduler/enqueued", 0) == enqueued_before
            assert scheduler.next_request() is None
        finally:
            await engine.close_async()

    @coroutine_test
    async def test_inline_follow_without_previous_lineage(self):
        RawInlineFollowMiddleware.emitted = []
        _crawler, engine = await _open_engine(
            {
                "DOWNLOADER_MIDDLEWARES": {
                    RawInlineFollowMiddleware: 650,
                }
            }
        )
        try:
            response = await engine.download_async(Request("data:text/html,hop0"))
            assert "hop1" in response.url
            follow = RawInlineFollowMiddleware.emitted[0]
            lineage = lineage_of(follow)
            assert lineage.admission == INLINE
            assert lineage.generator is None
        finally:
            await engine.close_async()


class LineageCrawlSpider(Spider):
    name = "lineage_crawl"
    start_urls = ["data:text/html,start"]

    async def parse(self, response, **kwargs):
        # only the start response produces output; child responses return here
        # again via the default callback and must not yield anything
        if response.url != "data:text/html,start":
            return
        # pre-filled ledgers must not make these look like retries/redirects
        yield Request(
            "data:text/html,child1",
            meta={"retry_times": 5, "redirect_times": 5},
        )
        yield Request("data:text/html,child2")


class TestStartAndCallbackGenerations:
    @coroutine_test
    async def test_start_and_callback_lineages(self):
        crawler = get_crawler(LineageCrawlSpider)
        scheduled: list[Request] = []

        def collect_scheduled(request, **kwargs):
            scheduled.append(request)

        crawler.signals.connect(collect_scheduled, signals.request_scheduled)

        await crawler.crawl_async()

        assert len(scheduled) == 3
        by_url = {request.url: request for request in scheduled}

        start_request = by_url["data:text/html,start"]
        start_lineage = lineage_of(start_request)
        assert start_lineage.generator == START
        assert start_lineage.admission == SCHEDULER
        assert start_lineage.body_action == PRESERVED
        assert start_lineage.parent is None
        assert start_request.dont_filter is True

        for url in ("data:text/html,child1", "data:text/html,child2"):
            child = by_url[url]
            lineage = lineage_of(child)
            assert lineage.generator == CALLBACK
            assert lineage.admission == SCHEDULER
            assert lineage.body_action == PRESERVED
            assert lineage.parent is start_lineage

        assert crawler.stats.get_value("item_scraped_count", 0) == 0
        assert crawler.stats.get_value("response_received_count") == 3
        assert crawler.stats.get_value("finish_reason") == "finished"


class SchedulerReadmitSpiderMiddleware:
    """A downloader middleware that emits a follow-up request without going
    through the redirect middleware (no 3xx, data: scheme)."""

    emitted: list[Request] = []

    def process_response(self, request, response, spider=None):
        if request.url == "data:text/html,start":
            follow = Request("data:text/html,followed")
            type(self).emitted.append(follow)
            return follow
        return response


class ReadmitCrawlSpider(Spider):
    name = "readmit_crawl"
    start_urls = ["data:text/html,start"]
    custom_settings = {
        "DOWNLOADER_MIDDLEWARES": {SchedulerReadmitSpiderMiddleware: 650}
    }
    seen_urls: list[str] = []
    followed_request: Request | None = None

    async def parse(self, response, **kwargs):
        type(self).seen_urls.append(response.url)
        if "followed" in response.url:
            type(self).followed_request = response.request
            yield Request("data:text/html,grand")


class TestMiddlewareReadmitInCrawl:
    @coroutine_test
    async def test_readmit_then_callback_parent(self):
        SchedulerReadmitSpiderMiddleware.emitted = []
        ReadmitCrawlSpider.seen_urls = []
        ReadmitCrawlSpider.followed_request = None
        crawler = get_crawler(ReadmitCrawlSpider)
        scheduled: list[Request] = []

        def collect_scheduled(request, **kwargs):
            scheduled.append(request)

        crawler.signals.connect(collect_scheduled, signals.request_scheduled)

        await crawler.crawl_async()

        follow = SchedulerReadmitSpiderMiddleware.emitted[0]
        follow_lineage = lineage_of(follow)
        assert follow_lineage.admission == READMIT
        # a third-party middleware request with no production lineage
        assert follow_lineage.generator is None

        by_url = {request.url: request for request in scheduled}
        grand = by_url["data:text/html,grand"]
        grand_lineage = lineage_of(grand)
        assert grand_lineage.generator == CALLBACK
        # direct parent is the re-admitted follow-up response.request,
        # which may itself already be a child generation
        followed_request = ReadmitCrawlSpider.followed_request
        assert followed_request is not None
        assert grand_lineage.parent is lineage_of(followed_request)
        assert grand_lineage.parent is follow_lineage
