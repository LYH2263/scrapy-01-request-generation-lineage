"""
This module implements the Request class which is used to represent HTTP
requests in Scrapy.

See documentation in docs/topics/request-response.rst
"""

from __future__ import annotations

import inspect
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Concatenate,
    NoReturn,
    TypeAlias,
    TypedDict,
    TypeVar,
    overload,
)

from w3lib.url import safe_url_string

# a workaround for the docs "more than one target found" problem
import scrapy  # noqa: TC001
from scrapy.http.headers import Headers
from scrapy.utils.curl import curl_to_request_kwargs
from scrapy.utils.python import to_bytes
from scrapy.utils.trackref import object_ref

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from twisted.python.failure import Failure

    # typing.NotRequired and typing.Self require Python 3.11
    from typing_extensions import NotRequired, Self

    # circular import
    from scrapy.http import Response

    CallbackT: TypeAlias = Callable[Concatenate[Response, ...], Any]


class VerboseCookie(TypedDict):
    name: str | bytes
    value: str | bytes | bool | float | int
    domain: NotRequired[str | bytes]
    path: NotRequired[str | bytes]
    secure: NotRequired[bool]


CookiesT: TypeAlias = (
    dict[str | bytes, str | bytes | bool | float | int] | list[VerboseCookie]
)


RequestTypeVar = TypeVar("RequestTypeVar", bound="Request")


class LineageGenerator(str, Enum):
    """How a :class:`Request` carrying a :class:`RequestLineage` was produced.

    The value is assigned at the production point, never inferred later from
    the request method, body or meta keys such as ``retry_times``.
    """

    #: yielded by ``Spider.start()`` (or a spider middleware ``process_start()``)
    START = "start"
    #: yielded by a spider callback/errback (or ``process_spider_output()``)
    CALLBACK = "callback"
    #: returned by :func:`scrapy.downloadermiddlewares.retry.get_retry_request`
    RETRY = "retry"
    #: redirect child that keeps the parent method and body (307/308, and
    #: 301/302/303 when the method was not changed)
    REDIRECT_PRESERVE = "redirect_preserve"
    #: redirect child forced to ``GET`` with an empty body (POST on 301/302,
    #: or a non-GET/HEAD method on 303)
    REDIRECT_TO_GET = "redirect_to_get"
    #: child produced by the meta refresh downloader middleware
    META_REFRESH = "meta_refresh"


class LineageAdmission(str, Enum):
    """How a request was admitted into the download pipeline."""

    #: went through ``ExecutionEngine.crawl()`` from ``Spider.start()`` or a
    #: spider callback
    SCHEDULER = "scheduler"
    #: a downloader middleware output request, re-admitted through
    #: ``ExecutionEngine._handle_downloader_output`` (dupefilter still applies)
    SCHEDULER_READMIT = "scheduler_readmit"
    #: a downloader middleware output request followed inline by
    #: ``ExecutionEngine.download_async()`` (never scheduled)
    INLINE_FOLLOW = "inline_follow"


class LineageBodyAction(str, Enum):
    """What happened to the request body at the production point."""

    #: method and body come from the parent request unchanged
    PRESERVED = "preserved"
    #: the request was forced to ``GET`` and its body was emptied
    FORCED_GET = "forced_get"


class RequestLineage:
    """Read-only record of how a :class:`~scrapy.Request` was generated.

    Every request that travels through Scrapy's crawl pipeline gets a lineage
    attached at the point where it is produced (``Spider.start()``, a spider
    callback, :func:`~scrapy.downloadermiddlewares.retry.get_retry_request`,
    the redirect or meta refresh middlewares) and an :attr:`admission`
    recorded at the engine boundary that admits it.

    Lineage objects are immutable: admitting a request replaces the
    :attr:`Request.lineage` reference with a new object instead of mutating
    the existing one. The :attr:`parent` is the lineage of the request from
    which this request was directly produced, or ``None`` when there is no
    parent (e.g. start requests). It points at the direct previous
    generation, not at the beginning of the chain.

    Requests built outside of the crawl pipeline (e.g. the initial request of
    :meth:`~scrapy.core.engine.ExecutionEngine.download_async`) may carry no
    lineage at all, in which case :attr:`Request.lineage` is ``None``.
    """

    __slots__ = ("_admission", "_body_action", "_generator", "_parent")

    def __init__(
        self,
        generator: LineageGenerator | None,
        admission: LineageAdmission | None,
        body_action: LineageBodyAction | None,
        parent: RequestLineage | None = None,
    ) -> None:
        if generator is not None and not isinstance(generator, LineageGenerator):
            raise TypeError(
                f"generator must be a LineageGenerator member, got {generator!r}"
            )
        if admission is not None and not isinstance(admission, LineageAdmission):
            raise TypeError(
                f"admission must be a LineageAdmission member, got {admission!r}"
            )
        if body_action is not None and not isinstance(body_action, LineageBodyAction):
            raise TypeError(
                f"body_action must be a LineageBodyAction member, got {body_action!r}"
            )
        if parent is not None and not isinstance(parent, RequestLineage):
            raise TypeError(f"parent must be a RequestLineage, got {parent!r}")
        self._generator: LineageGenerator | None = generator
        self._admission: LineageAdmission | None = admission
        self._body_action: LineageBodyAction | None = body_action
        self._parent: RequestLineage | None = parent

    @classmethod
    def spawn(
        cls,
        parent: RequestLineage | None,
        generator: LineageGenerator,
        *,
        body_action: LineageBodyAction = LineageBodyAction.PRESERVED,
        admission: LineageAdmission | None = None,
    ) -> RequestLineage:
        """Create a child generation with *parent* as its direct parent."""
        return cls(generator, admission, body_action, parent)

    @classmethod
    def for_admission(
        cls,
        lineage: RequestLineage | None,
        admission: LineageAdmission,
    ) -> RequestLineage:
        """Return a lineage carrying *admission*.

        If *lineage* is ``None`` (a request minted by a component that does
        not assign lineage), a bare lineage with unknown production
        information is returned.
        """
        if lineage is None:
            return cls(None, admission, None, None)
        return lineage.with_admission(admission)

    @property
    def generator(self) -> LineageGenerator | None:
        """The production point that created the request."""
        return self._generator

    @property
    def admission(self) -> LineageAdmission | None:
        """The admission path taken by the request, or ``None`` before it has
        been admitted."""
        return self._admission

    @property
    def body_action(self) -> LineageBodyAction | None:
        """Whether the body was preserved or the request was forced to GET."""
        return self._body_action

    @property
    def parent(self) -> RequestLineage | None:
        """The lineage of the directly previous generation, or ``None``."""
        return self._parent

    def with_admission(self, admission: LineageAdmission) -> RequestLineage:
        """Return a copy of this lineage with a different admission."""
        return RequestLineage(
            self._generator, admission, self._body_action, self._parent
        )

    def duplicate(self) -> RequestLineage:
        """Return a distinct object with the same data.

        Used by :meth:`Request.copy` and :meth:`Request.replace` so that a
        copy never shares the lineage object of the original request. The
        parent chain is shared because lineage objects are immutable.
        """
        return RequestLineage(
            self._generator, self._admission, self._body_action, self._parent
        )

    def __repr__(self) -> str:
        parts = [
            f"generator={self._generator.value if self._generator else None!r}",
            f"admission={self._admission.value if self._admission else None!r}",
            f"body_action={self._body_action.value if self._body_action else None!r}",
            f"parent={'yes' if self._parent is not None else 'no'}",
        ]
        return f"<RequestLineage {' '.join(parts)}>"


def NO_CALLBACK(*args: Any, **kwargs: Any) -> NoReturn:
    """When assigned to the ``callback`` parameter of
    :class:`~scrapy.Request`, it indicates that the request is not meant
    to have a spider callback at all.

    For example:

    .. code-block:: python

       Request("https://example.com", callback=NO_CALLBACK)

    This value should be used by :ref:`components <topics-components>` that
    create and handle their own requests, e.g. through
    :meth:`scrapy.core.engine.ExecutionEngine.download`, so that downloader
    middlewares handling such requests can treat them differently from requests
    intended for the :meth:`~scrapy.Spider.parse` callback.
    """
    raise RuntimeError(
        "The NO_CALLBACK callback has been called. This is a special callback "
        "value intended for requests whose callback is never meant to be "
        "called."
    )


class Request(object_ref):
    """Represents an HTTP request, which is usually generated in a Spider and
    executed by the Downloader, thus generating a :class:`~scrapy.http.Response`.
    """

    __attrs_and_slots = ("callback", "dont_filter", "errback", "method", "priority")
    attributes: tuple[str, ...] = (
        "url",
        "headers",
        "body",
        "cookies",
        "meta",
        "encoding",
        "flags",
        "cb_kwargs",
        *__attrs_and_slots,
    )
    """A tuple of :class:`str` objects containing the name of all public
    attributes of the class that are also keyword parameters of the
    ``__init__()`` method.

    Currently used by :meth:`.Request.replace`, :meth:`.Request.to_dict` and
    :func:`~scrapy.utils.request.request_from_dict`.
    """

    __slots__ = (
        "__weakref__",
        "_body",
        "_cb_kwargs",
        "_cookies",
        "_encoding",
        "_flags",
        "_headers",
        "_lineage",
        "_meta",
        "_url",
        *__attrs_and_slots,
    )
    del __attrs_and_slots

    def __init__(
        self,
        url: str,
        callback: CallbackT | None = None,
        method: str = "GET",
        headers: Mapping[str, Any]
        | Mapping[bytes, Any]
        | Iterable[tuple[str | bytes, Any]]
        | None = None,
        body: bytes | str | None = None,
        cookies: CookiesT | None = None,
        meta: dict[str, Any] | None = None,
        encoding: str = "utf-8",
        priority: int = 0,
        dont_filter: bool = False,
        errback: Callable[[Failure], Any] | None = None,
        flags: list[str] | None = None,
        cb_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._encoding: str = encoding  # this one has to be set first
        self.method: str = str(method).upper()
        self._meta: dict[str, Any] | None = dict(meta) if meta else None
        self._set_url(url)
        self._set_body(body)
        if not isinstance(priority, int):
            raise TypeError(f"Request priority not an integer: {priority!r}")

        #: Default: ``0``
        #:
        #: Value that the :ref:`scheduler <topics-scheduler>` may use for
        #: request prioritization.
        #:
        #: Built-in schedulers prioritize requests with a higher priority
        #: value.
        #:
        #: Negative values are allowed.
        self.priority: int = priority

        if not (callable(callback) or callback is None):
            raise TypeError(
                f"callback must be a callable, got {type(callback).__name__}"
            )
        if not (callable(errback) or errback is None):
            raise TypeError(f"errback must be a callable, got {type(errback).__name__}")

        #: :class:`~collections.abc.Callable` to parse the
        #: :class:`~scrapy.http.Response` to this request once received.
        #:
        #: The callable must expect the response as its first parameter, and
        #: support any additional keyword arguments set through
        #: :attr:`cb_kwargs`. See :ref:`writing-callbacks` and
        #: :ref:`callback-output`.
        #:
        #: In addition to an arbitrary callable, the following values are also
        #: supported:
        #:
        #: -   ``None`` (default), which indicates that the
        #:     :meth:`~scrapy.Spider.parse` method of the spider must be used.
        #:
        #: -   :func:`~scrapy.http.request.NO_CALLBACK`.
        #:
        #: If an unhandled exception is raised during request or response
        #: processing, i.e. by a :ref:`spider middleware
        #: <topics-spider-middleware>`, :ref:`downloader middleware
        #: <topics-downloader-middleware>` or download handler
        #: (:setting:`DOWNLOAD_HANDLERS`), :attr:`errback` is called instead.
        #:
        #: .. tip::
        #:     :class:`~scrapy.spidermiddlewares.httperror.HttpErrorMiddleware`
        #:     raises exceptions for non-2xx responses by default, sending them
        #:     to the :attr:`errback` instead.
        #:
        #: .. seealso:: :ref:`callbacks`
        self.callback: CallbackT | None = callback

        #: :class:`~collections.abc.Callable` to handle exceptions raised
        #: during request or response processing.
        #:
        #: The callable must expect a :exc:`~twisted.python.failure.Failure` as
        #: its first parameter.
        #:
        #: .. seealso:: :ref:`errbacks`
        self.errback: Callable[[Failure], Any] | None = errback

        self._cookies: CookiesT | None = cookies or None
        self._headers: Headers | None = (
            Headers(headers, encoding=encoding) if headers else None
        )

        #: Whether this request may be filtered out by :ref:`components
        #: <topics-components>` that support filtering out requests (``False``,
        #: default), or those components should not filter out this request
        #: (``True``).
        #:
        #: The following built-in components check this attribute:
        #:
        #: -   The :ref:`scheduler <topics-scheduler>` uses it to skip
        #:     duplicate request filtering (see
        #:     :setting:`DUPEFILTER_CLASS`). When set to ``True``, the
        #:     request is not checked against the duplicate filter,
        #:     allowing requests that would otherwise be considered duplicates
        #:     to be scheduled multiple times.
        #: -   :class:`~scrapy.downloadermiddlewares.offsite.OffsiteMiddleware`
        #:     uses it to allow requests to domains not in
        #:     :attr:`~scrapy.Spider.allowed_domains`. To skip only the offsite
        #:     filter without affecting other components, consider using the
        #:     :reqmeta:`allow_offsite` request meta key instead.
        #:
        #: Third-party components may also use this attribute to decide whether
        #: to filter out a request.
        #:
        #: When defining the start URLs of a spider through
        #: :attr:`~scrapy.Spider.start_urls`, this attribute is enabled by
        #: default. See :meth:`~scrapy.Spider.start`.
        self.dont_filter: bool = dont_filter

        self._cb_kwargs: dict[str, Any] | None = dict(cb_kwargs) if cb_kwargs else None
        self._flags: list[str] | None = list(flags) if flags else None
        self._lineage: RequestLineage | None = None

    @property
    def cb_kwargs(self) -> dict[str, Any]:
        if self._cb_kwargs is None:
            self._cb_kwargs = {}
        return self._cb_kwargs

    @property
    def meta(self) -> dict[str, Any]:
        if self._meta is None:
            self._meta = {}
        return self._meta

    @property
    def lineage(self) -> RequestLineage | None:
        """The :class:`RequestLineage` of this request, or ``None`` if it has
        not been produced through the crawl pipeline yet.

        It is assigned by the framework at the request production point
        (``Spider.start()``, spider callbacks, the retry and redirect
        downloader middlewares) and is not part of the request constructor
        arguments.
        """
        return self._lineage

    @lineage.setter
    def lineage(self, value: RequestLineage | None) -> None:
        if value is not None and not isinstance(value, RequestLineage):
            raise TypeError(
                f"lineage must be a RequestLineage or None, got {type(value).__name__}"
            )
        self._lineage = value

    @property
    def url(self) -> str:
        return self._url

    def _url_is_verbatim(self) -> bool:
        return bool(self._meta and self._meta.get("verbatim_url"))

    def _set_url(self, url: str) -> None:
        if not isinstance(url, str):
            raise TypeError(f"Request url must be str, got {type(url).__name__}")

        if self._url_is_verbatim():
            self._url = url
        else:
            self._url = safe_url_string(url, self.encoding)

        if (
            "://" not in self._url
            and not self._url.startswith("about:")
            and not self._url.startswith("data:")
        ):
            raise ValueError(f"Missing scheme in request url: {self._url}")

    @property
    def body(self) -> bytes:
        return self._body

    def _set_body(self, body: str | bytes | None) -> None:
        self._body = b"" if not body else to_bytes(body, self.encoding)

    @property
    def encoding(self) -> str:
        return self._encoding

    @property
    def flags(self) -> list[str]:
        if self._flags is None:
            self._flags = []
        return self._flags

    @flags.setter
    def flags(self, value: list[str] | None) -> None:
        self._flags = value

    @property
    def cookies(self) -> CookiesT:
        if self._cookies is None:
            self._cookies = {}
        return self._cookies

    @cookies.setter
    def cookies(self, value: CookiesT | None) -> None:
        self._cookies = value

    @property
    def headers(self) -> Headers:
        if self._headers is None:
            self._headers = Headers(encoding=self.encoding)
        return self._headers

    @headers.setter
    def headers(
        self,
        value: Mapping[str, Any]
        | Mapping[bytes, Any]
        | Iterable[tuple[str | bytes, Any]]
        | None,
    ) -> None:
        if isinstance(value, Headers):
            self._headers = value
        else:
            self._headers = (
                Headers(value, encoding=self.encoding) if value is not None else None
            )

    def __repr__(self) -> str:
        return f"<{self.method} {self.url}>"

    def copy(self) -> Self:
        return self.replace()

    @overload
    def replace(
        self, *args: Any, cls: type[RequestTypeVar], **kwargs: Any
    ) -> RequestTypeVar: ...

    @overload
    def replace(self, *args: Any, cls: None = None, **kwargs: Any) -> Self: ...

    def replace(
        self, *args: Any, cls: type[Request] | None = None, **kwargs: Any
    ) -> Request:
        """Create a new Request with the same attributes except for those given new values"""
        for x in self.attributes:
            kwargs.setdefault(x, getattr(self, x))
        if cls is None:
            cls = self.__class__
        new_request = cls(*args, **kwargs)
        # The copy gets its own lineage object so that changing its visible
        # state (e.g. its admission) cannot affect the original request.
        if self._lineage is not None:
            new_request._lineage = self._lineage.duplicate()
        return new_request

    @classmethod
    def from_curl(
        cls,
        curl_command: str,
        ignore_unknown_options: bool = True,
        **kwargs: Any,
    ) -> Self:
        """Create a Request object from a string containing a `cURL
        <https://curl.se/>`_ command. It populates the HTTP method, the
        URL, the headers, the cookies and the body. It accepts the same
        arguments as the :class:`Request` class, taking preference and
        overriding the values of the same arguments contained in the cURL
        command.

        Unrecognized options are ignored by default. To raise an error when
        finding unknown options call this method by passing
        ``ignore_unknown_options=False``.

        .. caution:: Using :meth:`from_curl` from :class:`~scrapy.Request`
                     subclasses, such as :class:`~scrapy.http.JsonRequest`, or
                     :class:`~scrapy.http.XmlRpcRequest`, as well as having
                     :ref:`downloader middlewares <topics-downloader-middleware>`
                     and
                     :ref:`spider middlewares <topics-spider-middleware>`
                     enabled, such as
                     :class:`~scrapy.downloadermiddlewares.defaultheaders.DefaultHeadersMiddleware`,
                     :class:`~scrapy.downloadermiddlewares.useragent.UserAgentMiddleware`,
                     or
                     :class:`~scrapy.downloadermiddlewares.httpcompression.HttpCompressionMiddleware`,
                     may modify the :class:`~scrapy.Request` object.

        To translate a cURL command into a Scrapy request,
        you may use `curl2scrapy <https://michael-shub.github.io/curl2scrapy/>`_.
        """
        request_kwargs = curl_to_request_kwargs(curl_command, ignore_unknown_options)
        request_kwargs.update(kwargs)
        return cls(**request_kwargs)

    def to_curl(self) -> str:
        """Return a string with a `cURL <https://curl.se/>`_ command equivalent
        to this request.

        Inverse of :meth:`from_curl`. See also
        :func:`scrapy.utils.request.request_to_curl`.

        .. versionadded:: 2.18.0
        """
        # Imported here to avoid a circular import.
        from scrapy.utils.request import request_to_curl  # noqa: PLC0415

        return request_to_curl(self)

    def to_dict(self, *, spider: scrapy.Spider | None = None) -> dict[str, Any]:
        """Return a dictionary containing the Request's data.

        Use :func:`~scrapy.utils.request.request_from_dict` to convert back into a :class:`~scrapy.Request` object.

        If a spider is given, this method will try to find out the name of the spider methods used as callback
        and errback and include them in the output dict, raising an exception if they cannot be found.
        """
        d = {
            "url": self.url,  # urls are safe (safe_string_url)
            "callback": (
                _find_method(spider, self.callback)
                if callable(self.callback)
                else self.callback
            ),
            "errback": (
                _find_method(spider, self.errback)
                if callable(self.errback)
                else self.errback
            ),
            "headers": dict(self.headers),
        }
        for attr in self.attributes:
            d.setdefault(attr, getattr(self, attr))
        if type(self) is not Request:  # pylint: disable=unidiomatic-typecheck
            d["_class"] = self.__module__ + "." + self.__class__.__name__
        return d


def _find_method(obj: Any, func: Callable[..., Any]) -> str:
    """Helper function for Request.to_dict"""
    # Only instance methods contain ``__func__``
    if obj and hasattr(func, "__func__"):
        members = inspect.getmembers(obj, predicate=inspect.ismethod)
        for name, obj_func in members:
            # We need to use __func__ to access the original function object because instance
            # method objects are generated each time attribute is retrieved from instance.
            #
            # Reference: The standard type hierarchy
            # https://docs.python.org/3/reference/datamodel.html
            if obj_func.__func__ is func.__func__:
                return name
    raise ValueError(f"Function {func} is not an instance method in: {obj}")
