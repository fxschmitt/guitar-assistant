"""Unit tests for guitar_assistant.wikipedia_client."""

from collections.abc import Callable
from typing import Any, Final

import httpx
import pytest

from guitar_assistant.wikipedia_client import (
    API_URL,
    ArticleNotFoundError,
    RequestLimitExceededError,
    WikipediaClient,
    default_user_agent,
)

_TEST_USER_AGENT: Final = "guitar-assistant-tests/0.1 (test@example.com)"
_TEST_REVISION_ID: Final = 1370144642


def _category_members_response(
    titles_and_kinds: list[tuple[str, bool]], *, continue_token: str | None = None
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "query": {
            "categorymembers": [
                {"title": title, "ns": 14 if is_subcategory else 0}
                for title, is_subcategory in titles_and_kinds
            ]
        }
    }
    if continue_token is not None:
        body["continue"] = {"cmcontinue": continue_token}
    return body


def _revisions_response(title: str, wikitext: str) -> dict[str, Any]:
    return {
        "query": {
            "pages": {
                "123": {
                    "title": title,
                    "revisions": [{"revid": _TEST_REVISION_ID, "slots": {"main": {"*": wikitext}}}],
                }
            }
        }
    }


def _missing_page_response(title: str) -> dict[str, Any]:
    return {"query": {"pages": {"-1": {"title": title, "missing": ""}}}}


def _client_with_handler(
    handler: Callable[[httpx.Request], httpx.Response], **kwargs
) -> WikipediaClient:
    return WikipediaClient(_TEST_USER_AGENT, transport=httpx.MockTransport(handler), **kwargs)


def test_walk_category_yields_articles_in_the_root_category():
    # GIVEN a category containing two articles and no subcategories
    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_category_members_response(
                [("Fender Stratocaster", False), ("Fender Telecaster", False)]
            ),
        )

    client = _client_with_handler(handle)
    # WHEN the category is walked
    titles = list(client.walk_category("Category:Electric guitars"))
    # THEN both articles are yielded
    assert titles == ["Fender Stratocaster", "Fender Telecaster"]


def test_walk_category_descends_into_subcategories_within_max_depth():
    # GIVEN a root category with one subcategory, which itself contains one article
    def handle(request: httpx.Request) -> httpx.Response:
        category = request.url.params["cmtitle"]
        if category == "Category:Electric guitars":
            return httpx.Response(
                200,
                json=_category_members_response([("Category:Fender electric guitars", True)]),
            )
        assert category == "Category:Fender electric guitars"
        return httpx.Response(200, json=_category_members_response([("Fender Jazzmaster", False)]))

    client = _client_with_handler(handle)
    # WHEN the category is walked with enough depth to reach the subcategory
    titles = list(client.walk_category("Category:Electric guitars", max_depth=1))
    # THEN the article inside the subcategory is yielded
    assert titles == ["Fender Jazzmaster"]


def test_walk_category_does_not_descend_past_max_depth():
    # GIVEN a root category with one subcategory
    def handle(request: httpx.Request) -> httpx.Response:
        category = request.url.params["cmtitle"]
        assert category == "Category:Electric guitars"
        return httpx.Response(
            200,
            json=_category_members_response([("Category:Fender electric guitars", True)]),
        )

    client = _client_with_handler(handle)
    # WHEN the category is walked with zero depth
    titles = list(client.walk_category("Category:Electric guitars", max_depth=0))
    # THEN no articles are yielded, since the only member found is a subcategory
    assert not titles


def test_walk_category_follows_pagination_continue_token():
    # GIVEN a category whose members are split across two pages
    call_count = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if "cmcontinue" not in request.url.params:
            return httpx.Response(
                200,
                json=_category_members_response(
                    [("Fender Stratocaster", False)], continue_token="next-page"
                ),
            )
        assert request.url.params["cmcontinue"] == "next-page"
        return httpx.Response(200, json=_category_members_response([("Fender Telecaster", False)]))

    client = _client_with_handler(handle)
    # WHEN the category is walked
    titles = list(client.walk_category("Category:Electric guitars"))
    # THEN both pages are fetched and all articles are yielded
    assert call_count == 2
    assert titles == ["Fender Stratocaster", "Fender Telecaster"]


def test_walk_category_does_not_revisit_an_already_seen_subcategory():
    # GIVEN two subcategories that both link back to the same shared subcategory
    def handle(request: httpx.Request) -> httpx.Response:
        category = request.url.params["cmtitle"]
        if category == "Category:Electric guitars":
            return httpx.Response(
                200,
                json=_category_members_response([("Category:A", True), ("Category:B", True)]),
            )
        if category in ("Category:A", "Category:B"):
            return httpx.Response(200, json=_category_members_response([("Category:Shared", True)]))
        assert category == "Category:Shared"
        return httpx.Response(200, json=_category_members_response([("Fender Jaguar", False)]))

    client = _client_with_handler(handle, max_requests=10)
    # WHEN the category is walked deep enough to reach the shared subcategory
    titles = list(client.walk_category("Category:Electric guitars", max_depth=2))
    # THEN the shared subcategory's article is only yielded once
    assert titles == ["Fender Jaguar"]


def test_fetch_wikitext_returns_the_article_content():
    # GIVEN an article with known wikitext content
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.params["titles"] == "Fender Stratocaster"
        return httpx.Response(
            200, json=_revisions_response("Fender Stratocaster", "{{Infobox musical instrument}}")
        )

    client = _client_with_handler(handle)
    # WHEN the article's wikitext is fetched
    article = client.fetch_wikitext("Fender Stratocaster")
    # THEN the raw wikitext content and revision ID are returned
    assert article.wikitext == "{{Infobox musical instrument}}"
    assert article.revision_id == _TEST_REVISION_ID


def test_fetch_wikitext_raises_article_not_found_for_a_missing_title():
    # GIVEN a title with no corresponding Wikipedia article
    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_missing_page_response("Not A Real Guitar"))

    client = _client_with_handler(handle)
    # WHEN its wikitext is fetched
    # THEN a clear error is raised instead of an opaque KeyError
    with pytest.raises(ArticleNotFoundError, match="Not A Real Guitar"):
        client.fetch_wikitext("Not A Real Guitar")


def test_requests_beyond_the_budget_raise_request_limit_exceeded_error():
    # GIVEN a client with a budget of a single request
    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_category_members_response([("Category:A", True)]))

    client = _client_with_handler(handle, max_requests=1)
    # WHEN the category is walked deep enough to require a second request
    # THEN the request budget is exhausted before completing the walk
    with pytest.raises(RequestLimitExceededError):
        list(client.walk_category("Category:Electric guitars", max_depth=1))


def test_requests_send_the_configured_user_agent_header():
    # GIVEN a client configured with an identifying User-Agent
    captured_requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(200, json=_category_members_response([]))

    client = _client_with_handler(handle)
    # WHEN a request is made
    list(client.walk_category("Category:Electric guitars"))
    # THEN the request carries the configured User-Agent header
    assert captured_requests[0].headers["User-Agent"] == _TEST_USER_AGENT
    assert captured_requests[0].url.copy_with(query=None) == httpx.URL(API_URL)


def test_context_manager_closes_the_underlying_http_client():
    # GIVEN a client used as a context manager
    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_category_members_response([]))

    # WHEN the context manager exits
    with _client_with_handler(handle) as client:
        pass
    # THEN the underlying HTTP client is closed
    assert client._client.is_closed


def test_default_user_agent_includes_the_configured_contact_email(monkeypatch: pytest.MonkeyPatch):
    # GIVEN a configured WIKIPEDIA_CONTACT_EMAIL
    monkeypatch.setenv("WIKIPEDIA_CONTACT_EMAIL", "test@example.com")
    # WHEN the default User-Agent is built
    user_agent = default_user_agent()
    # THEN it includes the configured contact email
    assert "test@example.com" in user_agent


def test_default_user_agent_raises_when_contact_email_is_not_configured(
    monkeypatch: pytest.MonkeyPatch,
):
    # GIVEN no WIKIPEDIA_CONTACT_EMAIL configured
    monkeypatch.delenv("WIKIPEDIA_CONTACT_EMAIL", raising=False)
    # WHEN building the default User-Agent
    # THEN a clear, actionable error is raised instead of sending an anonymous request
    with pytest.raises(RuntimeError, match="WIKIPEDIA_CONTACT_EMAIL"):
        default_user_agent()


def test_wikipedia_client_uses_the_default_user_agent_when_none_is_given(
    monkeypatch: pytest.MonkeyPatch,
):
    # GIVEN a configured WIKIPEDIA_CONTACT_EMAIL and no explicit user_agent
    monkeypatch.setenv("WIKIPEDIA_CONTACT_EMAIL", "test@example.com")
    captured_requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(200, json=_category_members_response([]))

    client = WikipediaClient(transport=httpx.MockTransport(handle))
    # WHEN a request is made
    list(client.walk_category("Category:Electric guitars"))
    # THEN the request carries the default User-Agent, built from the configured email
    assert "test@example.com" in captured_requests[0].headers["User-Agent"]
