"""Discover and fetch electric guitar articles from the Wikipedia API.

See the "Wikipedia ingestion" section of docs/scaling_strategy.md:
`walk_category` enumerates candidate article titles by walking a category's
subcategory tree — in practice, `ELECTRIC_GUITARS_BY_MANUFACTURER_CATEGORY`,
since `ELECTRIC_GUITARS_CATEGORY` itself holds mostly generic topic articles
rather than guitar models — and `fetch_wikitext` pulls one article's raw
wikitext and revision ID. Both go through `WikipediaClient`, which enforces
Wikipedia's API etiquette (an identifying `User-Agent`) and a hard cap on the
number of requests a run can make, so early test runs can't accidentally walk
the whole category tree.
"""

from __future__ import annotations

import os
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from importlib import metadata
from types import TracebackType
from typing import Any, Final, Self

from dotenv import load_dotenv
import httpx

API_URL: Final = "https://en.wikipedia.org/w/api.php"
ELECTRIC_GUITARS_CATEGORY: Final = "Category:Electric guitars"
# `ELECTRIC_GUITARS_CATEGORY`'s direct articles are mostly generic topics (e.g.
# "Guitar amplifier", "Distortion (music)"), not guitar models; the actual models
# live under this subcategory, one subcategory per manufacturer.
ELECTRIC_GUITARS_BY_MANUFACTURER_CATEGORY: Final = "Category:Electric guitars by manufacturer"
DEFAULT_MAX_REQUESTS: Final = 20
DEFAULT_MAX_DEPTH: Final = 3
_CATEGORY_MEMBERS_PAGE_SIZE: Final = 500
_CONTACT_EMAIL_ENV_VAR: Final = "WIKIPEDIA_CONTACT_EMAIL"

load_dotenv()


class RequestLimitExceededError(RuntimeError):
    """Raised when a `WikipediaClient` would exceed its configured request budget."""


class ArticleNotFoundError(RuntimeError):
    """Raised when `fetch_wikitext` is asked for a title Wikipedia has no article for.

    A title found via `walk_category` can still fail to resolve — e.g. a stale
    category listing pointing at a deleted or renamed page.
    """


@dataclass(frozen=True)
class FetchedArticle:
    """One article's wikitext, and the revision it was fetched at.

    Attributes:
        wikitext: The article's raw wikitext of the fetched revision.
        revision_id: The Wikipedia revision ID `wikitext` was fetched at, used by
            `IngestionManifest` to detect when an already-ingested article has
            since changed and needs re-processing.
    """

    wikitext: str
    revision_id: int


def default_user_agent() -> str:
    """Build the default `User-Agent` from the contact email configured in `.env`.

    Wikipedia's API etiquette (https://meta.wikimedia.org/wiki/User-Agent_policy)
    requires an identifying `User-Agent` with a way to reach the operator, so this
    is read from `WIKIPEDIA_CONTACT_EMAIL` rather than hard-coded, letting each
    person running this code register their own contact email.

    Returns:
        A `User-Agent` string identifying this project and its configured
        contact email.

    Raises:
        RuntimeError: If `WIKIPEDIA_CONTACT_EMAIL` is not set.
    """
    contact_email = os.environ.get(_CONTACT_EMAIL_ENV_VAR)
    if not contact_email:
        raise RuntimeError(
            f"{_CONTACT_EMAIL_ENV_VAR} is not set. Wikipedia's API etiquette requires an "
            "identifying User-Agent with a contact email; add it to your .env file, e.g. "
            f'`echo "{_CONTACT_EMAIL_ENV_VAR}=you@example.com" >> .env`.'
        )
    return f"guitar-assistant/{metadata.version('guitar-assistant')} ({contact_email})"


class WikipediaClient:
    """Thin wrapper around the Wikipedia API with a built-in request budget.

    Args:
        user_agent: Identifying `User-Agent` string sent with every request,
            per Wikipedia's API etiquette (https://meta.wikimedia.org/wiki/User-Agent_policy).
            Defaults to `default_user_agent()`, built from the `WIKIPEDIA_CONTACT_EMAIL`
            configured in `.env`.
        max_requests: Maximum number of API requests this client will make
            before raising `RequestLimitExceededError`. Defaults to a small
            value so exploratory runs can't walk the whole category tree.
        transport: Optional `httpx` transport override, for tests to substitute
            a network-free mock transport.
    """

    def __init__(
        self,
        user_agent: str | None = None,
        *,
        max_requests: int = DEFAULT_MAX_REQUESTS,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        resolved_user_agent = user_agent if user_agent is not None else default_user_agent()
        self._client = httpx.Client(
            headers={"User-Agent": resolved_user_agent}, transport=transport
        )
        self._max_requests = max_requests
        self._requests_made = 0

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._client.close()

    def walk_category(
        self, category: str = ELECTRIC_GUITARS_CATEGORY, *, max_depth: int = DEFAULT_MAX_DEPTH
    ) -> Iterator[str]:
        """Walk a category's subcategory tree and yield candidate article titles.

        Breadth-first over subcategories up to `max_depth`, so a mislabeled or
        unexpectedly broad subcategory can't pull in unrelated articles.

        Args:
            category: Root category to start from, e.g. `Category:Electric guitars`.
            max_depth: Maximum number of subcategory levels to descend into.
                `0` only lists articles directly in `category`.

        Yields:
            Article titles found in `category` or one of its subcategories,
            within `max_depth` levels.
        """
        seen_categories = {category}
        queue = deque([(category, 0)])
        while queue:
            current_category, depth = queue.popleft()
            for title, is_subcategory in self._iter_category_members(current_category):
                if is_subcategory:
                    if depth < max_depth and title not in seen_categories:
                        seen_categories.add(title)
                        queue.append((title, depth + 1))
                else:
                    yield title

    def fetch_wikitext(self, title: str) -> FetchedArticle:
        """Fetch one article's current wikitext and revision ID.

        Args:
            title: Exact Wikipedia article title, as returned by `walk_category`.

        Returns:
            The article's raw wikitext and revision ID, both of its latest revision.

        Raises:
            ArticleNotFoundError: If `title` has no corresponding Wikipedia article.
        """
        response = self._get(
            {
                "action": "query",
                "prop": "revisions",
                "titles": title,
                "rvslots": "main",
                "rvprop": "ids|content",
            }
        )
        pages = response["query"]["pages"]
        (page,) = pages.values()
        if "missing" in page:
            raise ArticleNotFoundError(f"No Wikipedia article found for title {title!r}.")
        revision = page["revisions"][0]
        return FetchedArticle(
            wikitext=revision["slots"]["main"]["*"], revision_id=revision["revid"]
        )

    def _iter_category_members(self, category: str) -> Iterator[tuple[str, bool]]:
        """Yield `(title, is_subcategory)` for every direct member of `category`."""
        continue_params: dict[str, str] = {}
        while True:
            response = self._get(
                {
                    "action": "query",
                    "list": "categorymembers",
                    "cmtitle": category,
                    "cmlimit": _CATEGORY_MEMBERS_PAGE_SIZE,
                    **continue_params,
                }
            )
            for member in response["query"]["categorymembers"]:
                yield member["title"], member["title"].startswith("Category:")
            if "continue" not in response:
                return
            continue_params = response["continue"]

    def _get(self, params: dict[str, Any]) -> dict[str, Any]:
        """Issue one API GET, enforcing the request budget.

        Args:
            params: Wikipedia API query parameters; `format=json` is added
                automatically.

        Returns:
            The parsed JSON response body.
        """
        if self._requests_made >= self._max_requests:
            raise RequestLimitExceededError(
                f"Wikipedia API request budget of {self._max_requests} requests exhausted."
            )
        self._requests_made += 1
        response = self._client.get(API_URL, params={**params, "format": "json"})
        response.raise_for_status()
        return response.json()
