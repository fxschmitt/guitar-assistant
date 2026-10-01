"""Parse Wikipedia wikitext into structured guitar-model data.

See the "Wikipedia ingestion" section of docs/scaling_strategy.md: `parse_article`
extracts the `Infobox Guitar model` template's fields (manufacturer, period, wood,
pickups, ...) and converts the rest of the article into clean Markdown, stripping
wiki markup (links, templates, references) and trailing boilerplate sections (See
also, References, ...). Returns `None` for pages that carry no matching infobox
(e.g. manufacturer overview pages, "List of ..." pages, disambiguation pages) —
those aren't guitar-model articles regardless of which category they were found
under, so filtering on infobox presence, not category membership, is what actually
separates guitar models from the noise `WikipediaClient.walk_category` picks up.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

import wikitextparser as wtp

_GUITAR_INFOBOX_TEMPLATE_NAME: Final = "infobox guitar model"
_CONVERT_TEMPLATE_NAME: Final = "convert"
# Sections that carry no article content (citations, cross-links), always at the
# end of a real guitar-model article; the body is truncated at the first one found.
_BOILERPLATE_SECTION_TITLES: Final = frozenset(
    {"see also", "notes", "references", "sources", "external links", "further reading"}
)
_HEADING_PATTERN: Final = re.compile(r"^(=+)\s*(.+?)\s*=+\s*$", re.MULTILINE)
# <gallery>...</gallery> blocks list bare "File:....jpg|caption" lines, which
# aren't wiki markup `plain_text()` recognizes and strips, only image filenames.
_GALLERY_PATTERN: Final = re.compile(r"<gallery\b.*?</gallery>", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class ParsedArticle:
    """One Wikipedia article, parsed into its infobox fields and clean body text.

    Attributes:
        title: The article's Wikipedia title.
        infobox: `Infobox Guitar model` fields, keyed by lowercased field name
            (e.g. `manufacturer`, `period`, `scale`), with wiki markup stripped.
        body_markdown: The article body as Markdown, with headings preserved
            (`##`/`###`), wiki markup stripped, and trailing boilerplate
            sections (References, See also, ...) removed.
    """

    title: str
    infobox: dict[str, str]
    body_markdown: str


def parse_article(title: str, wikitext: str) -> ParsedArticle | None:
    """Parse one article's wikitext into infobox fields and a clean Markdown body.

    Args:
        title: The article's Wikipedia title, carried through unchanged.
        wikitext: The article's raw wikitext, as returned by
            `WikipediaClient.fetch_wikitext`.

    Returns:
        A `ParsedArticle`, or `None` if the article carries no `Infobox Guitar
        model` template — i.e. it isn't a guitar-model article (a manufacturer
        overview page, a "List of ..." page, a disambiguation page, ...).
    """
    parsed = wtp.parse(wikitext)
    infobox_template = _find_guitar_infobox(parsed)
    if infobox_template is None:
        return None
    infobox = {
        argument.name.strip().lower(): _to_plain_text(argument.value)
        for argument in infobox_template.arguments
        if argument.value.strip()
    }
    return ParsedArticle(title=title, infobox=infobox, body_markdown=_to_body_markdown(wikitext))


def _find_guitar_infobox(parsed: wtp.WikiText) -> wtp.Template | None:
    """Return the article's `Infobox Guitar model` template, if it has one."""
    for template in parsed.templates:
        if template.name.strip().lower() == _GUITAR_INFOBOX_TEMPLATE_NAME:
            return template
    return None


def _to_body_markdown(wikitext: str) -> str:
    """Convert the article body into clean Markdown, minus trailing boilerplate."""
    body_wikitext = _GALLERY_PATTERN.sub("", _expand_convert_templates(wikitext))
    plain_text = wtp.parse(body_wikitext).plain_text()
    body = _HEADING_PATTERN.sub(
        lambda match: f"{'#' * len(match.group(1))} {match.group(2)}", plain_text
    )
    return _truncate_before_boilerplate(body).strip()


def _truncate_before_boilerplate(body: str) -> str:
    """Cut `body` at the first boilerplate section heading (References, ...)."""
    for heading in re.finditer(r"^#{2,}\s+(.+)$", body, re.MULTILINE):
        if heading.group(1).strip().lower() in _BOILERPLATE_SECTION_TITLES:
            return body[: heading.start()]
    return body


def _to_plain_text(wikitext_value: str) -> str:
    """Strip wiki markup from one infobox field value, expanding `{{convert}}`."""
    return wtp.parse(_expand_convert_templates(wikitext_value)).plain_text().strip()


def _expand_convert_templates(wikitext: str) -> str:
    """Rewrite every `{{convert|VALUE|UNIT|...}}` template to a plain "VALUE UNIT".

    `plain_text()` drops templates it can't evaluate rather than expanding them,
    which would otherwise leave dangling text (e.g. "scale length is .") wherever
    `{{convert}}` is used — both in infobox field values and in body prose.
    """
    parsed = wtp.parse(wikitext)
    for template in parsed.templates:
        if template.name.strip().lower() == _CONVERT_TEMPLATE_NAME:
            arguments = [
                argument.value.strip() for argument in template.arguments if argument.positional
            ]
            if len(arguments) >= 2:
                template.string = f"{arguments[0]} {arguments[1]}"
    # Re-parsing the mutated string is required: `plain_text()` on the same
    # `parsed` object doesn't reflect the `template.string` mutation above (a
    # wikitextparser quirk) — only a fresh parse of the mutated string does.
    return str(parsed)
