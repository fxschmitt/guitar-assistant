"""Unit tests for guitar_assistant.infobox_parser."""

from guitar_assistant.infobox_parser import parse_article

_STRATOCASTER_WIKITEXT = """{{Short description|Solid-body electric guitar}}
{{Use mdy dates|date=May 2022}}
{{Infobox Guitar model
|title= Fender Stratocaster
|manufacturer=[[Fender Musical Instruments Corporation|Fender]]
|period=1954-present
|scale={{convert|25.5|in|1}}
|woodbody=Most commonly [[alder]] or [[ash tree|ash]].
}}
The '''Fender Stratocaster''' is a model of [[electric guitar]].

==Overall design==
The archetypal Stratocaster is a solid-body electric guitar. Its scale length is {{convert|25.5|in|0}}.

<gallery>
File:Fender Stratocaster body.jpg|Body and electronics
File:Fender Stratocaster headstock.jpg|Headstock and tuning pegs
</gallery>

===Body===
Most commonly alder or ash.

==References==
<ref>Some citation.</ref>

==External links==
* [https://example.com Official site]
"""

_COMPANY_OVERVIEW_WIKITEXT = """{{Short description|American guitar brand}}
{{Infobox company
|name=B.C. Rich Guitars
|type=Private
}}
'''B.C. Rich''' is an American brand of guitars.
"""

_LIST_PAGE_WIKITEXT = """'''List of B.C. Rich guitars''' below.

* Mockingbird
* Warlock
"""


def test_parse_article_extracts_infobox_fields_with_wiki_markup_stripped():
    # GIVEN a real-shaped article with an Infobox Guitar model template
    # WHEN the article is parsed
    parsed = parse_article("Fender Stratocaster", _STRATOCASTER_WIKITEXT)
    # THEN the infobox fields are extracted with links resolved to their display text
    assert parsed is not None
    assert parsed.infobox["manufacturer"] == "Fender"
    assert parsed.infobox["period"] == "1954-present"
    assert parsed.infobox["woodbody"] == "Most commonly alder or ash."


def test_parse_article_expands_convert_templates_in_infobox_fields():
    # GIVEN an infobox field whose value is a {{convert}} template
    # WHEN the article is parsed
    parsed = parse_article("Fender Stratocaster", _STRATOCASTER_WIKITEXT)
    # THEN the template is expanded to a plain "value unit" string, not dropped
    assert parsed is not None
    assert parsed.infobox["scale"] == "25.5 in"


def test_parse_article_converts_wiki_headings_to_markdown_headings():
    # GIVEN an article with wiki-style section headings
    # WHEN the article is parsed
    parsed = parse_article("Fender Stratocaster", _STRATOCASTER_WIKITEXT)
    # THEN the headings are converted to Markdown, preserving their level
    assert parsed is not None
    assert "## Overall design" in parsed.body_markdown
    assert "### Body" in parsed.body_markdown


def test_parse_article_truncates_body_before_the_references_section():
    # GIVEN an article with trailing References/External links boilerplate
    # WHEN the article is parsed
    parsed = parse_article("Fender Stratocaster", _STRATOCASTER_WIKITEXT)
    # THEN the boilerplate sections are dropped from the body
    assert parsed is not None
    assert "References" not in parsed.body_markdown
    assert "External links" not in parsed.body_markdown


def test_parse_article_expands_convert_templates_in_the_body_text():
    # GIVEN body prose containing a {{convert}} template, not just the infobox
    # WHEN the article is parsed
    parsed = parse_article("Fender Stratocaster", _STRATOCASTER_WIKITEXT)
    # THEN it is expanded in place rather than left as a dangling empty template
    assert parsed is not None
    assert "Its scale length is 25.5 in." in parsed.body_markdown


def test_parse_article_strips_gallery_blocks_from_the_body():
    # GIVEN a <gallery> block listing raw "File:....jpg|caption" lines
    # WHEN the article is parsed
    parsed = parse_article("Fender Stratocaster", _STRATOCASTER_WIKITEXT)
    # THEN the gallery block is removed rather than leaking raw image markup
    assert parsed is not None
    assert "File:" not in parsed.body_markdown
    assert "gallery" not in parsed.body_markdown.lower()


def test_parse_article_returns_none_for_a_page_without_a_guitar_infobox():
    # GIVEN a manufacturer overview page carrying an unrelated Infobox company
    # WHEN the article is parsed
    parsed = parse_article("B.C. Rich", _COMPANY_OVERVIEW_WIKITEXT)
    # THEN it is not treated as a guitar-model article
    assert parsed is None


def test_parse_article_returns_none_for_a_page_with_no_infobox_at_all():
    # GIVEN a "List of ..." page with no infobox
    # WHEN the article is parsed
    parsed = parse_article("List of B.C. Rich guitars", _LIST_PAGE_WIKITEXT)
    # THEN it is not treated as a guitar-model article
    assert parsed is None
