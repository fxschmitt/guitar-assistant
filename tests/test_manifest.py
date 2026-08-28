"""Unit tests for guitar_assistant.manifest."""

from pathlib import Path

from guitar_assistant.manifest import IngestionManifest


def test_is_up_to_date_is_false_for_a_title_never_ingested():
    # GIVEN an empty manifest
    manifest = IngestionManifest()
    # WHEN checking a title that was never ingested
    # THEN it is reported as not up to date
    assert not manifest.is_up_to_date("Fender Stratocaster", 123)


def test_is_up_to_date_is_true_for_a_title_ingested_at_that_exact_revision():
    # GIVEN a manifest with one ingested article
    manifest = IngestionManifest()
    manifest.mark_ingested("Fender Stratocaster", 123)
    # WHEN checking that title at the same revision
    # THEN it is reported as up to date
    assert manifest.is_up_to_date("Fender Stratocaster", 123)


def test_is_up_to_date_is_false_for_a_title_ingested_at_a_different_revision():
    # GIVEN a manifest with one ingested article
    manifest = IngestionManifest()
    manifest.mark_ingested("Fender Stratocaster", 123)
    # WHEN checking that title at a newer revision
    # THEN it is reported as not up to date
    assert not manifest.is_up_to_date("Fender Stratocaster", 456)


def test_load_returns_an_empty_manifest_when_the_file_does_not_exist(tmp_path: Path):
    # GIVEN a path with no manifest file yet
    path = tmp_path / "does_not_exist.json"
    # WHEN a manifest is loaded from it
    manifest = IngestionManifest.load(path)
    # THEN an empty manifest is returned rather than raising
    assert not manifest.revisions


def test_save_then_load_round_trips_the_manifest_contents(tmp_path: Path):
    # GIVEN a manifest with some ingested articles
    path = tmp_path / "manifest.json"
    manifest = IngestionManifest()
    manifest.mark_ingested("Fender Stratocaster", 123)
    manifest.mark_ingested("Gibson SG", 456)
    # WHEN it is saved and reloaded
    manifest.save(path)
    reloaded = IngestionManifest.load(path)
    # THEN the reloaded manifest matches what was saved
    assert reloaded.revisions == {"Fender Stratocaster": 123, "Gibson SG": 456}
