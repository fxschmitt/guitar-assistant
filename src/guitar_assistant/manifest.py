"""Track which Wikipedia revision was last ingested for each article.

See the "Persistent vector store" section of docs/scaling_strategy.md (#2):
`IngestionManifest` maps each article's `source_uri` (its Wikipedia title) to the
revision ID last ingested, persisted as a local JSON file, so re-running the
ingestion script only re-fetches, re-chunks, and re-embeds pages whose revision
changed since the previous run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

DEFAULT_MANIFEST_PATH: Final = Path("ingestion_manifest.json")


@dataclass
class IngestionManifest:
    """Maps each ingested article's title to the revision ID last ingested.

    Attributes:
        revisions: Article title to last-ingested revision ID.
    """

    revisions: dict[str, int] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path = DEFAULT_MANIFEST_PATH) -> IngestionManifest:
        """Load a manifest from `path`, or return an empty one if it doesn't exist yet.

        Args:
            path: JSON file to load from. Defaults to `DEFAULT_MANIFEST_PATH`.

        Returns:
            The loaded manifest, or an empty one on a first-ever ingestion run.
        """
        if not path.exists():
            return cls()
        return cls(revisions=json.loads(path.read_text(encoding="utf-8")))

    def save(self, path: Path = DEFAULT_MANIFEST_PATH) -> None:
        """Persist this manifest to `path` as JSON.

        Args:
            path: JSON file to write to. Defaults to `DEFAULT_MANIFEST_PATH`.
        """
        path.write_text(json.dumps(self.revisions, indent=2, sort_keys=True), encoding="utf-8")

    def is_up_to_date(self, title: str, revision_id: int) -> bool:
        """Return whether `title` was already ingested at `revision_id`.

        Args:
            title: Wikipedia article title.
            revision_id: Revision ID the article was just fetched at.

        Returns:
            `True` if `title` was last ingested at exactly `revision_id`.
        """
        return self.revisions.get(title) == revision_id

    def mark_ingested(self, title: str, revision_id: int) -> None:
        """Record that `title` was ingested at `revision_id`.

        Args:
            title: Wikipedia article title.
            revision_id: Revision ID that was just ingested.
        """
        self.revisions[title] = revision_id
