"""Pub/sub message types."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Shared metadata helpers
# ---------------------------------------------------------------------------

@dataclass
class VectorMeta:
    model: str = ""
    dimensions: int = 0


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

@dataclass
class Message:
    type: str = "message"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_json(self) -> str:
        return json.dumps(asdict(self))


# ---------------------------------------------------------------------------
# Image pipeline
# ---------------------------------------------------------------------------

@dataclass
class ImageUploadRequested(Message):
    type: str = "image.upload_requested"
    path: str = ""
    vector_meta: VectorMeta = field(default_factory=VectorMeta)


@dataclass
class ImageAnnotated(Message):
    type: str = "image.annotated"
    image_id: str = ""
    path: str = ""
    tags: list[str] = field(default_factory=list)
    caption: str = ""


@dataclass
class ImageEmbedded(Message):
    type: str = "image.embedded"
    image_id: str = ""
    embedding: list[float] = field(default_factory=list)
    vector_meta: VectorMeta = field(default_factory=VectorMeta)


@dataclass
class ImagePipelineComplete(Message):
    type: str = "image.pipeline_complete"
    image_id: str = ""
    path: str = ""


# ---------------------------------------------------------------------------
# Search pipeline
# ---------------------------------------------------------------------------

@dataclass
class SearchRequested(Message):
    type: str = "search.requested"
    query: str = ""


@dataclass
class SearchResultsReady(Message):
    type: str = "search.results_ready"
    request_id: str = ""
    results: list[dict] = field(default_factory=list)
