"""Event message contracts exchanged over Redis pub/sub."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


@dataclass
class Message:
    type: str = "message"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class ImageUploadRequested(Message):
    type: str = "image.upload_requested"
    path: str = ""


@dataclass
class ImageAccepted(Message):
    type: str = "image.accepted"
    image_id: str = ""
    path: str = ""


@dataclass
class ImageEmbeddingRequested(Message):
    type: str = "image.embedding_requested"
    image_id: str = ""
    path: str = ""


@dataclass
class ImageEmbeddingFailed(Message):
    type: str = "image.embedding_failed"
    image_id: str = ""
    path: str = ""
    reason: str = ""


@dataclass
class ImageStored(Message):
    type: str = "image.stored"
    image_id: str = ""
    path: str = ""
    status: str = "stored"


@dataclass
class ImagePipelineComplete(Message):
    type: str = "image.pipeline_complete"
    image_id: str = ""
    path: str = ""


@dataclass
class ImageListRequested(Message):
    type: str = "image.list_requested"


@dataclass
class ImageListReady(Message):
    type: str = "image.list_ready"
    request_id: str = ""
    images: list[dict] = field(default_factory=list)


@dataclass
class SearchRequested(Message):
    type: str = "search.requested"
    query: str = ""
    top_k: int = 5


@dataclass
class SearchResultsReady(Message):
    type: str = "search.results_ready"
    request_id: str = ""
    results: list[dict] = field(default_factory=list)


@dataclass
class StorageClearRequested(Message):
    type: str = "storage.clear_requested"


@dataclass
class StorageClearCompleted(Message):
    type: str = "storage.clear_completed"
    request_id: str = ""
    service: str = ""
    deleted_count: int = 0
