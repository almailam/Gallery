import json
from unittest.mock import AsyncMock

import pytest

from gallery.broker.pubsub import Channels, RedisBroker
from gallery.services.document_db import DocumentDBService


@pytest.fixture
def broker():
    b = RedisBroker()
    b._client = AsyncMock()
    return b


@pytest.mark.asyncio
async def test_document_db_persists_accepted(tmp_path, broker):
    db_path = tmp_path / "documents.json"
    DocumentDBService(broker, db_path=db_path)
    handler = broker._handlers[Channels.IMAGE_ACCEPTED][0]
    await handler(
        {
            "type": "image.accepted",
            "id": "msg-1",
            "image_id": "img-1",
            "path": "/photos/a.jpg",
        }
    )

    data = json.loads(db_path.read_text(encoding="utf-8"))
    assert "img-1" in data["records"]
    assert data["records"]["img-1"]["path"] == "/photos/a.jpg"
    assert "updated_at" in data["records"]["img-1"]


@pytest.mark.asyncio
async def test_annotation_merges_with_prior_accept(tmp_path, broker):
    db_path = tmp_path / "documents.json"
    DocumentDBService(broker, db_path=db_path)
    on_ann = broker._handlers[Channels.IMAGE_ANNOTATION_REQUESTED][0]
    on_acc = broker._handlers[Channels.IMAGE_ACCEPTED][0]

    await on_ann({"image_id": "img-2", "path": "/b.jpg", "type": "image.annotation_requested"})
    await on_acc(
        {
            "type": "image.accepted",
            "image_id": "img-2",
            "path": "/b.jpg",
        }
    )

    rec = json.loads(db_path.read_text(encoding="utf-8"))["records"]["img-2"]
    assert rec["path"] == "/b.jpg"
    assert "annotation_requested_at" in rec
