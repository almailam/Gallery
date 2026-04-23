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


@pytest.mark.asyncio
async def test_list_merges_vector_annotations(tmp_path, broker):
    db_path = tmp_path / "documents.json"
    vector_path = tmp_path / "vector_db.json"
    vector_path.write_text(
        json.dumps(
            {
                "records": {
                    "img-1": {"annotations": ["apple", "fruit"], "path": "/a.jpg"},
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    DocumentDBService(broker, db_path=db_path, vector_db_path=vector_path)
    on_acc = broker._handlers[Channels.IMAGE_ACCEPTED][0]
    await on_acc(
        {
            "type": "image.accepted",
            "image_id": "img-1",
            "path": "/photos/a.jpg",
        }
    )

    published: list[tuple[str, dict]] = []

    async def capture(channel: str, payload: str) -> None:
        published.append((channel, json.loads(payload)))

    broker._client.publish = AsyncMock(side_effect=capture)

    on_list = broker._handlers[Channels.IMAGE_LIST_REQUESTED][0]
    await on_list({"type": "image.list_requested", "id": "req-1"})

    assert len(published) == 1
    channel, body = published[0]
    assert channel == Channels.IMAGE_LIST_READY
    assert body["type"] == "image.list_ready"
    assert body["request_id"] == "req-1"
    assert len(body["images"]) == 1
    assert body["images"][0]["image_id"] == "img-1"
    assert body["images"][0]["path"] == "/photos/a.jpg"
    assert body["images"][0]["annotations"] == ["apple", "fruit"]

    broker._client.publish.assert_awaited()
