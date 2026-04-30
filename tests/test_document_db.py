from unittest.mock import AsyncMock

import pytest

from gallery.broker.pubsub import Channels, RedisBroker
from gallery.services.document_db import DocumentDBService


class _FakeAsyncCursor:
    def __init__(self, records):
        self._records = list(records)
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._records):
            raise StopAsyncIteration
        value = self._records[self._index]
        self._index += 1
        return dict(value)


class _FakeCollection:
    def __init__(self, records=None):
        self.records = records or {}

    async def update_one(self, filter_doc, update_doc, *, upsert=False):
        key = filter_doc["_id"]
        if key not in self.records:
            if not upsert:
                return
            self.records[key] = dict(update_doc.get("$setOnInsert", {}))
        self.records[key].update(update_doc.get("$set", {}))

    def find(self, filter_doc):
        return _FakeAsyncCursor(self.records.values())


@pytest.fixture
def broker():
    b = RedisBroker()
    b._client = AsyncMock()
    return b


@pytest.mark.asyncio
async def test_document_db_persists_accepted(broker):
    collection = _FakeCollection()
    DocumentDBService(broker, collection=collection, vector_collection=_FakeCollection())
    handler = broker._handlers[Channels.IMAGE_ACCEPTED][0]
    await handler(
        {
            "type": "image.accepted",
            "id": "msg-1",
            "image_id": "img-1",
            "path": "/photos/a.jpg",
        }
    )

    assert "img-1" in collection.records
    assert collection.records["img-1"]["path"] == "/photos/a.jpg"
    assert "updated_at" in collection.records["img-1"]


@pytest.mark.asyncio
async def test_annotation_merges_with_prior_accept(broker):
    collection = _FakeCollection()
    DocumentDBService(broker, collection=collection, vector_collection=_FakeCollection())
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

    rec = collection.records["img-2"]
    assert rec["path"] == "/b.jpg"
    assert "annotation_requested_at" in rec


@pytest.mark.asyncio
async def test_list_merges_vector_embedding_metadata(broker):
    collection = _FakeCollection()
    vector_collection = _FakeCollection(
        {
            "img-1": {
                "_id": "img-1",
                "image_id": "img-1",
                "embedding": [1.0, 0.0],
                "embedding_model": "test-model",
                "embedding_dim": 2,
                "path": "/a.jpg",
            }
        }
    )
    DocumentDBService(broker, collection=collection, vector_collection=vector_collection)
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
        import json

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
    assert body["images"][0]["embedding_model"] == "test-model"
    assert body["images"][0]["embedding_dim"] == 2

    broker._client.publish.assert_awaited()
