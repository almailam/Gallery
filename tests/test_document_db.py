from unittest.mock import AsyncMock

import pytest

from gallery.broker.pubsub import Channels, RedisBroker
from gallery.services.document_db import DocumentDBService
from tests.helpers import FakeCollection


@pytest.fixture
def broker():
    b = RedisBroker()
    b._client = AsyncMock()
    return b


@pytest.mark.asyncio
async def test_document_db_persists_accepted(broker):
    collection = FakeCollection()
    DocumentDBService(broker, collection=collection, vector_collection=FakeCollection())
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
async def test_document_db_accepted_publishes_only_stored(broker):
    """DocumentDBService must publish IMAGE_STORED (not IMAGE_PIPELINE_COMPLETE)."""
    collection = FakeCollection()
    DocumentDBService(broker, collection=collection, vector_collection=FakeCollection())

    published: list[tuple[str, dict]] = []

    async def capture(channel: str, payload: str) -> None:
        import json

        published.append((channel, json.loads(payload)))

    broker._client.publish = AsyncMock(side_effect=capture)
    handler = broker._handlers[Channels.IMAGE_ACCEPTED][0]
    await handler({"type": "image.accepted", "image_id": "img-1", "path": "/photos/a.jpg"})

    channels = [ch for ch, _ in published]
    assert Channels.IMAGE_STORED in channels
    assert Channels.IMAGE_PIPELINE_COMPLETE not in channels


@pytest.mark.asyncio
async def test_list_merges_vector_embedding_metadata(broker):
    collection = FakeCollection()
    vector_collection = FakeCollection(
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


@pytest.mark.asyncio
async def test_document_db_clears_records_and_publishes_completion(broker):
    collection = FakeCollection({"img-1": {"_id": "img-1"}, "img-2": {"_id": "img-2"}})
    DocumentDBService(broker, collection=collection, vector_collection=FakeCollection())

    published: list[tuple[str, dict]] = []

    async def capture(channel: str, payload: str) -> None:
        import json

        published.append((channel, json.loads(payload)))

    broker._client.publish = AsyncMock(side_effect=capture)
    handler = broker._handlers[Channels.STORAGE_CLEAR_REQUESTED][0]
    await handler({"type": "storage.clear_requested", "id": "req-clear-1"})

    assert collection.records == {}
    assert len(published) == 1
    channel, body = published[0]
    assert channel == Channels.STORAGE_CLEAR_COMPLETED
    assert body["service"] == "documents"
    assert body["deleted_count"] == 2
    assert body["request_id"] == "req-clear-1"


@pytest.mark.asyncio
async def test_document_db_accepted_missing_image_id_is_skipped(broker):
    collection = FakeCollection()
    DocumentDBService(broker, collection=collection, vector_collection=FakeCollection())
    handler = broker._handlers[Channels.IMAGE_ACCEPTED][0]
    await handler({"type": "image.accepted", "path": "/photos/a.jpg"})

    assert collection.records == {}


@pytest.mark.asyncio
async def test_document_db_list_empty_collection(broker):
    collection = FakeCollection()
    DocumentDBService(broker, collection=collection, vector_collection=FakeCollection())

    published: list[tuple[str, dict]] = []

    async def capture(channel: str, payload: str) -> None:
        import json

        published.append((channel, json.loads(payload)))

    broker._client.publish = AsyncMock(side_effect=capture)
    on_list = broker._handlers[Channels.IMAGE_LIST_REQUESTED][0]
    await on_list({"type": "image.list_requested", "id": "req-empty"})

    assert len(published) == 1
    _, body = published[0]
    assert body["images"] == []
