import json
from unittest.mock import AsyncMock, patch

import pytest

from gallery.broker.pubsub import Channels, RedisBroker
from gallery.services.vector_db import (
    VectorDBService,
    _message_content_to_text,
    _parse_annotations_from_model_text,
)


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

    async def find_one(self, filter_doc):
        record = self.records.get(filter_doc["_id"])
        return dict(record) if record else None

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
async def test_embedding_reuses_stored_annotations_without_model_call(broker):
    collection = _FakeCollection(
        {
            "img-1": {
                "_id": "img-1",
                "image_id": "img-1",
                "path": "/old.jpg",
                "annotations": ["apple", "fruit"],
                "search_text": "apple fruit",
            }
        }
    )
    svc = VectorDBService(broker, collection=collection)
    handler = broker._handlers[Channels.IMAGE_EMBEDDING_REQUESTED][0]

    with patch.object(svc, "_annotate_image", new_callable=AsyncMock) as mock_ann:
        mock_ann.return_value = ["wrong"]
        await handler({"image_id": "img-1", "path": "/new.jpg", "type": "image.embedding_requested"})
        mock_ann.assert_not_awaited()

    assert collection.records["img-1"]["annotations"] == ["apple", "fruit"]
    assert collection.records["img-1"]["path"] == "/new.jpg"


@pytest.mark.asyncio
async def test_embedding_calls_model_once_then_reuses_for_same_image_id(broker):
    collection = _FakeCollection()
    svc = VectorDBService(broker, collection=collection)
    handler = broker._handlers[Channels.IMAGE_EMBEDDING_REQUESTED][0]

    with patch.object(svc, "_annotate_image", new_callable=AsyncMock) as mock_ann:
        mock_ann.return_value = ["a", "b"]
        await handler({"image_id": "img-1", "path": "/x.jpg", "type": "image.embedding_requested"})
        await handler({"image_id": "img-1", "path": "/x.jpg", "type": "image.embedding_requested"})
        assert mock_ann.await_count == 1


@pytest.mark.asyncio
async def test_embedding_reuses_annotations_for_same_file_path_different_id(tmp_path, broker):
    collection = _FakeCollection()
    image_file = tmp_path / "same.jpg"
    image_file.write_bytes(b"\xff\xd8\xff")

    svc = VectorDBService(broker, collection=collection)
    handler = broker._handlers[Channels.IMAGE_EMBEDDING_REQUESTED][0]

    with patch.object(svc, "_annotate_image", new_callable=AsyncMock) as mock_ann:
        mock_ann.return_value = ["sky", "clouds"]
        await handler(
            {"image_id": "first", "path": str(image_file), "type": "image.embedding_requested"}
        )
        await handler(
            {"image_id": "second", "path": str(image_file), "type": "image.embedding_requested"}
        )
        assert mock_ann.await_count == 1

    assert collection.records["second"]["annotations"] == ["sky", "clouds"]


@pytest.mark.asyncio
async def test_search_returns_scored_mongo_records(broker):
    collection = _FakeCollection(
        {
            "img-1": {
                "_id": "img-1",
                "image_id": "img-1",
                "path": "/photos/apple.jpg",
                "annotations": ["apple", "fruit"],
                "search_text": "apple fruit",
                "updated_at": "2026-01-01T00:00:00+00:00",
            }
        }
    )
    VectorDBService(broker, collection=collection)
    handler = broker._handlers[Channels.SEARCH_REQUESTED][0]
    published: list[tuple[str, dict]] = []

    async def capture(channel: str, payload: str) -> None:
        published.append((channel, json.loads(payload)))

    broker._client.publish = AsyncMock(side_effect=capture)
    await handler({"type": "search.requested", "id": "req-1", "query": "apple", "top_k": 5})

    assert len(published) == 1
    channel, body = published[0]
    assert channel == Channels.SEARCH_RESULTS_READY
    assert body["request_id"] == "req-1"
    assert body["results"] == [
        {
            "image_id": "img-1",
            "path": "/photos/apple.jpg",
            "annotations": ["apple", "fruit"],
            "score": 1,
        }
    ]


def test_message_content_to_text_supports_part_list():
    text = _message_content_to_text(
        {"content": [{"type": "text", "text": '{"annotations":["a","b"]}'}]}
    )
    assert "annotations" in text


def test_parse_annotations_from_fenced_json():
    raw = """```json
{"annotations": ["x", "y", "z", "w", "v"]}
```"""
    assert _parse_annotations_from_model_text(raw) == ["x", "y", "z", "w", "v"]
