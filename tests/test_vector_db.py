import json
from unittest.mock import AsyncMock, patch

import pytest

from gallery.broker.pubsub import Channels, RedisBroker
from gallery.services.vector_db import (
    VectorDBService,
    _cosine_similarity,
    _normalize_embedding,
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

    async def delete_many(self, filter_doc):
        del filter_doc
        self.records = {}


@pytest.fixture
def broker():
    b = RedisBroker()
    b._client = AsyncMock()
    return b


@pytest.mark.asyncio
async def test_embedding_reuses_stored_embedding_without_model_call(broker):
    collection = _FakeCollection(
        {
            "img-1": {
                "_id": "img-1",
                "image_id": "img-1",
                "path": "/old.jpg",
                "embedding": [1.0, 0.0],
                "embedding_model": "test-model",
                "embedding_dim": 2,
            }
        }
    )
    svc = VectorDBService(broker, collection=collection, embedding_model_name="test-model")
    handler = broker._handlers[Channels.IMAGE_EMBEDDING_REQUESTED][0]

    with patch.object(svc, "_embed_image", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = [0.0, 1.0]
        await handler({"image_id": "img-1", "path": "/new.jpg", "type": "image.embedding_requested"})
        mock_embed.assert_not_awaited()

    assert collection.records["img-1"]["embedding"] == [1.0, 0.0]
    assert collection.records["img-1"]["path"] == "/new.jpg"
    assert collection.records["img-1"]["embedding_dim"] == 2


@pytest.mark.asyncio
async def test_embedding_calls_model_once_then_reuses_for_same_image_id(broker):
    collection = _FakeCollection()
    svc = VectorDBService(broker, collection=collection, embedding_model_name="test-model")
    handler = broker._handlers[Channels.IMAGE_EMBEDDING_REQUESTED][0]

    with patch.object(svc, "_embed_image", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = [0.6, 0.8]
        await handler({"image_id": "img-1", "path": "/x.jpg", "type": "image.embedding_requested"})
        await handler({"image_id": "img-1", "path": "/x.jpg", "type": "image.embedding_requested"})
        assert mock_embed.await_count == 1

    assert collection.records["img-1"]["embedding"] == [0.6, 0.8]
    assert collection.records["img-1"]["embedding_model"] == "test-model"


@pytest.mark.asyncio
async def test_embedding_failure_publishes_visible_event(broker):
    collection = _FakeCollection()
    svc = VectorDBService(broker, collection=collection, embedding_model_name="test-model")
    handler = broker._handlers[Channels.IMAGE_EMBEDDING_REQUESTED][0]
    published: list[tuple[str, dict]] = []

    async def capture(channel: str, payload: str) -> None:
        published.append((channel, json.loads(payload)))

    broker._client.publish = AsyncMock(side_effect=capture)
    svc._last_embedding_error = "missing dependency"
    with patch.object(svc, "_embed_image", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = []
        await handler({"image_id": "img-1", "path": "/x.jpg", "type": "image.embedding_requested"})

    assert collection.records == {}
    assert len(published) == 1
    channel, body = published[0]
    assert channel == Channels.IMAGE_EMBEDDING_FAILED
    assert body["image_id"] == "img-1"
    assert body["path"] == "/x.jpg"
    assert body["reason"] == "missing dependency"


@pytest.mark.asyncio
async def test_embedding_reuses_embedding_for_same_file_path_different_id(tmp_path, broker):
    collection = _FakeCollection()
    image_file = tmp_path / "same.jpg"
    image_file.write_bytes(b"\xff\xd8\xff")

    svc = VectorDBService(broker, collection=collection, embedding_model_name="test-model")
    handler = broker._handlers[Channels.IMAGE_EMBEDDING_REQUESTED][0]

    with patch.object(svc, "_embed_image", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = [0.25, 0.75]
        await handler(
            {"image_id": "first", "path": str(image_file), "type": "image.embedding_requested"}
        )
        await handler(
            {"image_id": "second", "path": str(image_file), "type": "image.embedding_requested"}
        )
        assert mock_embed.await_count == 1

    assert collection.records["second"]["embedding"] == [0.25, 0.75]


@pytest.mark.asyncio
async def test_search_returns_similarity_ranked_records(broker):
    collection = _FakeCollection(
        {
            "img-1": {
                "_id": "img-1",
                "image_id": "img-1",
                "path": "/photos/apple.jpg",
                "embedding": [1.0, 0.0],
                "embedding_model": "test-model",
                "embedding_dim": 2,
                "updated_at": "2026-01-01T00:00:00+00:00",
            },
            "img-2": {
                "_id": "img-2",
                "image_id": "img-2",
                "path": "/photos/banana.jpg",
                "embedding": [0.0, 1.0],
                "embedding_model": "test-model",
                "embedding_dim": 2,
                "updated_at": "2026-01-02T00:00:00+00:00",
            },
        }
    )
    svc = VectorDBService(broker, collection=collection, embedding_model_name="test-model")
    svc._search_min_score = 0.0
    handler = broker._handlers[Channels.SEARCH_REQUESTED][0]
    published: list[tuple[str, dict]] = []

    async def capture(channel: str, payload: str) -> None:
        published.append((channel, json.loads(payload)))

    broker._client.publish = AsyncMock(side_effect=capture)
    with patch.object(svc, "_embed_text", new_callable=AsyncMock) as mock_embed_text:
        mock_embed_text.return_value = [1.0, 0.0]
        await handler({"type": "search.requested", "id": "req-1", "query": "apple", "top_k": 5})

    assert len(published) == 1
    channel, body = published[0]
    assert channel == Channels.SEARCH_RESULTS_READY
    assert body["request_id"] == "req-1"
    assert [row["image_id"] for row in body["results"]] == ["img-1", "img-2"]
    assert body["results"][0]["score"] == pytest.approx(1.0)
    assert body["results"][1]["score"] == pytest.approx(0.0)
    assert body["results"][0]["embedding_model"] == "test-model"
    assert body["results"][0]["embedding_dim"] == 2


@pytest.mark.asyncio
async def test_search_filters_results_below_similarity_threshold(broker):
    collection = _FakeCollection(
        {
            "img-1": {
                "_id": "img-1",
                "image_id": "img-1",
                "path": "/photos/apple.jpg",
                "embedding": [0.0, 1.0],
                "embedding_model": "test-model",
                "embedding_dim": 2,
            }
        }
    )
    svc = VectorDBService(broker, collection=collection, embedding_model_name="test-model")
    svc._search_min_score = 0.5
    handler = broker._handlers[Channels.SEARCH_REQUESTED][0]
    published: list[tuple[str, dict]] = []

    async def capture(channel: str, payload: str) -> None:
        published.append((channel, json.loads(payload)))

    broker._client.publish = AsyncMock(side_effect=capture)
    with patch.object(svc, "_embed_text", new_callable=AsyncMock) as mock_embed_text:
        mock_embed_text.return_value = [1.0, 0.0]
        await handler({"type": "search.requested", "id": "req-1", "query": "apple", "top_k": 5})

    assert published[0][1]["results"] == []


@pytest.mark.asyncio
async def test_search_backfills_legacy_record_embedding(broker):
    collection = _FakeCollection(
        {
            "img-1": {
                "_id": "img-1",
                "image_id": "img-1",
                "path": "/photos/apple.jpg",
            }
        }
    )
    svc = VectorDBService(broker, collection=collection, embedding_model_name="test-model")
    handler = broker._handlers[Channels.SEARCH_REQUESTED][0]
    published: list[tuple[str, dict]] = []

    async def capture(channel: str, payload: str) -> None:
        published.append((channel, json.loads(payload)))

    broker._client.publish = AsyncMock(side_effect=capture)
    with (
        patch.object(svc, "_embed_text", new_callable=AsyncMock) as mock_embed_text,
        patch.object(svc, "_embed_image", new_callable=AsyncMock) as mock_embed_image,
    ):
        mock_embed_text.return_value = [1.0, 0.0]
        mock_embed_image.return_value = [1.0, 0.0]
        await handler({"type": "search.requested", "id": "req-1", "query": "apple", "top_k": 5})

    assert published[0][1]["results"][0]["image_id"] == "img-1"
    assert collection.records["img-1"]["embedding"] == [1.0, 0.0]
    assert collection.records["img-1"]["embedding_model"] == "test-model"


@pytest.mark.asyncio
async def test_search_skips_unusable_records_for_other_embedding_models(broker):
    collection = _FakeCollection(
        {
            "img-1": {
                "_id": "img-1",
                "image_id": "img-1",
                "path": "",
                "embedding": [1.0, 0.0],
                "embedding_model": "other-model",
                "embedding_dim": 2,
            }
        }
    )
    svc = VectorDBService(broker, collection=collection, embedding_model_name="test-model")
    handler = broker._handlers[Channels.SEARCH_REQUESTED][0]
    published: list[tuple[str, dict]] = []

    async def capture(channel: str, payload: str) -> None:
        published.append((channel, json.loads(payload)))

    broker._client.publish = AsyncMock(side_effect=capture)
    with patch.object(svc, "_embed_text", new_callable=AsyncMock) as mock_embed_text:
        mock_embed_text.return_value = [1.0, 0.0]
        await handler({"type": "search.requested", "id": "req-1", "query": "apple", "top_k": 5})

    assert published[0][1]["results"] == []


def test_normalize_embedding_rejects_invalid_values():
    assert _normalize_embedding([1, "2.5"]) == [1.0, 2.5]
    assert _normalize_embedding([1, float("nan")]) == []
    assert _normalize_embedding([True, 1.0]) == []


def test_cosine_similarity_handles_vectors():
    assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert _cosine_similarity([1.0], [1.0, 0.0]) == 0.0


@pytest.mark.asyncio
async def test_vector_db_clears_records_and_publishes_completion(broker):
    collection = _FakeCollection(
        {
            "img-1": {"_id": "img-1", "embedding": [1.0, 0.0]},
            "img-2": {"_id": "img-2", "embedding": [0.0, 1.0]},
        }
    )
    VectorDBService(broker, collection=collection, embedding_model_name="test-model")

    published: list[tuple[str, dict]] = []

    async def capture(channel: str, payload: str) -> None:
        published.append((channel, json.loads(payload)))

    broker._client.publish = AsyncMock(side_effect=capture)
    handler = broker._handlers[Channels.STORAGE_CLEAR_REQUESTED][0]
    await handler({"type": "storage.clear_requested", "id": "req-clear-2"})

    assert collection.records == {}
    assert len(published) == 1
    channel, body = published[0]
    assert channel == Channels.STORAGE_CLEAR_COMPLETED
    assert body["service"] == "vectors"
    assert body["deleted_count"] == 2
    assert body["request_id"] == "req-clear-2"
