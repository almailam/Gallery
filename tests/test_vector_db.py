import json
from unittest.mock import AsyncMock, patch

import pytest

from gallery.broker.pubsub import Channels, RedisBroker
from gallery.services.vector_db import (
    VectorDBService,
    _cosine_similarity,
    _embedding_model_cache_dir,
    _embedding_model_name,
    _embedding_model_repo_id,
    _filename_match_score,
    _normalize_embedding,
    _query_variants,
    _safe_model_dir_name,
)
from tests.helpers import FakeCollection


@pytest.fixture
def broker():
    b = RedisBroker()
    b._client = AsyncMock()
    return b


@pytest.mark.asyncio
async def test_embedding_reuses_stored_embedding_without_model_call(broker):
    collection = FakeCollection(
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
    collection = FakeCollection()
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
    collection = FakeCollection()
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
    collection = FakeCollection()
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
    collection = FakeCollection(
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
        mock_embed_text.return_value = [[1.0, 0.0]]
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
    collection = FakeCollection(
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
        mock_embed_text.return_value = [[1.0, 0.0]]
        await handler({"type": "search.requested", "id": "req-1", "query": "apple", "top_k": 5})

    assert published[0][1]["results"] == []


@pytest.mark.asyncio
async def test_search_backfills_legacy_record_embedding(broker):
    collection = FakeCollection(
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
        mock_embed_text.return_value = [[1.0, 0.0]]
        mock_embed_image.return_value = [1.0, 0.0]
        await handler({"type": "search.requested", "id": "req-1", "query": "apple", "top_k": 5})

    assert published[0][1]["results"][0]["image_id"] == "img-1"
    assert collection.records["img-1"]["embedding"] == [1.0, 0.0]
    assert collection.records["img-1"]["embedding_model"] == "test-model"


@pytest.mark.asyncio
async def test_search_skips_unusable_records_for_other_embedding_models(broker):
    collection = FakeCollection(
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
        mock_embed_text.return_value = [[1.0, 0.0]]
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


def test_query_variants_include_clip_friendly_prompts():
    assert _query_variants("  red dog  ") == [
        "red dog",
        "a photo of red dog",
        "a picture of red dog",
        "an image of red dog",
    ]


def test_filename_match_score_boosts_path_name_matches():
    assert _filename_match_score("patrick", "/photos/patrick.png") > 0
    assert _filename_match_score("red dog", "/photos/red-dog.jpg") > 0
    assert _filename_match_score("cat", "/photos/red-dog.jpg") == 0.0


def test_embedding_defaults_use_larger_local_model_cache(monkeypatch):
    monkeypatch.delenv("GALLERY_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("GALLERY_EMBEDDING_MODEL_CACHE", raising=False)

    assert _embedding_model_name() == "clip-ViT-L-14"
    assert _embedding_model_cache_dir().name == ".gallery_models"


def test_model_repo_and_local_dir_names():
    assert _embedding_model_repo_id("clip-ViT-L-14") == "sentence-transformers/clip-ViT-L-14"
    assert _embedding_model_repo_id("org/model") == "org/model"
    assert _safe_model_dir_name("clip-ViT-L-14") == "sentence-transformers__clip-ViT-L-14"


def test_local_model_path_uses_configured_cache(broker, tmp_path, monkeypatch):
    monkeypatch.setenv("GALLERY_EMBEDDING_MODEL_CACHE", str(tmp_path / "models"))
    svc = VectorDBService(broker, collection=FakeCollection(), embedding_model_name="clip-ViT-L-14")

    assert svc._local_model_path() == tmp_path / "models" / "sentence-transformers__clip-ViT-L-14"


def test_local_model_path_reuses_huggingface_snapshot_cache(broker, tmp_path, monkeypatch):
    monkeypatch.setenv("GALLERY_EMBEDDING_MODEL_CACHE", str(tmp_path / "models"))
    snapshot = (
        tmp_path
        / "models"
        / "models--sentence-transformers--clip-ViT-L-14"
        / "snapshots"
        / "abc123"
    )
    snapshot.mkdir(parents=True)
    (snapshot / "modules.json").write_text("{}", encoding="utf-8")
    svc = VectorDBService(broker, collection=FakeCollection(), embedding_model_name="clip-ViT-L-14")

    assert svc._local_model_path() == snapshot


def test_ensure_local_model_reuses_downloaded_snapshot(broker, tmp_path, monkeypatch):
    monkeypatch.setenv("GALLERY_EMBEDDING_MODEL_CACHE", str(tmp_path / "models"))
    local_model = tmp_path / "models" / "sentence-transformers__clip-ViT-L-14"
    local_model.mkdir(parents=True)
    (local_model / "modules.json").write_text("{}", encoding="utf-8")
    svc = VectorDBService(broker, collection=FakeCollection(), embedding_model_name="clip-ViT-L-14")

    with patch("huggingface_hub.snapshot_download") as mock_snapshot:
        assert svc._ensure_local_model() == local_model
        mock_snapshot.assert_not_called()


@pytest.mark.asyncio
async def test_prepare_model_loads_model_once_when_enabled(broker, tmp_path, monkeypatch):
    monkeypatch.setenv("GALLERY_PRELOAD_EMBEDDING_MODEL", "1")
    monkeypatch.setenv("GALLERY_EMBEDDING_MODEL_CACHE", str(tmp_path / "models"))
    model = object()
    svc = VectorDBService(
        broker,
        collection=FakeCollection(),
        embedding_model=model,
        embedding_model_name="test-model",
    )

    assert await svc.prepare_model() is True
    assert svc._model() is model


@pytest.mark.asyncio
async def test_prepare_model_can_be_disabled(broker, monkeypatch):
    monkeypatch.setenv("GALLERY_PRELOAD_EMBEDDING_MODEL", "0")
    svc = VectorDBService(broker, collection=FakeCollection(), embedding_model_name="test-model")

    assert await svc.prepare_model() is False


@pytest.mark.asyncio
async def test_vector_db_clears_records_and_publishes_completion(broker):
    collection = FakeCollection(
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


@pytest.mark.asyncio
async def test_embedding_success_publishes_pipeline_complete(broker):
    collection = FakeCollection()
    svc = VectorDBService(broker, collection=collection, embedding_model_name="test-model")
    handler = broker._handlers[Channels.IMAGE_EMBEDDING_REQUESTED][0]
    published: list[tuple[str, dict]] = []

    async def capture(channel: str, payload: str) -> None:
        published.append((channel, json.loads(payload)))

    broker._client.publish = AsyncMock(side_effect=capture)
    with patch.object(svc, "_embed_image", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = [0.6, 0.8]
        await handler({"image_id": "img-1", "path": "/x.jpg", "type": "image.embedding_requested"})

    channels = [ch for ch, _ in published]
    assert Channels.IMAGE_PIPELINE_COMPLETE in channels
    pipeline_body = next(body for ch, body in published if ch == Channels.IMAGE_PIPELINE_COMPLETE)
    assert pipeline_body["image_id"] == "img-1"
    assert pipeline_body["path"] == "/x.jpg"


@pytest.mark.asyncio
async def test_embedding_failure_does_not_publish_pipeline_complete(broker):
    collection = FakeCollection()
    svc = VectorDBService(broker, collection=collection, embedding_model_name="test-model")
    handler = broker._handlers[Channels.IMAGE_EMBEDDING_REQUESTED][0]
    published: list[tuple[str, dict]] = []

    async def capture(channel: str, payload: str) -> None:
        published.append((channel, json.loads(payload)))

    broker._client.publish = AsyncMock(side_effect=capture)
    with patch.object(svc, "_embed_image", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = []
        await handler({"image_id": "img-1", "path": "/x.jpg", "type": "image.embedding_requested"})

    channels = [ch for ch, _ in published]
    assert Channels.IMAGE_PIPELINE_COMPLETE not in channels
    assert Channels.IMAGE_EMBEDDING_FAILED in channels


@pytest.mark.asyncio
async def test_prepare_model_handles_import_error(broker, monkeypatch):
    svc = VectorDBService(broker, collection=FakeCollection(), embedding_model_name="test-model")

    def _raise_import(*_args, **_kwargs):
        raise ImportError("sentence_transformers not installed")

    with patch.object(svc, "_model", side_effect=_raise_import):
        result = await svc.prepare_model()

    assert result is False
    assert "unavailable" in svc._last_embedding_error


def test_normalize_embedding_empty_list():
    assert _normalize_embedding([]) == []


def test_normalize_embedding_non_list():
    assert _normalize_embedding("not-a-list") == []
    assert _normalize_embedding(None) == []


def test_cosine_similarity_identical_vectors():
    assert _cosine_similarity([0.6, 0.8], [0.6, 0.8]) == pytest.approx(1.0)


def test_cosine_similarity_zero_norm():
    assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_query_variants_empty_string():
    assert _query_variants("") == []
    assert _query_variants("   ") == []
