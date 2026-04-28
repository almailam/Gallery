import json
from unittest.mock import AsyncMock, patch

import pytest

from gallery.broker.pubsub import Channels, RedisBroker
from gallery.services.vector_db import (
    VectorDBService,
    _message_content_to_text,
    _parse_annotations_from_model_text,
    _stem,
    _token_matches,
)


@pytest.fixture
def broker():
    b = RedisBroker()
    b._client = AsyncMock()
    return b


@pytest.mark.asyncio
async def test_embedding_reuses_stored_annotations_without_model_call(tmp_path, broker):
    db_path = tmp_path / "vector_db.json"
    db_path.write_text(
        json.dumps(
            {
                "records": {
                    "img-1": {
                        "image_id": "img-1",
                        "path": "/old.jpg",
                        "annotations": ["apple", "fruit"],
                        "search_text": "apple fruit",
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    svc = VectorDBService(broker, db_path=db_path)
    handler = broker._handlers[Channels.IMAGE_EMBEDDING_REQUESTED][0]

    with patch.object(svc, "_annotate_image", new_callable=AsyncMock) as mock_ann:
        mock_ann.return_value = ["wrong"]
        await handler({"image_id": "img-1", "path": "/new.jpg", "type": "image.embedding_requested"})
        mock_ann.assert_not_awaited()

    data = json.loads(db_path.read_text(encoding="utf-8"))
    assert data["records"]["img-1"]["annotations"] == ["apple", "fruit"]
    assert data["records"]["img-1"]["path"] == "/new.jpg"


@pytest.mark.asyncio
async def test_embedding_calls_model_once_then_reuses_for_same_image_id(tmp_path, broker):
    db_path = tmp_path / "vector_db.json"
    db_path.write_text(json.dumps({"records": {}}, ensure_ascii=False), encoding="utf-8")
    svc = VectorDBService(broker, db_path=db_path)
    handler = broker._handlers[Channels.IMAGE_EMBEDDING_REQUESTED][0]

    with patch.object(svc, "_annotate_image", new_callable=AsyncMock) as mock_ann:
        mock_ann.return_value = ["a", "b"]
        await handler({"image_id": "img-1", "path": "/x.jpg", "type": "image.embedding_requested"})
        await handler({"image_id": "img-1", "path": "/x.jpg", "type": "image.embedding_requested"})
        assert mock_ann.await_count == 1


@pytest.mark.asyncio
async def test_embedding_reuses_annotations_for_same_file_path_different_id(tmp_path, broker):
    db_path = tmp_path / "vector_db.json"
    db_path.write_text(json.dumps({"records": {}}, ensure_ascii=False), encoding="utf-8")
    image_file = tmp_path / "same.jpg"
    image_file.write_bytes(b"\xff\xd8\xff")

    svc = VectorDBService(broker, db_path=db_path)
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

    data = json.loads(db_path.read_text(encoding="utf-8"))
    assert data["records"]["second"]["annotations"] == ["sky", "clouds"]


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


# ---------------------------------------------------------------------------
# _stem tests
# ---------------------------------------------------------------------------


def test_stem_plural_s():
    assert _stem("dogs") == "dog"
    assert _stem("cats") == "cat"


def test_stem_plural_ies():
    assert _stem("puppies") == "puppy"
    assert _stem("babies") == "baby"


def test_stem_plural_s_no_strip_ss():
    # Words ending in 'ss' must not be modified.
    assert _stem("grass") == "grass"
    assert _stem("class") == "class"


def test_stem_ing():
    assert _stem("running") == "run"
    assert _stem("jumping") == "jump"
    assert _stem("walking") == "walk"


def test_stem_ed():
    assert _stem("walked") == "walk"
    assert _stem("stopped") == "stop"


def test_stem_short_words_unchanged():
    # Words too short to stem safely should come back unchanged.
    assert _stem("go") == "go"
    assert _stem("the") == "the"


def test_stem_no_change_when_no_suffix_matches():
    assert _stem("beach") == "beach"
    assert _stem("dog") == "dog"


# ---------------------------------------------------------------------------
# _token_matches tests
# ---------------------------------------------------------------------------


def test_token_matches_exact_substring():
    tokens = frozenset(["hot", "dog", "on", "the", "beach"])
    assert _token_matches("dog", "hot dog on the beach", tokens)


def test_token_matches_plural_query_singular_annotation():
    """Searching 'dogs' should match an annotation containing 'dog'."""
    tokens = frozenset(["a", "dog", "running"])
    assert _token_matches("dogs", "a dog running", tokens)


def test_token_matches_singular_query_plural_annotation():
    """Searching 'dog' should match an annotation containing 'dogs'."""
    tokens = frozenset(["two", "dogs", "playing"])
    assert _token_matches("dog", "two dogs playing", tokens)


def test_token_matches_inflected_verb():
    tokens = frozenset(["person", "running", "fast"])
    assert _token_matches("run", "person running fast", tokens)
    assert _token_matches("runs", "person running fast", tokens)


def test_token_matches_no_match():
    tokens = frozenset(["cat", "sleeping"])
    assert not _token_matches("dog", "cat sleeping", tokens)


# ---------------------------------------------------------------------------
# Integration: on_search_requested with fuzzy matching
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_matches_plural_query_against_singular_annotation(tmp_path, broker):
    """Querying 'dogs' should find a record annotated with 'dog'."""
    db_path = tmp_path / "vector_db.json"
    db_path.write_text(
        json.dumps(
            {
                "records": {
                    "img-dog": {
                        "image_id": "img-dog",
                        "path": "/photos/pet.jpg",
                        "annotations": ["dog", "grass", "park"],
                        "search_text": "dog grass park",
                        "updated_at": "2024-01-01T00:00:00+00:00",
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    svc = VectorDBService(broker, db_path=db_path)
    handler = broker._handlers[Channels.SEARCH_REQUESTED][0]

    published: list[dict] = []
    broker._client.publish = AsyncMock(
        side_effect=lambda ch, msg: published.append({"channel": ch, "data": json.loads(msg)})
    )

    await handler({"query": "dogs", "top_k": 5, "id": "req-1"})

    assert published, "Expected a SEARCH_RESULTS_READY publish"
    results = published[-1]["data"]["results"]
    assert any(r["image_id"] == "img-dog" for r in results), (
        "Expected img-dog in results when searching for 'dogs'"
    )


@pytest.mark.asyncio
async def test_search_matches_singular_query_against_plural_annotation(tmp_path, broker):
    """Querying 'dog' should find a record annotated with 'dogs'."""
    db_path = tmp_path / "vector_db.json"
    db_path.write_text(
        json.dumps(
            {
                "records": {
                    "img-dogs": {
                        "image_id": "img-dogs",
                        "path": "/photos/pack.jpg",
                        "annotations": ["dogs", "field", "play"],
                        "search_text": "dogs field play",
                        "updated_at": "2024-01-01T00:00:00+00:00",
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    svc = VectorDBService(broker, db_path=db_path)
    handler = broker._handlers[Channels.SEARCH_REQUESTED][0]

    published: list[dict] = []
    broker._client.publish = AsyncMock(
        side_effect=lambda ch, msg: published.append({"channel": ch, "data": json.loads(msg)})
    )

    await handler({"query": "dog", "top_k": 5, "id": "req-2"})

    assert published
    results = published[-1]["data"]["results"]
    assert any(r["image_id"] == "img-dogs" for r in results), (
        "Expected img-dogs in results when searching for 'dog'"
    )

