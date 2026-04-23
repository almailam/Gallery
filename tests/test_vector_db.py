import json
from unittest.mock import AsyncMock, patch

import pytest

from gallery.broker.pubsub import Channels, RedisBroker
from gallery.services.vector_db import (
    VectorDBService,
    _message_content_to_text,
    _parse_annotations_from_model_text,
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
