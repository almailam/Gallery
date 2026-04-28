import json

import pytest

from gallery.services.json_collection import JsonCollection


@pytest.mark.asyncio
async def test_json_collection_upserts_and_persists_records(tmp_path):
    path = tmp_path / "documents.json"
    collection = JsonCollection(path)

    await collection.update_one(
        {"_id": "img-1"},
        {
            "$setOnInsert": {"_id": "img-1"},
            "$set": {"image_id": "img-1", "path": "/image.jpg"},
        },
        upsert=True,
    )

    assert await collection.find_one({"_id": "img-1"}) == {
        "_id": "img-1",
        "image_id": "img-1",
        "path": "/image.jpg",
    }
    assert json.loads(path.read_text(encoding="utf-8"))["records"]["img-1"]["path"] == "/image.jpg"


@pytest.mark.asyncio
async def test_json_collection_find_returns_async_snapshot(tmp_path):
    path = tmp_path / "vectors.json"
    path.write_text(
        json.dumps(
            {
                "records": {
                    "img-1": {"_id": "img-1", "annotations": ["apple"]},
                    "img-2": {"_id": "img-2", "annotations": ["banana"]},
                }
            }
        ),
        encoding="utf-8",
    )
    collection = JsonCollection(path)

    records = [record async for record in collection.find({})]

    assert records == [
        {"_id": "img-1", "annotations": ["apple"]},
        {"_id": "img-2", "annotations": ["banana"]},
    ]
