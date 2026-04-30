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
async def test_json_collection_delete_many_clears_records(tmp_path):
    path = tmp_path / "vectors.json"
    collection = JsonCollection(path)
    await collection.update_one(
        {"_id": "img-1"},
        {"$setOnInsert": {"_id": "img-1"}, "$set": {"image_id": "img-1"}},
        upsert=True,
    )

    await collection.delete_many({})

    assert [record async for record in collection.find({})] == []
    assert json.loads(path.read_text(encoding="utf-8"))["records"] == {}


@pytest.mark.asyncio
async def test_json_collection_find_returns_async_snapshot(tmp_path):
    path = tmp_path / "vectors.json"
    path.write_text(
        json.dumps(
            {
                "records": {
                    "img-1": {"_id": "img-1", "embedding": [1.0, 0.0]},
                    "img-2": {"_id": "img-2", "embedding": [0.0, 1.0]},
                }
            }
        ),
        encoding="utf-8",
    )
    collection = JsonCollection(path)

    records = [record async for record in collection.find({})]

    assert records == [
        {"_id": "img-1", "embedding": [1.0, 0.0]},
        {"_id": "img-2", "embedding": [0.0, 1.0]},
    ]


@pytest.mark.asyncio
async def test_json_collection_find_one_returns_none_for_missing_key(tmp_path):
    collection = JsonCollection(tmp_path / "data.json")
    assert await collection.find_one({"_id": "nonexistent"}) is None


@pytest.mark.asyncio
async def test_json_collection_find_one_returns_none_for_missing_id_field(tmp_path):
    collection = JsonCollection(tmp_path / "data.json")
    # filter without _id should return None
    assert await collection.find_one({}) is None


@pytest.mark.asyncio
async def test_json_collection_update_one_no_upsert_does_not_create(tmp_path):
    collection = JsonCollection(tmp_path / "data.json")
    await collection.update_one(
        {"_id": "img-1"},
        {"$set": {"path": "/img.jpg"}},
        upsert=False,
    )
    assert await collection.find_one({"_id": "img-1"}) is None


@pytest.mark.asyncio
async def test_json_collection_loads_existing_file(tmp_path):
    import json as _json

    path = tmp_path / "data.json"
    path.write_text(
        _json.dumps({"records": {"img-1": {"_id": "img-1", "path": "/existing.jpg"}}}),
        encoding="utf-8",
    )
    collection = JsonCollection(path)
    record = await collection.find_one({"_id": "img-1"})
    assert record is not None
    assert record["path"] == "/existing.jpg"
