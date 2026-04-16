import json

from gallery.messages import (
    ImageAnnotated,
    ImageEmbedded,
    ImageUploadRequested,
    Message,
    SearchRequested,
    SearchResultsReady,
    VectorMeta,
)


def test_message_base_defaults():
    m = Message()
    assert m.type == "message"
    assert m.id
    assert m.timestamp


def test_message_unique_ids():
    assert Message().id != Message().id


def test_message_to_json():
    m = Message()
    data = json.loads(m.to_json())
    assert data["type"] == "message"
    assert data["id"] == m.id
    assert data["timestamp"] == m.timestamp


def test_image_upload_requested_type():
    m = ImageUploadRequested(path="/photos/dog.jpg")
    assert m.type == "image.upload_requested"
    assert m.path == "/photos/dog.jpg"


def test_image_upload_requested_vector_meta_default():
    m = ImageUploadRequested()
    assert isinstance(m.vector_meta, VectorMeta)
    assert m.vector_meta.model == ""
    assert m.vector_meta.dimensions == 0


def test_image_upload_requested_vector_meta_custom():
    meta = VectorMeta(model="clip", dimensions=512)
    m = ImageUploadRequested(path="/img.jpg", vector_meta=meta)
    data = json.loads(m.to_json())
    assert data["vector_meta"] == {"model": "clip", "dimensions": 512}


def test_image_annotated_fields():
    m = ImageAnnotated(image_id="abc", path="/img.jpg", tags=["dog"], caption="a dog")
    assert m.type == "image.annotated"
    data = json.loads(m.to_json())
    assert data["tags"] == ["dog"]
    assert data["caption"] == "a dog"


def test_image_embedded_fields():
    m = ImageEmbedded(image_id="abc", embedding=[0.1, 0.2])
    assert m.type == "image.embedded"
    assert m.embedding == [0.1, 0.2]


def test_search_requested_fields():
    m = SearchRequested(query="dogs playing outside")
    assert m.type == "search.requested"
    assert m.query == "dogs playing outside"


def test_search_results_ready_fields():
    results = [{"image_id": "abc", "path": "/img.jpg"}]
    m = SearchResultsReady(request_id="req-1", results=results)
    assert m.type == "search.results_ready"
    data = json.loads(m.to_json())
    assert data["results"] == results
