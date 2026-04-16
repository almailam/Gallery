import json

from gallery.messages import (
    ImageAccepted,
    ImageAnnotationRequested,
    ImageEmbeddingRequested,
    ImagePipelineComplete,
    ImageStored,
    ImageUploadRequested,
    Message,
    SearchRequested,
    SearchResultsReady,
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


def test_image_accepted_fields():
    m = ImageAccepted(image_id="abc", path="/img.jpg")
    assert m.type == "image.accepted"
    data = json.loads(m.to_json())
    assert data["image_id"] == "abc"
    assert data["path"] == "/img.jpg"


def test_image_stored_fields():
    m = ImageStored(image_id="abc", path="/img.jpg")
    assert m.type == "image.stored"
    data = json.loads(m.to_json())
    assert data["status"] == "stored"


def test_image_pipeline_complete_fields():
    m = ImagePipelineComplete(image_id="abc", path="/tmp/x.jpg")
    assert m.type == "image.pipeline_complete"
    data = json.loads(m.to_json())
    assert data["image_id"] == "abc"
    assert data["path"] == "/tmp/x.jpg"


def test_image_annotation_requested_fields():
    m = ImageAnnotationRequested(image_id="abc", path="/img.jpg")
    assert m.type == "image.annotation_requested"
    data = json.loads(m.to_json())
    assert data["image_id"] == "abc"


def test_image_embedding_requested_fields():
    m = ImageEmbeddingRequested(image_id="abc", path="/img.jpg")
    assert m.type == "image.embedding_requested"
    data = json.loads(m.to_json())
    assert data["path"] == "/img.jpg"


def test_search_requested_fields():
    m = SearchRequested(query="cats", top_k=3)
    assert m.type == "search.requested"
    data = json.loads(m.to_json())
    assert data["query"] == "cats"
    assert data["top_k"] == 3


def test_search_results_ready_fields():
    m = SearchResultsReady(request_id="r1", results=[])
    assert m.type == "search.results_ready"
    data = json.loads(m.to_json())
    assert data["request_id"] == "r1"
    assert "note" in data
