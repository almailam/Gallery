import pytest

from event_generator import build_trigger, parse_events
from gallery.broker.pubsub import Channels


def test_parse_events_accepts_comma_list():
    assert parse_events("upload, search,list") == ["upload", "search", "list"]


def test_parse_events_rejects_unknown():
    with pytest.raises(ValueError):
        parse_events("upload,unknown")


def test_build_trigger_upload():
    channel, msg = build_trigger(
        "upload",
        upload_path="samples/bananas.jpg",
        search_query="fruit",
        search_top_k=5,
    )
    assert channel == Channels.IMAGE_UPLOAD_REQUESTED
    assert msg.type == "image.upload_requested"
    assert msg.path == "samples/bananas.jpg"


def test_build_trigger_search():
    channel, msg = build_trigger(
        "search",
        upload_path="samples/bananas.jpg",
        search_query="cats",
        search_top_k=3,
    )
    assert channel == Channels.SEARCH_REQUESTED
    assert msg.type == "search.requested"
    assert msg.query == "cats"
    assert msg.top_k == 3


def test_build_trigger_list():
    channel, msg = build_trigger(
        "list",
        upload_path="samples/bananas.jpg",
        search_query="cats",
        search_top_k=3,
    )
    assert channel == Channels.IMAGE_LIST_REQUESTED
    assert msg.type == "image.list_requested"
