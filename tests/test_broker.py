from unittest.mock import AsyncMock

import pytest

from gallery.broker.pubsub import (
    Channels,
    RedisBroker,
    _ansi_for_message_type,
    _as_str,
    _format_connected_line,
    _format_incoming_redis_log_line,
    _format_listening_line,
)
from gallery.messages import ImageUploadRequested


@pytest.fixture
def broker():
    b = RedisBroker()
    b._client = AsyncMock()
    return b


def test_format_connected_and_listening_plain():
    c = _format_connected_line("redis://localhost:6379", use_color=False)
    assert "Connected to Redis" in c
    assert "redis://localhost:6379" in c
    assert "\033[" not in c

    lst = _format_listening_line(["gallery:a", "gallery:b"], use_color=False)
    assert "Listening on channels" in lst
    assert "gallery:a" in lst
    assert "\033[" not in lst


def test_format_connected_and_listening_colored():
    c = _format_connected_line("redis://localhost:6379", use_color=True)
    assert "\033[" in c
    assert "[REDIS]" in c

    lst = _format_listening_line(["gallery:image:x"], use_color=True)
    assert "\033[" in lst
    assert "gallery:image:x" in lst


def test_format_incoming_redis_plain_no_escapes():
    line = _format_incoming_redis_log_line(
        "gallery:image:accepted",
        {"type": "image.accepted", "id": "x"},
        use_color=False,
    )
    assert "[REDIS]" in line
    assert "gallery:image:accepted" in line
    assert "image.accepted" in line
    assert "\033[" not in line


def test_format_incoming_redis_colored_has_escapes():
    line = _format_incoming_redis_log_line(
        "gallery:image:accepted",
        {"type": "image.accepted"},
        use_color=True,
    )
    assert "\033[" in line
    assert "[REDIS]" in line


def test_ansi_for_message_type_known_types_distinct():
    types = (
        "image.upload_requested",
        "image.accepted",
        "image.annotation_requested",
        "image.embedding_requested",
        "image.stored",
        "image.pipeline_complete",
        "search.requested",
        "search.results_ready",
    )
    codes = {_ansi_for_message_type(t) for t in types}
    assert len(codes) == len(types)


def test_as_str_normalizes_redis_channel_keys():
    assert _as_str("gallery:image:upload_requested") == "gallery:image:upload_requested"
    assert _as_str(b"gallery:image:upload_requested") == "gallery:image:upload_requested"


def test_as_str_non_bytes_coerced_to_str():
    assert _as_str(123) == "123"


def test_ansi_for_unknown_message_type_uses_default():
    unknown = "totally.unknown.type"
    assert _ansi_for_message_type(unknown) == _ansi_for_message_type("also.unknown")


def test_channels_constants():
    assert Channels.IMAGE_UPLOAD_REQUESTED == "gallery:image:upload_requested"
    assert Channels.IMAGE_ACCEPTED == "gallery:image:accepted"
    assert Channels.IMAGE_ANNOTATION_REQUESTED == "gallery:image:annotation_requested"
    assert Channels.IMAGE_EMBEDDING_REQUESTED == "gallery:image:embedding_requested"
    assert Channels.IMAGE_STORED == "gallery:image:stored"
    assert Channels.IMAGE_PIPELINE_COMPLETE == "gallery:image:pipeline_complete"
    assert Channels.SEARCH_REQUESTED == "gallery:search:requested"
    assert Channels.SEARCH_RESULTS_READY == "gallery:search:results_ready"


def test_handler_registration(broker):
    @broker.on(Channels.IMAGE_UPLOAD_REQUESTED)
    async def handler(msg): ...

    assert Channels.IMAGE_UPLOAD_REQUESTED in broker._handlers
    assert handler in broker._handlers[Channels.IMAGE_UPLOAD_REQUESTED]


def test_multiple_handlers_same_channel(broker):
    @broker.on(Channels.IMAGE_UPLOAD_REQUESTED)
    async def h1(msg): ...

    @broker.on(Channels.IMAGE_UPLOAD_REQUESTED)
    async def h2(msg): ...

    assert len(broker._handlers[Channels.IMAGE_UPLOAD_REQUESTED]) == 2


@pytest.mark.asyncio
async def test_publish_calls_redis(broker):
    msg = ImageUploadRequested(path="/img.jpg")
    await broker.publish(Channels.IMAGE_UPLOAD_REQUESTED, msg)

    broker._client.publish.assert_awaited_once_with(
        Channels.IMAGE_UPLOAD_REQUESTED,
        msg.to_json(),
    )


@pytest.mark.asyncio
async def test_publish_uses_message_json(broker):
    msg = ImageUploadRequested(path="/cats.jpg")
    await broker.publish(Channels.IMAGE_UPLOAD_REQUESTED, msg)

    _, call_payload = broker._client.publish.call_args.args
    import json
    data = json.loads(call_payload)
    assert data["type"] == "image.upload_requested"
    assert data["path"] == "/cats.jpg"


@pytest.mark.asyncio
async def test_listen_no_handlers_returns_early(broker):
    # Should return immediately without touching the client.
    await broker.listen()
    broker._client.pubsub.assert_not_called()
