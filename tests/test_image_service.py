"""Unit tests for ImageService."""

import json
from unittest.mock import AsyncMock

import pytest

from gallery.broker.pubsub import Channels, RedisBroker
from gallery.services.image_service import ImageService


@pytest.fixture
def broker():
    b = RedisBroker()
    b._client = AsyncMock()
    return b


@pytest.mark.asyncio
async def test_image_service_publishes_accepted_on_upload(broker):
    ImageService(broker)
    handler = broker._handlers[Channels.IMAGE_UPLOAD_REQUESTED][0]

    published: list[tuple[str, dict]] = []

    async def capture(channel: str, payload: str) -> None:
        published.append((channel, json.loads(payload)))

    broker._client.publish = AsyncMock(side_effect=capture)
    await handler({"type": "image.upload_requested", "path": "/photos/dog.jpg"})

    channels = [ch for ch, _ in published]
    assert Channels.IMAGE_ACCEPTED in channels


@pytest.mark.asyncio
async def test_image_service_publishes_embedding_requested_on_upload(broker):
    ImageService(broker)
    handler = broker._handlers[Channels.IMAGE_UPLOAD_REQUESTED][0]

    published: list[tuple[str, dict]] = []

    async def capture(channel: str, payload: str) -> None:
        published.append((channel, json.loads(payload)))

    broker._client.publish = AsyncMock(side_effect=capture)
    await handler({"type": "image.upload_requested", "path": "/photos/dog.jpg"})

    channels = [ch for ch, _ in published]
    assert Channels.IMAGE_EMBEDDING_REQUESTED in channels


@pytest.mark.asyncio
async def test_image_service_assigns_uuid_image_id(broker):
    ImageService(broker)
    handler = broker._handlers[Channels.IMAGE_UPLOAD_REQUESTED][0]

    published: list[tuple[str, dict]] = []

    async def capture(channel: str, payload: str) -> None:
        published.append((channel, json.loads(payload)))

    broker._client.publish = AsyncMock(side_effect=capture)
    await handler({"type": "image.upload_requested", "path": "/photos/dog.jpg"})

    accepted = next(body for ch, body in published if ch == Channels.IMAGE_ACCEPTED)
    assert accepted["image_id"]
    # UUID format: 36 chars with hyphens
    assert len(accepted["image_id"]) == 36


@pytest.mark.asyncio
async def test_image_service_same_image_id_for_accepted_and_embedding(broker):
    """Both IMAGE_ACCEPTED and IMAGE_EMBEDDING_REQUESTED must carry the same image_id."""
    ImageService(broker)
    handler = broker._handlers[Channels.IMAGE_UPLOAD_REQUESTED][0]

    published: list[tuple[str, dict]] = []

    async def capture(channel: str, payload: str) -> None:
        published.append((channel, json.loads(payload)))

    broker._client.publish = AsyncMock(side_effect=capture)
    await handler({"type": "image.upload_requested", "path": "/photos/cat.jpg"})

    by_channel = {ch: body for ch, body in published}
    assert by_channel[Channels.IMAGE_ACCEPTED]["image_id"] == by_channel[Channels.IMAGE_EMBEDDING_REQUESTED]["image_id"]


@pytest.mark.asyncio
async def test_image_service_preserves_path(broker):
    ImageService(broker)
    handler = broker._handlers[Channels.IMAGE_UPLOAD_REQUESTED][0]

    published: list[tuple[str, dict]] = []

    async def capture(channel: str, payload: str) -> None:
        published.append((channel, json.loads(payload)))

    broker._client.publish = AsyncMock(side_effect=capture)
    await handler({"type": "image.upload_requested", "path": "/photos/my image.jpg"})

    accepted = next(body for ch, body in published if ch == Channels.IMAGE_ACCEPTED)
    embedding_req = next(body for ch, body in published if ch == Channels.IMAGE_EMBEDDING_REQUESTED)
    assert accepted["path"] == "/photos/my image.jpg"
    assert embedding_req["path"] == "/photos/my image.jpg"


@pytest.mark.asyncio
async def test_image_service_generates_unique_ids_per_upload(broker):
    ImageService(broker)
    handler = broker._handlers[Channels.IMAGE_UPLOAD_REQUESTED][0]

    ids: list[str] = []

    async def capture(channel: str, payload: str) -> None:
        body = json.loads(payload)
        if channel == Channels.IMAGE_ACCEPTED:
            ids.append(body["image_id"])

    broker._client.publish = AsyncMock(side_effect=capture)
    await handler({"type": "image.upload_requested", "path": "/a.jpg"})
    await handler({"type": "image.upload_requested", "path": "/b.jpg"})

    assert len(ids) == 2
    assert ids[0] != ids[1]
