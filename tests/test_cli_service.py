"""Unit tests for CLIService publish methods and display helpers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gallery.broker.pubsub import Channels, RedisBroker
from gallery.services.cli import CLIService


@pytest.fixture
def broker():
    b = RedisBroker()
    b._client = AsyncMock()
    return b


@pytest.fixture
def cli(broker):
    return CLIService(broker)


# ---------------------------------------------------------------------------
# upload_image
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upload_image_publishes_upload_requested(cli, broker, tmp_path):
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"\xff\xd8\xff")

    published: list[tuple[str, dict]] = []

    async def capture(channel: str, payload: str) -> None:
        published.append((channel, json.loads(payload)))

    broker._client.publish = AsyncMock(side_effect=capture)
    await cli.upload_image(str(image))

    assert len(published) == 1
    channel, body = published[0]
    assert channel == Channels.IMAGE_UPLOAD_REQUESTED
    assert body["type"] == "image.upload_requested"
    assert body["path"] == str(image.resolve())


@pytest.mark.asyncio
async def test_upload_image_missing_file_prints_error_not_publish(cli, broker, tmp_path):
    missing = tmp_path / "does_not_exist.jpg"
    await cli.upload_image(str(missing))
    broker._client.publish.assert_not_awaited()


@pytest.mark.asyncio
async def test_upload_image_broker_error_does_not_raise(cli, broker, tmp_path):
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"\xff\xd8\xff")
    broker._client.publish = AsyncMock(side_effect=RuntimeError("redis down"))
    # Should not raise; error is handled gracefully.
    await cli.upload_image(str(image))


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_publishes_search_requested(cli, broker):
    published: list[tuple[str, dict]] = []

    async def capture(channel: str, payload: str) -> None:
        published.append((channel, json.loads(payload)))

    broker._client.publish = AsyncMock(side_effect=capture)
    await cli.search("fruit bowl")

    assert len(published) == 1
    channel, body = published[0]
    assert channel == Channels.SEARCH_REQUESTED
    assert body["query"] == "fruit bowl"


@pytest.mark.asyncio
async def test_search_broker_error_does_not_raise(cli, broker):
    broker._client.publish = AsyncMock(side_effect=RuntimeError("redis down"))
    await cli.search("cats")


# ---------------------------------------------------------------------------
# list_images
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_images_publishes_list_requested(cli, broker):
    published: list[tuple[str, dict]] = []

    async def capture(channel: str, payload: str) -> None:
        published.append((channel, json.loads(payload)))

    broker._client.publish = AsyncMock(side_effect=capture)
    await cli.list_images()

    assert len(published) == 1
    channel, body = published[0]
    assert channel == Channels.IMAGE_LIST_REQUESTED
    assert body["type"] == "image.list_requested"


@pytest.mark.asyncio
async def test_list_images_broker_error_does_not_raise(cli, broker):
    broker._client.publish = AsyncMock(side_effect=RuntimeError("redis down"))
    await cli.list_images()


# ---------------------------------------------------------------------------
# clear_storage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clear_storage_publishes_clear_requested(cli, broker):
    published: list[tuple[str, dict]] = []

    async def capture(channel: str, payload: str) -> None:
        published.append((channel, json.loads(payload)))

    broker._client.publish = AsyncMock(side_effect=capture)
    await cli.clear_storage()

    assert len(published) == 1
    channel, body = published[0]
    assert channel == Channels.STORAGE_CLEAR_REQUESTED
    assert body["type"] == "storage.clear_requested"


@pytest.mark.asyncio
async def test_clear_storage_broker_error_does_not_raise(cli, broker):
    broker._client.publish = AsyncMock(side_effect=RuntimeError("redis down"))
    await cli.clear_storage()


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def test_embedding_summary_with_dim_and_model():
    row = {"embedding_dim": 768, "embedding_model": "clip-ViT-L-14"}
    summary = CLIService._embedding_summary(row)
    assert "768" in summary
    assert "clip-ViT-L-14" in summary


def test_embedding_summary_not_ready_when_dim_zero():
    assert CLIService._embedding_summary({"embedding_dim": 0}) == "not ready"
    assert CLIService._embedding_summary({}) == "not ready"


def test_format_score_valid_float():
    assert CLIService._format_score(0.9876) == "0.988"
    assert CLIService._format_score("0.5") == "0.500"


def test_format_score_invalid_value():
    assert CLIService._format_score(None) == "n/a"
    assert CLIService._format_score("bad") == "n/a"
