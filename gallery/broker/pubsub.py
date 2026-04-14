"""Async Redis pub/sub broker."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Awaitable, Callable

import redis.asyncio as aioredis

from gallery.messages import Message

log = logging.getLogger(__name__)

Handler = Callable[[dict], Awaitable[None]]


class Channels:
    IMAGE_UPLOAD_REQUESTED = "gallery:image:upload_requested"
    IMAGE_ANNOTATED = "gallery:image:annotated"
    IMAGE_EMBEDDED = "gallery:image:embedded"
    SEARCH_REQUESTED = "gallery:search:requested"
    SEARCH_RESULTS_READY = "gallery:search:results_ready"


class RedisBroker:
    """
    Thin wrapper around Redis pub/sub.

    Usage
    -----
    broker = RedisBroker()
    await broker.connect()

    # subscribe
    @broker.on(Channels.IMAGE_UPLOAD_REQUESTED)
    async def handle(msg: dict): ...

    # publish
    await broker.publish(Channels.IMAGE_UPLOAD_REQUESTED, some_message)

    # run the listener loop (blocks until cancelled)
    await broker.listen()
    """

    def __init__(self, url: str = "redis://localhost:6379") -> None:
        self._url = url
        self._client: aioredis.Redis | None = None
        self._handlers: dict[str, list[Handler]] = {}

    async def connect(self) -> None:
        self._client = aioredis.from_url(self._url, decode_responses=True)
        log.info("Connected to Redis at %s", self._url)

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()

    def on(self, channel: str) -> Callable[[Handler], Handler]:
        """Decorator that registers a handler for a channel."""

        def decorator(fn: Handler) -> Handler:
            self._handlers.setdefault(channel, []).append(fn)
            return fn

        return decorator

    async def publish(self, channel: str, message: Message) -> None:
        assert self._client, "Call connect() first"
        await self._client.publish(channel, message.to_json())
        log.debug("Published %s -> %s", message.type, channel)

    async def listen(self) -> None:
        """Subscribe to all registered channels and dispatch incoming messages."""
        assert self._client, "Call connect() first"

        channels = list(self._handlers)
        if not channels:
            return

        pubsub = self._client.pubsub()
        await pubsub.subscribe(*channels)
        log.info("Listening on channels: %s", channels)

        async for raw in pubsub.listen():
            if raw["type"] != "message":
                continue

            channel: str = raw["channel"]
            try:
                data: dict = json.loads(raw["data"])
            except json.JSONDecodeError:
                log.warning("Malformed message on %s", channel)
                continue

            for handler in self._handlers.get(channel, []):
                asyncio.create_task(handler(data))
