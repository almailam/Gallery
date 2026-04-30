"""Async Redis pub/sub broker."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from typing import Awaitable, Callable

import redis.asyncio as aioredis

from gallery.messages import Message

log = logging.getLogger(__name__)

Handler = Callable[[dict], Awaitable[None]]


def _as_str(value: object) -> str:
    """Redis pub/sub may return channel names as bytes even with decode_responses."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _use_ansi_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR", "") == ""


_RESET = "\033[0m"
_DIM = "\033[2m"
_BOLD = "\033[1m"

# Distinct hues per application message.type (payload JSON), not Redis pub/sub "type".
_TYPE_COLOR: dict[str, str] = {
    "image.upload_requested": "\033[38;5;214m",  # orange
    "image.accepted": "\033[32m",  # green
    "image.embedding_requested": "\033[38;5;99m",  # purple
    "image.stored": "\033[38;5;118m",  # yellow-green
    "image.pipeline_complete": "\033[38;5;51m",  # cyan
    "image.list_requested": "\033[38;5;75m",
    "image.list_ready": "\033[38;5;81m",
    "search.requested": "\033[38;5;208m",  # dark orange
    "search.results_ready": "\033[35m",  # magenta
    "storage.clear_requested": "\033[38;5;196m",  # bright red
    "storage.clear_completed": "\033[38;5;202m",  # red-orange
}
_DEFAULT_TYPE_COLOR = "\033[36m"


def _ansi_for_message_type(mtype: str) -> str:
    return _TYPE_COLOR.get(mtype, _DEFAULT_TYPE_COLOR)


def _redis_tag(*, use_color: bool) -> str:
    if not use_color:
        return "[REDIS]"
    return f"{_BOLD}\033[95m[REDIS]{_RESET}"


def _format_connected_line(url: str, *, use_color: bool) -> str:
    if not use_color:
        return f"Connected to Redis at {url}"
    tag = _redis_tag(use_color=use_color)
    return f"{tag} {_BOLD}Connected{_RESET} to Redis at {_DIM}{url}{_RESET}"


def _format_listening_line(channels: list[str], *, use_color: bool) -> str:
    if not use_color:
        return f"Listening on channels: {channels}"
    tag = _redis_tag(use_color=use_color)
    listed = ", ".join(f"{_DIM}{c}{_RESET}" for c in channels)
    return f"{tag} {_BOLD}Listening{_RESET} on channels: {listed}"


def _format_incoming_redis_log_line(channel: str, data: dict, *, use_color: bool) -> str:
    """One log line per incoming Redis message: label, channel, colored type, full JSON body."""
    mtype = str(data.get("type", "") or "?")
    payload = json.dumps(data, ensure_ascii=False)

    if not use_color:
        return f"[REDIS] channel={channel} type={mtype} raw={payload}"

    tag = _redis_tag(use_color=use_color)
    ch = f"{_DIM}{channel}{_RESET}"
    tc = _ansi_for_message_type(mtype)
    type_seg = f"type={tc}{mtype}{_RESET}"
    body = f"raw={_DIM}{payload}{_RESET}"
    return f"{tag} {ch}  {type_seg}  {body}"


class Channels:
    IMAGE_UPLOAD_REQUESTED = "gallery:image:upload_requested"
    IMAGE_ACCEPTED = "gallery:image:accepted"
    IMAGE_EMBEDDING_REQUESTED = "gallery:image:embedding_requested"
    IMAGE_STORED = "gallery:image:stored"
    IMAGE_PIPELINE_COMPLETE = "gallery:image:pipeline_complete"
    IMAGE_LIST_REQUESTED = "gallery:image:list_requested"
    IMAGE_LIST_READY = "gallery:image:list_ready"
    SEARCH_REQUESTED = "gallery:search:requested"
    SEARCH_RESULTS_READY = "gallery:search:results_ready"
    STORAGE_CLEAR_REQUESTED = "gallery:storage:clear_requested"
    STORAGE_CLEAR_COMPLETED = "gallery:storage:clear_completed"


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
        # Separate clients so the subscriber connection is not shared with PUBLISH.
        self._client: aioredis.Redis | None = None
        self._listener: aioredis.Redis | None = None
        self._handlers: dict[str, list[Handler]] = {}

    async def connect(self) -> None:
        self._client = aioredis.from_url(self._url, decode_responses=True)
        self._listener = aioredis.from_url(self._url, decode_responses=True)
        # Force a real connection now (redis-py otherwise connects lazily).
        await self._client.ping()
        await self._listener.ping()
        log.info("%s", _format_connected_line(self._url, use_color=_use_ansi_color()))

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
        if self._listener:
            await self._listener.aclose()
            self._listener = None

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
        channels = list(self._handlers)
        if not channels:
            return

        assert self._listener, "Call connect() first"

        pubsub = self._listener.pubsub()
        await pubsub.subscribe(*channels)
        log.info("%s", _format_listening_line(channels, use_color=_use_ansi_color()))

        async for raw in pubsub.listen():
            if raw["type"] != "message":
                continue

            channel = _as_str(raw["channel"])
            payload = raw["data"]
            if isinstance(payload, bytes):
                payload = payload.decode("utf-8", errors="replace")
            try:
                data: dict = json.loads(payload)
            except json.JSONDecodeError:
                log.warning("Malformed message on %s", channel)
                continue

            log.info("%s", _format_incoming_redis_log_line(channel, data, use_color=_use_ansi_color()))

            for handler in self._handlers.get(channel, []):

                async def _run(
                    h: Handler = handler,
                    ch: str = channel,
                    msg: dict = data,
                ) -> None:
                    try:
                        await h(msg)
                    except Exception:
                        log.exception("Handler failed on %s", ch)

                asyncio.create_task(_run())
