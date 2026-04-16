"""CLI service - publishes user commands and prints results."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from gallery.broker.pubsub import Channels, RedisBroker
from gallery.messages import ImageUploadRequested, SearchRequested

log = logging.getLogger(__name__)


class CLIService:
    def __init__(self, broker: RedisBroker) -> None:
        self.broker = broker
        self._register_handlers()

    def _register_handlers(self) -> None:
        @self.broker.on(Channels.SEARCH_RESULTS_READY)
        async def on_results(msg: dict) -> None:
            print(f"[CLI] Search results:")
            for r in msg.get("results", []):
                print(f"       {r}")

        @self.broker.on(Channels.IMAGE_PIPELINE_COMPLETE)
        async def on_upload_done(msg: dict) -> None:
            print(
                f"[CLI] Upload finished: image_id={msg.get('image_id')} path={msg.get('path')}"
            )

    async def upload_image(self, path: str) -> None:
        try:
            resolved = Path(path).expanduser().resolve()
        except (OSError, RuntimeError) as e:
            print(f"[CLI] Error: invalid path {path!r}: {e}")
            return
        if not resolved.is_file():
            print(f"[CLI] Error: file not found: {path}")
            print(
                "[CLI] Hint: use an absolute path, or run the app from the project "
                "folder and try e.g. image_samples/apple.jpeg"
            )
            return
        msg = ImageUploadRequested(path=str(resolved))
        try:
            await self.broker.publish(Channels.IMAGE_UPLOAD_REQUESTED, msg)
        except Exception as e:
            print(f"[CLI] Error: could not send upload to Redis: {e}")
            log.exception("Upload publish failed")
            return
        log.info("Requested upload for %s", resolved)

    async def search(self, query: str) -> None:
        msg = SearchRequested(query=query)
        await self.broker.publish(Channels.SEARCH_REQUESTED, msg)
        log.info("Search requested: %r", query)

    async def run_interactive(self) -> None:
        loop = asyncio.get_event_loop()
        print("\nCommands:  upload <path>  |  search <query>  |  quit")

        while True:
            line: str = await loop.run_in_executor(None, input, "> ")
            line = line.strip()
            if not line:
                continue
            if line in ("quit", "exit", "q"):
                break
            parts = line.split(maxsplit=1)
            cmd, arg = parts[0].lower(), parts[1] if len(parts) > 1 else ""

            if cmd == "upload" and arg:
                await self.upload_image(arg)
            elif cmd == "search" and arg:
                await self.search(arg)
            else:
                print(f"Usage:  upload <path>  |  search <query>")
