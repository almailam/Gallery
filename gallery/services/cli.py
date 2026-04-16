"""CLI service - publishes user commands and prints results."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion, PathCompleter
from prompt_toolkit.document import Document
from prompt_toolkit.patch_stdout import patch_stdout

from gallery.broker.pubsub import Channels, RedisBroker
from gallery.messages import ImageUploadRequested, SearchRequested

log = logging.getLogger(__name__)


class _CLICompleter(Completer):
    """Tab completion for top-level commands and upload paths."""

    _commands = ("upload", "search", "quit", "exit", "q")

    def __init__(self) -> None:
        self._path_completer = PathCompleter(expanduser=True)

    def get_completions(self, document: Document, complete_event):
        text = document.text_before_cursor
        parts = text.split(maxsplit=1)

        # Complete command name when still typing first token.
        if len(parts) <= 1 and not text.endswith(" "):
            prefix = parts[0] if parts else ""
            for cmd in self._commands:
                if cmd.startswith(prefix):
                    yield Completion(cmd, start_position=-len(prefix))
            return

        cmd = parts[0].lower() if parts else ""
        if cmd != "upload":
            return

        # For upload <path>, complete filesystem paths.
        path_text = parts[1] if len(parts) > 1 else ""
        path_doc = Document(path_text, cursor_position=len(path_text))
        yield from self._path_completer.get_completions(path_doc, complete_event)


class CLIService:
    def __init__(self, broker: RedisBroker) -> None:
        self.broker = broker
        self._register_handlers()

    def _register_handlers(self) -> None:
        @self.broker.on(Channels.IMAGE_ACCEPTED)
        async def on_accepted(msg: dict) -> None:
            print(
                f"[CLI] image.accepted   image_id={msg.get('image_id')} path={msg.get('path')}"
            )

        @self.broker.on(Channels.IMAGE_STORED)
        async def on_stored(msg: dict) -> None:
            print(
                f"[CLI] image.stored     image_id={msg.get('image_id')} path={msg.get('path')}"
            )

        @self.broker.on(Channels.IMAGE_PIPELINE_COMPLETE)
        async def on_upload_done(msg: dict) -> None:
            print(
                f"[CLI] image.pipeline_complete image_id={msg.get('image_id')} path={msg.get('path')}"
            )

        @self.broker.on(Channels.SEARCH_RESULTS_READY)
        async def on_search_results(msg: dict) -> None:
            print(
                f"[CLI] search.results_ready request_id={msg.get('request_id')} "
                f"count={len(msg.get('results', []))}"
            )
            if msg.get("note"):
                print(f"[CLI] {msg.get('note')}")

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
        try:
            await self.broker.publish(Channels.SEARCH_REQUESTED, msg)
        except Exception as e:
            print(f"[CLI] Error: could not send search to Redis: {e}")
            log.exception("Search publish failed")
            return
        log.info("Requested search for %r", query)

    async def run_interactive(self) -> None:
        session = PromptSession(
            completer=_CLICompleter(),
            complete_while_typing=False,
        )
        print("\nCommands:  upload <path>  |  search <query>  |  quit")

        while True:
            # Keep prompt responsive even while background event handlers print logs.
            with patch_stdout():
                line: str = await session.prompt_async("> ")
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
                print("Usage:  upload <path>  |  search <query>")
