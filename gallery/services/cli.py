"""CLI service - publishes user commands and prints results."""

from __future__ import annotations

import asyncio
import html
import logging
import os
import sys
from pathlib import Path

from prompt_toolkit import PromptSession, print_formatted_text
from prompt_toolkit.completion import Completer, Completion, PathCompleter
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.patch_stdout import patch_stdout

from gallery.broker.pubsub import Channels, RedisBroker
from gallery.messages import (
    ImageListRequested,
    ImageUploadRequested,
    SearchRequested,
    StorageClearRequested,
)

log = logging.getLogger(__name__)

COMMAND_SPECS: tuple[tuple[str, str], ...] = (
    ("upload <path>", "queue an image for processing"),
    ("search <query>", "search indexed images by visual similarity"),
    ("list", "show known images and embedding status"),
    ("clear", "delete all stored images and embeddings"),
)
EXIT_COMMANDS = ("quit", "exit", "q")
COMMAND_NAMES = tuple(spec.split(maxsplit=1)[0] for spec, _ in COMMAND_SPECS)
COMPLETION_COMMANDS = (*COMMAND_NAMES, *EXIT_COMMANDS)
COMMAND_USAGE = "Usage:  " + "  |  ".join((*[spec for spec, _ in COMMAND_SPECS], "quit"))
COMMAND_BANNER = "\nCommands:  " + "  |  ".join((*[spec for spec, _ in COMMAND_SPECS], "quit"))


def _use_ansi_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR", "") == ""


def _html(value: object) -> str:
    return html.escape(str(value), quote=False)


class _CLICompleter(Completer):
    """Tab completion for top-level commands and upload paths."""

    _commands = COMPLETION_COMMANDS

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
            self._print_event(
                "image.accepted",
                f"image_id={msg.get('image_id')} path={msg.get('path')}",
            )

        @self.broker.on(Channels.IMAGE_STORED)
        async def on_stored(msg: dict) -> None:
            self._print_event(
                "image.stored",
                f"image_id={msg.get('image_id')} path={msg.get('path')}",
            )

        @self.broker.on(Channels.IMAGE_PIPELINE_COMPLETE)
        async def on_upload_done(msg: dict) -> None:
            self._print_event(
                "image.pipeline_complete",
                f"image_id={msg.get('image_id')} path={msg.get('path')}",
            )

        @self.broker.on(Channels.SEARCH_RESULTS_READY)
        async def on_search_results(msg: dict) -> None:
            results = msg.get("results", [])
            self._print_event(
                "search.results_ready",
                f"request_id={msg.get('request_id')} count={len(results)}",
            )
            if not results:
                self._print_hint("No results above the similarity threshold.")
                return
            for row in results:
                self._print_search_result(row)

        @self.broker.on(Channels.STORAGE_CLEAR_COMPLETED)
        async def on_clear_completed(msg: dict) -> None:
            service = str(msg.get("service") or "?")
            count = int(msg.get("deleted_count") or 0)
            self._print_event(
                "storage.clear_completed",
                f"service={service} deleted_count={count}",
            )

        @self.broker.on(Channels.IMAGE_LIST_READY)
        async def on_image_list(msg: dict) -> None:
            images = msg.get("images", [])
            if _use_ansi_color():
                print_formatted_text(
                    HTML(
                        "<ansicyan>image.list_ready</ansicyan> "
                        f"count=<b><ansigreen>{len(images)}</ansigreen></b>"
                    )
                )
            else:
                self._print_event("image.list_ready", f"count={len(images)}")
            for row in images:
                if _use_ansi_color():
                    print_formatted_text(
                        HTML(
                            f"  <style fg='ansibrightblack'>id=</style>"
                            f"<ansimagenta>{_html(row.get('image_id'))}</ansimagenta> "
                            f"<style fg='ansibrightblack'>path=</style>{_html(row.get('path'))}\n"
                            f"      <style fg='ansibrightblack'>embedding:</style> "
                            f"<ansigreen>{_html(self._embedding_summary(row))}</ansigreen>"
                        )
                    )
                else:
                    print(
                        f"  id={row.get('image_id')} path={row.get('path')}\n"
                        f"      embedding: {self._embedding_summary(row)}"
                    )

    @staticmethod
    def _embedding_summary(row: dict) -> str:
        dim = row.get("embedding_dim") or 0
        model = row.get("embedding_model") or "unknown"
        return f"{dim} dimensions ({model})" if dim else "not ready"

    @staticmethod
    def _format_score(value: object) -> str:
        try:
            return f"{float(value):.3f}"
        except (TypeError, ValueError):
            return "n/a"

    @classmethod
    def _print_search_result(cls, row: dict) -> None:
        score = cls._format_score(row.get("score"))
        if _use_ansi_color():
            print_formatted_text(
                HTML(
                    "  <style fg='ansibrightblack'>similarity=</style>"
                    f"<b><ansigreen>{_html(score)}</ansigreen></b> "
                    "<style fg='ansibrightblack'>id=</style>"
                    f"<ansimagenta>{_html(row.get('image_id'))}</ansimagenta> "
                    f"<style fg='ansibrightblack'>path=</style>{_html(row.get('path'))}\n"
                    "      <style fg='ansibrightblack'>embedding:</style> "
                    f"<ansigreen>{_html(cls._embedding_summary(row))}</ansigreen>"
                )
            )
            return
        print(
            f"  similarity={score} id={row.get('image_id')} path={row.get('path')}\n"
            f"      embedding: {cls._embedding_summary(row)}"
        )

    @staticmethod
    def _print_event(name: str, details: str = "") -> None:
        line = f"{name}: {details}".strip()
        if _use_ansi_color():
            print_formatted_text(
                HTML(
                    f"<ansicyan>{_html(name)}</ansicyan>"
                    f"{': ' if details else ''}"
                    f"<style fg='ansibrightblack'>{_html(details)}</style>"
                )
            )
        else:
            print(line)

    @staticmethod
    def _print_error(message: str) -> None:
        if _use_ansi_color():
            print_formatted_text(HTML(f"<ansired>Error:</ansired> {_html(message)}"))
        else:
            print(f"Error: {message}")

    @staticmethod
    def _print_hint(message: str) -> None:
        if _use_ansi_color():
            print_formatted_text(HTML(f"<ansiyellow>Hint:</ansiyellow> {_html(message)}"))
        else:
            print(f"Hint: {message}")

    @staticmethod
    def _print_banner() -> None:
        if _use_ansi_color():
            commands = COMMAND_BANNER.removeprefix("\nCommands:  ")
            print_formatted_text(HTML(f"\n<ansicyan>Commands:</ansicyan> {_html(commands)}"))
        else:
            print(COMMAND_BANNER)

    async def upload_image(self, path: str) -> None:
        try:
            resolved = Path(path).expanduser().resolve()
        except (OSError, RuntimeError) as e:
            self._print_error(f"invalid path {path!r}: {e}")
            return
        if not resolved.is_file():
            self._print_error(f"file not found: {path}")
            self._print_hint("use an absolute path, or try e.g. samples/apple.jpeg")
            return
        msg = ImageUploadRequested(path=str(resolved))
        try:
            await self.broker.publish(Channels.IMAGE_UPLOAD_REQUESTED, msg)
        except Exception as e:
            self._print_error(f"could not send upload to Redis: {e}")
            log.exception("Upload publish failed")
            return
        log.info("Requested upload for %s", resolved)

    async def search(self, query: str) -> None:
        msg = SearchRequested(query=query)
        try:
            await self.broker.publish(Channels.SEARCH_REQUESTED, msg)
        except Exception as e:
            self._print_error(f"could not send search to Redis: {e}")
            log.exception("Search publish failed")
            return
        log.info("Requested search for %r", query)

    async def list_images(self) -> None:
        msg = ImageListRequested()
        try:
            await self.broker.publish(Channels.IMAGE_LIST_REQUESTED, msg)
        except Exception as e:
            self._print_error(f"could not request image list: {e}")
            log.exception("List publish failed")
            return
        log.info("Requested image list")

    async def clear_storage(self) -> None:
        msg = StorageClearRequested()
        try:
            await self.broker.publish(Channels.STORAGE_CLEAR_REQUESTED, msg)
        except Exception as e:
            self._print_error(f"could not send clear request to Redis: {e}")
            log.exception("Clear publish failed")
            return
        log.info("Requested storage clear")

    async def run_interactive(self) -> None:
        session = PromptSession(
            completer=_CLICompleter(),
            complete_while_typing=False,
        )
        self._print_banner()

        while True:
            # Keep prompt responsive even while background event handlers print logs.
            with patch_stdout():
                line: str = await session.prompt_async("> ")
            line = line.strip()
            if not line:
                continue
            if line.lower() in EXIT_COMMANDS:
                break
            parts = line.split(maxsplit=1)
            cmd, arg = parts[0].lower(), parts[1] if len(parts) > 1 else ""

            if cmd == "upload" and arg:
                await self.upload_image(arg)
            elif cmd == "search" and arg:
                await self.search(arg)
            elif cmd == "list":
                await self.list_images()
            elif cmd == "clear":
                if await self._confirm_clear(session, arg):
                    await self.clear_storage()
                else:
                    self._print_hint("clear cancelled")
            else:
                if _use_ansi_color():
                    print_formatted_text(HTML(f"<ansiyellow>{_html(COMMAND_USAGE)}</ansiyellow>"))
                else:
                    print(COMMAND_USAGE)

    @staticmethod
    async def _confirm_clear(session: PromptSession, arg: str) -> bool:
        if arg.strip().lower() in {"y", "yes", "--yes", "-y"}:
            return True
        prompt_text: object
        if _use_ansi_color():
            prompt_text = HTML(
                "<ansired>This will delete all stored images and embeddings.</ansired> "
                "Type <b>yes</b> to confirm: "
            )
        else:
            prompt_text = "This will delete all stored images and embeddings. Type 'yes' to confirm: "
        with patch_stdout():
            answer = await session.prompt_async(prompt_text)
        return answer.strip().lower() in {"y", "yes"}
