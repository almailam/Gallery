"""Entry point - wires up all services and starts the broker listener."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import shutil
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pymongo import AsyncMongoClient

from gallery.broker.pubsub import RedisBroker
from gallery.services import CLIService, DocumentDBService, ImageService, VectorDBService
from gallery.services.json_collection import JsonCollection

_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")

_LOG_FORMAT = "%(asctime)s  %(levelname)-7s  %(name)s - %(message)s"
_DEFAULT_DEBUG_LOG_PATH = _PROJECT_ROOT / "gallery_data" / "debug.log"
_DEFAULT_DATA_DIR = _PROJECT_ROOT / "gallery_data"

log = logging.getLogger(__name__)


def _debug_log_path() -> Path:
    configured = os.environ.get("GALLERY_DEBUG_LOG")
    if configured:
        return Path(configured).expanduser()
    return _DEFAULT_DEBUG_LOG_PATH


def _configure_debug_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)


def _debug_terminal_command(log_path: Path) -> str:
    quoted_path = shlex.quote(str(log_path))
    return (
        "clear; "
        "printf '\\033]0;Gallery Redis Debug\\007'; "
        "echo 'Gallery Redis/debug messages'; "
        f"echo 'Log file: {quoted_path}'; "
        f"tail -n +1 -f {quoted_path}"
    )


async def _open_debug_terminal(log_path: Path) -> bool:
    if os.environ.get("GALLERY_OPEN_DEBUG_TERMINAL", "1") == "0":
        return False
    if sys.platform != "darwin" or not shutil.which("osascript"):
        return False

    script = (
        'tell application "Terminal"\n'
        f"    do script {json.dumps(_debug_terminal_command(log_path))}\n"
        "    activate\n"
        "end tell"
    )
    proc = await asyncio.create_subprocess_exec(
        "osascript",
        "-e",
        script,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    return await proc.wait() == 0


async def _try_connect(broker: RedisBroker, *, attempts: int = 10) -> bool:
    for _ in range(attempts):
        try:
            await broker.connect()
            return True
        except Exception:
            await asyncio.sleep(0.25)
    return False


async def _start_redis_server() -> asyncio.subprocess.Process | None:
    redis_server = shutil.which("redis-server")
    if not redis_server:
        return None

    # Start a local ephemeral redis. Keeps data in memory only.
    return await asyncio.create_subprocess_exec(
        redis_server,
        "--port",
        "6379",
        "--save",
        "",
        "--appendonly",
        "no",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )


def _mongo_timeout_ms() -> int:
    raw = os.environ.get("MONGO_TIMEOUT_MS", "1000")
    try:
        return max(int(raw), 1)
    except ValueError:
        return 1000


async def _open_storage() -> tuple[AsyncMongoClient | None, Any, Any, str]:
    mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
    timeout_ms = _mongo_timeout_ms()
    mongo_client = AsyncMongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=timeout_ms,
        connectTimeoutMS=timeout_ms,
    )
    try:
        await mongo_client.admin.command("ping")
    except Exception as exc:
        await mongo_client.close()
        data_dir = _DEFAULT_DATA_DIR
        log.warning(
            "MongoDB unavailable at %s; using local JSON storage in %s",
            mongo_uri,
            data_dir,
        )
        log.debug("MongoDB connection failure detail", exc_info=True)
        return (
            None,
            JsonCollection(data_dir / "documents.json"),
            JsonCollection(data_dir / "vector_db.json"),
            f"local JSON storage ({data_dir})",
        )

    mongo_db = mongo_client[os.environ.get("MONGO_DB", "gallery")]
    return mongo_client, mongo_db.documents, mongo_db.vectors, f"MongoDB ({mongo_uri})"


async def main() -> None:
    debug_log = _debug_log_path()
    _configure_debug_logging(debug_log)
    if await _open_debug_terminal(debug_log):
        print(f"Redis/debug messages are streaming in a separate Terminal window ({debug_log}).")
    else:
        print(f"Redis/debug messages are written to {debug_log}.")

    broker = RedisBroker()
    redis_proc: asyncio.subprocess.Process | None = None
    mongo_client: AsyncMongoClient | None = None

    connected = await _try_connect(broker, attempts=2)
    if not connected:
        redis_proc = await _start_redis_server()
        if redis_proc:
            log.info("Started redis-server in background (pid=%s)", redis_proc.pid)
            connected = await _try_connect(broker, attempts=20)

    if not connected:
        log.error(
            "Cannot connect to Redis at localhost:6379. "
            "Either start Redis manually, or install it so `redis-server` is available."
        )
        raise SystemExit(1)

    mongo_client, document_collection, vector_collection, storage_label = await _open_storage()
    print(f"Storage: {storage_label}")

    # Instantiate services - each one registers its own handlers on the broker.
    ImageService(broker)
    DocumentDBService(broker, collection=document_collection, vector_collection=vector_collection)
    vector_service = VectorDBService(broker, collection=vector_collection)
    if await vector_service.prepare_model():
        print("Embedding model: ready")
    else:
        print("Embedding model: will load on first use")
    cli = CLIService(broker)

    # Run the broker listener and the interactive CLI concurrently.
    # When the user quits the CLI, cancel the listener so the process can exit
    # (broker.listen() would otherwise run forever).
    listen_task = asyncio.create_task(broker.listen())
    try:
        await cli.run_interactive()
    finally:
        listen_task.cancel()
        try:
            await listen_task
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("Broker listener exited with error")
        await broker.disconnect()
        if mongo_client is not None:
            await mongo_client.close()
        if redis_proc and redis_proc.returncode is None:
            redis_proc.terminate()
            with suppress(ProcessLookupError):
                await redis_proc.wait()


if __name__ == "__main__":
    asyncio.run(main())
