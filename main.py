"""Entry point - wires up all services and starts the broker listener."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from contextlib import suppress
from pathlib import Path

from dotenv import load_dotenv
from pymongo import AsyncMongoClient

from gallery.broker.pubsub import RedisBroker
from gallery.services import CLIService, DocumentDBService, ImageService, VectorDBService

_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s - %(message)s",
)

log = logging.getLogger(__name__)

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


async def main() -> None:
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

    mongo_client = AsyncMongoClient(os.environ.get("MONGO_URI", "mongodb://localhost:27017"))
    mongo_db = mongo_client[os.environ.get("MONGO_DB", "gallery")]

    # Instantiate services - each one registers its own handlers on the broker.
    ImageService(broker)
    DocumentDBService(broker, collection=mongo_db.documents, vector_collection=mongo_db.vectors)
    VectorDBService(broker, collection=mongo_db.vectors)
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
