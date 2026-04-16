"""Entry point - wires up all services and starts the broker listener."""

from __future__ import annotations

import asyncio
import logging
import shutil
from contextlib import suppress

from gallery.broker.pubsub import RedisBroker
from gallery.services import CLIService, DocumentDBService, ImageService, VectorDBService

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

    # Instantiate services - each one registers its own handlers on the broker.
    image_svc = ImageService(broker)          # noqa: F841
    doc_db = DocumentDBService(broker)        # noqa: F841
    vec_db = VectorDBService(broker)          # noqa: F841
    cli = CLIService(broker)

    # Run the broker listener and the interactive CLI concurrently.
    try:
        await asyncio.gather(
            broker.listen(),
            cli.run_interactive(),
        )
    finally:
        await broker.disconnect()
        if redis_proc and redis_proc.returncode is None:
            redis_proc.terminate()
            with suppress(ProcessLookupError):
                await redis_proc.wait()


if __name__ == "__main__":
    asyncio.run(main())
