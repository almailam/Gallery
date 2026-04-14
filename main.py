"""Entry point - wires up all services and starts the broker listener."""

from __future__ import annotations

import asyncio
import logging

from gallery.broker.pubsub import RedisBroker
from gallery.services import CLIService, DocumentDBService, ImageService, VectorDBService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s - %(message)s",
)


async def main() -> None:
    broker = RedisBroker()
    await broker.connect()

    # Instantiate services - each one registers its own handlers on the broker.
    image_svc = ImageService(broker)          # noqa: F841
    doc_db = DocumentDBService(broker)        # noqa: F841
    vec_db = VectorDBService(broker)          # noqa: F841
    cli = CLIService(broker)

    # Run the broker listener and the interactive CLI concurrently.
    await asyncio.gather(
        broker.listen(),
        cli.run_interactive(),
    )

    await broker.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
