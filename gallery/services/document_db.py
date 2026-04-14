"""Document DB service - stores image metadata and handles keyword search."""

from __future__ import annotations

import logging

from gallery.broker.pubsub import Channels, RedisBroker
from gallery.messages import SearchResultsReady

log = logging.getLogger(__name__)


class DocumentDBService:
    def __init__(self, broker: RedisBroker) -> None:
        self.broker = broker
        self._store: dict[str, dict] = {}  # image_id -> metadata
        self._register_handlers()

    def _register_handlers(self) -> None:
        @self.broker.on(Channels.IMAGE_ANNOTATED)
        async def on_annotated(msg: dict) -> None:
            self._store[msg["image_id"]] = msg
            log.info("[DocumentDB] Stored %s", msg["image_id"])

        @self.broker.on(Channels.SEARCH_REQUESTED)
        async def on_search(msg: dict) -> None:
            q = msg.get("query", "").lower()
            results = [
                doc for doc in self._store.values()
                if q in doc.get("caption", "").lower()
                or any(q in t.lower() for t in doc.get("tags", []))
            ]
            log.info("[DocumentDB] %r -> %d result(s)", q, len(results))
            await self.broker.publish(
                Channels.SEARCH_RESULTS_READY,
                SearchResultsReady(request_id=msg["id"], results=results),
            )
