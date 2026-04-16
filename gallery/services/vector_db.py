"""Vector DB service placeholder for future embedding search."""

from __future__ import annotations

import logging

from gallery.broker.pubsub import Channels, RedisBroker
from gallery.messages import SearchResultsReady

log = logging.getLogger(__name__)


class VectorDBService:
    def __init__(self, broker: RedisBroker) -> None:
        self.broker = broker
        self._register_handlers()

    def _register_handlers(self) -> None:
        @self.broker.on(Channels.IMAGE_EMBEDDING_REQUESTED)
        async def on_embedding_requested(msg: dict) -> None:
            # Placeholder: later this service will persist embeddings by image_id.
            log.info("[VectorDB] embedding requested image_id=%s", msg.get("image_id"))

        @self.broker.on(Channels.SEARCH_REQUESTED)
        async def on_search_requested(msg: dict) -> None:
            # Placeholder: for now we only prove messaging and response contracts.
            query = msg.get("query", "")
            top_k = msg.get("top_k", 5)
            log.info("[VectorDB] search requested query=%r top_k=%s", query, top_k)
            await self.broker.publish(
                Channels.SEARCH_RESULTS_READY,
                SearchResultsReady(
                    request_id=msg.get("id", ""),
                    results=[],
                ),
            )
