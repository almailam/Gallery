"""Vector DB service - stores embeddings and handles semantic search."""

from __future__ import annotations

import logging
import math

from gallery.broker.pubsub import Channels, RedisBroker
from gallery.messages import SearchResultsReady

log = logging.getLogger(__name__)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b))
    return dot / mag if mag else 0.0


class VectorDBService:
    def __init__(self, broker: RedisBroker) -> None:
        self.broker = broker
        self._store: dict[str, dict] = {}  # image_id -> {image_id, embedding}
        self._register_handlers()

    def _register_handlers(self) -> None:
        @self.broker.on(Channels.IMAGE_EMBEDDED)
        async def on_embedded(msg: dict) -> None:
            self._store[msg["image_id"]] = msg
            log.info("[VectorDB] Stored %s", msg["image_id"])

        @self.broker.on(Channels.SEARCH_REQUESTED)
        async def on_search(msg: dict) -> None:
            # TODO: encode msg["query"] with the same model used at ingest
            dims = next(
                (len(d["embedding"]) for d in self._store.values()), 512
            )
            query_vec = [0.0] * dims
            ranked = sorted(
                self._store.values(),
                key=lambda d: _cosine(query_vec, d["embedding"]),
                reverse=True,
            )
            results = [{"image_id": d["image_id"]} for d in ranked[:10]]
            log.info("[VectorDB] %r -> %d result(s)", msg.get("query"), len(results))
            await self.broker.publish(
                Channels.SEARCH_RESULTS_READY,
                SearchResultsReady(request_id=msg["id"], results=results),
            )
