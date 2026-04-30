"""Document DB service backed by MongoDB."""

from __future__ import annotations

import logging
from typing import Any

from gallery.broker.pubsub import Channels, RedisBroker
from gallery.messages import (
    ImageListReady,
    ImageStored,
    StorageClearCompleted,
)
from gallery.utils import _utc_now_iso

log = logging.getLogger(__name__)


class DocumentDBService:
    """Persists accepted image metadata to MongoDB."""

    def __init__(
        self,
        broker: RedisBroker,
        *,
        collection: Any,
        vector_collection: Any,
    ) -> None:
        self.broker = broker
        self._collection = collection
        self._vector_collection = vector_collection
        self._register_handlers()

    async def _read_vector_records(self) -> dict[str, dict]:
        records: dict[str, dict] = {}
        async for record in self._vector_collection.find({}):
            image_id = record.get("image_id") or record.get("_id")
            if image_id:
                records[str(image_id)] = record
        return records

    def _register_handlers(self) -> None:
        @self.broker.on(Channels.IMAGE_ACCEPTED)
        async def on_accepted(msg: dict) -> None:
            image_id = msg.get("image_id")
            if not image_id:
                log.warning("[DocumentDB] image.accepted missing image_id: %s", msg)
                return
            doc = {**msg, "image_id": image_id, "updated_at": _utc_now_iso()}
            await self._collection.update_one(
                {"_id": image_id},
                {"$set": doc, "$setOnInsert": {"_id": image_id}},
                upsert=True,
            )
            log.info("[DocumentDB] Stored %s", image_id)
            await self.broker.publish(
                Channels.IMAGE_STORED,
                ImageStored(image_id=image_id, path=msg.get("path", "")),
            )

        @self.broker.on(Channels.IMAGE_LIST_REQUESTED)
        async def on_list_requested(msg: dict) -> None:
            vector_by_id = await self._read_vector_records()
            snapshot = []
            async for record in self._collection.find({}):
                image_id = record.get("image_id") or record.get("_id")
                if image_id:
                    snapshot.append((str(image_id), record))
            images: list[dict] = []
            for image_id, doc in snapshot:
                vrec = vector_by_id.get(image_id, {}) if isinstance(vector_by_id, dict) else {}
                embedding_model = ""
                embedding_dim = 0
                if isinstance(vrec, dict):
                    embedding_model = str(vrec.get("embedding_model") or "")
                    raw_dim = vrec.get("embedding_dim", 0)
                    if isinstance(raw_dim, int):
                        embedding_dim = raw_dim
                    elif isinstance(raw_dim, str) and raw_dim.isdigit():
                        embedding_dim = int(raw_dim)
                images.append(
                    {
                        "image_id": image_id,
                        "path": doc.get("path", ""),
                        "embedding_model": embedding_model,
                        "embedding_dim": embedding_dim,
                        "updated_at": doc.get("updated_at", ""),
                    }
                )
            images.sort(key=lambda row: row.get("updated_at") or "", reverse=True)
            await self.broker.publish(
                Channels.IMAGE_LIST_READY,
                ImageListReady(request_id=msg.get("id", ""), images=images),
            )
            log.info("[DocumentDB] list ready count=%s", len(images))

        @self.broker.on(Channels.STORAGE_CLEAR_REQUESTED)
        async def on_clear_requested(msg: dict) -> None:
            request_id = msg.get("id", "")
            deleted = 0
            async for _ in self._collection.find({}):
                deleted += 1
            await self._collection.delete_many({})
            log.info("[DocumentDB] cleared documents count=%s", deleted)
            await self.broker.publish(
                Channels.STORAGE_CLEAR_COMPLETED,
                StorageClearCompleted(
                    request_id=request_id,
                    service="documents",
                    deleted_count=deleted,
                ),
            )
