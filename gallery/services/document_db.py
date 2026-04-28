"""Document DB service backed by MongoDB."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from gallery.broker.pubsub import Channels, RedisBroker
from gallery.messages import ImageListReady, ImagePipelineComplete, ImageStored

log = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
            await self.broker.publish(
                Channels.IMAGE_PIPELINE_COMPLETE,
                ImagePipelineComplete(image_id=image_id, path=msg.get("path", "")),
            )

        @self.broker.on(Channels.IMAGE_ANNOTATION_REQUESTED)
        async def on_annotation_requested(msg: dict) -> None:
            image_id = msg.get("image_id")
            if not image_id:
                return
            now = _utc_now_iso()
            doc = {
                **{k: v for k, v in msg.items() if v is not None},
                "image_id": image_id,
                "annotation_requested_at": now,
                "updated_at": now,
            }
            await self._collection.update_one(
                {"_id": image_id},
                {"$set": doc, "$setOnInsert": {"_id": image_id}},
                upsert=True,
            )
            log.info(
                "[DocumentDB] annotation requested image_id=%s path=%s",
                image_id,
                msg.get("path"),
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
                annotations: list[str] = []
                if isinstance(vrec, dict):
                    raw_ann = vrec.get("annotations", [])
                    if isinstance(raw_ann, list):
                        annotations = [str(x) for x in raw_ann if str(x).strip()]
                images.append(
                    {
                        "image_id": image_id,
                        "path": doc.get("path", ""),
                        "annotations": annotations,
                        "updated_at": doc.get("updated_at", ""),
                    }
                )
            images.sort(key=lambda row: row.get("updated_at") or "", reverse=True)
            await self.broker.publish(
                Channels.IMAGE_LIST_READY,
                ImageListReady(request_id=msg.get("id", ""), images=images),
            )
            log.info("[DocumentDB] list ready count=%s", len(images))
