"""Document DB service - stores upload records and acknowledges persistence."""

from __future__ import annotations

import logging

from gallery.broker.pubsub import Channels, RedisBroker
from gallery.messages import ImagePipelineComplete, ImageStored

log = logging.getLogger(__name__)


class DocumentDBService:
    def __init__(self, broker: RedisBroker) -> None:
        self.broker = broker
        self._store: dict[str, dict] = {}  # image_id -> record
        self._register_handlers()

    def _register_handlers(self) -> None:
        @self.broker.on(Channels.IMAGE_ACCEPTED)
        async def on_accepted(msg: dict) -> None:
            self._store[msg["image_id"]] = msg
            log.info("[DocumentDB] Stored %s", msg["image_id"])
            await self.broker.publish(
                Channels.IMAGE_STORED,
                ImageStored(image_id=msg["image_id"], path=msg["path"]),
            )
            await self.broker.publish(
                Channels.IMAGE_PIPELINE_COMPLETE,
                ImagePipelineComplete(image_id=msg["image_id"], path=msg["path"]),
            )

        @self.broker.on(Channels.IMAGE_ANNOTATION_REQUESTED)
        async def on_annotation_requested(msg: dict) -> None:
            # Placeholder: annotation details will be written here later.
            log.info(
                "[DocumentDB] annotation requested image_id=%s path=%s",
                msg.get("image_id"),
                msg.get("path"),
            )
