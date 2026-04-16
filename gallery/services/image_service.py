"""Image service - accepts uploads and emits lifecycle events."""

from __future__ import annotations

import logging
import uuid

from gallery.broker.pubsub import Channels, RedisBroker
from gallery.messages import (
    ImageAccepted,
    ImageAnnotationRequested,
    ImageEmbeddingRequested,
)

log = logging.getLogger(__name__)


class ImageService:
    def __init__(self, broker: RedisBroker) -> None:
        self.broker = broker
        self._register_handlers()

    def _register_handlers(self) -> None:
        @self.broker.on(Channels.IMAGE_UPLOAD_REQUESTED)
        async def on_upload(msg: dict) -> None:
            await self._process(msg)

    async def _process(self, msg: dict) -> None:
        image_id = str(uuid.uuid4())
        path = msg.get("path", "")
        log.info("[ImageService] Accepted upload %s -> %s", path, image_id)

        await self.broker.publish(
            Channels.IMAGE_ACCEPTED,
            ImageAccepted(
                image_id=image_id,
                path=path,
            ),
        )
        await self.broker.publish(
            Channels.IMAGE_ANNOTATION_REQUESTED,
            ImageAnnotationRequested(
                image_id=image_id,
                path=path,
            ),
        )
        await self.broker.publish(
            Channels.IMAGE_EMBEDDING_REQUESTED,
            ImageEmbeddingRequested(
                image_id=image_id,
                path=path,
            ),
        )

