"""Image service - stubs for annotation and embedding."""

from __future__ import annotations

import logging
import uuid

from gallery.broker.pubsub import Channels, RedisBroker
from gallery.messages import (
    ImageAnnotated,
    ImageEmbedded,
    ImagePipelineComplete,
    VectorMeta,
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
        vector_meta = VectorMeta(model="clip-vit-base", dimensions=512)
        log.info("[ImageService] Processing %s -> %s", path, image_id)

        await self.broker.publish(
            Channels.IMAGE_ANNOTATED,
            ImageAnnotated(
                image_id=image_id,
                path=path,
                tags=["stub-tag"],       # TODO: vision model
                caption="stub caption",  # TODO: captioning model
            ),
        )

        await self.broker.publish(
            Channels.IMAGE_EMBEDDED,
            ImageEmbedded(
                image_id=image_id,
                embedding=[0.0] * vector_meta.dimensions,  # TODO: CLIP encoder
                vector_meta=vector_meta,
            ),
        )

        await self.broker.publish(
            Channels.IMAGE_PIPELINE_COMPLETE,
            ImagePipelineComplete(image_id=image_id, path=path),
        )
