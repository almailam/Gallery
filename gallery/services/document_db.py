"""Document DB service — simple JSON file store for image metadata."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from gallery.broker.pubsub import Channels, RedisBroker
from gallery.messages import ImagePipelineComplete, ImageStored

log = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path("gallery_data") / "documents.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class DocumentDBService:
    """Persists accepted image metadata to a single JSON file (keyed by image_id)."""

    def __init__(self, broker: RedisBroker, *, db_path: Path | None = None) -> None:
        self.broker = broker
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._records: dict[str, dict] = {}
        self._lock = asyncio.Lock()
        self._load_from_disk()
        self._register_handlers()

    def _load_from_disk(self) -> None:
        if not self._db_path.exists():
            return
        try:
            raw = self._db_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            self._records = data.get("records", {})
            log.info("[DocumentDB] Loaded %s records from %s", len(self._records), self._db_path)
        except (OSError, json.JSONDecodeError) as e:
            log.warning("[DocumentDB] Could not load %s: %s", self._db_path, e)

    def _flush_to_disk_locked(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"records": self._records}
        self._db_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _register_handlers(self) -> None:
        @self.broker.on(Channels.IMAGE_ACCEPTED)
        async def on_accepted(msg: dict) -> None:
            image_id = msg.get("image_id")
            if not image_id:
                log.warning("[DocumentDB] image.accepted missing image_id: %s", msg)
                return
            async with self._lock:
                prior = self._records.get(image_id, {})
                doc = {**prior, **msg, "updated_at": _utc_now_iso()}
                self._records[image_id] = doc
                self._flush_to_disk_locked()
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
            async with self._lock:
                prior = self._records.get(image_id, {})
                doc = {
                    **prior,
                    **{k: v for k, v in msg.items() if v is not None},
                    "annotation_requested_at": _utc_now_iso(),
                    "updated_at": _utc_now_iso(),
                }
                self._records[image_id] = doc
                self._flush_to_disk_locked()
            log.info(
                "[DocumentDB] annotation requested image_id=%s path=%s",
                image_id,
                msg.get("path"),
            )
