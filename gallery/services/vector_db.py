"""Vector DB service with Mongo-backed image embeddings."""

from __future__ import annotations

import asyncio
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gallery.broker.pubsub import Channels, RedisBroker
from gallery.messages import ImageEmbeddingFailed, SearchResultsReady, StorageClearCompleted

log = logging.getLogger(__name__)

_DEFAULT_EMBEDDING_MODEL = "clip-ViT-B-32"
_DEFAULT_SEARCH_MIN_SCORE = 0.24


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _embedding_model_name() -> str:
    return os.environ.get("GALLERY_EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL).strip() or _DEFAULT_EMBEDDING_MODEL


def _search_min_score() -> float:
    raw = os.environ.get("GALLERY_SEARCH_MIN_SCORE", str(_DEFAULT_SEARCH_MIN_SCORE))
    try:
        return float(raw)
    except ValueError:
        return _DEFAULT_SEARCH_MIN_SCORE


def _normalize_embedding(raw: object) -> list[float]:
    if not isinstance(raw, (list, tuple)):
        return []
    values: list[float] = []
    for item in raw:
        if isinstance(item, bool):
            return []
        try:
            value = float(item)
        except (TypeError, ValueError):
            return []
        if not math.isfinite(value):
            return []
        values.append(value)
    return values


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


class VectorDBService:
    def __init__(
        self,
        broker: RedisBroker,
        *,
        collection: Any | None = None,
        embedding_model: Any | None = None,
        embedding_model_name: str | None = None,
    ) -> None:
        self.broker = broker
        self._collection = collection
        self._embedding_model = embedding_model
        self._embedding_model_name = embedding_model_name or _embedding_model_name()
        self._search_min_score = _search_min_score()
        self._last_embedding_error = ""
        self._register_handlers()

    def _model(self) -> Any:
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer(self._embedding_model_name)
        return self._embedding_model

    @staticmethod
    def _vector_to_list(vector: Any) -> list[float]:
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        return _normalize_embedding(vector)

    def _encode_image_sync(self, path: str) -> list[float]:
        image_path = Path(path)
        if not image_path.is_file():
            log.warning("[VectorDB] Cannot embed missing file: %s", path)
            return []

        try:
            from PIL import Image

            with Image.open(image_path) as image:
                image.load()
                vector = self._model().encode(
                    image.convert("RGB"),
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            return self._vector_to_list(vector)
        except OSError as e:
            self._last_embedding_error = f"failed reading image: {e}"
            log.warning("[VectorDB] Failed reading image %s: %s", path, e)
            return []
        except ImportError as e:
            self._last_embedding_error = f"embedding dependencies are unavailable: {e}"
            log.warning("[VectorDB] Embedding dependencies are unavailable: %s", e)
            return []
        except Exception as e:
            self._last_embedding_error = f"image embedding failed: {e}"
            log.warning("[VectorDB] Image embedding failed for %s: %s", path, e)
            return []

    async def _embed_image(self, path: str) -> list[float]:
        return await asyncio.to_thread(self._encode_image_sync, path)

    def _encode_text_sync(self, query: str) -> list[float]:
        try:
            vector = self._model().encode(
                query,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return self._vector_to_list(vector)
        except ImportError as e:
            self._last_embedding_error = f"embedding dependencies are unavailable: {e}"
            log.warning("[VectorDB] Embedding dependencies are unavailable: %s", e)
            return []
        except Exception as e:
            self._last_embedding_error = f"text embedding failed: {e}"
            log.warning("[VectorDB] Text embedding failed for query=%r: %s", query, e)
            return []

    async def _embed_text(self, query: str) -> list[float]:
        return await asyncio.to_thread(self._encode_text_sync, query)

    def _embedding_from_record(self, record: dict) -> list[float] | None:
        if record.get("embedding_model") != self._embedding_model_name:
            return None
        embedding = _normalize_embedding(record.get("embedding"))
        return embedding if embedding else None

    async def _cached_embedding_for_image(self, image_id: str) -> list[float] | None:
        if self._collection is None:
            return None
        prior = await self._collection.find_one({"_id": image_id}) or {}
        return self._embedding_from_record(prior)

    async def _cached_embedding_for_same_path(self, path: str, exclude_id: str) -> list[float] | None:
        if self._collection is None:
            return None
        if not path:
            return None
        try:
            target = str(Path(path).expanduser().resolve())
        except (OSError, RuntimeError):
            target = path
        async for rec in self._collection.find({}):
            rid = rec.get("image_id") or rec.get("_id")
            if rid == exclude_id:
                continue
            other_path = rec.get("path", "")
            if not other_path:
                continue
            try:
                other_resolved = str(Path(other_path).expanduser().resolve())
            except (OSError, RuntimeError):
                other_resolved = other_path
            if other_resolved != target:
                continue
            embedding = self._embedding_from_record(rec)
            if embedding:
                return embedding
        return None

    async def _store_embedding(self, image_id: str, path: str, embedding: list[float]) -> None:
        if self._collection is None:
            return
        await self._collection.update_one(
            {"_id": image_id},
            {
                "$set": {
                    "image_id": image_id,
                    "path": path,
                    "embedding": embedding,
                    "embedding_model": self._embedding_model_name,
                    "embedding_dim": len(embedding),
                    "updated_at": _utc_now_iso(),
                },
                "$setOnInsert": {"_id": image_id},
            },
            upsert=True,
        )

    async def _embedding_for_search_record(self, record: dict) -> list[float] | None:
        embedding = self._embedding_from_record(record)
        if embedding:
            return embedding

        image_id = record.get("image_id") or record.get("_id")
        path = record.get("path", "")
        if not image_id or not path:
            return None

        embedding = await self._embed_image(str(path))
        if not embedding:
            return None
        await self._store_embedding(str(image_id), str(path), embedding)
        record["image_id"] = str(image_id)
        record["path"] = str(path)
        record["embedding"] = embedding
        record["embedding_model"] = self._embedding_model_name
        record["embedding_dim"] = len(embedding)
        return embedding

    def _register_handlers(self) -> None:
        @self.broker.on(Channels.IMAGE_EMBEDDING_REQUESTED)
        async def on_embedding_requested(msg: dict) -> None:
            image_id = msg.get("image_id")
            path = msg.get("path", "")
            if not image_id:
                log.warning("[VectorDB] embedding requested missing image_id")
                return
            if self._collection is None:
                log.warning("[VectorDB] no Mongo collection configured")
                return

            reuse = await self._cached_embedding_for_image(image_id)
            if reuse is None:
                reuse = await self._cached_embedding_for_same_path(path, image_id)
            from_cache = reuse is not None

            if from_cache:
                embedding = reuse
                log.info(
                    "[VectorDB] using stored embedding for image_id=%s (no model call)",
                    image_id,
                )
            else:
                embedding = await self._embed_image(path)
            if not embedding:
                reason = self._last_embedding_error or "embedding generation returned no vector"
                log.warning("[VectorDB] no embedding stored for image_id=%s path=%s", image_id, path)
                await self.broker.publish(
                    Channels.IMAGE_EMBEDDING_FAILED,
                    ImageEmbeddingFailed(
                        image_id=image_id,
                        path=path,
                        reason=reason,
                    ),
                )
                return

            await self._store_embedding(image_id, path, embedding)
            log.info(
                "[VectorDB] stored image_id=%s embedding_dim=%s model=%s",
                image_id,
                len(embedding),
                self._embedding_model_name,
            )

        @self.broker.on(Channels.SEARCH_REQUESTED)
        async def on_search_requested(msg: dict) -> None:
            query = msg.get("query", "")
            top_k = msg.get("top_k", 5)
            if self._collection is None:
                log.warning("[VectorDB] no Mongo collection configured")
                return

            query_embedding = await self._embed_text(query) if query.strip() else []
            scored: list[tuple[float, dict]] = []
            async for record in self._collection.find({}):
                embedding = await self._embedding_for_search_record(record)
                if not embedding:
                    continue
                score = _cosine_similarity(query_embedding, embedding) if query_embedding else 0.0
                if not query_embedding or score >= self._search_min_score:
                    scored.append((score, record))

            scored.sort(
                key=lambda item: (
                    item[0],
                    item[1].get("updated_at", ""),
                ),
                reverse=True,
            )
            limit = max(int(top_k), 1)
            results = [
                {
                    "image_id": rec.get("image_id"),
                    "path": rec.get("path"),
                    "embedding_model": rec.get("embedding_model"),
                    "embedding_dim": rec.get("embedding_dim", len(_normalize_embedding(rec.get("embedding")))),
                    "score": score,
                }
                for score, rec in scored[:limit]
            ]

            log.info("[VectorDB] search requested query=%r top_k=%s", query, top_k)
            await self.broker.publish(
                Channels.SEARCH_RESULTS_READY,
                SearchResultsReady(
                    request_id=msg.get("id", ""),
                    results=results,
                ),
            )

        @self.broker.on(Channels.STORAGE_CLEAR_REQUESTED)
        async def on_clear_requested(msg: dict) -> None:
            if self._collection is None:
                log.warning("[VectorDB] no Mongo collection configured")
                return
            request_id = msg.get("id", "")
            deleted = 0
            async for _ in self._collection.find({}):
                deleted += 1
            await self._collection.delete_many({})
            log.info("[VectorDB] cleared embeddings count=%s", deleted)
            await self.broker.publish(
                Channels.STORAGE_CLEAR_COMPLETED,
                StorageClearCompleted(
                    request_id=request_id,
                    service="vectors",
                    deleted_count=deleted,
                ),
            )
