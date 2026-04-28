"""Vector DB service with Mongo-backed annotation search."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request

from gallery.broker.pubsub import Channels, RedisBroker
from gallery.messages import SearchResultsReady

log = logging.getLogger(__name__)

_GITHUB_MODELS_CHAT_COMPLETIONS_URL = "https://models.github.ai/inference/chat/completions"
_GITHUB_MODELS_MODEL = "openai/gpt-4.1-mini"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_annotation_list(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(x).strip() for x in raw if str(x).strip()]


def _message_content_to_text(message: dict) -> str:
    """OpenAI-compatible APIs may return message.content as a string or a list of parts."""
    raw = message.get("content")
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, list):
        parts: list[str] = []
        for block in raw:
            if isinstance(block, dict):
                if block.get("type") == "text" and block.get("text") is not None:
                    parts.append(str(block["text"]))
                elif "text" in block:
                    parts.append(str(block["text"]))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts).strip()
    return str(raw).strip()


def _unwrap_json_blob(text: str) -> str:
    """Strip optional ```json ... ``` fences from model output."""
    t = text.strip()
    if not t.startswith("```"):
        return t
    lines = t.split("\n")
    if not lines:
        return t
    # Drop opening fence (``` or ```json)
    lines = lines[1:]
    while lines and lines[-1].strip() == "":
        lines.pop()
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_annotations_from_model_text(text: str) -> list[str]:
    if not text:
        return []
    try:
        blob = _unwrap_json_blob(text)
        parsed = json.loads(blob)
        annotations = parsed.get("annotations", [])
        if not isinstance(annotations, list):
            return []
        return [str(x).strip() for x in annotations if str(x).strip()][:5]
    except (json.JSONDecodeError, TypeError, AttributeError):
        return []


class VectorDBService:
    def __init__(self, broker: RedisBroker, *, collection: Any | None = None) -> None:
        self.broker = broker
        self._collection = collection
        self._register_handlers()

    async def _annotate_image(self, path: str) -> list[str]:
        token = os.environ.get("MODELS_TOKEN", "")
        if not token:
            log.warning("[VectorDB] MODELS_TOKEN is not set; skipping annotation")
            return []

        image_path = Path(path)
        if not image_path.is_file():
            log.warning("[VectorDB] Cannot annotate missing file: %s", path)
            return []

        try:
            image_bytes = image_path.read_bytes()
        except OSError as e:
            log.warning("[VectorDB] Failed reading image %s: %s", path, e)
            return []

        ext = image_path.suffix.lower().lstrip(".")
        mime = f"image/{ext}" if ext in {"png", "jpg", "jpeg", "webp", "gif"} else "image/jpeg"
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:{mime};base64,{image_b64}"

        payload = {
            "model": _GITHUB_MODELS_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Look at this image and return exactly 5 short tags that describe "
                                "what is visible. Return strict JSON with this shape only: "
                                '{"annotations":["tag1","tag2","tag3","tag4","tag5"]}.'
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "temperature": 0.2,
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            _GITHUB_MODELS_CHAT_COMPLETIONS_URL,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
            method="POST",
        )

        try:
            response_body = await asyncio.to_thread(self._urlopen_read, req)
            response_json = json.loads(response_body)
            message = response_json.get("choices", [{}])[0].get("message", {})
            if not isinstance(message, dict):
                message = {}
            content_text = _message_content_to_text(message)
            annotations = _parse_annotations_from_model_text(content_text)
            if not annotations:
                log.warning(
                    "[VectorDB] No annotations parsed for %s (content preview=%r)",
                    path,
                    (content_text[:300] + "…") if len(content_text) > 300 else content_text,
                )
            return annotations
        except error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            log.warning(
                "[VectorDB] Annotation HTTP %s for %s: %s",
                e.code,
                path,
                (err_body[:800] + "…") if len(err_body) > 800 else err_body,
            )
            return []
        except (error.URLError, json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
            log.warning("[VectorDB] Annotation request failed for %s: %s", path, e)
            return []

    @staticmethod
    def _urlopen_read(req: request.Request) -> str:
        with request.urlopen(req, timeout=120) as resp:
            return resp.read().decode("utf-8")

    async def _cached_annotations_for_image(self, image_id: str) -> list[str] | None:
        if self._collection is None:
            return None
        prior = await self._collection.find_one({"_id": image_id}) or {}
        ann = _normalize_annotation_list(prior.get("annotations"))
        return ann if ann else None

    async def _cached_annotations_for_same_path(self, path: str, exclude_id: str) -> list[str] | None:
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
            ann = _normalize_annotation_list(rec.get("annotations"))
            if ann:
                return ann
        return None

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

            reuse = await self._cached_annotations_for_image(image_id)
            if reuse is None:
                reuse = await self._cached_annotations_for_same_path(path, image_id)
            from_cache = reuse is not None

            if from_cache:
                annotations = reuse
                log.info(
                    "[VectorDB] using stored annotations for image_id=%s (no model call)",
                    image_id,
                )
            else:
                annotations = await self._annotate_image(path)

            search_text = " ".join(annotations)
            await self._collection.update_one(
                {"_id": image_id},
                {
                    "$set": {
                        "image_id": image_id,
                        "path": path,
                        "annotations": annotations,
                        "search_text": search_text,
                        "updated_at": _utc_now_iso(),
                    },
                    "$setOnInsert": {"_id": image_id},
                },
                upsert=True,
            )
            log.info(
                "[VectorDB] stored image_id=%s annotations=%s",
                image_id,
                annotations,
            )

        @self.broker.on(Channels.SEARCH_REQUESTED)
        async def on_search_requested(msg: dict) -> None:
            query = msg.get("query", "")
            top_k = msg.get("top_k", 5)
            query_tokens = [t for t in query.lower().split() if t]
            if self._collection is None:
                log.warning("[VectorDB] no Mongo collection configured")
                return

            scored: list[tuple[int, dict]] = []
            async for record in self._collection.find({}):
                searchable = (
                    f"{record.get('search_text', '')} {Path(record.get('path', '')).name}"
                ).lower()
                score = sum(1 for token in query_tokens if token in searchable)
                if score > 0 or not query_tokens:
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
                    "annotations": rec.get("annotations", []),
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
