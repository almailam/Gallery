"""Small JSON-backed collection used when MongoDB is unavailable locally."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any


class _JsonAsyncCursor:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = records
        self._index = 0

    def __aiter__(self) -> "_JsonAsyncCursor":
        return self

    async def __anext__(self) -> dict[str, Any]:
        if self._index >= len(self._records):
            raise StopAsyncIteration
        record = self._records[self._index]
        self._index += 1
        return dict(record)


class JsonCollection:
    """Minimal async collection API matching the Mongo calls used by services."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = asyncio.Lock()
        self._records = self._load_records()

    def _load_records(self) -> dict[str, dict[str, Any]]:
        if not self._path.exists():
            return {}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        records = raw.get("records", {})
        if not isinstance(records, dict):
            return {}
        return {
            str(key): value
            for key, value in records.items()
            if isinstance(value, dict)
        }

    def _write_records(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"records": self._records}
        self._path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    async def find_one(self, filter_doc: dict[str, Any]) -> dict[str, Any] | None:
        key = filter_doc.get("_id")
        if key is None:
            return None
        async with self._lock:
            record = self._records.get(str(key))
            return dict(record) if record else None

    async def update_one(
        self,
        filter_doc: dict[str, Any],
        update_doc: dict[str, Any],
        *,
        upsert: bool = False,
    ) -> None:
        key = filter_doc.get("_id")
        if key is None:
            return
        key = str(key)
        async with self._lock:
            if key not in self._records:
                if not upsert:
                    return
                self._records[key] = dict(update_doc.get("$setOnInsert", {}))
            self._records[key].update(update_doc.get("$set", {}))
            self._write_records()

    def find(self, filter_doc: dict[str, Any]) -> _JsonAsyncCursor:
        del filter_doc
        return _JsonAsyncCursor([dict(record) for record in self._records.values()])
