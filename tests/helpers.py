"""Shared test helpers for service tests."""

from __future__ import annotations

from typing import Any


class FakeAsyncCursor:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = list(records)
        self._index = 0

    def __aiter__(self) -> "FakeAsyncCursor":
        return self

    async def __anext__(self) -> dict[str, Any]:
        if self._index >= len(self._records):
            raise StopAsyncIteration
        value = self._records[self._index]
        self._index += 1
        return dict(value)


class FakeCollection:
    """Minimal in-memory MongoDB collection stub for unit tests."""

    def __init__(self, records: dict[str, dict[str, Any]] | None = None) -> None:
        self.records: dict[str, dict[str, Any]] = records or {}

    async def find_one(self, filter_doc: dict[str, Any]) -> dict[str, Any] | None:
        key = filter_doc.get("_id")
        if key is None:
            return None
        record = self.records.get(str(key))
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
        if key not in self.records:
            if not upsert:
                return
            self.records[key] = dict(update_doc.get("$setOnInsert", {}))
        self.records[key].update(update_doc.get("$set", {}))

    def find(self, filter_doc: dict[str, Any]) -> FakeAsyncCursor:
        del filter_doc
        return FakeAsyncCursor(list(self.records.values()))

    async def delete_many(self, filter_doc: dict[str, Any]) -> None:
        del filter_doc
        self.records = {}
