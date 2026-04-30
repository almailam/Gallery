"""Unit tests for gallery.utils shared helpers."""

from __future__ import annotations

import os
from datetime import datetime, timezone

from gallery.utils import _use_ansi_color, _utc_now_iso


def test_utc_now_iso_returns_parseable_timestamp():
    ts = _utc_now_iso()
    dt = datetime.fromisoformat(ts)
    assert dt.tzinfo is not None


def test_utc_now_iso_advances_over_time():
    first = _utc_now_iso()
    second = _utc_now_iso()
    assert second >= first


def test_use_ansi_color_disabled_by_no_color_env(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert _use_ansi_color() is False


def test_use_ansi_color_disabled_when_not_a_tty(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    # In the test runner stdout is not a TTY.
    assert _use_ansi_color() is False
