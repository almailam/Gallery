"""Shared utilities used across gallery services and broker."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _use_ansi_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR", "") == ""
