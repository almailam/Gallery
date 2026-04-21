from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from main import _try_connect


@pytest.mark.asyncio
async def test_try_connect_succeeds_first_attempt():
    broker = MagicMock()
    broker.connect = AsyncMock()
    ok = await _try_connect(broker, attempts=5)
    assert ok is True
    broker.connect.assert_awaited_once()


@pytest.mark.asyncio
async def test_try_connect_false_after_exhausted_attempts():
    broker = MagicMock()
    broker.connect = AsyncMock(side_effect=RuntimeError("unavailable"))
    with patch("main.asyncio.sleep", new_callable=AsyncMock):
        ok = await _try_connect(broker, attempts=3)
    assert ok is False
    assert broker.connect.await_count == 3
