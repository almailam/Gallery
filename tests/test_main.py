from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from main import _debug_log_path, _debug_terminal_command, _mongo_timeout_ms, _open_storage, _try_connect
from gallery.services.json_collection import JsonCollection


def test_debug_log_path_defaults_to_gallery_data():
    with patch.dict("main.os.environ", {}, clear=True):
        path = _debug_log_path()

    assert path.name == "debug.log"
    assert path.parent.name == "gallery_data"


def test_debug_log_path_uses_environment_override():
    with patch.dict("main.os.environ", {"GALLERY_DEBUG_LOG": "~/gallery-debug.log"}):
        path = _debug_log_path()

    assert path == Path("~/gallery-debug.log").expanduser()


def test_debug_terminal_command_tails_quoted_log_path():
    command = _debug_terminal_command(Path("/tmp/Gallery Logs/debug.log"))

    assert command.startswith("clear; ")
    assert "Gallery Redis/debug messages" in command
    assert "tail -n +1 -f '/tmp/Gallery Logs/debug.log'" in command


def test_mongo_timeout_ms_defaults_for_invalid_environment():
    with patch.dict("main.os.environ", {"MONGO_TIMEOUT_MS": "not-a-number"}):
        assert _mongo_timeout_ms() == 1000


@pytest.mark.asyncio
async def test_open_storage_falls_back_to_json_when_mongo_unavailable(tmp_path):
    class _Admin:
        async def command(self, name):
            assert name == "ping"
            raise RuntimeError("mongo down")

    class _Client:
        admin = _Admin()

        async def close(self):
            return None

    with (
        patch("main.AsyncMongoClient", return_value=_Client()) as client_cls,
        patch("main._DEFAULT_DATA_DIR", tmp_path),
    ):
        client, documents, vectors, label = await _open_storage()

    client_cls.assert_called_once()
    assert client is None
    assert isinstance(documents, JsonCollection)
    assert isinstance(vectors, JsonCollection)
    assert "local JSON storage" in label


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
