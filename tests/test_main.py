from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import signal

import pytest

from main import (
    _debug_log_path,
    _debug_terminal_command,
    _install_signal_handlers,
    _mongo_timeout_ms,
    _open_storage,
    _remove_signal_handlers,
    _try_connect,
)
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


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------


def _capture_signal_handlers(loop_mock, stop_event):
    """Helper: install handlers on a MagicMock loop and return the captured callbacks."""
    registered: dict[signal.Signals, tuple] = {}

    def fake_add(sig, fn, *args):
        registered[sig] = (fn, args)

    loop_mock.add_signal_handler.side_effect = fake_add
    _install_signal_handlers(loop_mock, stop_event)
    return registered


def test_install_signal_handlers_registers_sigint():
    loop = MagicMock()
    stop_event = MagicMock()
    stop_event.is_set.return_value = False

    registered = _capture_signal_handlers(loop, stop_event)

    assert signal.SIGINT in registered


def test_install_signal_handlers_registers_sigterm():
    loop = MagicMock()
    stop_event = MagicMock()
    stop_event.is_set.return_value = False

    registered = _capture_signal_handlers(loop, stop_event)

    assert signal.SIGTERM in registered


def test_install_signal_handlers_sigint_sets_stop_event():
    loop = MagicMock()
    stop_event = MagicMock()
    stop_event.is_set.return_value = False

    registered = _capture_signal_handlers(loop, stop_event)
    fn, args = registered[signal.SIGINT]
    fn(*args)

    stop_event.set.assert_called_once()


def test_install_signal_handlers_sigterm_sets_stop_event():
    loop = MagicMock()
    stop_event = MagicMock()
    stop_event.is_set.return_value = False

    registered = _capture_signal_handlers(loop, stop_event)
    fn, args = registered[signal.SIGTERM]
    fn(*args)

    stop_event.set.assert_called_once()


def test_install_signal_handlers_idempotent_second_signal(capsys):
    """Calling the handler a second time while already stopped does not print again."""
    loop = MagicMock()
    stop_event = MagicMock()
    # First call: not set yet; second call: already set.
    stop_event.is_set.side_effect = [False, True]

    registered = _capture_signal_handlers(loop, stop_event)
    fn, args = registered[signal.SIGINT]
    fn(*args)
    fn(*args)

    # set() called only once
    assert stop_event.set.call_count == 1
    captured = capsys.readouterr().out
    assert captured.count("Shutting down") == 1


def test_install_signal_handlers_tolerates_not_implemented():
    """On platforms where add_signal_handler is unsupported the call silently passes."""
    loop = MagicMock()
    loop.add_signal_handler = MagicMock(side_effect=NotImplementedError)
    stop_event = MagicMock()
    # Must not raise.
    _install_signal_handlers(loop, stop_event)


def test_remove_signal_handlers_calls_remove_for_sigint_and_sigterm():
    loop = MagicMock()
    _remove_signal_handlers(loop)
    calls = [call.args[0] for call in loop.remove_signal_handler.call_args_list]
    assert signal.SIGINT in calls
    assert signal.SIGTERM in calls


def test_remove_signal_handlers_tolerates_not_implemented():
    loop = MagicMock()
    loop.remove_signal_handler = MagicMock(side_effect=NotImplementedError)
    # Must not raise.
    _remove_signal_handlers(loop)


def test_mongo_timeout_ms_returns_valid_int():
    with patch.dict("main.os.environ", {"MONGO_TIMEOUT_MS": "500"}):
        assert _mongo_timeout_ms() == 500


def test_mongo_timeout_ms_clamps_below_one():
    with patch.dict("main.os.environ", {"MONGO_TIMEOUT_MS": "0"}):
        assert _mongo_timeout_ms() == 1

