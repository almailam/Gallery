from unittest.mock import MagicMock

from prompt_toolkit.document import Document

from gallery.services.cli import (
    COMMAND_BANNER,
    COMMAND_SPECS,
    COMMAND_USAGE,
    COMPLETION_COMMANDS,
    EXIT_COMMANDS,
    _CLICompleter,
)


def test_cli_completer_matches_upload_prefix():
    c = _CLICompleter()
    doc = Document("u", cursor_position=1)
    texts = [comp.text for comp in c.get_completions(doc, MagicMock())]
    assert "upload" in texts


def test_cli_completer_matches_search_prefix():
    c = _CLICompleter()
    doc = Document("sea", cursor_position=3)
    texts = [comp.text for comp in c.get_completions(doc, MagicMock())]
    assert "search" in texts


def test_cli_completer_matches_list_prefix():
    c = _CLICompleter()
    doc = Document("li", cursor_position=2)
    texts = [comp.text for comp in c.get_completions(doc, MagicMock())]
    assert "list" in texts


def test_cli_completer_uses_shared_command_list():
    assert _CLICompleter._commands == COMPLETION_COMMANDS
    assert all(command in COMPLETION_COMMANDS for command in EXIT_COMMANDS)


def test_cli_usage_and_banner_include_command_specs_once():
    for spec, _description in COMMAND_SPECS:
        assert COMMAND_USAGE.count(spec) == 1
        assert COMMAND_BANNER.count(spec) == 1
    assert COMMAND_USAGE.startswith("Usage:")
    assert COMMAND_BANNER.startswith("\nCommands:")
