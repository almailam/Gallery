from unittest.mock import MagicMock

from prompt_toolkit.document import Document

from gallery.services.cli import _CLICompleter


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
