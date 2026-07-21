import pytest
from summarizer import _parse_numbered_list, check_ollama_connection


def test_parse_numbered_list_standard():
    raw = "1. First point\n2. Second point\n3. Third point"
    result = _parse_numbered_list(raw)
    assert len(result) == 3
    assert result[0] == "First point"
    assert result[1] == "Second point"
    assert result[2] == "Third point"


def test_parse_numbered_list_parenthesis_format():
    raw = "1) Alpha\n2) Beta\n3) Gamma"
    result = _parse_numbered_list(raw)
    assert len(result) == 3
    assert "Alpha" in result[0]


def test_parse_numbered_list_bullet_format():
    raw = "- First item\n- Second item\n- Third item"
    result = _parse_numbered_list(raw)
    assert len(result) == 3
    assert "First item" in result[0]


def test_parse_numbered_list_empty_input():
    result = _parse_numbered_list("")
    assert result == []


def test_parse_numbered_list_filters_short():
    raw = "1. Hi\n2. This is a proper point that should pass the filter"
    result = _parse_numbered_list(raw)
    # "Hi" is too short (< 5 chars), should be filtered
    assert len(result) == 1
    assert "proper point" in result[0]


def test_ollama_connection_returns_tuple():
    ok, msg = check_ollama_connection()
    assert isinstance(ok, bool)
    assert isinstance(msg, str)
    assert len(msg) > 0