import pytest
from utils import validate_youtube_url, clean_transcript_text
from extractor import _parse_vtt_to_text


def test_valid_standard_url():
    ok, vid = validate_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert ok is True
    assert vid == "dQw4w9WgXcQ"


def test_valid_short_url():
    ok, vid = validate_youtube_url("https://youtu.be/dQw4w9WgXcQ")
    assert ok is True
    assert vid == "dQw4w9WgXcQ"


def test_valid_shorts_url():
    ok, vid = validate_youtube_url("https://www.youtube.com/shorts/dQw4w9WgXcQ")
    assert ok is True
    assert vid == "dQw4w9WgXcQ"


def test_invalid_url_random_string():
    ok, vid = validate_youtube_url("not_a_url_at_all")
    assert ok is False
    assert vid == ""


def test_invalid_url_empty():
    ok, vid = validate_youtube_url("")
    assert ok is False


def test_invalid_url_none():
    ok, vid = validate_youtube_url(None)
    assert ok is False


def test_clean_transcript_removes_html():
    raw = "Hello <b>world</b> this is <i>a test</i>"
    cleaned = clean_transcript_text(raw)
    assert "<b>" not in cleaned
    assert "<i>" not in cleaned
    assert "Hello" in cleaned
    assert "world" in cleaned


def test_clean_transcript_removes_timestamps():
    raw = "[00:01] Hello there [1:23] how are you"
    cleaned = clean_transcript_text(raw)
    assert "[00:01]" not in cleaned
    assert "[1:23]" not in cleaned


def test_clean_transcript_normalizes_whitespace():
    raw = "Hello    world   this   is   spaced"
    cleaned = clean_transcript_text(raw)
    assert "  " not in cleaned


def test_parse_vtt_removes_headers():
    vtt = """WEBVTT
Kind: captions
Language: en

1
00:00:01.000 --> 00:00:03.000
Hello world

2
00:00:03.500 --> 00:00:05.000
This is a test
"""
    result = _parse_vtt_to_text(vtt)
    assert "WEBVTT" not in result
    assert "00:00:01" not in result
    assert "Hello world" in result
    assert "This is a test" in result


def test_parse_vtt_deduplicates():
    vtt = """WEBVTT

00:00:01.000 --> 00:00:03.000
Hello world

00:00:02.000 --> 00:00:04.000
Hello world
"""
    result = _parse_vtt_to_text(vtt)
    # Should appear only once (case-insensitive dedup)
    assert result.lower().count("hello world") == 1