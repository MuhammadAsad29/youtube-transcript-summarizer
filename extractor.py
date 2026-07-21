import subprocess
import json
import os
import re
import tempfile
from pathlib import Path
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from utils import get_logger, clean_transcript_text, validate_youtube_url

logger = get_logger(__name__)

PREFERRED_LANGUAGES = ["en", "en-US", "en-GB", "en-CA", "en-AU"]


def extract_transcript(url: str) -> tuple[str, str]:
    """
    Main entry point for transcript extraction.
    Returns (clean_transcript_text, method_used).
    Tries primary method first, falls back to secondary.

    Raises:
        ValueError: If URL is invalid.
        RuntimeError: If all extraction methods fail.
    """
    is_valid, video_id = validate_youtube_url(url)
    if not is_valid:
        raise ValueError(
            f"Invalid YouTube URL: '{url}'. "
            "Please provide a valid YouTube watch, short, or embed URL."
        )

    logger.info(f"Starting transcript extraction for video ID: {video_id}")

    # Primary: youtube-transcript-api
    try:
        raw_text = _extract_via_transcript_api(video_id)
        logger.info("Primary extraction successful (youtube-transcript-api).")
        return clean_transcript_text(raw_text), "youtube-transcript-api"
    except Exception as e:
        logger.warning(f"Primary extraction failed: {e}. Trying fallback (yt-dlp).")

    # Fallback: yt-dlp
    try:
        raw_text = _extract_via_ytdlp(url)
        logger.info("Fallback extraction successful (yt-dlp).")
        return clean_transcript_text(raw_text), "yt-dlp"
    except Exception as e:
        logger.error(f"Fallback extraction also failed: {e}")
        raise RuntimeError(
            "Could not extract transcript from this video. "
            "Possible reasons: captions are disabled, the video is private or age-restricted, "
            "or the video has no available subtitles. Please try a different video."
        )


def _extract_via_transcript_api(video_id: str) -> str:
    """
    Uses youtube-transcript-api to fetch transcript segments.
    Tries preferred languages first, then falls back to any available language.
    """
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

    transcript = None

    # Try manually uploaded transcripts in preferred languages first
    try:
        transcript = transcript_list.find_manually_created_transcript(PREFERRED_LANGUAGES)
        logger.info("Found manually created transcript.")
    except NoTranscriptFound:
        pass

    # Try auto-generated transcripts in preferred languages
    if transcript is None:
        try:
            transcript = transcript_list.find_generated_transcript(PREFERRED_LANGUAGES)
            logger.info("Found auto-generated transcript.")
        except NoTranscriptFound:
            pass

    # Take whatever is available and translate if necessary
    if transcript is None:
        available = list(transcript_list)
        if not available:
            raise NoTranscriptFound(video_id, PREFERRED_LANGUAGES, [])
        transcript = available[0]
        logger.info(
            f"Using transcript in language: {transcript.language} ({transcript.language_code}). "
            "Translation to English will be attempted."
        )
        if transcript.language_code not in [lang.split("-")[0] for lang in PREFERRED_LANGUAGES]:
            try:
                transcript = transcript.translate("en")
                logger.info("Transcript translated to English.")
            except Exception as te:
                logger.warning(f"Translation failed: {te}. Using original language.")

    segments = transcript.fetch()

    # Concatenate all text segments
    full_text = " ".join(
        segment["text"].replace("\n", " ")
        for segment in segments
        if segment.get("text")
    )

    if not full_text.strip():
        raise ValueError("Transcript fetched but contained no text content.")

    logger.info(f"Transcript extracted. Total characters: {len(full_text)}")
    return full_text


def _extract_via_ytdlp(url: str) -> str:
    """
    Uses yt-dlp CLI to download subtitle files, then parses them to plain text.
    Works with VTT format auto-generated subtitles.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        output_template = os.path.join(tmpdir, "subtitle")

        command = [
            "yt-dlp",
            "--write-auto-sub",
            "--write-sub",
            "--sub-lang", "en",
            "--sub-format", "vtt",
            "--skip-download",
            "--no-playlist",
            "--output", output_template,
            url
        ]

        logger.info(f"Running yt-dlp command: {' '.join(command)}")

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            logger.error(f"yt-dlp stderr: {result.stderr}")
            raise RuntimeError(f"yt-dlp failed with return code {result.returncode}.")

        # Find the downloaded VTT file
        vtt_files = list(Path(tmpdir).glob("*.vtt"))
        if not vtt_files:
            raise FileNotFoundError(
                "yt-dlp ran successfully but no subtitle file was produced. "
                "The video may not have auto-generated captions."
            )

        vtt_file = vtt_files[0]
        logger.info(f"VTT file found: {vtt_file.name}")

        raw_content = vtt_file.read_text(encoding="utf-8", errors="ignore")
        return _parse_vtt_to_text(raw_content)


def _parse_vtt_to_text(vtt_content: str) -> str:
    """
    Parses VTT subtitle file content to plain text.
    Removes all timestamp cues, headers, and metadata.
    Deduplicates repeated lines common in auto-generated VTT files.
    """
    lines = vtt_content.splitlines()
    text_lines = []
    seen_lines = set()

    timestamp_pattern = re.compile(
        r"\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}"
    )
    cue_setting_pattern = re.compile(
        r"(align|position|size|line|vertical):"
    )

    for line in lines:
        line = line.strip()

        # Skip empty lines, headers, timestamps, cue settings, and numeric cue IDs
        if not line:
            continue
        if line.startswith("WEBVTT") or line.startswith("Kind:") or line.startswith("Language:"):
            continue
        if timestamp_pattern.search(line):
            continue
        if cue_setting_pattern.search(line):
            continue
        if line.isdigit():
            continue

        # Deduplicate (auto-generated VTT often repeats lines)
        normalized = line.lower().strip()
        if normalized not in seen_lines:
            seen_lines.add(normalized)
            text_lines.append(line)

    return " ".join(text_lines)


def get_video_metadata(url: str) -> dict:
    """
    Fetches basic video metadata using yt-dlp --dump-json.
    Returns a dict with title, duration, uploader, upload_date.
    Does not raise — returns empty dict on failure.
    """
    try:
        result = subprocess.run(
            ["yt-dlp", "--dump-json", "--no-playlist", "--skip-download", url],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0 and result.stdout:
            data = json.loads(result.stdout)
            return {
                "title": data.get("title", "Unknown Title"),
                "duration_seconds": data.get("duration", 0),
                "uploader": data.get("uploader", "Unknown"),
                "upload_date": data.get("upload_date", ""),
                "view_count": data.get("view_count", 0),
                "description": (data.get("description", "") or "")[:300]
            }
    except Exception as e:
        logger.warning(f"Metadata fetch failed (non-critical): {e}")
    return {}