import re
import logging
import time
from pathlib import Path
from urllib.parse import urlparse, parse_qs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def validate_youtube_url(url: str) -> tuple[bool, str]:
    """
    Validates a YouTube URL and returns (is_valid, video_id).
    Supports standard, shortened, and embed URLs.
    """
    if not url or not isinstance(url, str):
        return False, ""

    url = url.strip()

    patterns = [
        r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})",
        r"(?:https?://)?youtu\.be/([a-zA-Z0-9_-]{11})",
        r"(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})",
        r"(?:https?://)?(?:www\.)?youtube\.com/shorts/([a-zA-Z0-9_-]{11})",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            video_id = match.group(1)
            logger.info(f"Valid YouTube URL. Video ID: {video_id}")
            return True, video_id

    logger.warning(f"Invalid YouTube URL: {url}")
    return False, ""


def clean_transcript_text(raw_text: str) -> str:
    """
    Cleans raw transcript text:
    - Removes HTML tags
    - Strips timestamps and speaker labels
    - Normalizes whitespace
    - Removes filler characters
    """
    if not raw_text:
        return ""

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", raw_text)

    # Remove VTT/SRT timestamp patterns like 00:00:01.000 --> 00:00:03.000
    text = re.sub(
        r"\d{2}:\d{2}:\d{2}[.,]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[.,]\d{3}",
        " ", text
    )

    # Remove standalone timestamps like [00:01] or (1:23)
    text = re.sub(r"[\[\(]\d{1,2}:\d{2}(?::\d{2})?[\]\)]", " ", text)

    # Remove WEBVTT header and numeric cue identifiers
    text = re.sub(r"WEBVTT.*?\n", " ", text)
    text = re.sub(r"^\d+\s*$", " ", text, flags=re.MULTILINE)

    # Remove speaker labels like "SPEAKER_00:" or "[Music]"
    text = re.sub(r"[A-Z_]+\d*\s*:", " ", text)
    text = re.sub(r"\[(?:Music|Applause|Laughter|Inaudible)\]", " ", text, flags=re.IGNORECASE)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    logger.info(f"Cleaned transcript. Character count: {len(text)}")
    return text


def count_words(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


def compute_compression_ratio(original: str, summary: str) -> float:
    original_words = count_words(original)
    summary_words = count_words(summary)
    if summary_words == 0:
        return 0.0
    return round(original_words / summary_words, 2)


def format_output_as_markdown(result: dict) -> str:
    """
    Formats the final result dictionary into a clean Markdown string
    suitable for download.
    """
    lines = []
    lines.append("# YouTube Transcript Summary")
    lines.append(f"\n**Model Used:** {result.get('model_used', 'N/A')}")
    lines.append(f"**Processing Time:** {result.get('processing_time_seconds', 0):.1f} seconds")
    lines.append(f"**Compression Ratio:** {result.get('compression_ratio', 0)}x")
    lines.append(f"**Original Word Count:** {result.get('word_count_original', 0)}")
    lines.append(f"**Summary Word Count:** {result.get('word_count_summary', 0)}")
    lines.append(f"**Chunks Processed:** {result.get('chunk_count', 0)}")

    lines.append("\n---\n")
    lines.append("## Executive Summary")
    lines.append(result.get("executive_summary", ""))

    lines.append("\n---\n")
    lines.append("## Key Points")
    for i, point in enumerate(result.get("key_points", []), 1):
        lines.append(f"{i}. {point}")

    lines.append("\n---\n")
    lines.append("## Main Themes")
    for theme in result.get("themes", []):
        lines.append(f"- {theme}")

    lines.append("\n---\n")
    lines.append("## Full Transcript")
    lines.append(result.get("full_transcript", ""))

    return "\n".join(lines)


def save_result_to_file(result: dict, output_dir: str = "outputs") -> str:
    """
    Saves the formatted result to a .md file.
    Returns the file path as a string.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"summary_{timestamp}.md"
    filepath = Path(output_dir) / filename

    content = format_output_as_markdown(result)
    filepath.write_text(content, encoding="utf-8")

    logger.info(f"Result saved to {filepath}")
    return str(filepath)


def truncate_text_preview(text: str, max_chars: int = 500) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."