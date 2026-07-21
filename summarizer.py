import gc
import time
import requests
import json
from config import (
    OLLAMA_HOST, MODEL_NAME, TEMPERATURE, TOP_P,
    NUM_CTX, NUM_THREAD, TIMEOUT,
    MAX_SUMMARY_WORDS, MIN_KEY_POINTS, MAX_KEY_POINTS
)
from utils import get_logger, count_words, compute_compression_ratio

logger = get_logger(__name__)

GENERATE_ENDPOINT = f"{OLLAMA_HOST}/api/generate"
TAGS_ENDPOINT = f"{OLLAMA_HOST}/api/tags"


def check_ollama_connection() -> tuple[bool, str]:
    """
    Checks if Ollama server is reachable and the configured model is available.
    Returns (is_ok, message).
    """
    try:
        response = requests.get(TAGS_ENDPOINT, timeout=5)
        if response.status_code != 200:
            return False, f"Ollama server returned status {response.status_code}."

        data = response.json()
        available_models = [m["name"] for m in data.get("models", [])]

        if not any(MODEL_NAME in m for m in available_models):
            return False, (
                f"Model '{MODEL_NAME}' is not available locally. "
                f"Available models: {available_models}. "
                f"Run: ollama pull {MODEL_NAME}"
            )

        return True, f"Ollama connected. Model '{MODEL_NAME}' is ready."

    except requests.exceptions.ConnectionError:
        return False, (
            "Cannot connect to Ollama server at "
            f"{OLLAMA_HOST}. "
            "Make sure Ollama is installed and running. "
            "Start it with: ollama serve"
        )
    except Exception as e:
        return False, f"Unexpected error checking Ollama: {e}"


def list_available_models() -> list[str]:
    """Returns list of locally available Ollama model names."""
    try:
        response = requests.get(TAGS_ENDPOINT, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        pass
    return []


def _call_ollama(prompt: str, system_prompt: str = "") -> str:
    """
    Makes a single blocking call to the Ollama /api/generate endpoint.
    Returns the generated text string.
    Raises RuntimeError on failure.
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "num_ctx": NUM_CTX,
            "num_thread": NUM_THREAD,
        }
    }

    if system_prompt:
        payload["system"] = system_prompt

    try:
        response = requests.post(
            GENERATE_ENDPOINT,
            json=payload,
            timeout=TIMEOUT
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama returned HTTP {response.status_code}: {response.text[:200]}"
            )

        result = response.json()
        generated_text = result.get("response", "").strip()

        if not generated_text:
            raise RuntimeError("Ollama returned an empty response.")

        return generated_text

    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"Ollama request timed out after {TIMEOUT} seconds. "
            "Consider reducing chunk size or increasing TIMEOUT in .env."
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Lost connection to Ollama server during inference. "
            "Ensure Ollama is still running."
        )


def summarize_chunk(chunk: str, chunk_index: int, total_chunks: int) -> str:
    """
    Summarizes a single transcript chunk.
    Returns a 3-5 sentence summary string.
    """
    system_prompt = (
        "You are an expert content summarizer. "
        "Your task is to summarize sections of a YouTube video transcript. "
        "Be concise, accurate, and preserve all technical terms and key facts. "
        "Do not add information that is not in the provided text. "
        "Do not introduce yourself or add any preamble."
    )

    prompt = (
        f"This is chunk {chunk_index} of {total_chunks} from a YouTube video transcript.\n\n"
        f"TRANSCRIPT CHUNK:\n{chunk}\n\n"
        f"Write a concise summary of this chunk in 3 to 5 sentences. "
        f"Focus on the most important information, key facts, and main ideas presented. "
        f"Output only the summary, no headings or labels."
    )

    logger.info(f"Summarizing chunk {chunk_index}/{total_chunks}...")
    summary = _call_ollama(prompt, system_prompt)
    logger.info(f"Chunk {chunk_index} summarized. Output length: {len(summary)} chars.")

    # Free memory between chunk calls
    gc.collect()

    return summary


def generate_final_summary(
    chunk_summaries: list[str],
    video_title: str = ""
) -> dict:
    """
    Takes all per-chunk summaries and generates:
    - Executive summary (150-200 words)
    - Key points (5-10 bullet points)
    - Main themes (3 topics)

    Returns a dict with keys: executive_summary, key_points, themes.
    """
    combined_summaries = "\n\n".join(
        f"[Section {i+1}]: {summary}"
        for i, summary in enumerate(chunk_summaries)
    )

    system_prompt = (
        "You are an expert analyst who synthesizes information from video content "
        "into clear, structured summaries for busy professionals. "
        "Be precise, factual, and use plain English. "
        "Do not repeat information unnecessarily. "
        "Always respond in the exact format requested."
    )

    title_context = f'The video is titled: "{video_title}"\n\n' if video_title else ""

    # --- Executive Summary ---
    executive_prompt = (
        f"{title_context}"
        f"Below are summaries from different sections of a YouTube video transcript:\n\n"
        f"{combined_summaries}\n\n"
        f"Write a single cohesive executive summary of the entire video in "
        f"{MAX_SUMMARY_WORDS - 20} to {MAX_SUMMARY_WORDS} words. "
        f"The summary should flow as natural paragraphs, cover the main subject, "
        f"key arguments, and conclusions. "
        f"Output only the summary text, no labels or headings."
    )

    logger.info("Generating executive summary...")
    executive_summary = _call_ollama(executive_prompt, system_prompt)
    gc.collect()

    # --- Key Points ---
    keypoints_prompt = (
        f"{title_context}"
        f"Based on these video section summaries:\n\n"
        f"{combined_summaries}\n\n"
        f"Extract {MIN_KEY_POINTS} to {MAX_KEY_POINTS} key takeaways from the video. "
        f"Each takeaway should be a single, complete, actionable or informative sentence. "
        f"Format your response as a numbered list like:\n"
        f"1. [First key point]\n"
        f"2. [Second key point]\n"
        f"Continue until all key points are listed. "
        f"Output only the numbered list, nothing else."
    )

    logger.info("Generating key points...")
    raw_key_points = _call_ollama(keypoints_prompt, system_prompt)
    key_points = _parse_numbered_list(raw_key_points)
    gc.collect()

    # --- Themes ---
    themes_prompt = (
        f"{title_context}"
        f"Based on these video section summaries:\n\n"
        f"{combined_summaries}\n\n"
        f"Identify exactly 3 main themes or topics that this video covers. "
        f"Each theme should be a short phrase (3-8 words). "
        f"Format your response as:\n"
        f"1. [First theme]\n"
        f"2. [Second theme]\n"
        f"3. [Third theme]\n"
        f"Output only the 3 themes, nothing else."
    )

    logger.info("Generating themes...")
    raw_themes = _call_ollama(themes_prompt, system_prompt)
    themes = _parse_numbered_list(raw_themes)[:3]
    gc.collect()

    return {
        "executive_summary": executive_summary,
        "key_points": key_points,
        "themes": themes
    }


def run_summarization_pipeline(
    transcript: str,
    chunks: list[str],
    video_title: str = "",
    progress_callback=None
) -> dict:
    """
    Orchestrates the full Map-Reduce summarization pipeline.

    Args:
        transcript: Full original transcript text.
        chunks: List of text chunks from chunker.py.
        video_title: Optional video title for context.
        progress_callback: Optional callable(stage: str, current: int, total: int).
                           Used by Streamlit to update a progress bar.

    Returns:
        Complete result dict ready for display and export.
    """
    start_time = time.time()
    total_chunks = len(chunks)

    if total_chunks == 0:
        raise ValueError("No chunks provided for summarization.")

    logger.info(
        f"Starting summarization pipeline. "
        f"Total chunks: {total_chunks}, Model: {MODEL_NAME}"
    )

    # --- Stage 1: Chunk Summarization (Map) ---
    chunk_summaries = []
    for i, chunk in enumerate(chunks, 1):
        if progress_callback:
            progress_callback("chunk", i, total_chunks)

        chunk_summary = summarize_chunk(chunk, i, total_chunks)
        chunk_summaries.append(chunk_summary)

    logger.info(f"All {total_chunks} chunks summarized. Starting final synthesis.")

    # --- Stage 2: Final Synthesis (Reduce) ---
    if progress_callback:
        progress_callback("synthesis", 1, 1)

    final = generate_final_summary(chunk_summaries, video_title)

    elapsed = round(time.time() - start_time, 1)

    result = {
        "executive_summary": final["executive_summary"],
        "key_points": final["key_points"],
        "themes": final["themes"],
        "full_transcript": transcript,
        "word_count_original": count_words(transcript),
        "word_count_summary": count_words(final["executive_summary"]),
        "compression_ratio": compute_compression_ratio(transcript, final["executive_summary"]),
        "processing_time_seconds": elapsed,
        "model_used": MODEL_NAME,
        "chunk_count": total_chunks,
        "chunk_summaries": chunk_summaries
    }

    logger.info(
        f"Pipeline complete. "
        f"Time: {elapsed}s, "
        f"Compression: {result['compression_ratio']}x, "
        f"Original: {result['word_count_original']} words, "
        f"Summary: {result['word_count_summary']} words."
    )

    return result


def _parse_numbered_list(raw_text: str) -> list[str]:
    """
    Parses a numbered list response from the LLM into a Python list.
    Handles various numbering formats: '1.', '1)', '- 1.', etc.
    """
    import re
    lines = raw_text.strip().splitlines()
    items = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Remove leading numbering like "1.", "1)", "- 1.", "• "
        cleaned = re.sub(r"^[\-\•\*]?\s*\d+[\.\)]\s*", "", line).strip()
        # Also handle plain bullet points
        cleaned = re.sub(r"^[\-\•\*]\s+", "", cleaned).strip()
        if cleaned and len(cleaned) > 5:
            items.append(cleaned)

    return items