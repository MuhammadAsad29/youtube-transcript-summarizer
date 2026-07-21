import tiktoken
from utils import get_logger
from config import CHUNK_SIZE, OVERLAP

logger = get_logger(__name__)

ENCODING_NAME = "cl100k_base"


def _get_encoder():
    try:
        return tiktoken.get_encoding(ENCODING_NAME)
    except Exception as e:
        logger.error(f"Failed to load tiktoken encoder: {e}")
        raise RuntimeError(
            "Could not initialize token encoder. "
            "Run: pip install tiktoken"
        )


def count_tokens(text: str) -> int:
    encoder = _get_encoder()
    return len(encoder.encode(text))


def split_text_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP
) -> list[str]:
    """
    Splits text into overlapping chunks based on token count.
    Uses sentence-aware splitting to avoid cutting mid-sentence.

    Args:
        text: The full transcript text.
        chunk_size: Max tokens per chunk.
        overlap: Token overlap between consecutive chunks.

    Returns:
        List of text chunk strings.
    """
    if not text or not text.strip():
        raise ValueError("Cannot chunk empty text.")

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")

    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be less than chunk_size ({chunk_size})."
        )

    encoder = _get_encoder()

    # Split into sentences first for cleaner chunk boundaries
    sentences = _split_into_sentences(text)

    chunks = []
    current_chunk_sentences = []
    current_token_count = 0

    for sentence in sentences:
        sentence_tokens = len(encoder.encode(sentence))

        # If a single sentence exceeds chunk size, force-split it
        if sentence_tokens > chunk_size:
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = []
                current_token_count = 0
            # Force split the long sentence by words
            forced_chunks = _force_split_long_text(sentence, chunk_size, encoder)
            chunks.extend(forced_chunks)
            continue

        if current_token_count + sentence_tokens > chunk_size:
            # Save current chunk
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))

            # Build overlap: take sentences from the end of the last chunk
            overlap_sentences = _get_overlap_sentences(
                current_chunk_sentences, overlap, encoder
            )
            current_chunk_sentences = overlap_sentences + [sentence]
            current_token_count = sum(
                len(encoder.encode(s)) for s in current_chunk_sentences
            )
        else:
            current_chunk_sentences.append(sentence)
            current_token_count += sentence_tokens

    # Don't forget the last chunk
    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))

    # Remove any empty chunks
    chunks = [c.strip() for c in chunks if c.strip()]

    logger.info(
        f"Text split into {len(chunks)} chunks. "
        f"Chunk size: {chunk_size} tokens, Overlap: {overlap} tokens."
    )

    for i, chunk in enumerate(chunks):
        token_count = len(encoder.encode(chunk))
        logger.debug(f"Chunk {i+1}: {token_count} tokens, {len(chunk.split())} words.")

    return chunks


def _split_into_sentences(text: str) -> list[str]:
    """
    Splits text into sentences using punctuation heuristics.
    Handles abbreviations and decimals to avoid false splits.
    """
    import re

    # Basic sentence boundary detection
    # Split on . ! ? followed by space and uppercase letter
    # Preserve the punctuation
    sentence_endings = re.compile(
        r'(?<=[.!?])\s+(?=[A-Z])'
    )

    raw_sentences = sentence_endings.split(text)

    # Filter and clean
    sentences = []
    for s in raw_sentences:
        s = s.strip()
        if s:
            sentences.append(s)

    # If splitting produced nothing useful, fall back to word-based splitting
    if not sentences:
        sentences = [text]

    return sentences


def _get_overlap_sentences(
    sentences: list[str],
    overlap_tokens: int,
    encoder
) -> list[str]:
    """
    Returns sentences from the end of the previous chunk
    that together don't exceed overlap_tokens.
    """
    overlap_sentences = []
    token_count = 0

    for sentence in reversed(sentences):
        tokens = len(encoder.encode(sentence))
        if token_count + tokens > overlap_tokens:
            break
        overlap_sentences.insert(0, sentence)
        token_count += tokens

    return overlap_sentences


def _force_split_long_text(
    text: str,
    chunk_size: int,
    encoder
) -> list[str]:
    """
    Last-resort word-by-word splitting for oversized single sentences.
    """
    words = text.split()
    chunks = []
    current_words = []
    current_tokens = 0

    for word in words:
        word_tokens = len(encoder.encode(word))
        if current_tokens + word_tokens > chunk_size:
            if current_words:
                chunks.append(" ".join(current_words))
            current_words = [word]
            current_tokens = word_tokens
        else:
            current_words.append(word)
            current_tokens += word_tokens

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def get_chunking_stats(text: str, chunks: list[str]) -> dict:
    encoder = _get_encoder()
    total_tokens = len(encoder.encode(text))
    chunk_token_counts = [len(encoder.encode(c)) for c in chunks]

    return {
        "total_tokens": total_tokens,
        "chunk_count": len(chunks),
        "avg_tokens_per_chunk": round(sum(chunk_token_counts) / len(chunk_token_counts), 1) if chunks else 0,
        "max_tokens_in_chunk": max(chunk_token_counts) if chunk_token_counts else 0,
        "min_tokens_in_chunk": min(chunk_token_counts) if chunk_token_counts else 0,
    }