import pytest
from chunker import split_text_into_chunks, count_tokens, get_chunking_stats


SAMPLE_TEXT = (
    "Artificial intelligence is transforming industries across the globe. "
    "Machine learning models can now recognize images with superhuman accuracy. "
    "Natural language processing has enabled machines to understand and generate human text. "
    "These technologies are being applied in healthcare, finance, education, and transportation. "
    "Researchers continue to push the boundaries of what is computationally possible. "
) * 50  # ~1500 words


def test_count_tokens_returns_int():
    result = count_tokens("Hello world this is a test.")
    assert isinstance(result, int)
    assert result > 0


def test_count_tokens_empty_string():
    result = count_tokens("")
    assert result == 0


def test_split_produces_chunks():
    chunks = split_text_into_chunks(SAMPLE_TEXT, chunk_size=500, overlap=50)
    assert isinstance(chunks, list)
    assert len(chunks) > 1


def test_no_chunk_exceeds_size():
    chunk_size = 500
    chunks = split_text_into_chunks(SAMPLE_TEXT, chunk_size=chunk_size, overlap=50)
    for i, chunk in enumerate(chunks):
        token_count = count_tokens(chunk)
        assert token_count <= chunk_size + 50, (
            f"Chunk {i} has {token_count} tokens, exceeds limit of {chunk_size}"
        )


def test_no_empty_chunks():
    chunks = split_text_into_chunks(SAMPLE_TEXT, chunk_size=500, overlap=50)
    for chunk in chunks:
        assert chunk.strip() != ""


def test_split_empty_text_raises():
    with pytest.raises(ValueError):
        split_text_into_chunks("", chunk_size=500, overlap=50)


def test_split_invalid_overlap_raises():
    with pytest.raises(ValueError):
        split_text_into_chunks(SAMPLE_TEXT, chunk_size=500, overlap=500)


def test_split_invalid_chunk_size_raises():
    with pytest.raises(ValueError):
        split_text_into_chunks(SAMPLE_TEXT, chunk_size=0, overlap=50)


def test_chunking_stats_structure():
    chunks = split_text_into_chunks(SAMPLE_TEXT, chunk_size=500, overlap=50)
    stats = get_chunking_stats(SAMPLE_TEXT, chunks)
    assert "total_tokens" in stats
    assert "chunk_count" in stats
    assert "avg_tokens_per_chunk" in stats
    assert "max_tokens_in_chunk" in stats
    assert "min_tokens_in_chunk" in stats
    assert stats["chunk_count"] == len(chunks)


def test_all_text_covered():
    """
    Verify content is not silently dropped during chunking.
    Check that first and last meaningful words appear in some chunk.
    """
    chunks = split_text_into_chunks(SAMPLE_TEXT, chunk_size=500, overlap=50)
    combined = " ".join(chunks)
    assert "Artificial intelligence" in combined
    assert "computationally possible" in combined