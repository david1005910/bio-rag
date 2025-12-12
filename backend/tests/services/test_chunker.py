"""Tests for text chunking module"""

import pytest

from ml.chunking.chunker import ChunkingConfig, TextChunker


@pytest.fixture
def chunker():
    """Create chunker instance with default config"""
    config = ChunkingConfig(
        chunk_size=100,
        chunk_overlap=20,
        min_chunk_size=30,
        max_chunks_per_doc=10,
    )
    return TextChunker(config)


@pytest.fixture
def sample_paper():
    """Sample paper data"""
    return {
        "pmid": "12345678",
        "title": "Test Paper Title",
        "abstract": "This is the abstract of the test paper. " * 10,
        "content": "This is the main content section. " * 50,
    }


class TestTextChunker:
    """Tests for TextChunker"""

    def test_chunk_short_text(self, chunker):
        """Test chunking short text"""
        text = "This is a short text."
        chunks = chunker.chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0].content == text

    def test_chunk_long_text(self, chunker):
        """Test chunking long text"""
        text = "This is a sentence. " * 50  # Long text
        chunks = chunker.chunk_text(text)
        assert len(chunks) > 1

    def test_chunk_overlap(self, chunker):
        """Test that chunks have overlap"""
        text = "Word " * 100  # Enough words to create multiple chunks
        chunks = chunker.chunk_text(text)

        if len(chunks) >= 2:
            # Check that there's some overlap in content
            first_end = chunks[0].content[-20:]
            second_start = chunks[1].content[:40]
            # Some overlap should exist
            assert any(word in second_start for word in first_end.split())

    def test_chunk_metadata(self, chunker):
        """Test that chunks have correct metadata"""
        text = "This is test content. " * 20
        chunks = chunker.chunk_text(
            text, metadata={"pmid": "12345678", "section": "abstract"}
        )

        for chunk in chunks:
            assert chunk.metadata.get("pmid") == "12345678"
            assert chunk.metadata.get("section") == "abstract"

    def test_chunk_indices(self, chunker):
        """Test that chunk indices are sequential"""
        text = "This is test content. " * 20
        chunks = chunker.chunk_text(text)

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_max_chunks_limit(self, chunker):
        """Test that max chunks limit is respected"""
        text = "Word " * 1000  # Very long text
        chunks = chunker.chunk_text(text)
        assert len(chunks) <= chunker.config.max_chunks_per_doc

    def test_min_chunk_size(self, chunker):
        """Test that minimum chunk size is respected"""
        text = "This is test content. " * 20
        chunks = chunker.chunk_text(text)

        for chunk in chunks:
            # Last chunk might be smaller
            if chunk.chunk_index < len(chunks) - 1:
                assert len(chunk.content) >= chunker.config.min_chunk_size


class TestSectionDetection:
    """Tests for section detection"""

    def test_detect_abstract_section(self, chunker):
        """Test detecting abstract section"""
        text = """
        Abstract:
        This is the abstract content.

        Introduction:
        This is the introduction.
        """
        chunks = chunker.chunk_text(text, detect_sections=True)
        # At least one chunk should have abstract section
        sections = [c.metadata.get("detected_section") for c in chunks]
        assert any("abstract" in str(s).lower() for s in sections if s)

    def test_detect_methods_section(self, chunker):
        """Test detecting methods section"""
        text = """
        Methods:
        We performed the following experiments.

        Results:
        The results showed significant improvement.
        """
        chunks = chunker.chunk_text(text, detect_sections=True)
        sections = [c.metadata.get("detected_section") for c in chunks]
        assert any("methods" in str(s).lower() for s in sections if s)


class TestChunkingConfig:
    """Tests for ChunkingConfig"""

    def test_default_config(self):
        """Test default configuration values"""
        config = ChunkingConfig()
        assert config.chunk_size == 512
        assert config.chunk_overlap == 50
        assert config.min_chunk_size == 100
        assert config.max_chunks_per_doc == 20

    def test_custom_config(self):
        """Test custom configuration"""
        config = ChunkingConfig(
            chunk_size=256,
            chunk_overlap=25,
            min_chunk_size=50,
            max_chunks_per_doc=10,
        )
        assert config.chunk_size == 256
        assert config.chunk_overlap == 25
