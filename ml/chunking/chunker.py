import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """Text chunk with metadata"""

    content: str
    chunk_id: str
    chunk_index: int
    section: str | None = None
    token_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkingConfig:
    """Chunking configuration"""

    chunk_size: int = 512  # Target tokens per chunk
    chunk_overlap: int = 50  # Overlap tokens between chunks
    min_chunk_size: int = 100  # Minimum chunk size
    max_chunks_per_doc: int = 20  # Maximum chunks per document

    # Section weights for relevance scoring
    section_weights: dict[str, float] = field(default_factory=lambda: {
        "title": 2.0,
        "abstract": 1.5,
        "introduction": 1.0,
        "methods": 0.8,
        "results": 1.2,
        "discussion": 1.0,
        "conclusion": 1.3,
    })


class TextChunker:
    """Text chunking for RAG pipeline"""

    # Section detection patterns
    SECTION_PATTERNS = {
        "abstract": r"(?:ABSTRACT|Abstract)[:\s]*(.+?)(?=(?:INTRODUCTION|Introduction|BACKGROUND|1\.|$))",
        "introduction": r"(?:INTRODUCTION|Introduction|BACKGROUND|1\.)[:\s]*(.+?)(?=(?:METHODS|Methods|MATERIALS|2\.|$))",
        "methods": r"(?:METHODS|Methods|MATERIALS AND METHODS|METHODOLOGY)[:\s]*(.+?)(?=(?:RESULTS|Results|3\.|$))",
        "results": r"(?:RESULTS|Results|FINDINGS|3\.)[:\s]*(.+?)(?=(?:DISCUSSION|Discussion|4\.|$))",
        "discussion": r"(?:DISCUSSION|Discussion|4\.)[:\s]*(.+?)(?=(?:CONCLUSION|Conclusion|REFERENCES|5\.|$))",
        "conclusion": r"(?:CONCLUSION|Conclusion|CONCLUSIONS|5\.)[:\s]*(.+?)(?=(?:REFERENCES|ACKNOWLEDGMENT|$))",
    }

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    def chunk_document(
        self,
        text: str,
        doc_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """
        Chunk a document into smaller pieces

        Args:
            text: Document text
            doc_id: Document identifier (e.g., PMID)
            metadata: Additional metadata to attach to chunks

        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            return []

        metadata = metadata or {}

        # Try section-based chunking first
        sections = self._detect_sections(text)

        if sections:
            chunks = self._chunk_by_sections(sections, doc_id, metadata)
        else:
            # Fall back to size-based chunking
            chunks = self._chunk_by_size(text, doc_id, None, metadata)

        # Limit number of chunks
        if len(chunks) > self.config.max_chunks_per_doc:
            chunks = chunks[: self.config.max_chunks_per_doc]

        return chunks

    def chunk_paper(
        self,
        title: str,
        abstract: str | None,
        pmid: str,
        additional_metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """
        Chunk a paper (title + abstract)

        Args:
            title: Paper title
            abstract: Paper abstract
            pmid: PubMed ID
            additional_metadata: Additional metadata

        Returns:
            List of Chunk objects
        """
        chunks: list[Chunk] = []
        metadata = additional_metadata or {}
        metadata["pmid"] = pmid
        metadata["title"] = title

        chunk_index = 0

        # Title chunk (always include)
        title_chunk = Chunk(
            content=title,
            chunk_id=f"{pmid}_title",
            chunk_index=chunk_index,
            section="title",
            token_count=self._estimate_tokens(title),
            metadata={**metadata, "section_weight": self.config.section_weights["title"]},
        )
        chunks.append(title_chunk)
        chunk_index += 1

        # Abstract chunks
        if abstract:
            abstract_chunks = self._chunk_by_size(
                abstract,
                pmid,
                "abstract",
                metadata,
                start_index=chunk_index,
            )
            chunks.extend(abstract_chunks)

        return chunks

    def _detect_sections(self, text: str) -> dict[str, str]:
        """Detect sections in text"""
        sections: dict[str, str] = {}

        for section_name, pattern in self.SECTION_PATTERNS.items():
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                if content and len(content) > 50:  # Minimum content length
                    sections[section_name] = content

        return sections

    def _chunk_by_sections(
        self,
        sections: dict[str, str],
        doc_id: str,
        metadata: dict[str, Any],
    ) -> list[Chunk]:
        """Chunk document by sections"""
        chunks: list[Chunk] = []
        chunk_index = 0

        for section_name, section_text in sections.items():
            section_chunks = self._chunk_by_size(
                section_text,
                doc_id,
                section_name,
                metadata,
                start_index=chunk_index,
            )
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)

        return chunks

    def _chunk_by_size(
        self,
        text: str,
        doc_id: str,
        section: str | None,
        metadata: dict[str, Any],
        start_index: int = 0,
    ) -> list[Chunk]:
        """Chunk text by size"""
        sentences = self._split_sentences(text)
        chunks: list[Chunk] = []

        current_chunk: list[str] = []
        current_tokens = 0
        chunk_index = start_index

        for sentence in sentences:
            sentence_tokens = self._estimate_tokens(sentence)

            # Check if adding this sentence exceeds chunk size
            if current_tokens + sentence_tokens > self.config.chunk_size and current_chunk:
                # Create chunk
                chunk_text = " ".join(current_chunk)
                chunk = self._create_chunk(
                    chunk_text,
                    doc_id,
                    chunk_index,
                    section,
                    metadata,
                )
                chunks.append(chunk)
                chunk_index += 1

                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(
                    current_chunk,
                    self.config.chunk_overlap,
                )
                current_chunk = overlap_sentences
                current_tokens = sum(self._estimate_tokens(s) for s in overlap_sentences)

            current_chunk.append(sentence)
            current_tokens += sentence_tokens

        # Handle remaining content
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if self._estimate_tokens(chunk_text) >= self.config.min_chunk_size:
                chunk = self._create_chunk(
                    chunk_text,
                    doc_id,
                    chunk_index,
                    section,
                    metadata,
                )
                chunks.append(chunk)
            elif chunks:
                # Merge with previous chunk if too small
                chunks[-1].content += " " + chunk_text
                chunks[-1].token_count = self._estimate_tokens(chunks[-1].content)

        return chunks

    def _create_chunk(
        self,
        text: str,
        doc_id: str,
        index: int,
        section: str | None,
        metadata: dict[str, Any],
    ) -> Chunk:
        """Create a Chunk object"""
        section_weight = 1.0
        if section and section in self.config.section_weights:
            section_weight = self.config.section_weights[section]

        return Chunk(
            content=text,
            chunk_id=f"{doc_id}_{index}",
            chunk_index=index,
            section=section,
            token_count=self._estimate_tokens(text),
            metadata={
                **metadata,
                "section_weight": section_weight,
            },
        )

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences"""
        # Handle common abbreviations
        text = re.sub(r'(\w\.)\s+(\w)', r'\1<SPACE>\2', text)

        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Restore spaces
        sentences = [s.replace('<SPACE>', ' ') for s in sentences]

        return [s.strip() for s in sentences if s.strip()]

    def _get_overlap_sentences(
        self,
        sentences: list[str],
        target_tokens: int,
    ) -> list[str]:
        """Get sentences for overlap"""
        overlap: list[str] = []
        current_tokens = 0

        for sentence in reversed(sentences):
            sentence_tokens = self._estimate_tokens(sentence)
            if current_tokens + sentence_tokens > target_tokens:
                break
            overlap.insert(0, sentence)
            current_tokens += sentence_tokens

        return overlap

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count"""
        # Rough estimate: ~4 characters per token
        return len(text) // 4


# Default chunker instance
text_chunker = TextChunker()
