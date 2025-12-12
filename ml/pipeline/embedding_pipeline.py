import logging
from dataclasses import dataclass
from typing import Any

from ml.chunking.chunker import Chunk, TextChunker, text_chunker
from ml.embeddings.service import EmbeddingService, embedding_service

logger = logging.getLogger(__name__)


@dataclass
class ProcessedChunk:
    """Chunk with embedding"""

    chunk_id: str
    content: str
    embedding: list[float]
    metadata: dict[str, Any]


@dataclass
class PipelineResult:
    """Result of pipeline processing"""

    doc_id: str
    chunks_processed: int
    success: bool
    error: str | None = None


class EmbeddingPipeline:
    """
    Pipeline for processing documents into embeddings

    Flow: Document → Chunking → Embedding → Vector Store
    """

    def __init__(
        self,
        chunker: TextChunker | None = None,
        embedding_svc: EmbeddingService | None = None,
    ) -> None:
        self.chunker = chunker or text_chunker
        self.embedding_svc = embedding_svc or embedding_service

    def process_paper(
        self,
        pmid: str,
        title: str,
        abstract: str | None,
        additional_metadata: dict[str, Any] | None = None,
    ) -> list[ProcessedChunk]:
        """
        Process a single paper through the pipeline

        Args:
            pmid: PubMed ID
            title: Paper title
            abstract: Paper abstract
            additional_metadata: Additional metadata to store

        Returns:
            List of ProcessedChunk objects ready for vector store
        """
        metadata = additional_metadata or {}

        # Step 1: Chunk the paper
        chunks = self.chunker.chunk_paper(
            title=title,
            abstract=abstract,
            pmid=pmid,
            additional_metadata=metadata,
        )

        if not chunks:
            logger.warning(f"No chunks generated for paper {pmid}")
            return []

        # Step 2: Generate embeddings
        processed_chunks = self._embed_chunks(chunks)

        logger.info(f"Processed paper {pmid}: {len(processed_chunks)} chunks")
        return processed_chunks

    def process_papers_batch(
        self,
        papers: list[dict[str, Any]],
        batch_size: int = 32,
    ) -> list[PipelineResult]:
        """
        Process multiple papers in batch

        Args:
            papers: List of paper dicts with 'pmid', 'title', 'abstract'
            batch_size: Batch size for embedding generation

        Returns:
            List of PipelineResult objects
        """
        results: list[PipelineResult] = []
        all_chunks: list[Chunk] = []
        chunk_to_paper: dict[str, str] = {}  # chunk_id -> pmid

        # Step 1: Chunk all papers
        for paper in papers:
            pmid = paper["pmid"]
            try:
                chunks = self.chunker.chunk_paper(
                    title=paper["title"],
                    abstract=paper.get("abstract"),
                    pmid=pmid,
                    additional_metadata=paper.get("metadata", {}),
                )
                all_chunks.extend(chunks)
                for chunk in chunks:
                    chunk_to_paper[chunk.chunk_id] = pmid
            except Exception as e:
                logger.error(f"Error chunking paper {pmid}: {e}")
                results.append(PipelineResult(
                    doc_id=pmid,
                    chunks_processed=0,
                    success=False,
                    error=str(e),
                ))

        if not all_chunks:
            return results

        # Step 2: Generate embeddings in batches
        processed_chunks = self._embed_chunks_batch(all_chunks, batch_size)

        # Count chunks per paper
        paper_chunk_counts: dict[str, int] = {}
        for chunk in processed_chunks:
            pmid = chunk_to_paper.get(chunk.chunk_id, "unknown")
            paper_chunk_counts[pmid] = paper_chunk_counts.get(pmid, 0) + 1

        # Generate results
        for paper in papers:
            pmid = paper["pmid"]
            if pmid not in paper_chunk_counts:
                continue

            results.append(PipelineResult(
                doc_id=pmid,
                chunks_processed=paper_chunk_counts[pmid],
                success=True,
            ))

        return results

    def _embed_chunks(self, chunks: list[Chunk]) -> list[ProcessedChunk]:
        """Generate embeddings for chunks"""
        if not chunks:
            return []

        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_svc.embed_batch(texts)

        processed: list[ProcessedChunk] = []
        for chunk, embedding in zip(chunks, embeddings):
            processed.append(ProcessedChunk(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                embedding=embedding,
                metadata={
                    "section": chunk.section,
                    "token_count": chunk.token_count,
                    "chunk_index": chunk.chunk_index,
                    **chunk.metadata,
                },
            ))

        return processed

    def _embed_chunks_batch(
        self,
        chunks: list[Chunk],
        batch_size: int,
    ) -> list[ProcessedChunk]:
        """Generate embeddings in batches"""
        processed: list[ProcessedChunk] = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_processed = self._embed_chunks(batch)
            processed.extend(batch_processed)

            if i + batch_size < len(chunks):
                logger.info(f"Processed {i + len(batch)}/{len(chunks)} chunks")

        return processed


# Singleton instance
embedding_pipeline = EmbeddingPipeline()
