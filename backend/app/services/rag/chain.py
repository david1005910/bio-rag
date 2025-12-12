"""RAG Chain - Main orchestrator for RAG pipeline"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, AsyncIterator

from app.services.rag.llm import LLMService, llm_service
from app.services.rag.prompts import build_prompt, format_context
from app.services.rag.retriever import RAGRetriever, RetrievedDocument, rag_retriever
from app.services.rag.validator import ResponseValidator, ValidationResult, response_validator

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Complete RAG response with metadata"""

    answer: str
    sources: list[RetrievedDocument]
    validation: ValidationResult
    query: str
    language: str
    metadata: dict[str, Any]


@dataclass
class RAGConfig:
    """Configuration for RAG chain"""

    top_k: int = 10
    min_score: float = 0.5
    validate_response: bool = True
    clean_invalid_citations: bool = True
    language: str = "en"


class RAGChain:
    """Main RAG chain orchestrator"""

    def __init__(
        self,
        retriever: RAGRetriever | None = None,
        llm: LLMService | None = None,
        validator: ResponseValidator | None = None,
        config: RAGConfig | None = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._validator = validator
        self.config = config or RAGConfig()

    @property
    def retriever(self) -> RAGRetriever:
        """Get retriever instance"""
        if self._retriever is None:
            self._retriever = rag_retriever
        return self._retriever

    @property
    def llm(self) -> LLMService:
        """Get LLM instance"""
        if self._llm is None:
            self._llm = llm_service
        return self._llm

    @property
    def validator(self) -> ResponseValidator:
        """Get validator instance"""
        if self._validator is None:
            self._validator = response_validator
        return self._validator

    def set_embedding_func(self, func: Callable) -> None:
        """Set embedding function for retriever and validator"""
        self.retriever.set_embedding_func(func)
        self.validator.set_embedding_func(func)

    async def invoke(
        self,
        query: str,
        filter_metadata: dict[str, Any] | None = None,
        language: str | None = None,
    ) -> RAGResponse:
        """
        Execute RAG pipeline

        Args:
            query: User question
            filter_metadata: Optional metadata filter for retrieval
            language: Response language (en/ko)

        Returns:
            RAGResponse with answer and metadata
        """
        lang = language or self.config.language
        logger.info(f"RAG invoke: query='{query[:50]}...', language={lang}")

        # Step 1: Retrieve relevant documents
        documents = self.retriever.retrieve(
            query=query,
            top_k=self.config.top_k,
            filter_metadata=filter_metadata,
        )
        logger.info(f"Retrieved {len(documents)} documents")

        if not documents:
            return self._create_no_context_response(query, lang)

        # Step 2: Format context and build prompts
        context_chunks = self._documents_to_chunks(documents)
        context = format_context(context_chunks)
        system_prompt, user_prompt = build_prompt(query, context, lang)

        # Step 3: Generate response
        answer = await self.llm.generate(system_prompt, user_prompt)
        logger.info(f"Generated response: {len(answer)} chars")

        # Step 4: Validate response
        validation = self.validator.validate(answer, documents)

        # Step 5: Clean response if needed
        if self.config.clean_invalid_citations and validation.invalid_citations:
            answer = self.validator.clean_response(answer, validation)

        # Step 6: Check for hallucinations
        has_hallucination, suspicious = self.validator.check_hallucination(
            answer, documents
        )
        if has_hallucination:
            logger.warning(f"Potential hallucination detected: {len(suspicious)} claims")

        return RAGResponse(
            answer=answer,
            sources=documents,
            validation=validation,
            query=query,
            language=lang,
            metadata={
                "num_documents": len(documents),
                "unique_papers": self.retriever.get_unique_papers(documents),
                "has_hallucination": has_hallucination,
                "suspicious_claims": suspicious if has_hallucination else [],
                "token_estimate": self.llm.count_tokens(answer),
            },
        )

    async def stream(
        self,
        query: str,
        filter_metadata: dict[str, Any] | None = None,
        language: str | None = None,
    ) -> AsyncIterator[str]:
        """
        Execute RAG pipeline with streaming response

        Args:
            query: User question
            filter_metadata: Optional metadata filter
            language: Response language

        Yields:
            Response chunks
        """
        lang = language or self.config.language
        logger.info(f"RAG stream: query='{query[:50]}...', language={lang}")

        # Step 1: Retrieve documents
        documents = self.retriever.retrieve(
            query=query,
            top_k=self.config.top_k,
            filter_metadata=filter_metadata,
        )

        if not documents:
            yield "Based on the available papers, I cannot find relevant information to answer this question."
            return

        # Step 2: Format context
        context_chunks = self._documents_to_chunks(documents)
        context = format_context(context_chunks)
        system_prompt, user_prompt = build_prompt(query, context, lang)

        # Step 3: Stream response
        async for chunk in self.llm.generate_stream(system_prompt, user_prompt):
            yield chunk

    async def retrieve_only(
        self,
        query: str,
        top_k: int | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[RetrievedDocument]:
        """
        Retrieve documents without generation

        Args:
            query: Search query
            top_k: Number of results
            filter_metadata: Metadata filter

        Returns:
            List of retrieved documents
        """
        return self.retriever.retrieve(
            query=query,
            top_k=top_k or self.config.top_k,
            filter_metadata=filter_metadata,
        )

    def _documents_to_chunks(
        self,
        documents: list[RetrievedDocument],
    ) -> list[dict]:
        """Convert RetrievedDocument to chunk dict for prompt formatting"""
        return [
            {
                "pmid": doc.pmid,
                "title": doc.title,
                "section": doc.section or "content",
                "content": doc.content,
            }
            for doc in documents
        ]

    def _create_no_context_response(
        self,
        query: str,
        language: str,
    ) -> RAGResponse:
        """Create response when no context is available"""
        if language == "ko":
            answer = "제공된 논문에서 관련 정보를 찾을 수 없습니다. 다른 검색어로 시도해 주세요."
        else:
            answer = "I could not find relevant information in the available papers. Please try a different search query."

        return RAGResponse(
            answer=answer,
            sources=[],
            validation=ValidationResult(
                is_valid=False,
                confidence_score=0.0,
                cited_pmids=[],
                valid_citations=[],
                invalid_citations=[],
                warnings=["No context documents available"],
            ),
            query=query,
            language=language,
            metadata={
                "num_documents": 0,
                "unique_papers": [],
                "has_hallucination": False,
                "suspicious_claims": [],
                "token_estimate": 0,
            },
        )


# Singleton instance
rag_chain = RAGChain()
