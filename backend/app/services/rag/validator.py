"""Response validation for hallucination detection and citation verification"""

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from app.services.rag.retriever import RetrievedDocument

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of response validation"""

    is_valid: bool
    confidence_score: float
    cited_pmids: list[str]
    valid_citations: list[str]
    invalid_citations: list[str]
    warnings: list[str]


class ResponseValidator:
    """Validator for RAG responses"""

    # Regex pattern for PMID citations
    PMID_PATTERN = re.compile(r'\[?PMID[:\s]*(\d+)\]?', re.IGNORECASE)

    def __init__(
        self,
        similarity_threshold: float = 0.5,
        min_citations: int = 1,
        embedding_func: Callable | None = None,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.min_citations = min_citations
        self._embedding_func = embedding_func

    def set_embedding_func(self, func: Callable) -> None:
        """Set embedding function for similarity checks"""
        self._embedding_func = func

    def validate(
        self,
        response: str,
        context_documents: list[RetrievedDocument],
    ) -> ValidationResult:
        """
        Validate LLM response

        Args:
            response: Generated response text
            context_documents: Documents used as context

        Returns:
            ValidationResult with validation details
        """
        warnings: list[str] = []

        # Extract cited PMIDs from response
        cited_pmids = self._extract_pmids(response)

        # Get valid PMIDs from context
        valid_pmids = {doc.pmid for doc in context_documents}

        # Check citation validity
        valid_citations = [pmid for pmid in cited_pmids if pmid in valid_pmids]
        invalid_citations = [pmid for pmid in cited_pmids if pmid not in valid_pmids]

        # Add warnings for invalid citations
        if invalid_citations:
            warnings.append(
                f"Invalid citations found: {', '.join(invalid_citations)}"
            )

        # Check minimum citations
        if len(valid_citations) < self.min_citations:
            warnings.append(
                f"Response has fewer than {self.min_citations} valid citations"
            )

        # Calculate confidence score
        confidence_score = self._calculate_confidence(
            response=response,
            context_documents=context_documents,
            valid_citations=valid_citations,
        )

        # Determine validity
        is_valid = (
            len(invalid_citations) == 0
            and len(valid_citations) >= self.min_citations
            and confidence_score >= self.similarity_threshold
        )

        return ValidationResult(
            is_valid=is_valid,
            confidence_score=confidence_score,
            cited_pmids=cited_pmids,
            valid_citations=valid_citations,
            invalid_citations=invalid_citations,
            warnings=warnings,
        )

    def _extract_pmids(self, text: str) -> list[str]:
        """Extract PMID citations from text"""
        matches = self.PMID_PATTERN.findall(text)
        return list(dict.fromkeys(matches))  # Preserve order, remove duplicates

    def _calculate_confidence(
        self,
        response: str,
        context_documents: list[RetrievedDocument],
        valid_citations: list[str],
    ) -> float:
        """
        Calculate confidence score based on response-context similarity

        Args:
            response: Generated response
            context_documents: Context documents
            valid_citations: Valid cited PMIDs

        Returns:
            Confidence score between 0 and 1
        """
        if not context_documents:
            return 0.0

        # Base score from citations
        citation_score = min(len(valid_citations) / max(self.min_citations, 1), 1.0)

        # If embedding function available, compute semantic similarity
        if self._embedding_func is not None:
            try:
                similarity_score = self._compute_similarity(response, context_documents)
                # Weighted average
                return 0.4 * citation_score + 0.6 * similarity_score
            except Exception as e:
                logger.warning(f"Error computing similarity: {e}")

        return citation_score

    def _compute_similarity(
        self,
        response: str,
        context_documents: list[RetrievedDocument],
    ) -> float:
        """Compute semantic similarity between response and context"""
        if self._embedding_func is None:
            return 0.5

        # Embed response
        response_embedding = np.array(self._embedding_func(response))

        # Embed context (use average of document embeddings)
        context_text = " ".join([doc.content for doc in context_documents])
        context_embedding = np.array(self._embedding_func(context_text))

        # Cosine similarity
        similarity = np.dot(response_embedding, context_embedding) / (
            np.linalg.norm(response_embedding) * np.linalg.norm(context_embedding)
        )

        return float(max(0, similarity))

    def check_hallucination(
        self,
        response: str,
        context_documents: list[RetrievedDocument],
    ) -> tuple[bool, list[str]]:
        """
        Check for potential hallucinations

        Args:
            response: Generated response
            context_documents: Context documents

        Returns:
            Tuple of (has_hallucination, suspicious_sentences)
        """
        suspicious: list[str] = []

        # Split response into sentences
        sentences = re.split(r'[.!?]\s+', response)

        # Check for factual claims without citations
        for sentence in sentences:
            # Skip short sentences
            if len(sentence) < 20:
                continue

            # Check if sentence makes a claim but has no citation
            claim_indicators = [
                "study shows", "research indicates", "evidence suggests",
                "has been shown", "demonstrated that", "found that",
                "according to", "reported that", "discovered",
            ]

            has_claim = any(ind in sentence.lower() for ind in claim_indicators)
            has_citation = bool(self.PMID_PATTERN.search(sentence))

            if has_claim and not has_citation:
                suspicious.append(sentence)

        has_hallucination = len(suspicious) > 0
        return has_hallucination, suspicious

    def clean_response(
        self,
        response: str,
        validation_result: ValidationResult,
    ) -> str:
        """
        Clean response by removing or flagging invalid citations

        Args:
            response: Original response
            validation_result: Validation result

        Returns:
            Cleaned response
        """
        cleaned = response

        # Remove invalid citations
        for pmid in validation_result.invalid_citations:
            patterns = [
                rf'\[PMID[:\s]*{pmid}\]',
                rf'PMID[:\s]*{pmid}',
            ]
            for pattern in patterns:
                cleaned = re.sub(pattern, '[citation needed]', cleaned, flags=re.IGNORECASE)

        # Add disclaimer if warnings exist
        if validation_result.warnings:
            cleaned += "\n\n⚠️ Note: Some information could not be fully verified against the provided sources."

        return cleaned


# Singleton instance
response_validator = ResponseValidator()
