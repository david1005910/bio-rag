import sys
import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================================
# Define test-compatible versions of the classes
# ============================================================================

@dataclass
class MockSearchResult:
    """Mock of SearchResult from vector_store.py"""
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]


@dataclass
class RAGResponse:
    """Test-compatible copy of RAGResponse from rag_service.py"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    chunks_used: List[MockSearchResult]
    reasoning_steps: List[Dict[str, Any]] = None


# ============================================================================
# RAGService test implementation with real logic but mocked dependencies
# ============================================================================

import re
import json


class TestableRAGService:
    """
    A testable version of RAGService that contains the actual logic
    but doesn't require database/external dependencies.
    """

    def __init__(self, embedding_service, vector_store, llm_client):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.llm_client = llm_client
        self._translator = None

    def _get_translator(self):
        return self._translator

    def _translate_if_korean(self, text: str) -> str:
        translator = self._get_translator()
        if translator and translator.is_korean(text):
            return translator.translate_to_english(text)
        return text

    def index_paper(
        self,
        pmid: str,
        title: str,
        abstract: str,
        authors: List[str] = None,
        journal: str = None,
        publication_date: str = None,
        full_text: str = None
    ) -> List[str]:
        chunks = self._create_chunks(pmid, title, abstract, full_text)

        texts = [c["text"] for c in chunks]
        embeddings = self.embedding_service.batch_encode(texts)

        metadatas = []
        for chunk in chunks:
            metadatas.append({
                "pmid": pmid,
                "title": title,
                "section": chunk["section"],
                "authors": ", ".join(authors) if authors else "",
                "journal": journal or "",
                "publication_date": publication_date or ""
            })

        ids = self.vector_store.add_documents(texts, embeddings, metadatas)
        return ids

    def _create_chunks(
        self,
        pmid: str,
        title: str,
        abstract: str,
        full_text: str = None,
        max_chunk_size: int = 500
    ) -> List[Dict[str, str]]:
        chunks = []

        if title:
            chunks.append({
                "text": f"Title: {title}",
                "section": "title"
            })

        if abstract:
            if len(abstract) > max_chunk_size * 2:
                sentences = re.split(r'(?<=[.!?])\s+', abstract)
                current_chunk = ""
                chunk_idx = 0

                for sentence in sentences:
                    if len(current_chunk) + len(sentence) < max_chunk_size:
                        current_chunk += " " + sentence
                    else:
                        if current_chunk.strip():
                            chunks.append({
                                "text": current_chunk.strip(),
                                "section": f"abstract_{chunk_idx}"
                            })
                            chunk_idx += 1
                        current_chunk = sentence

                if current_chunk.strip():
                    chunks.append({
                        "text": current_chunk.strip(),
                        "section": f"abstract_{chunk_idx}"
                    })
            else:
                chunks.append({
                    "text": abstract,
                    "section": "abstract"
                })

        if full_text:
            sections = self._extract_sections(full_text)
            for section_name, section_text in sections.items():
                if len(section_text) > max_chunk_size:
                    sub_chunks = self._split_text(section_text, max_chunk_size)
                    for i, sub_chunk in enumerate(sub_chunks):
                        chunks.append({
                            "text": sub_chunk,
                            "section": f"{section_name}_{i}"
                        })
                else:
                    chunks.append({
                        "text": section_text,
                        "section": section_name
                    })

        return chunks

    def _extract_sections(self, text: str) -> Dict[str, str]:
        return {"full_text": text}

    def _split_text(self, text: str, max_size: int) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_size:
                current_chunk += " " + sentence
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    async def query(
        self,
        question: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> RAGResponse:
        search_query = self._translate_if_korean(question)
        question_embedding = self.embedding_service.encode(search_query)

        search_results = self.vector_store.search(
            query_embedding=question_embedding,
            top_k=top_k,
            filter_dict=filter_dict
        )

        if not search_results:
            return RAGResponse(
                answer="I couldn't find any relevant papers in the database for your question. Please try a different query or add more papers to the database.",
                sources=[],
                confidence=0.0,
                chunks_used=[]
            )

        context = self._build_context(search_results)
        answer = await self._generate_answer(question, context)

        sources = self._extract_sources(search_results)
        confidence = self._calculate_confidence(search_results, answer)

        return RAGResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            chunks_used=search_results
        )

    def _build_context(self, results: List[MockSearchResult]) -> str:
        context_parts = []

        for i, result in enumerate(results, 1):
            pmid = result.metadata.get("pmid", "Unknown")
            title = result.metadata.get("title", "Unknown")
            section = result.metadata.get("section", "Unknown")

            context_parts.append(
                f"[Paper {i}] PMID: {pmid}\n"
                f"Title: {title}\n"
                f"Section: {section}\n"
                f"Content: {result.text}\n"
            )

        return "\n\n".join(context_parts)

    async def _generate_answer(self, question: str, context: str) -> str:
        user_prompt = f"""Based on the following research paper excerpts, please answer the question.

Context from research papers:
{context}

Question: {question}

Please provide a detailed, accurate answer with citations to the relevant papers using [PMID: xxxxx] format:"""

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert biomedical researcher."},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2000,
                temperature=0.1
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"Error generating response: {str(e)}"

    def _extract_sources(self, results: List[MockSearchResult]) -> List[Dict[str, Any]]:
        sources = []
        seen_pmids = set()

        for result in results:
            pmid = result.metadata.get("pmid")
            if pmid and pmid not in seen_pmids:
                seen_pmids.add(pmid)
                sources.append({
                    "pmid": pmid,
                    "title": result.metadata.get("title", ""),
                    "journal": result.metadata.get("journal", ""),
                    "relevance": result.score,
                    "excerpt": result.text[:200] + "..." if len(result.text) > 200 else result.text
                })

        return sources

    def _calculate_confidence(self, results: List[MockSearchResult], answer: str) -> float:
        if not results:
            return 0.0

        avg_score = sum(r.score for r in results) / len(results)

        cited_pmids = re.findall(r'PMID:\s*(\d+)', answer)
        source_pmids = [r.metadata.get("pmid") for r in results]

        if cited_pmids:
            valid_citations = sum(1 for pmid in cited_pmids if pmid in source_pmids)
            citation_score = valid_citations / len(cited_pmids)
        else:
            citation_score = 0.5

        confidence = (avg_score * 0.6) + (citation_score * 0.4)
        return min(confidence, 1.0)

    def semantic_search(
        self,
        query: str,
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        search_query = self._translate_if_korean(query)
        query_embedding = self.embedding_service.encode(search_query)

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k * 2,
            filter_dict=filter_dict
        )

        papers = {}
        for result in results:
            pmid = result.metadata.get("pmid")
            if pmid not in papers:
                papers[pmid] = {
                    "pmid": pmid,
                    "title": result.metadata.get("title", ""),
                    "journal": result.metadata.get("journal", ""),
                    "publication_date": result.metadata.get("publication_date", ""),
                    "relevance": result.score,
                    "excerpt": result.text[:300] + "..." if len(result.text) > 300 else result.text
                }
            else:
                papers[pmid]["relevance"] = max(papers[pmid]["relevance"], result.score)

        return sorted(papers.values(), key=lambda x: x["relevance"], reverse=True)[:top_k]

    async def reasoning_query(
        self,
        question: str,
        top_k: int = 5,
        max_iterations: int = 3,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> RAGResponse:
        reasoning_steps = []
        all_search_results = []
        accumulated_context = []

        step1 = await self._decompose_question(question)
        reasoning_steps.append({
            "step": 1,
            "type": "decomposition",
            "description": "Question decomposition",
            "content": step1
        })

        sub_questions = step1.get("sub_questions", [question])

        for i, sub_q in enumerate(sub_questions[:max_iterations]):
            search_sub_q = self._translate_if_korean(sub_q)
            query_embedding = self.embedding_service.encode(search_sub_q)
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filter_dict=filter_dict
            )

            for r in results:
                if r not in all_search_results:
                    all_search_results.append(r)

            if results:
                context = self._build_context(results)
                accumulated_context.append(f"[Sub-question {i+1}]: {sub_q}\n\n{context}")

                sub_answer = await self._generate_sub_answer(sub_q, context)
                reasoning_steps.append({
                    "step": i + 2,
                    "type": "sub_answer",
                    "description": f"Sub-question {i+1} analysis",
                    "sub_question": sub_q,
                    "content": sub_answer,
                    "sources_found": len(results)
                })

        if not all_search_results:
            return RAGResponse(
                answer="No relevant information found in indexed papers.",
                sources=[],
                confidence=0.0,
                chunks_used=[],
                reasoning_steps=reasoning_steps
            )

        full_context = "\n\n---\n\n".join(accumulated_context)
        final_answer = await self._synthesize_reasoning_answer(question, full_context, reasoning_steps)

        reasoning_steps.append({
            "step": len(reasoning_steps) + 1,
            "type": "synthesis",
            "description": "Final answer synthesis",
            "content": "Synthesized all sub-analysis results"
        })

        sources = self._extract_sources(all_search_results)
        confidence = self._calculate_confidence(all_search_results, final_answer)
        confidence = min(confidence * 1.1, 1.0)

        return RAGResponse(
            answer=final_answer,
            sources=sources,
            confidence=confidence,
            chunks_used=all_search_results,
            reasoning_steps=reasoning_steps
        )

    async def _decompose_question(self, question: str) -> Dict[str, Any]:
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": f"Decompose: {question}"}],
                max_tokens=1000,
                temperature=0.2
            )

            result_text = response.choices[0].message.content.strip()
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]

            return json.loads(result_text)
        except Exception:
            return {
                "complexity": "simple",
                "main_concepts": [],
                "sub_questions": [question],
                "reasoning_approach": "Direct search and answer"
            }

    async def _generate_sub_answer(self, question: str, context: str) -> str:
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a biomedical research assistant."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
                ],
                max_tokens=500,
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

    async def _synthesize_reasoning_answer(
        self,
        original_question: str,
        accumulated_context: str,
        reasoning_steps: List[Dict[str, Any]]
    ) -> str:
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert biomedical researcher."},
                    {"role": "user", "content": f"Question: {original_question}\n\nContext:\n{accumulated_context}"}
                ],
                max_tokens=2500,
                temperature=0.15
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error synthesizing answer: {str(e)}"


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_embedding_service():
    """Mock embedding service that returns deterministic embeddings."""
    mock = MagicMock()
    mock.encode.return_value = np.random.rand(768).astype(np.float32)
    mock.batch_encode.return_value = np.random.rand(3, 768).astype(np.float32)
    mock.dimension = 768
    return mock


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing without Qdrant."""
    mock = MagicMock()
    mock.add_documents.return_value = ["id1", "id2", "id3"]
    mock.search.return_value = []
    mock.get_collection_info.return_value = {"name": "test", "points_count": 0}
    return mock


@pytest.fixture
def mock_llm_client():
    """Mock OpenAI client for testing LLM calls."""
    mock = MagicMock()

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "This is a test answer based on the research [PMID: 12345]."

    mock.chat.completions.create.return_value = mock_response
    return mock


@pytest.fixture
def sample_search_results():
    """Sample search results for testing."""
    return [
        MockSearchResult(
            id="chunk1",
            text="This study examines the effects of drug X on cancer cells.",
            score=0.95,
            metadata={
                "pmid": "12345",
                "title": "Effects of Drug X on Cancer",
                "section": "abstract",
                "journal": "Nature Medicine",
                "authors": "Smith J, Doe A"
            }
        ),
        MockSearchResult(
            id="chunk2",
            text="The results show significant tumor reduction in treated mice.",
            score=0.87,
            metadata={
                "pmid": "12345",
                "title": "Effects of Drug X on Cancer",
                "section": "results_0",
                "journal": "Nature Medicine",
                "authors": "Smith J, Doe A"
            }
        ),
        MockSearchResult(
            id="chunk3",
            text="Gene therapy approaches have shown promise in treating genetic disorders.",
            score=0.75,
            metadata={
                "pmid": "67890",
                "title": "Gene Therapy Advances",
                "section": "abstract",
                "journal": "Cell",
                "authors": "Johnson B"
            }
        ),
    ]


@pytest.fixture
def sample_paper_data():
    """Sample paper data for indexing tests."""
    return {
        "pmid": "99999",
        "title": "Novel Treatment for Alzheimer's Disease",
        "abstract": "This study presents a novel treatment approach. We tested the drug in animal models. Results showed significant improvement. The treatment was well tolerated. Further clinical trials are needed.",
        "authors": ["Dr. Smith", "Dr. Johnson"],
        "journal": "Lancet Neurology",
        "publication_date": "2024-01-15",
        "full_text": None
    }


@pytest.fixture
def long_abstract():
    """Long abstract for testing chunking."""
    sentences = [
        "This is the first sentence of a very long abstract.",
        "It contains multiple paragraphs of scientific content.",
        "The study methodology was carefully designed.",
        "Participants were recruited from multiple centers.",
        "Statistical analysis was performed using standard methods.",
        "Results showed significant differences between groups.",
        "The treatment group had better outcomes.",
        "Side effects were minimal and manageable.",
        "Discussion of the findings follows.",
        "These results are consistent with prior research.",
        "Limitations of the study include sample size.",
        "Future research should address these gaps.",
        "In conclusion, the treatment shows promise.",
        "Clinical implications are discussed.",
        "The authors declare no conflicts of interest.",
    ]
    return " ".join(sentences * 10)


@pytest.fixture
def mock_translation_service():
    """Mock translation service."""
    mock = MagicMock()
    mock.is_korean.return_value = False
    mock.translate_to_english.return_value = "translated text"
    return mock


@pytest.fixture
def rag_service_with_mocks(mock_embedding_service, mock_vector_store, mock_llm_client):
    """Create a testable RAGService instance with all dependencies mocked."""
    return TestableRAGService(
        embedding_service=mock_embedding_service,
        vector_store=mock_vector_store,
        llm_client=mock_llm_client
    )


@pytest.fixture
def rag_response_class():
    """Return the RAGResponse class for direct instantiation tests."""
    return RAGResponse
