import pytest
from unittest.mock import MagicMock

from tests.conftest import MockSearchResult


class TestRAGResponse:
    """Tests for the RAGResponse dataclass."""

    def test_rag_response_creation(self, rag_response_class):
        """Test basic RAGResponse instantiation."""
        RAGResponse = rag_response_class

        response = RAGResponse(
            answer="Test answer",
            sources=[{"pmid": "123", "title": "Test"}],
            confidence=0.85,
            chunks_used=[]
        )

        assert response.answer == "Test answer"
        assert len(response.sources) == 1
        assert response.confidence == 0.85
        assert response.reasoning_steps is None

    def test_rag_response_with_reasoning_steps(self, rag_response_class):
        """Test RAGResponse with reasoning steps."""
        RAGResponse = rag_response_class

        reasoning = [{"step": 1, "type": "decomposition", "content": "Test"}]
        response = RAGResponse(
            answer="Test answer",
            sources=[],
            confidence=0.5,
            chunks_used=[],
            reasoning_steps=reasoning
        )

        assert response.reasoning_steps is not None
        assert len(response.reasoning_steps) == 1


class TestChunking:
    """Tests for text chunking functionality."""

    def test_create_chunks_title_only(self, rag_service_with_mocks):
        """Test chunking with just a title."""
        service = rag_service_with_mocks

        chunks = service._create_chunks(
            pmid="12345",
            title="Test Title",
            abstract=None
        )

        assert len(chunks) == 1
        assert chunks[0]["section"] == "title"
        assert "Test Title" in chunks[0]["text"]

    def test_create_chunks_with_short_abstract(self, rag_service_with_mocks):
        """Test chunking with a short abstract (no splitting needed)."""
        service = rag_service_with_mocks

        chunks = service._create_chunks(
            pmid="12345",
            title="Test Title",
            abstract="This is a short abstract."
        )

        assert len(chunks) == 2
        sections = [c["section"] for c in chunks]
        assert "title" in sections
        assert "abstract" in sections

    def test_create_chunks_with_long_abstract(self, rag_service_with_mocks, long_abstract):
        """Test chunking splits long abstracts."""
        service = rag_service_with_mocks

        chunks = service._create_chunks(
            pmid="12345",
            title="Test Title",
            abstract=long_abstract,
            max_chunk_size=500
        )

        # Should have title + multiple abstract chunks
        assert len(chunks) > 2
        abstract_chunks = [c for c in chunks if "abstract" in c["section"]]
        assert len(abstract_chunks) > 1

    def test_create_chunks_with_full_text(self, rag_service_with_mocks):
        """Test chunking includes full text sections."""
        service = rag_service_with_mocks

        full_text = "This is the full text of the paper. It has multiple sentences. More content here."

        chunks = service._create_chunks(
            pmid="12345",
            title="Test Title",
            abstract="Short abstract.",
            full_text=full_text
        )

        # Should have title, abstract, and full_text sections
        sections = [c["section"] for c in chunks]
        assert any("full_text" in s for s in sections)

    def test_split_text_respects_max_size(self, rag_service_with_mocks):
        """Test _split_text respects max chunk size."""
        service = rag_service_with_mocks

        long_text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five. " * 20

        chunks = service._split_text(long_text, max_size=100)

        for chunk in chunks:
            # Allow some flexibility for sentence boundaries
            assert len(chunk) < 200

    def test_split_text_handles_empty_input(self, rag_service_with_mocks):
        """Test _split_text handles empty strings."""
        service = rag_service_with_mocks

        chunks = service._split_text("", max_size=500)
        assert chunks == []


class TestContextBuilding:
    """Tests for context building from search results."""

    def test_build_context_formats_correctly(self, rag_service_with_mocks, sample_search_results):
        """Test context is formatted with paper info."""
        service = rag_service_with_mocks

        context = service._build_context(sample_search_results)

        assert "PMID: 12345" in context
        assert "PMID: 67890" in context
        assert "Effects of Drug X on Cancer" in context
        assert "[Paper 1]" in context
        assert "[Paper 2]" in context

    def test_build_context_empty_results(self, rag_service_with_mocks):
        """Test context building with no results."""
        service = rag_service_with_mocks

        context = service._build_context([])
        assert context == ""

    def test_build_context_includes_section_info(self, rag_service_with_mocks, sample_search_results):
        """Test context includes section information."""
        service = rag_service_with_mocks

        context = service._build_context(sample_search_results)

        assert "Section: abstract" in context
        assert "Section: results_0" in context


class TestSourceExtraction:
    """Tests for source extraction from search results."""

    def test_extract_sources_deduplicates_pmids(self, rag_service_with_mocks, sample_search_results):
        """Test sources are deduplicated by PMID."""
        service = rag_service_with_mocks

        sources = service._extract_sources(sample_search_results)

        # Should have 2 unique PMIDs (12345 and 67890)
        pmids = [s["pmid"] for s in sources]
        assert len(pmids) == 2
        assert "12345" in pmids
        assert "67890" in pmids

    def test_extract_sources_includes_metadata(self, rag_service_with_mocks, sample_search_results):
        """Test extracted sources include required metadata."""
        service = rag_service_with_mocks

        sources = service._extract_sources(sample_search_results)

        for source in sources:
            assert "pmid" in source
            assert "title" in source
            assert "journal" in source
            assert "relevance" in source
            assert "excerpt" in source

    def test_extract_sources_truncates_long_excerpts(self, rag_service_with_mocks):
        """Test long text excerpts are truncated."""
        long_text = "A" * 500
        results = [
            MockSearchResult(
                id="1",
                text=long_text,
                score=0.9,
                metadata={"pmid": "123", "title": "Test", "journal": "Test"}
            )
        ]

        service = rag_service_with_mocks
        sources = service._extract_sources(results)

        assert len(sources[0]["excerpt"]) <= 203  # 200 chars + "..."

    def test_extract_sources_empty_results(self, rag_service_with_mocks):
        """Test source extraction with no results."""
        service = rag_service_with_mocks

        sources = service._extract_sources([])
        assert sources == []


class TestConfidenceCalculation:
    """Tests for confidence score calculation."""

    def test_confidence_high_scores_with_citations(self, rag_service_with_mocks, sample_search_results):
        """Test confidence is high when scores are high and citations match."""
        service = rag_service_with_mocks

        answer = "The study [PMID: 12345] showed significant results."
        confidence = service._calculate_confidence(sample_search_results, answer)

        assert 0.7 <= confidence <= 1.0

    def test_confidence_empty_results_returns_zero(self, rag_service_with_mocks):
        """Test confidence is 0 with no search results."""
        service = rag_service_with_mocks

        confidence = service._calculate_confidence([], "Some answer")
        assert confidence == 0.0

    def test_confidence_no_citations_gives_lower_score(self, rag_service_with_mocks, sample_search_results):
        """Test confidence is lower when answer has no citations."""
        service = rag_service_with_mocks

        answer_with_citation = "Test [PMID: 12345]."
        answer_without = "Test answer without any citations."

        conf_with = service._calculate_confidence(sample_search_results, answer_with_citation)
        conf_without = service._calculate_confidence(sample_search_results, answer_without)

        # Answer with valid citation should have higher confidence
        assert conf_with >= conf_without

    def test_confidence_invalid_citations(self, rag_service_with_mocks, sample_search_results):
        """Test confidence handles citations to non-existent PMIDs."""
        service = rag_service_with_mocks

        # Reference a PMID not in results
        answer = "According to [PMID: 99999], this is wrong."
        confidence = service._calculate_confidence(sample_search_results, answer)

        # Should still return a valid confidence (just lower)
        assert 0 <= confidence <= 1.0

    def test_confidence_capped_at_one(self, rag_service_with_mocks):
        """Test confidence never exceeds 1.0."""
        # Create results with perfect scores
        perfect_results = [
            MockSearchResult(
                id="1",
                text="Perfect match",
                score=1.0,
                metadata={"pmid": "123"}
            )
        ]

        service = rag_service_with_mocks
        answer = "[PMID: 123] citation"

        confidence = service._calculate_confidence(perfect_results, answer)
        assert confidence <= 1.0


class TestQuery:
    """Tests for the main query method."""

    @pytest.mark.asyncio
    async def test_query_returns_rag_response(self, rag_service_with_mocks, sample_search_results):
        """Test query returns proper RAGResponse."""
        service = rag_service_with_mocks
        service.vector_store.search.return_value = sample_search_results

        response = await service.query("What are the effects of Drug X?")

        assert response.answer is not None
        assert isinstance(response.sources, list)
        assert isinstance(response.confidence, float)

    @pytest.mark.asyncio
    async def test_query_no_results_returns_helpful_message(self, rag_service_with_mocks):
        """Test query with no results returns appropriate message."""
        service = rag_service_with_mocks
        service.vector_store.search.return_value = []

        response = await service.query("Obscure question with no matches")

        assert "couldn't find" in response.answer.lower() or "no relevant" in response.answer.lower()
        assert response.confidence == 0.0
        assert response.sources == []

    @pytest.mark.asyncio
    async def test_query_uses_embedding_service(self, rag_service_with_mocks, sample_search_results):
        """Test query calls embedding service to encode question."""
        service = rag_service_with_mocks
        service.vector_store.search.return_value = sample_search_results

        await service.query("Test question")

        service.embedding_service.encode.assert_called()

    @pytest.mark.asyncio
    async def test_query_with_filter(self, rag_service_with_mocks, sample_search_results):
        """Test query passes filter to vector store."""
        service = rag_service_with_mocks
        service.vector_store.search.return_value = sample_search_results

        filter_dict = {"journal": "Nature Medicine"}
        await service.query("Test question", filter_dict=filter_dict)

        call_kwargs = service.vector_store.search.call_args[1]
        assert call_kwargs["filter_dict"] == filter_dict


class TestReasoningQuery:
    """Tests for the multi-step reasoning query."""

    @pytest.mark.asyncio
    async def test_reasoning_query_includes_steps(self, rag_service_with_mocks, sample_search_results):
        """Test reasoning query returns reasoning steps."""
        service = rag_service_with_mocks
        service.vector_store.search.return_value = sample_search_results

        # Mock decomposition response
        decompose_response = MagicMock()
        decompose_response.choices = [MagicMock()]
        decompose_response.choices[0].message.content = '''
        {
            "complexity": "moderate",
            "main_concepts": ["drug X", "cancer"],
            "sub_questions": ["What is drug X?", "How does it affect cancer?"],
            "reasoning_approach": "Two-step analysis"
        }
        '''

        service.llm_client.chat.completions.create.return_value = decompose_response

        response = await service.reasoning_query("Complex question about drug mechanisms")

        assert response.reasoning_steps is not None
        assert len(response.reasoning_steps) > 0

    @pytest.mark.asyncio
    async def test_reasoning_query_no_results(self, rag_service_with_mocks):
        """Test reasoning query handles no search results."""
        service = rag_service_with_mocks
        service.vector_store.search.return_value = []

        # Mock decomposition
        decompose_response = MagicMock()
        decompose_response.choices = [MagicMock()]
        decompose_response.choices[0].message.content = '{"complexity": "simple", "sub_questions": ["test"]}'
        service.llm_client.chat.completions.create.return_value = decompose_response

        response = await service.reasoning_query("No results question")

        assert response.confidence == 0.0
        assert response.reasoning_steps is not None

    @pytest.mark.asyncio
    async def test_reasoning_query_decomposition_step(self, rag_service_with_mocks, sample_search_results):
        """Test reasoning includes decomposition step."""
        service = rag_service_with_mocks
        service.vector_store.search.return_value = sample_search_results

        decompose_response = MagicMock()
        decompose_response.choices = [MagicMock()]
        decompose_response.choices[0].message.content = '''
        {"complexity": "complex", "main_concepts": ["test"], "sub_questions": ["Q1"], "reasoning_approach": "test"}
        '''
        service.llm_client.chat.completions.create.return_value = decompose_response

        response = await service.reasoning_query("Test")

        step_types = [s["type"] for s in response.reasoning_steps]
        assert "decomposition" in step_types


class TestSemanticSearch:
    """Tests for semantic search functionality."""

    def test_semantic_search_deduplicates_papers(self, rag_service_with_mocks, sample_search_results):
        """Test semantic search deduplicates results by PMID."""
        service = rag_service_with_mocks
        service.vector_store.search.return_value = sample_search_results

        results = service.semantic_search("cancer treatment", top_k=10)

        # Should have 2 unique papers
        pmids = [r["pmid"] for r in results]
        assert len(pmids) == len(set(pmids))

    def test_semantic_search_keeps_highest_score(self, rag_service_with_mocks, sample_search_results):
        """Test semantic search keeps highest relevance score per paper."""
        service = rag_service_with_mocks
        service.vector_store.search.return_value = sample_search_results

        results = service.semantic_search("test", top_k=10)

        # PMID 12345 appears twice with scores 0.95 and 0.87
        paper_12345 = next(r for r in results if r["pmid"] == "12345")
        assert paper_12345["relevance"] == 0.95

    def test_semantic_search_respects_top_k(self, rag_service_with_mocks, sample_search_results):
        """Test semantic search respects top_k limit."""
        service = rag_service_with_mocks
        service.vector_store.search.return_value = sample_search_results

        results = service.semantic_search("test", top_k=1)

        assert len(results) <= 1

    def test_semantic_search_includes_metadata(self, rag_service_with_mocks, sample_search_results):
        """Test semantic search results include paper metadata."""
        service = rag_service_with_mocks
        service.vector_store.search.return_value = sample_search_results

        results = service.semantic_search("test")

        for result in results:
            assert "pmid" in result
            assert "title" in result
            assert "journal" in result
            assert "relevance" in result
            assert "excerpt" in result


class TestIndexPaper:
    """Tests for paper indexing."""

    def test_index_paper_creates_chunks(self, rag_service_with_mocks, sample_paper_data):
        """Test indexing creates and stores chunks."""
        service = rag_service_with_mocks

        ids = service.index_paper(**sample_paper_data)

        assert isinstance(ids, list)
        assert len(ids) > 0
        service.vector_store.add_documents.assert_called_once()

    def test_index_paper_includes_metadata(self, rag_service_with_mocks, sample_paper_data):
        """Test indexed chunks include paper metadata."""
        service = rag_service_with_mocks

        service.index_paper(**sample_paper_data)

        call_args = service.vector_store.add_documents.call_args[0]
        metadatas = call_args[2]

        for metadata in metadatas:
            assert metadata["pmid"] == sample_paper_data["pmid"]
            assert metadata["title"] == sample_paper_data["title"]
            assert metadata["journal"] == sample_paper_data["journal"]

    def test_index_paper_uses_embedding_service(self, rag_service_with_mocks, sample_paper_data):
        """Test indexing uses embedding service for batch encoding."""
        service = rag_service_with_mocks

        service.index_paper(**sample_paper_data)

        service.embedding_service.batch_encode.assert_called_once()


class TestTranslation:
    """Tests for Korean translation functionality."""

    def test_translate_if_korean_passes_through_english(self, rag_service_with_mocks):
        """Test English text is not translated."""
        service = rag_service_with_mocks
        service._translator = None

        result = service._translate_if_korean("This is English text")

        assert result == "This is English text"

    def test_translate_if_korean_with_mock_translator(self, rag_service_with_mocks, mock_translation_service):
        """Test Korean text goes through translator."""
        service = rag_service_with_mocks
        service._translator = mock_translation_service

        mock_translation_service.is_korean.return_value = True
        mock_translation_service.translate_to_english.return_value = "translated query"

        result = service._translate_if_korean("한국어 텍스트")

        mock_translation_service.translate_to_english.assert_called_once()
        assert result == "translated query"


class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_generate_answer_handles_llm_error(self, rag_service_with_mocks):
        """Test graceful handling of LLM errors."""
        service = rag_service_with_mocks
        service.llm_client.chat.completions.create.side_effect = Exception("API Error")

        result = await service._generate_answer("test", "test context")

        assert "Error" in result

    @pytest.mark.asyncio
    async def test_decompose_question_handles_invalid_json(self, rag_service_with_mocks):
        """Test decomposition handles invalid JSON from LLM."""
        service = rag_service_with_mocks

        bad_response = MagicMock()
        bad_response.choices = [MagicMock()]
        bad_response.choices[0].message.content = "This is not valid JSON"
        service.llm_client.chat.completions.create.return_value = bad_response

        result = await service._decompose_question("test question")

        # Should return fallback structure
        assert "sub_questions" in result
        assert result["complexity"] == "simple"
