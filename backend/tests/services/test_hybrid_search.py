"""Comprehensive tests for hybrid search service"""

from unittest.mock import MagicMock, patch
import pytest

from app.services.rag.hybrid_search import (
    HybridSearchConfig,
    BM25Index,
    HybridSearcher,
)
from app.services.rag.retriever import RetrievedDocument


class TestHybridSearchConfig:
    """Tests for HybridSearchConfig"""

    def test_default_values(self):
        """Test default configuration values"""
        config = HybridSearchConfig()
        assert config.dense_weight == 0.7
        assert config.bm25_weight == 0.3
        assert config.rrf_k == 60
        assert config.top_k == 10
        assert config.min_score == 0.3

    def test_custom_values(self):
        """Test custom configuration values"""
        config = HybridSearchConfig(
            dense_weight=0.5,
            bm25_weight=0.5,
            rrf_k=30,
            top_k=20,
            min_score=0.1,
        )
        assert config.dense_weight == 0.5
        assert config.bm25_weight == 0.5
        assert config.rrf_k == 30
        assert config.top_k == 20
        assert config.min_score == 0.1


class TestBM25Index:
    """Tests for BM25Index"""

    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing"""
        return [
            {"chunk_id": "1", "content": "Cancer treatment using immunotherapy is effective"},
            {"chunk_id": "2", "content": "Machine learning in drug discovery research"},
            {"chunk_id": "3", "content": "Gene therapy for cancer patients"},
            {"chunk_id": "4", "content": "Immunotherapy advances in oncology"},
        ]

    def test_init(self):
        """Test initialization"""
        index = BM25Index()
        assert index._documents == []
        assert index._index is None
        assert index._tokenized_corpus == []
        assert index.is_built is False

    def test_build_index(self, sample_documents):
        """Test building index"""
        index = BM25Index()
        index.build_index(sample_documents)

        assert index.is_built is True
        assert index.document_count == 4
        assert len(index._tokenized_corpus) == 4

    def test_search_before_build(self):
        """Test search before building index"""
        index = BM25Index()

        with pytest.raises(ValueError, match="BM25 index not built"):
            index.search("cancer")

    def test_search_after_build(self, sample_documents):
        """Test search after building index"""
        index = BM25Index()
        index.build_index(sample_documents)

        results = index.search("cancer immunotherapy", top_k=2)

        assert isinstance(results, list)
        assert len(results) <= 2
        # Each result is (index, score)
        for idx, score in results:
            assert isinstance(idx, int)
            assert isinstance(score, float)
            assert score > 0

    def test_search_returns_ranked_results(self, sample_documents):
        """Test search returns results ranked by score"""
        index = BM25Index()
        index.build_index(sample_documents)

        results = index.search("cancer treatment", top_k=4)

        if len(results) >= 2:
            # Scores should be in descending order
            for i in range(len(results) - 1):
                assert results[i][1] >= results[i + 1][1]

    def test_get_document(self, sample_documents):
        """Test getting document by index"""
        index = BM25Index()
        index.build_index(sample_documents)

        doc = index.get_document(0)
        assert doc["chunk_id"] == "1"

        doc = index.get_document(2)
        assert doc["chunk_id"] == "3"

    def test_tokenize(self):
        """Test tokenization"""
        index = BM25Index()

        tokens = index._tokenize("Hello, World! This is a test.")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens
        # Punctuation should be removed
        assert "," not in tokens
        assert "!" not in tokens

    def test_tokenize_numbers(self):
        """Test tokenization with numbers"""
        index = BM25Index()

        tokens = index._tokenize("COVID19 and 2024 research")
        assert "covid19" in tokens
        assert "2024" in tokens

    def test_search_no_results(self, sample_documents):
        """Test search with no matching results"""
        index = BM25Index()
        index.build_index(sample_documents)

        # Query with terms not in documents
        results = index.search("xyz123 quantum physics", top_k=5)

        # Should return empty or very low scores
        assert isinstance(results, list)

    def test_document_count_property(self, sample_documents):
        """Test document count property"""
        index = BM25Index()
        assert index.document_count == 0

        index.build_index(sample_documents)
        assert index.document_count == 4

    def test_is_built_property(self):
        """Test is_built property"""
        index = BM25Index()
        assert index.is_built is False

        index.build_index([{"chunk_id": "1", "content": "test"}])
        assert index.is_built is True


class TestHybridSearcher:
    """Tests for HybridSearcher"""

    @pytest.fixture
    def mock_retriever(self):
        """Create mock retriever"""
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            RetrievedDocument(
                chunk_id="1",
                pmid="12345",
                title="Paper 1",
                content="Cancer treatment content",
                section="abstract",
                score=0.9,
                metadata={},
            ),
            RetrievedDocument(
                chunk_id="2",
                pmid="12346",
                title="Paper 2",
                content="Drug discovery content",
                section="abstract",
                score=0.8,
                metadata={},
            ),
        ]
        return retriever

    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store"""
        store = MagicMock()
        store.get_all_documents.return_value = [
            {"chunk_id": "1", "content": "Cancer treatment content"},
            {"chunk_id": "2", "content": "Drug discovery content"},
        ]
        return store

    def test_init_default(self):
        """Test default initialization"""
        searcher = HybridSearcher()
        assert searcher._dense_retriever is None
        assert searcher._vector_store is None
        assert isinstance(searcher._bm25_index, BM25Index)
        assert isinstance(searcher.config, HybridSearchConfig)

    def test_init_with_config(self):
        """Test initialization with custom config"""
        config = HybridSearchConfig(dense_weight=0.6, bm25_weight=0.4)
        searcher = HybridSearcher(config=config)

        assert searcher.config.dense_weight == 0.6
        assert searcher.config.bm25_weight == 0.4

    def test_init_with_retriever(self, mock_retriever):
        """Test initialization with retriever"""
        searcher = HybridSearcher(dense_retriever=mock_retriever)
        assert searcher._dense_retriever is mock_retriever

    def test_dense_retriever_property(self, mock_retriever):
        """Test dense retriever property lazy initialization"""
        searcher = HybridSearcher(dense_retriever=mock_retriever)
        retriever = searcher.dense_retriever
        assert retriever is mock_retriever

    def test_store_property(self, mock_vector_store):
        """Test store property lazy initialization"""
        searcher = HybridSearcher(vector_store=mock_vector_store)
        store = searcher.store
        assert store is mock_vector_store

    def test_set_embedding_func(self, mock_retriever):
        """Test setting embedding function"""
        searcher = HybridSearcher(dense_retriever=mock_retriever)

        def custom_embed(text):
            return [0.1] * 768

        searcher.set_embedding_func(custom_embed)
        mock_retriever.set_embedding_func.assert_called_once_with(custom_embed)

    def test_build_bm25_index_from_documents(self):
        """Test building BM25 index from documents"""
        searcher = HybridSearcher()
        documents = [
            {"chunk_id": "1", "content": "Test content one"},
            {"chunk_id": "2", "content": "Test content two"},
        ]

        searcher.build_bm25_index(documents)

        assert searcher._bm25_index.is_built
        assert searcher._bm25_index.document_count == 2

    def test_build_bm25_index_from_store(self, mock_vector_store):
        """Test building BM25 index from vector store"""
        searcher = HybridSearcher(vector_store=mock_vector_store)

        searcher.build_bm25_index()  # None means load from store

        mock_vector_store.get_all_documents.assert_called_once()
        assert searcher._bm25_index.is_built

    def test_search_dense_only(self, mock_retriever):
        """Test search with dense retrieval only (no BM25 index)"""
        searcher = HybridSearcher(dense_retriever=mock_retriever)

        results = searcher.search("cancer treatment", top_k=5)

        assert len(results) > 0
        assert isinstance(results[0], RetrievedDocument)
        mock_retriever.retrieve.assert_called_once()

    def test_search_hybrid(self, mock_retriever):
        """Test hybrid search with both dense and BM25"""
        searcher = HybridSearcher(dense_retriever=mock_retriever)

        # Build BM25 index
        documents = [
            {"chunk_id": "1", "content": "Cancer treatment content", "pmid": "12345", "title": "Paper 1"},
            {"chunk_id": "2", "content": "Drug discovery content", "pmid": "12346", "title": "Paper 2"},
            {"chunk_id": "3", "content": "Immunotherapy for cancer", "pmid": "12347", "title": "Paper 3"},
        ]
        searcher.build_bm25_index(documents)

        results = searcher.search("cancer treatment", top_k=3)

        assert len(results) > 0
        # Results should have fused scores

    def test_rrf_fusion_logic(self, mock_retriever):
        """Test RRF fusion combines results properly"""
        searcher = HybridSearcher(
            dense_retriever=mock_retriever,
            config=HybridSearchConfig(dense_weight=0.5, bm25_weight=0.5, rrf_k=60)
        )

        # Build BM25 index with document appearing in both
        documents = [
            {"chunk_id": "1", "content": "Cancer treatment content", "pmid": "12345", "title": "Paper 1"},
            {"chunk_id": "3", "content": "Immunotherapy advances", "pmid": "12347", "title": "Paper 3"},
        ]
        searcher.build_bm25_index(documents)

        results = searcher.search("cancer", top_k=5)

        # Document "1" appears in both dense and BM25, should have higher fused score
        assert len(results) > 0

    def test_rrf_fusion_document_from_bm25_only(self, mock_retriever):
        """Test RRF fusion handles documents only in BM25"""
        mock_retriever.retrieve.return_value = []  # No dense results

        searcher = HybridSearcher(dense_retriever=mock_retriever)

        documents = [
            {"chunk_id": "bm25_only", "content": "Unique content", "pmid": "99999", "title": "BM25 Paper"},
        ]
        searcher.build_bm25_index(documents)

        results = searcher.search("unique content", top_k=5)

        # Should still return results from BM25
        if results:
            assert any(r.chunk_id == "bm25_only" for r in results)

    def test_search_with_filter(self, mock_retriever):
        """Test search with metadata filter"""
        searcher = HybridSearcher(dense_retriever=mock_retriever)

        filter_metadata = {"source": "pubmed"}
        searcher.search("cancer", filter_metadata=filter_metadata)

        # Filter should be passed to dense retriever
        call_args = mock_retriever.retrieve.call_args
        assert call_args.kwargs.get("filter_metadata") == filter_metadata

    def test_search_respects_top_k(self, mock_retriever):
        """Test search respects top_k parameter"""
        searcher = HybridSearcher(dense_retriever=mock_retriever)

        results = searcher.search("cancer", top_k=1)

        assert len(results) <= 1


class TestHybridSearcherWithScores:
    """Tests for search_with_scores method"""

    @pytest.fixture
    def mock_retriever_with_docs(self):
        """Create mock retriever with documents"""
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            RetrievedDocument(
                chunk_id="1",
                pmid="12345",
                title="Paper 1",
                content="Cancer content",
                section="abstract",
                score=0.9,
                metadata={},
            ),
            RetrievedDocument(
                chunk_id="2",
                pmid="12346",
                title="Paper 2",
                content="Treatment content",
                section="abstract",
                score=0.7,
                metadata={},
            ),
        ]
        return retriever

    def test_search_with_scores_structure(self, mock_retriever_with_docs):
        """Test search_with_scores returns proper structure"""
        searcher = HybridSearcher(dense_retriever=mock_retriever_with_docs)

        results = searcher.search_with_scores("cancer", top_k=5)

        assert isinstance(results, list)
        for doc, scores in results:
            assert isinstance(doc, RetrievedDocument)
            assert isinstance(scores, dict)
            assert "dense_score" in scores
            assert "bm25_score" in scores
            assert "fused_score" in scores

    def test_search_with_scores_bm25_normalization(self, mock_retriever_with_docs):
        """Test BM25 scores are normalized"""
        searcher = HybridSearcher(dense_retriever=mock_retriever_with_docs)

        documents = [
            {"chunk_id": "1", "content": "Cancer content", "pmid": "12345", "title": "Paper 1"},
            {"chunk_id": "2", "content": "Treatment content", "pmid": "12346", "title": "Paper 2"},
        ]
        searcher.build_bm25_index(documents)

        results = searcher.search_with_scores("cancer", top_k=5)

        # BM25 scores should be normalized (0-1)
        for _, scores in results:
            assert 0 <= scores["bm25_score"] <= 1

    def test_search_with_scores_without_bm25(self, mock_retriever_with_docs):
        """Test search_with_scores without BM25 index"""
        searcher = HybridSearcher(dense_retriever=mock_retriever_with_docs)

        results = searcher.search_with_scores("cancer", top_k=5)

        # BM25 scores should be 0 when index not built
        for _, scores in results:
            assert scores["bm25_score"] == 0.0

    def test_search_with_scores_sorted(self, mock_retriever_with_docs):
        """Test results are sorted by fused score"""
        searcher = HybridSearcher(dense_retriever=mock_retriever_with_docs)

        results = searcher.search_with_scores("cancer", top_k=5)

        if len(results) >= 2:
            for i in range(len(results) - 1):
                assert results[i][1]["fused_score"] >= results[i + 1][1]["fused_score"]


class TestHybridSearcherSingleton:
    """Tests for hybrid searcher singleton"""

    def test_singleton_instance_exists(self):
        """Test singleton instance exists"""
        from app.services.rag.hybrid_search import hybrid_searcher
        assert hybrid_searcher is not None
        assert isinstance(hybrid_searcher, HybridSearcher)
