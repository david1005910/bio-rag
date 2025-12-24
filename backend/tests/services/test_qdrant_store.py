"""Comprehensive tests for Qdrant hybrid store"""

from unittest.mock import MagicMock, patch, PropertyMock
import pytest

from app.services.vector.qdrant_store import (
    HybridSearchResult,
    HybridSearchResponse,
    SPLADEEncoder,
    BGEM3Encoder,
    PubMedBERTEncoder,
    QdrantHybridStore,
)


class TestHybridSearchResult:
    """Tests for HybridSearchResult dataclass"""

    def test_default_values(self):
        """Test default values"""
        result = HybridSearchResult(
            doc_id="test_1",
            content="Test content",
            metadata={},
        )
        assert result.doc_id == "test_1"
        assert result.dense_score == 0.0
        assert result.sparse_score == 0.0
        assert result.rrf_score == 0.0
        assert result.dense_rank == 0
        assert result.sparse_rank == 0

    def test_with_scores(self):
        """Test with all scores"""
        result = HybridSearchResult(
            doc_id="test_2",
            content="Content",
            metadata={"pmid": "123"},
            dense_score=0.85,
            sparse_score=0.72,
            rrf_score=0.80,
            dense_rank=1,
            sparse_rank=2,
        )
        assert result.dense_score == 0.85
        assert result.sparse_score == 0.72
        assert result.rrf_score == 0.80


class TestHybridSearchResponse:
    """Tests for HybridSearchResponse dataclass"""

    def test_response_creation(self):
        """Test creating response"""
        results = [
            HybridSearchResult(doc_id="1", content="c1", metadata={}),
            HybridSearchResult(doc_id="2", content="c2", metadata={}),
        ]
        response = HybridSearchResponse(
            results=results,
            query="test query",
            total_results=2,
            dense_weight=0.7,
            sparse_weight=0.3,
            rrf_k=60,
        )
        assert response.query == "test query"
        assert response.total_results == 2
        assert response.dense_weight == 0.7
        assert len(response.results) == 2
        assert response.score_distribution == {}

    def test_response_with_distribution(self):
        """Test response with score distribution"""
        response = HybridSearchResponse(
            results=[],
            query="test",
            total_results=0,
            dense_weight=0.7,
            sparse_weight=0.3,
            rrf_k=60,
            score_distribution={"avg_dense": 0.8},
        )
        assert response.score_distribution == {"avg_dense": 0.8}


class TestSPLADEEncoder:
    """Tests for SPLADEEncoder"""

    def test_init(self):
        """Test encoder initialization"""
        encoder = SPLADEEncoder()
        assert encoder.model_name == "naver/splade-cocondenser-ensembledistil"
        assert encoder._model is None
        assert encoder._tokenizer is None

    def test_init_custom_model(self):
        """Test encoder with custom model name"""
        encoder = SPLADEEncoder(model_name="custom/model")
        assert encoder.model_name == "custom/model"

    def test_encode_bm25_fallback(self):
        """Test BM25-style fallback encoding"""
        encoder = SPLADEEncoder()
        encoder._model = "fallback"  # Force fallback

        indices, values = encoder.encode("This is a test sentence for encoding")
        assert isinstance(indices, list)
        assert isinstance(values, list)
        assert len(indices) == len(values)
        assert len(indices) > 0

    def test_encode_bm25_with_query_boost(self):
        """Test BM25 encoding with query boost"""
        encoder = SPLADEEncoder()
        encoder._model = "fallback"

        indices1, values1 = encoder.encode("test query", is_query=False)
        indices2, values2 = encoder.encode("test query", is_query=True)

        # Query values should be boosted (1.5x)
        if values1 and values2:
            assert values2[0] > values1[0]

    def test_encode_bm25_empty_after_stopwords(self):
        """Test encoding with only stopwords"""
        encoder = SPLADEEncoder()
        encoder._model = "fallback"

        indices, values = encoder.encode("the and or")
        assert indices == []
        assert values == []

    def test_encode_bm25_style_tokenization(self):
        """Test BM25 tokenization"""
        encoder = SPLADEEncoder()
        encoder._model = "fallback"

        # Test stopword removal
        indices1, values1 = encoder.encode("cancer treatment")
        indices2, values2 = encoder.encode("the cancer and treatment")

        # Should have similar results (stopwords removed)
        assert len(indices1) == len(indices2)


class TestBGEM3Encoder:
    """Tests for BGEM3Encoder"""

    def test_init(self):
        """Test encoder initialization"""
        encoder = BGEM3Encoder()
        assert encoder.model_name == "BAAI/bge-m3"
        assert encoder._model is None
        assert encoder.dimension == 1024

    def test_init_custom_model(self):
        """Test with custom model"""
        encoder = BGEM3Encoder(model_name="custom/model")
        assert encoder.model_name == "custom/model"


class TestPubMedBERTEncoder:
    """Tests for PubMedBERTEncoder"""

    def test_init(self):
        """Test encoder initialization"""
        encoder = PubMedBERTEncoder()
        assert "PubMedBERT" in encoder.model_name
        assert encoder._model is None
        assert encoder._tokenizer is None
        assert encoder.dimension == 768
        assert encoder._device == "cpu"

    def test_init_custom_model(self):
        """Test with custom model"""
        encoder = PubMedBERTEncoder(model_name="custom/bert")
        assert encoder.model_name == "custom/bert"


class TestQdrantHybridStore:
    """Tests for QdrantHybridStore"""

    def test_init_default(self):
        """Test default initialization"""
        with patch("app.services.vector.qdrant_store.settings") as mock_settings:
            mock_settings.QDRANT_COLLECTION = "test_collection"
            mock_settings.QDRANT_HOST = "localhost"
            mock_settings.QDRANT_PORT = 6333
            mock_settings.QDRANT_API_KEY = None
            mock_settings.QDRANT_USE_MEMORY = True

            store = QdrantHybridStore()
            assert store.collection_name == "test_collection"
            assert store.host == "localhost"
            assert store.port == 6333
            assert store.use_memory is True
            assert store.use_multilingual is True

    def test_init_custom_params(self):
        """Test initialization with custom parameters"""
        store = QdrantHybridStore(
            collection_name="custom_collection",
            host="custom_host",
            port=1234,
            api_key="test_key",
            use_memory=True,
            use_multilingual=False,
        )
        assert store.collection_name == "custom_collection"
        assert store.host == "custom_host"
        assert store.port == 1234
        assert store.api_key == "test_key"
        assert store.use_multilingual is False

    def test_dense_dim_multilingual(self):
        """Test dense dimension for multilingual"""
        store = QdrantHybridStore(use_multilingual=True, use_memory=True)
        assert store.dense_dim == 1024

    def test_dense_dim_biomedical(self):
        """Test dense dimension for biomedical"""
        store = QdrantHybridStore(use_multilingual=False, use_memory=True)
        assert store.dense_dim == 768

    def test_hybrid_search_config(self):
        """Test default hybrid search config"""
        store = QdrantHybridStore(use_memory=True)
        assert store.dense_weight == 0.7
        assert store.sparse_weight == 0.3
        assert store.rrf_k == 60

    def test_encoder_properties(self):
        """Test encoder property lazy initialization"""
        store = QdrantHybridStore(use_memory=True)

        # Properties should create encoders lazily
        assert store._bgem3_encoder is None
        assert store._dense_encoder is None
        assert store._sparse_encoder is None

    def test_sparse_encoder_property(self):
        """Test sparse encoder property"""
        store = QdrantHybridStore(use_memory=True)
        encoder = store.sparse_encoder
        assert isinstance(encoder, SPLADEEncoder)
        # Should reuse same instance
        assert store.sparse_encoder is encoder

    def test_hash_id(self):
        """Test ID hashing"""
        store = QdrantHybridStore(use_memory=True)

        id1 = store._hash_id("document_1")
        id2 = store._hash_id("document_2")
        id3 = store._hash_id("document_1")  # Same as id1

        assert isinstance(id1, int)
        assert id1 != id2
        assert id1 == id3  # Same input produces same hash

    def test_calculate_score_distribution_empty(self):
        """Test score distribution with empty results"""
        store = QdrantHybridStore(use_memory=True)
        dist = store._calculate_score_distribution([], 0.7, 0.3)
        assert dist == {}

    def test_calculate_score_distribution(self):
        """Test score distribution calculation"""
        store = QdrantHybridStore(use_memory=True)

        results = [
            HybridSearchResult(
                doc_id="1",
                content="c1",
                metadata={},
                dense_score=0.9,
                sparse_score=0.8,
                rrf_score=0.85,
            ),
            HybridSearchResult(
                doc_id="2",
                content="c2",
                metadata={},
                dense_score=0.7,
                sparse_score=0.6,
                rrf_score=0.65,
            ),
        ]

        dist = store._calculate_score_distribution(results, 0.7, 0.3)

        assert "dense_scores" in dist
        assert "sparse_scores" in dist
        assert "rrf_scores" in dist
        assert "contributions" in dist
        assert "config" in dist

        assert dist["dense_scores"]["min"] == 0.7
        assert dist["dense_scores"]["max"] == 0.9
        assert dist["config"]["dense_weight"] == 0.7
        assert len(dist["contributions"]) == 2

    def test_calculate_score_distribution_zero_total(self):
        """Test score distribution when total is zero"""
        store = QdrantHybridStore(use_memory=True)

        results = [
            HybridSearchResult(
                doc_id="1",
                content="c1",
                metadata={},
                dense_score=0.0,
                sparse_score=0.0,
                rrf_score=0.0,
            ),
        ]

        dist = store._calculate_score_distribution(results, 0.7, 0.3)

        # When total is 0, should default to 50/50 contribution
        assert dist["contributions"][0]["dense_contribution"] == 50.0
        assert dist["contributions"][0]["sparse_contribution"] == 50.0


class TestQdrantClientMemoryMode:
    """Tests for Qdrant client in memory mode"""

    @patch("app.services.vector.qdrant_store.QdrantClient")
    def test_client_memory_mode(self, mock_qdrant):
        """Test client creation in memory mode"""
        mock_client = MagicMock()
        mock_qdrant.return_value = mock_client
        mock_client.get_collections.return_value.collections = []

        store = QdrantHybridStore(use_memory=True)
        client = store.client

        assert client is not None
        mock_qdrant.assert_called_once_with(":memory:")

    @patch("app.services.vector.qdrant_store.QdrantClient")
    def test_client_server_mode(self, mock_qdrant):
        """Test client creation in server mode"""
        mock_client = MagicMock()
        mock_qdrant.return_value = mock_client
        mock_client.get_collections.return_value.collections = []

        store = QdrantHybridStore(
            host="localhost",
            port=6333,
            use_memory=False,
        )
        client = store.client

        assert client is not None
        mock_qdrant.assert_called_once_with(
            host="localhost",
            port=6333,
            api_key=None,
        )

    @patch("app.services.vector.qdrant_store.QdrantClient")
    def test_collection_creation(self, mock_qdrant):
        """Test collection is created when not exists"""
        mock_client = MagicMock()
        mock_qdrant.return_value = mock_client
        mock_client.get_collections.return_value.collections = []

        store = QdrantHybridStore(
            collection_name="new_collection",
            use_memory=True,
        )
        _ = store.client

        # Should call create_collection for new collection
        mock_client.create_collection.assert_called_once()

    @patch("app.services.vector.qdrant_store.QdrantClient")
    def test_collection_exists(self, mock_qdrant):
        """Test when collection already exists"""
        mock_client = MagicMock()
        mock_qdrant.return_value = mock_client

        # Mock existing collection
        mock_collection = MagicMock()
        mock_collection.name = "existing_collection"
        mock_client.get_collections.return_value.collections = [mock_collection]

        store = QdrantHybridStore(
            collection_name="existing_collection",
            use_memory=True,
        )
        _ = store.client

        # Should NOT call create_collection
        mock_client.create_collection.assert_not_called()


class TestQdrantStoreOperations:
    """Tests for Qdrant store operations with mocks"""

    @pytest.fixture
    def mock_store(self):
        """Create store with mocked client and encoders"""
        with patch("app.services.vector.qdrant_store.QdrantClient") as mock_qdrant:
            mock_client = MagicMock()
            mock_qdrant.return_value = mock_client
            mock_client.get_collections.return_value.collections = []

            store = QdrantHybridStore(use_memory=True, use_multilingual=False)
            store._client = mock_client

            # Mock sparse encoder to use fallback
            store._sparse_encoder = SPLADEEncoder()
            store._sparse_encoder._model = "fallback"

            yield store

    def test_get_sparse_vector_multilingual(self, mock_store):
        """Test sparse vector with multilingual encoder"""
        mock_store.use_multilingual = True
        mock_store._bgem3_encoder = MagicMock()
        mock_store._bgem3_encoder.encode_sparse.return_value = [([1, 2], [0.5, 0.3])]

        indices, values = mock_store._get_sparse_vector("test text")
        assert indices == [1, 2]
        assert values == [0.5, 0.3]

    def test_get_sparse_vector_splade(self, mock_store):
        """Test sparse vector with SPLADE encoder"""
        mock_store.use_multilingual = False

        indices, values = mock_store._get_sparse_vector("cancer treatment research")
        assert len(indices) > 0
        assert len(values) > 0


class TestQdrantSearchOperations:
    """Tests for search operations"""

    def test_search_dense_mock(self):
        """Test dense search with mock"""
        with patch("app.services.vector.qdrant_store.QdrantClient") as mock_qdrant:
            mock_client = MagicMock()
            mock_qdrant.return_value = mock_client
            mock_client.get_collections.return_value.collections = []

            store = QdrantHybridStore(use_memory=True)
            store._client = mock_client

            # Mock query_points response
            mock_hit = MagicMock()
            mock_hit.payload = {"doc_id": "1", "content": "test"}
            mock_hit.score = 0.9
            mock_client.query_points.return_value.points = [mock_hit]

            # Mock embedding
            with patch.object(store, "_get_dense_embedding", return_value=[0.1] * 768):
                results = store.search_dense("test query", top_k=5)

            assert len(results) == 1
            assert results[0][0]["doc_id"] == "1"
            assert results[0][1] == 0.9

    def test_search_sparse_empty(self):
        """Test sparse search with empty indices"""
        with patch("app.services.vector.qdrant_store.QdrantClient") as mock_qdrant:
            mock_client = MagicMock()
            mock_qdrant.return_value = mock_client
            mock_client.get_collections.return_value.collections = []

            store = QdrantHybridStore(use_memory=True)
            store._client = mock_client

            # Mock empty sparse vector
            with patch.object(store, "_get_sparse_vector", return_value=([], [])):
                results = store.search_sparse("test", top_k=5)

            assert results == []


class TestQdrantHybridSearch:
    """Tests for hybrid search"""

    def test_hybrid_search_logic(self):
        """Test hybrid search RRF fusion logic"""
        with patch("app.services.vector.qdrant_store.QdrantClient") as mock_qdrant:
            mock_client = MagicMock()
            mock_qdrant.return_value = mock_client
            mock_client.get_collections.return_value.collections = []

            store = QdrantHybridStore(use_memory=True)
            store._client = mock_client

            # Mock dense search results
            dense_results = [
                ({"doc_id": "doc1", "content": "c1"}, 0.9),
                ({"doc_id": "doc2", "content": "c2"}, 0.8),
            ]

            # Mock sparse search results
            sparse_results = [
                ({"doc_id": "doc2", "content": "c2"}, 0.7),
                ({"doc_id": "doc3", "content": "c3"}, 0.6),
            ]

            with patch.object(store, "search_dense", return_value=dense_results):
                with patch.object(store, "search_sparse", return_value=sparse_results):
                    response = store.hybrid_search("test query", top_k=3)

            assert isinstance(response, HybridSearchResponse)
            assert response.query == "test query"
            assert response.dense_weight == 0.7
            assert response.sparse_weight == 0.3
            # doc2 should rank high because it appears in both
            doc_ids = [r.doc_id for r in response.results]
            assert "doc2" in doc_ids

    def test_hybrid_search_custom_weights(self):
        """Test hybrid search with custom weights"""
        with patch("app.services.vector.qdrant_store.QdrantClient") as mock_qdrant:
            mock_client = MagicMock()
            mock_qdrant.return_value = mock_client
            mock_client.get_collections.return_value.collections = []

            store = QdrantHybridStore(use_memory=True)
            store._client = mock_client

            with patch.object(store, "search_dense", return_value=[]):
                with patch.object(store, "search_sparse", return_value=[]):
                    response = store.hybrid_search(
                        "test",
                        top_k=5,
                        dense_weight=0.5,
                        sparse_weight=0.5,
                        rrf_k=30,
                    )

            assert response.dense_weight == 0.5
            assert response.sparse_weight == 0.5
            assert response.rrf_k == 30
