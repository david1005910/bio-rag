"""Comprehensive tests for cross-encoder reranker"""

from unittest.mock import MagicMock, patch
import pytest
import numpy as np

from app.services.rag.reranker import (
    RerankerConfig,
    CrossEncoderReranker,
    TwoStageRetriever,
    create_two_stage_retriever,
    DEFAULT_RERANKER_MODEL,
    BIOMEDICAL_RERANKER_MODEL,
)
from app.services.rag.retriever import RetrievedDocument


class TestRerankerConfig:
    """Tests for RerankerConfig"""

    def test_default_values(self):
        """Test default configuration values"""
        config = RerankerConfig()
        assert config.model_name == DEFAULT_RERANKER_MODEL
        assert config.top_k == 5
        assert config.batch_size == 32
        assert config.max_length == 512
        assert config.device is None
        assert config.score_threshold == 0.0

    def test_custom_values(self):
        """Test custom configuration values"""
        config = RerankerConfig(
            model_name=BIOMEDICAL_RERANKER_MODEL,
            top_k=10,
            batch_size=16,
            max_length=256,
            device="cuda",
            score_threshold=0.5,
        )
        assert config.model_name == BIOMEDICAL_RERANKER_MODEL
        assert config.top_k == 10
        assert config.batch_size == 16
        assert config.max_length == 256
        assert config.device == "cuda"
        assert config.score_threshold == 0.5


class TestCrossEncoderReranker:
    """Tests for CrossEncoderReranker"""

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing"""
        return [
            RetrievedDocument(
                chunk_id="1",
                pmid="12345",
                title="Paper 1",
                content="Cancer treatment using immunotherapy shows promising results",
                section="abstract",
                score=0.9,
                metadata={},
            ),
            RetrievedDocument(
                chunk_id="2",
                pmid="12346",
                title="Paper 2",
                content="Machine learning for drug discovery",
                section="abstract",
                score=0.8,
                metadata={},
            ),
            RetrievedDocument(
                chunk_id="3",
                pmid="12347",
                title="Paper 3",
                content="Gene therapy advances in oncology",
                section="abstract",
                score=0.7,
                metadata={},
            ),
        ]

    def test_init_default(self):
        """Test default initialization"""
        with patch("app.services.rag.reranker.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            reranker = CrossEncoderReranker()

            assert isinstance(reranker.config, RerankerConfig)
            assert reranker._model is None
            assert reranker._device == "cpu"

    def test_init_with_config(self):
        """Test initialization with custom config"""
        config = RerankerConfig(
            model_name="custom/model",
            device="cuda",
        )
        reranker = CrossEncoderReranker(config=config)

        assert reranker.config.model_name == "custom/model"
        assert reranker._device == "cuda"

    def test_init_auto_device_cuda(self):
        """Test auto device selection with CUDA available"""
        with patch("app.services.rag.reranker.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True

            config = RerankerConfig(device=None)
            reranker = CrossEncoderReranker(config=config)

            assert reranker._device == "cuda"

    def test_init_auto_device_cpu(self):
        """Test auto device selection without CUDA"""
        with patch("app.services.rag.reranker.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False

            config = RerankerConfig(device=None)
            reranker = CrossEncoderReranker(config=config)

            assert reranker._device == "cpu"

    @patch("app.services.rag.reranker.CrossEncoder")
    def test_model_lazy_loading(self, mock_cross_encoder):
        """Test model is loaded lazily"""
        mock_model = MagicMock()
        mock_cross_encoder.return_value = mock_model

        reranker = CrossEncoderReranker()
        assert reranker._model is None

        # Access model property
        model = reranker.model
        assert model is mock_model
        mock_cross_encoder.assert_called_once()

    def test_rerank_empty_documents(self):
        """Test reranking empty document list"""
        reranker = CrossEncoderReranker()
        result = reranker.rerank("query", [])
        assert result == []

    @patch("app.services.rag.reranker.CrossEncoder")
    def test_rerank_documents(self, mock_cross_encoder, sample_documents):
        """Test reranking documents"""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.9, 0.5, 0.8])
        mock_cross_encoder.return_value = mock_model

        config = RerankerConfig(top_k=3)
        reranker = CrossEncoderReranker(config=config)

        results = reranker.rerank("cancer immunotherapy", sample_documents)

        assert len(results) == 3
        # Should be sorted by cross-encoder score (0.9, 0.8, 0.5)
        assert results[0].score == 0.9
        assert results[1].score == 0.8
        assert results[2].score == 0.5

    @patch("app.services.rag.reranker.CrossEncoder")
    def test_rerank_respects_top_k(self, mock_cross_encoder, sample_documents):
        """Test reranking respects top_k"""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.9, 0.8, 0.7])
        mock_cross_encoder.return_value = mock_model

        config = RerankerConfig(top_k=2)
        reranker = CrossEncoderReranker(config=config)

        results = reranker.rerank("query", sample_documents)

        assert len(results) == 2

    @patch("app.services.rag.reranker.CrossEncoder")
    def test_rerank_override_top_k(self, mock_cross_encoder, sample_documents):
        """Test reranking with overridden top_k"""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.9, 0.8, 0.7])
        mock_cross_encoder.return_value = mock_model

        config = RerankerConfig(top_k=3)
        reranker = CrossEncoderReranker(config=config)

        results = reranker.rerank("query", sample_documents, top_k=1)

        assert len(results) == 1

    @patch("app.services.rag.reranker.CrossEncoder")
    def test_rerank_score_threshold(self, mock_cross_encoder, sample_documents):
        """Test reranking with score threshold"""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.9, 0.3, 0.8])
        mock_cross_encoder.return_value = mock_model

        config = RerankerConfig(top_k=3, score_threshold=0.5)
        reranker = CrossEncoderReranker(config=config)

        results = reranker.rerank("query", sample_documents)

        # Should filter out score < 0.5
        assert len(results) == 2
        for doc in results:
            assert doc.score >= 0.5

    @patch("app.services.rag.reranker.CrossEncoder")
    def test_rerank_preserves_metadata(self, mock_cross_encoder, sample_documents):
        """Test reranking preserves and adds metadata"""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.9, 0.8, 0.7])
        mock_cross_encoder.return_value = mock_model

        reranker = CrossEncoderReranker()
        results = reranker.rerank("query", sample_documents)

        for result in results:
            assert "original_score" in result.metadata
            assert "rerank_score" in result.metadata

    def test_rerank_with_scores_empty(self):
        """Test rerank_with_scores with empty documents"""
        reranker = CrossEncoderReranker()
        result = reranker.rerank_with_scores("query", [])
        assert result == []

    @patch("app.services.rag.reranker.CrossEncoder")
    def test_rerank_with_scores(self, mock_cross_encoder, sample_documents):
        """Test rerank_with_scores returns score tuples"""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.9, 0.8, 0.7])
        mock_cross_encoder.return_value = mock_model

        reranker = CrossEncoderReranker()
        results = reranker.rerank_with_scores("query", sample_documents)

        assert len(results) == 3
        for doc, original_score, rerank_score in results:
            assert isinstance(doc, RetrievedDocument)
            assert isinstance(original_score, float)
            assert isinstance(rerank_score, float)

    @patch("app.services.rag.reranker.CrossEncoder")
    def test_rerank_with_scores_sorted(self, mock_cross_encoder, sample_documents):
        """Test rerank_with_scores returns sorted results"""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.5, 0.9, 0.7])
        mock_cross_encoder.return_value = mock_model

        reranker = CrossEncoderReranker()
        results = reranker.rerank_with_scores("query", sample_documents)

        # Should be sorted by rerank score (0.9, 0.7, 0.5)
        rerank_scores = [score for _, _, score in results]
        assert rerank_scores == sorted(rerank_scores, reverse=True)

    @patch("app.services.rag.reranker.CrossEncoder")
    def test_compute_relevance_score(self, mock_cross_encoder):
        """Test computing single relevance score"""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.85])
        mock_cross_encoder.return_value = mock_model

        reranker = CrossEncoderReranker()
        score = reranker.compute_relevance_score("cancer treatment", "Immunotherapy for cancer")

        assert score == 0.85
        mock_model.predict.assert_called_once_with([["cancer treatment", "Immunotherapy for cancer"]])

    @patch("app.services.rag.reranker.CrossEncoder")
    def test_batch_score(self, mock_cross_encoder):
        """Test batch scoring"""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.9, 0.8])
        mock_cross_encoder.return_value = mock_model

        reranker = CrossEncoderReranker()
        scores = reranker.batch_score(
            queries=["query1", "query2"],
            documents=["doc1", "doc2"],
        )

        assert isinstance(scores, np.ndarray)
        assert len(scores) == 2

    def test_batch_score_length_mismatch(self):
        """Test batch_score raises error for length mismatch"""
        reranker = CrossEncoderReranker()

        with pytest.raises(ValueError, match="same length"):
            reranker.batch_score(
                queries=["q1", "q2"],
                documents=["d1"],
            )


class TestTwoStageRetriever:
    """Tests for TwoStageRetriever"""

    @pytest.fixture
    def mock_retriever_func(self):
        """Create mock retriever function"""
        def retriever(query, top_k, filter_metadata=None):
            return [
                RetrievedDocument(
                    chunk_id=f"doc_{i}",
                    pmid=f"1234{i}",
                    title=f"Paper {i}",
                    content=f"Content {i}",
                    section="abstract",
                    score=0.9 - i * 0.1,
                    metadata={},
                )
                for i in range(min(top_k, 5))
            ]
        return retriever

    def test_init_default(self):
        """Test default initialization"""
        func = MagicMock()
        retriever = TwoStageRetriever(retriever_func=func)

        assert retriever._retriever_func is func
        assert retriever._reranker is None
        assert retriever.initial_k == 50
        assert retriever.final_k == 10

    def test_init_custom(self):
        """Test custom initialization"""
        func = MagicMock()
        reranker = MagicMock()
        retriever = TwoStageRetriever(
            retriever_func=func,
            reranker=reranker,
            initial_k=100,
            final_k=20,
        )

        assert retriever.initial_k == 100
        assert retriever.final_k == 20
        assert retriever._reranker is reranker

    def test_reranker_property_lazy(self):
        """Test reranker property lazy initialization"""
        with patch("app.services.rag.reranker.CrossEncoderReranker") as mock_class:
            mock_reranker = MagicMock()
            mock_class.return_value = mock_reranker

            retriever = TwoStageRetriever(retriever_func=MagicMock())
            assert retriever._reranker is None

            # Access property
            reranker = retriever.reranker
            assert reranker is mock_reranker

    def test_reranker_property_existing(self):
        """Test reranker property with existing reranker"""
        mock_reranker = MagicMock()
        retriever = TwoStageRetriever(
            retriever_func=MagicMock(),
            reranker=mock_reranker,
        )

        assert retriever.reranker is mock_reranker

    @patch("app.services.rag.reranker.CrossEncoder")
    def test_retrieve_two_stage(self, mock_cross_encoder, mock_retriever_func):
        """Test two-stage retrieval"""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        mock_cross_encoder.return_value = mock_model

        retriever = TwoStageRetriever(
            retriever_func=mock_retriever_func,
            initial_k=5,
            final_k=3,
        )

        results = retriever.retrieve("cancer treatment")

        assert len(results) == 3

    def test_retrieve_empty_initial(self):
        """Test retrieval with empty initial results"""
        def empty_retriever(query, top_k, filter_metadata=None):
            return []

        retriever = TwoStageRetriever(
            retriever_func=empty_retriever,
            initial_k=50,
            final_k=10,
        )

        results = retriever.retrieve("query")
        assert results == []

    @patch("app.services.rag.reranker.CrossEncoder")
    def test_retrieve_with_filter(self, mock_cross_encoder):
        """Test retrieval with metadata filter"""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.9])
        mock_cross_encoder.return_value = mock_model

        mock_func = MagicMock()
        mock_func.return_value = [
            RetrievedDocument(
                chunk_id="1",
                pmid="12345",
                title="Paper",
                content="Content",
                section="abstract",
                score=0.9,
                metadata={},
            )
        ]

        retriever = TwoStageRetriever(retriever_func=mock_func)
        filter_meta = {"source": "pubmed"}

        retriever.retrieve("query", filter_metadata=filter_meta)

        # Check filter was passed
        mock_func.assert_called_once()
        call_kwargs = mock_func.call_args.kwargs
        assert call_kwargs.get("filter_metadata") == filter_meta

    @patch("app.services.rag.reranker.CrossEncoder")
    def test_retrieve_custom_top_k(self, mock_cross_encoder, mock_retriever_func):
        """Test retrieval with custom top_k"""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        mock_cross_encoder.return_value = mock_model

        retriever = TwoStageRetriever(
            retriever_func=mock_retriever_func,
            initial_k=5,
            final_k=5,
        )

        results = retriever.retrieve("query", top_k=2)
        assert len(results) <= 2


class TestCreateTwoStageRetriever:
    """Tests for create_two_stage_retriever factory"""

    def test_create_with_defaults(self):
        """Test creating with default settings"""
        func = MagicMock()
        retriever = create_two_stage_retriever(func)

        assert isinstance(retriever, TwoStageRetriever)
        assert retriever.initial_k == 50
        assert retriever.final_k == 10

    def test_create_with_custom_settings(self):
        """Test creating with custom settings"""
        func = MagicMock()
        retriever = create_two_stage_retriever(
            retriever_func=func,
            initial_k=100,
            final_k=20,
            reranker_model=BIOMEDICAL_RERANKER_MODEL,
        )

        assert retriever.initial_k == 100
        assert retriever.final_k == 20
        assert retriever._reranker.config.model_name == BIOMEDICAL_RERANKER_MODEL


class TestSingletonInstance:
    """Tests for singleton instances"""

    def test_cross_encoder_reranker_singleton(self):
        """Test singleton instance exists"""
        from app.services.rag.reranker import cross_encoder_reranker
        assert cross_encoder_reranker is not None
        assert isinstance(cross_encoder_reranker, CrossEncoderReranker)
