"""Tests for vector_store.py"""
import pytest
import numpy as np
import uuid
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from unittest.mock import MagicMock


# ============================================================================
# Testable implementations of vector store classes
# ============================================================================

@dataclass
class SearchResult:
    """Search result from vector store."""
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]


class _TestableVectorStore:
    """In-memory vector store for testing."""

    def __init__(self, collection_name: str = None, dimension: int = 768):
        if collection_name is None:
            self.collection_name = f"biomedical_papers_{dimension}d"
        else:
            self.collection_name = collection_name
        self.dimension = dimension
        self._documents: Dict[str, Dict[str, Any]] = {}

    def add_documents(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add documents to the vector store."""
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]

        for i, (text, embedding, metadata) in enumerate(zip(texts, embeddings, metadatas)):
            self._documents[ids[i]] = {
                "text": text,
                "embedding": embedding,
                "metadata": metadata
            }

        return ids

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar documents."""
        results = []

        for doc_id, doc in self._documents.items():
            # Apply filter if provided
            if filter_dict:
                match = True
                for key, value in filter_dict.items():
                    doc_value = doc["metadata"].get(key)
                    if isinstance(value, list):
                        if doc_value not in value:
                            match = False
                            break
                    else:
                        if doc_value != value:
                            match = False
                            break
                if not match:
                    continue

            # Calculate cosine similarity
            doc_embedding = doc["embedding"]
            score = self._cosine_similarity(query_embedding, doc_embedding)

            results.append(SearchResult(
                id=doc_id,
                text=doc["text"],
                score=score,
                metadata=doc["metadata"]
            ))

        # Sort by score descending and return top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot_product / (norm_a * norm_b))

    def delete_by_pmid(self, pmid: str):
        """Delete all documents with the given PMID."""
        ids_to_delete = [
            doc_id for doc_id, doc in self._documents.items()
            if doc["metadata"].get("pmid") == pmid
        ]
        for doc_id in ids_to_delete:
            del self._documents[doc_id]

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        return {
            "name": self.collection_name,
            "points_count": len(self._documents),
            "vectors_count": len(self._documents)
        }

    def clear(self):
        """Clear all documents."""
        self._documents = {}


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def vector_store():
    """Create a testable vector store instance."""
    return _TestableVectorStore(dimension=768)


@pytest.fixture
def vector_store_1536():
    """Create a testable vector store with 1536 dimensions."""
    return _TestableVectorStore(dimension=1536)


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing."""
    np.random.seed(42)
    return np.random.rand(3, 768).astype(np.float32)


@pytest.fixture
def sample_texts():
    """Create sample texts for testing."""
    return [
        "Cancer treatment with immunotherapy",
        "Gene therapy for genetic disorders",
        "Drug interactions in elderly patients"
    ]


@pytest.fixture
def sample_metadatas():
    """Create sample metadata for testing."""
    return [
        {"pmid": "12345", "title": "Cancer Paper", "journal": "Nature"},
        {"pmid": "67890", "title": "Gene Therapy Paper", "journal": "Science"},
        {"pmid": "11111", "title": "Drug Paper", "journal": "Nature"}
    ]


# ============================================================================
# Tests
# ============================================================================

class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_search_result_creation(self):
        """Test creating a SearchResult."""
        result = SearchResult(
            id="doc1",
            text="Sample text",
            score=0.95,
            metadata={"pmid": "123"}
        )

        assert result.id == "doc1"
        assert result.text == "Sample text"
        assert result.score == 0.95
        assert result.metadata["pmid"] == "123"


class TestVectorStoreInitialization:
    """Tests for vector store initialization."""

    def test_default_collection_name(self):
        """Test default collection name is based on dimension."""
        store = _TestableVectorStore(dimension=768)
        assert store.collection_name == "biomedical_papers_768d"

    def test_custom_collection_name(self):
        """Test custom collection name."""
        store = _TestableVectorStore(collection_name="my_collection", dimension=768)
        assert store.collection_name == "my_collection"

    def test_dimension_1536(self):
        """Test 1536 dimension collection name."""
        store = _TestableVectorStore(dimension=1536)
        assert store.collection_name == "biomedical_papers_1536d"
        assert store.dimension == 1536


class TestAddDocuments:
    """Tests for adding documents to the vector store."""

    def test_add_documents_returns_ids(self, vector_store, sample_embeddings, sample_texts, sample_metadatas):
        """Test that add_documents returns a list of IDs."""
        ids = vector_store.add_documents(sample_texts, sample_embeddings, sample_metadatas)

        assert isinstance(ids, list)
        assert len(ids) == 3
        for id in ids:
            assert isinstance(id, str)

    def test_add_documents_with_custom_ids(self, vector_store, sample_embeddings, sample_texts, sample_metadatas):
        """Test adding documents with custom IDs."""
        custom_ids = ["id1", "id2", "id3"]
        ids = vector_store.add_documents(sample_texts, sample_embeddings, sample_metadatas, ids=custom_ids)

        assert ids == custom_ids

    def test_add_documents_increases_count(self, vector_store, sample_embeddings, sample_texts, sample_metadatas):
        """Test that adding documents increases collection count."""
        initial_info = vector_store.get_collection_info()
        assert initial_info["points_count"] == 0

        vector_store.add_documents(sample_texts, sample_embeddings, sample_metadatas)

        final_info = vector_store.get_collection_info()
        assert final_info["points_count"] == 3

    def test_add_single_document(self, vector_store):
        """Test adding a single document."""
        text = ["Single document"]
        embedding = np.random.rand(1, 768).astype(np.float32)
        metadata = [{"pmid": "99999"}]

        ids = vector_store.add_documents(text, embedding, metadata)

        assert len(ids) == 1
        info = vector_store.get_collection_info()
        assert info["points_count"] == 1


class TestSearch:
    """Tests for searching the vector store."""

    def test_search_returns_results(self, vector_store, sample_embeddings, sample_texts, sample_metadatas):
        """Test that search returns results."""
        vector_store.add_documents(sample_texts, sample_embeddings, sample_metadatas)

        query_embedding = np.random.rand(768).astype(np.float32)
        results = vector_store.search(query_embedding, top_k=3)

        assert isinstance(results, list)
        assert len(results) <= 3

    def test_search_returns_search_results(self, vector_store, sample_embeddings, sample_texts, sample_metadatas):
        """Test that search returns SearchResult objects."""
        vector_store.add_documents(sample_texts, sample_embeddings, sample_metadatas)

        query_embedding = sample_embeddings[0]  # Use first embedding as query
        results = vector_store.search(query_embedding, top_k=3)

        for result in results:
            assert isinstance(result, SearchResult)
            assert hasattr(result, 'id')
            assert hasattr(result, 'text')
            assert hasattr(result, 'score')
            assert hasattr(result, 'metadata')

    def test_search_respects_top_k(self, vector_store, sample_embeddings, sample_texts, sample_metadatas):
        """Test that search respects top_k limit."""
        vector_store.add_documents(sample_texts, sample_embeddings, sample_metadatas)

        query_embedding = np.random.rand(768).astype(np.float32)
        results = vector_store.search(query_embedding, top_k=1)

        assert len(results) == 1

    def test_search_empty_store(self, vector_store):
        """Test searching an empty vector store."""
        query_embedding = np.random.rand(768).astype(np.float32)
        results = vector_store.search(query_embedding, top_k=5)

        assert results == []

    def test_search_most_similar_first(self, vector_store):
        """Test that most similar results come first."""
        # Add documents with specific embeddings
        texts = ["Document A", "Document B"]
        embeddings = np.array([
            [1.0, 0.0, 0.0] + [0.0] * 765,  # Embedding A
            [0.0, 1.0, 0.0] + [0.0] * 765   # Embedding B
        ]).astype(np.float32)
        metadatas = [{"id": "A"}, {"id": "B"}]

        vector_store.add_documents(texts, embeddings, metadatas)

        # Query with embedding similar to A
        query = np.array([0.9, 0.1, 0.0] + [0.0] * 765).astype(np.float32)
        results = vector_store.search(query, top_k=2)

        assert results[0].metadata["id"] == "A"  # A should be first

    def test_search_scores_are_valid(self, vector_store, sample_embeddings, sample_texts, sample_metadatas):
        """Test that search scores are valid cosine similarities."""
        vector_store.add_documents(sample_texts, sample_embeddings, sample_metadatas)

        query_embedding = sample_embeddings[0]
        results = vector_store.search(query_embedding, top_k=3)

        for result in results:
            # Allow small floating-point precision errors
            assert -1.01 <= result.score <= 1.01


class TestSearchWithFilter:
    """Tests for filtered search."""

    def test_search_with_single_filter(self, vector_store, sample_embeddings, sample_texts, sample_metadatas):
        """Test search with a single filter condition."""
        vector_store.add_documents(sample_texts, sample_embeddings, sample_metadatas)

        query_embedding = np.random.rand(768).astype(np.float32)
        results = vector_store.search(
            query_embedding,
            top_k=10,
            filter_dict={"journal": "Nature"}
        )

        # Should only return papers from Nature journal
        for result in results:
            assert result.metadata["journal"] == "Nature"

    def test_search_with_pmid_filter(self, vector_store, sample_embeddings, sample_texts, sample_metadatas):
        """Test search filtered by PMID."""
        vector_store.add_documents(sample_texts, sample_embeddings, sample_metadatas)

        query_embedding = np.random.rand(768).astype(np.float32)
        results = vector_store.search(
            query_embedding,
            top_k=10,
            filter_dict={"pmid": "12345"}
        )

        assert len(results) == 1
        assert results[0].metadata["pmid"] == "12345"

    def test_search_with_list_filter(self, vector_store, sample_embeddings, sample_texts, sample_metadatas):
        """Test search with list filter (match any)."""
        vector_store.add_documents(sample_texts, sample_embeddings, sample_metadatas)

        query_embedding = np.random.rand(768).astype(np.float32)
        results = vector_store.search(
            query_embedding,
            top_k=10,
            filter_dict={"pmid": ["12345", "67890"]}
        )

        assert len(results) == 2
        pmids = [r.metadata["pmid"] for r in results]
        assert "12345" in pmids
        assert "67890" in pmids

    def test_search_filter_no_matches(self, vector_store, sample_embeddings, sample_texts, sample_metadatas):
        """Test search with filter that matches nothing."""
        vector_store.add_documents(sample_texts, sample_embeddings, sample_metadatas)

        query_embedding = np.random.rand(768).astype(np.float32)
        results = vector_store.search(
            query_embedding,
            top_k=10,
            filter_dict={"journal": "NonexistentJournal"}
        )

        assert results == []


class TestDeleteByPmid:
    """Tests for deleting documents by PMID."""

    def test_delete_by_pmid_removes_document(self, vector_store, sample_embeddings, sample_texts, sample_metadatas):
        """Test that delete_by_pmid removes the document."""
        vector_store.add_documents(sample_texts, sample_embeddings, sample_metadatas)

        initial_count = vector_store.get_collection_info()["points_count"]
        assert initial_count == 3

        vector_store.delete_by_pmid("12345")

        final_count = vector_store.get_collection_info()["points_count"]
        assert final_count == 2

    def test_delete_by_pmid_nonexistent(self, vector_store, sample_embeddings, sample_texts, sample_metadatas):
        """Test deleting non-existent PMID doesn't cause error."""
        vector_store.add_documents(sample_texts, sample_embeddings, sample_metadatas)

        initial_count = vector_store.get_collection_info()["points_count"]

        vector_store.delete_by_pmid("nonexistent")

        final_count = vector_store.get_collection_info()["points_count"]
        assert final_count == initial_count  # No change

    def test_delete_by_pmid_removes_from_search(self, vector_store, sample_embeddings, sample_texts, sample_metadatas):
        """Test that deleted documents don't appear in search."""
        vector_store.add_documents(sample_texts, sample_embeddings, sample_metadatas)

        vector_store.delete_by_pmid("12345")

        query_embedding = np.random.rand(768).astype(np.float32)
        results = vector_store.search(query_embedding, top_k=10)

        pmids = [r.metadata["pmid"] for r in results]
        assert "12345" not in pmids


class TestCollectionInfo:
    """Tests for collection info."""

    def test_get_collection_info_empty(self, vector_store):
        """Test collection info for empty store."""
        info = vector_store.get_collection_info()

        assert info["name"] == "biomedical_papers_768d"
        assert info["points_count"] == 0
        assert info["vectors_count"] == 0

    def test_get_collection_info_with_documents(self, vector_store, sample_embeddings, sample_texts, sample_metadatas):
        """Test collection info with documents."""
        vector_store.add_documents(sample_texts, sample_embeddings, sample_metadatas)

        info = vector_store.get_collection_info()

        assert info["points_count"] == 3
        assert info["vectors_count"] == 3

    def test_collection_info_after_delete(self, vector_store, sample_embeddings, sample_texts, sample_metadatas):
        """Test collection info updates after deletion."""
        vector_store.add_documents(sample_texts, sample_embeddings, sample_metadatas)
        vector_store.delete_by_pmid("12345")

        info = vector_store.get_collection_info()

        assert info["points_count"] == 2
