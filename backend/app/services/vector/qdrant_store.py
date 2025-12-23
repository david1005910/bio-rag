"""
Qdrant Vector Store with Hybrid Search (Dense + Sparse)
- Dense vectors: BGE-M3 (multilingual, 1024 dim) or PubMedBERT (biomedical, 768 dim)
- Sparse vectors: BGE-M3 sparse or SPLADE-based learned sparse representation
- RRF (Reciprocal Rank Fusion) for score combination
- Supports Korean queries with multilingual embeddings
"""

import hashlib
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class HybridSearchResult:
    """Hybrid search result with detailed score breakdown"""

    doc_id: str
    content: str
    metadata: dict[str, Any]
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rrf_score: float = 0.0
    dense_rank: int = 0
    sparse_rank: int = 0


@dataclass
class HybridSearchResponse:
    """Response containing results and score visualization data"""

    results: list[HybridSearchResult]
    query: str
    total_results: int
    dense_weight: float
    sparse_weight: float
    rrf_k: int
    score_distribution: dict[str, Any] = field(default_factory=dict)


class SPLADEEncoder:
    """
    SPLADE (Sparse Lexical AnD Expansion) Encoder
    Uses MLM (Masked Language Model) for learned sparse representations
    """

    def __init__(self, model_name: str = "naver/splade-cocondenser-ensembledistil"):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._device = "cpu"

    def _load_model(self):
        """Lazy load SPLADE model"""
        if self._model is None:
            try:
                import torch
                from transformers import AutoModelForMaskedLM, AutoTokenizer

                logger.info(f"Loading SPLADE model: {self.model_name}")
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForMaskedLM.from_pretrained(self.model_name)

                # Check for GPU/MPS
                if torch.cuda.is_available():
                    self._device = "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self._device = "mps"

                self._model.to(self._device)
                self._model.eval()
                logger.info(f"SPLADE model loaded on {self._device}")

            except Exception as e:
                logger.warning(f"Failed to load SPLADE model: {e}. Falling back to BM25-style encoding.")
                self._model = "fallback"

    def encode(self, text: str, is_query: bool = False) -> tuple[list[int], list[float]]:
        """
        Encode text using SPLADE.
        Returns (indices, values) for sparse vector.
        """
        self._load_model()

        # Fallback to BM25-style if SPLADE not available
        if self._model == "fallback":
            return self._encode_bm25_style(text, is_query)

        try:
            import torch

            # Tokenize
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                max_length=256,
                truncation=True,
                padding=True,
            ).to(self._device)

            # Get MLM logits
            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits

            # SPLADE: log(1 + ReLU(logits)) * attention_mask
            # Take max over sequence length
            weights = torch.log1p(torch.relu(logits))
            weights = weights * inputs["attention_mask"].unsqueeze(-1)
            weights = torch.max(weights, dim=1)[0].squeeze()

            # Get non-zero indices and values
            nonzero_mask = weights > 0.1  # Threshold for sparsity
            indices = torch.where(nonzero_mask)[0].cpu().numpy().tolist()
            values = weights[nonzero_mask].cpu().numpy().tolist()

            # Query boosting
            if is_query:
                values = [v * 1.2 for v in values]

            return indices, values

        except Exception as e:
            logger.warning(f"SPLADE encoding failed: {e}. Using fallback.")
            return self._encode_bm25_style(text, is_query)

    def _encode_bm25_style(self, text: str, is_query: bool = False) -> tuple[list[int], list[float]]:
        """Fallback BM25-style encoding"""
        # Stopwords
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'this', 'that', 'it', 'its', 'they', 'we', 'you', 'not',
        }

        # Tokenize
        text = text.lower()
        tokens = re.findall(r'\b[a-zA-Z]{2,}\b', text)
        tokens = [t for t in tokens if t not in stopwords]

        if not tokens:
            return [], []

        # Count term frequency
        tf = Counter(tokens)
        total = len(tokens)

        indices = []
        values = []

        for token, count in tf.items():
            # Hash to vocab space
            idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % 30522  # BERT vocab size
            # TF-IDF like weight with log scaling
            weight = (1 + math.log(1 + count)) * (1 + 1/total)
            if is_query:
                weight *= 1.5
            indices.append(idx)
            values.append(weight)

        return indices, values


class BGEM3Encoder:
    """
    BGE-M3 Multilingual Encoder (BAAI/bge-m3)
    - Supports 100+ languages including Korean
    - Provides both dense (1024 dim) and sparse embeddings
    - Best for cross-lingual retrieval
    """

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model_name = model_name
        self._model = None
        self.dimension = 1024  # BGE-M3 dimension

    def _load_model(self):
        """Lazy load BGE-M3 model"""
        if self._model is None:
            try:
                from FlagEmbedding import BGEM3FlagModel

                logger.info(f"Loading BGE-M3 model: {self.model_name}")
                self._model = BGEM3FlagModel(
                    self.model_name,
                    use_fp16=True,  # Use FP16 for efficiency
                )
                logger.info("BGE-M3 model loaded successfully")

            except Exception as e:
                logger.error(f"Failed to load BGE-M3: {e}")
                raise

    def encode_dense(self, texts: str | list[str]) -> list[list[float]]:
        """Get dense embeddings"""
        self._load_model()

        if isinstance(texts, str):
            texts = [texts]

        # BGE-M3 encode returns dict with 'dense_vecs'
        embeddings = self._model.encode(
            texts,
            batch_size=8,
            max_length=512,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

        return embeddings["dense_vecs"].tolist()

    def encode_sparse(self, texts: str | list[str]) -> list[tuple[list[int], list[float]]]:
        """Get sparse embeddings (lexical weights)"""
        self._load_model()

        if isinstance(texts, str):
            texts = [texts]

        # BGE-M3 encode returns dict with 'lexical_weights'
        embeddings = self._model.encode(
            texts,
            batch_size=8,
            max_length=512,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        results = []
        for lexical_weight in embeddings["lexical_weights"]:
            # lexical_weight is dict: {token_id: weight}
            indices = list(lexical_weight.keys())
            values = list(lexical_weight.values())
            results.append((indices, values))

        return results

    def encode(self, texts: str | list[str]) -> dict[str, Any]:
        """Get both dense and sparse embeddings"""
        self._load_model()

        if isinstance(texts, str):
            texts = [texts]

        embeddings = self._model.encode(
            texts,
            batch_size=8,
            max_length=512,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        dense_vecs = embeddings["dense_vecs"].tolist()
        sparse_vecs = []
        for lexical_weight in embeddings["lexical_weights"]:
            indices = list(lexical_weight.keys())
            values = list(lexical_weight.values())
            sparse_vecs.append((indices, values))

        return {
            "dense": dense_vecs,
            "sparse": sparse_vecs,
        }


class PubMedBERTEncoder:
    """
    PubMedBERT Dense Encoder for biomedical domain
    Uses microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
    """

    def __init__(self, model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._device = "cpu"
        self.dimension = 768  # BERT base dimension

    def _load_model(self):
        """Lazy load PubMedBERT model"""
        if self._model is None:
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer

                logger.info(f"Loading PubMedBERT model: {self.model_name}")
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModel.from_pretrained(self.model_name)

                # Check for GPU/MPS
                if torch.cuda.is_available():
                    self._device = "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self._device = "mps"

                self._model.to(self._device)
                self._model.eval()
                logger.info(f"PubMedBERT model loaded on {self._device}")

            except Exception as e:
                logger.error(f"Failed to load PubMedBERT: {e}")
                raise

    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling with attention mask"""
        import torch
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def encode(self, text: str) -> list[float]:
        """Encode single text to dense vector"""
        self._load_model()

        import torch
        import torch.nn.functional as F

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        embedding = self._mean_pooling(outputs, inputs["attention_mask"])
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding.cpu().numpy().tolist()[0]

    def encode_batch(self, texts: list[str], batch_size: int = 8) -> list[list[float]]:
        """Encode batch of texts to dense vectors"""
        self._load_model()

        import torch
        import torch.nn.functional as F

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            inputs = self._tokenizer(
                batch,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True,
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)

            embeddings = self._mean_pooling(outputs, inputs["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.extend(embeddings.cpu().numpy().tolist())

        return all_embeddings


class QdrantHybridStore:
    """
    Qdrant vector store with hybrid search capabilities

    - Dense: BGE-M3 (1024 dim, multilingual) or PubMedBERT (768 dim, biomedical)
    - Sparse: BGE-M3 sparse or SPLADE for learned sparse representations
    - Fusion: RRF (Reciprocal Rank Fusion)
    """

    def __init__(
        self,
        collection_name: str | None = None,
        host: str | None = None,
        port: int | None = None,
        api_key: str | None = None,
        use_memory: bool | None = None,
        use_multilingual: bool = True,  # Use BGE-M3 for multilingual support
    ):
        self.collection_name = collection_name or settings.QDRANT_COLLECTION
        self.host = host or settings.QDRANT_HOST
        self.port = port or settings.QDRANT_PORT
        self.api_key = api_key or settings.QDRANT_API_KEY
        self.use_memory = use_memory if use_memory is not None else settings.QDRANT_USE_MEMORY
        self.use_multilingual = use_multilingual

        self._client: QdrantClient | None = None
        self._bgem3_encoder: BGEM3Encoder | None = None
        self._dense_encoder: PubMedBERTEncoder | None = None
        self._sparse_encoder: SPLADEEncoder | None = None

        # Hybrid search config
        self.dense_weight = 0.7
        self.sparse_weight = 0.3
        self.rrf_k = 60  # RRF constant

    @property
    def dense_dim(self) -> int:
        """Embedding dimension based on model"""
        return 1024 if self.use_multilingual else 768

    @property
    def client(self) -> QdrantClient:
        """Get or create Qdrant client"""
        if self._client is None:
            if self.use_memory:
                self._client = QdrantClient(":memory:")
                logger.info("Qdrant client initialized (in-memory mode)")
            else:
                self._client = QdrantClient(
                    host=self.host,
                    port=self.port,
                    api_key=self.api_key if self.api_key else None,
                )
                logger.info(f"Qdrant client initialized: {self.host}:{self.port}")

            # Ensure collection exists
            self._ensure_collection()

        return self._client

    @property
    def bgem3_encoder(self) -> BGEM3Encoder:
        """Get or create BGE-M3 encoder (multilingual)"""
        if self._bgem3_encoder is None:
            self._bgem3_encoder = BGEM3Encoder()
        return self._bgem3_encoder

    @property
    def dense_encoder(self) -> PubMedBERTEncoder:
        """Get or create PubMedBERT encoder (biomedical)"""
        if self._dense_encoder is None:
            self._dense_encoder = PubMedBERTEncoder()
        return self._dense_encoder

    @property
    def sparse_encoder(self) -> SPLADEEncoder:
        """Get or create SPLADE encoder"""
        if self._sparse_encoder is None:
            self._sparse_encoder = SPLADEEncoder()
        return self._sparse_encoder

    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist"""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)

            if not exists:
                model_name = "BGE-M3 (multilingual)" if self.use_multilingual else "PubMedBERT"
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "dense": models.VectorParams(
                            size=self.dense_dim,  # 1024 for BGE-M3, 768 for PubMedBERT
                            distance=models.Distance.COSINE,
                        )
                    },
                    sparse_vectors_config={
                        "sparse": models.SparseVectorParams(
                            index=models.SparseIndexParams(on_disk=False)
                        )
                    },
                )
                logger.info(f"Created Qdrant collection: {self.collection_name} ({model_name})")
            else:
                logger.info(f"Qdrant collection exists: {self.collection_name}")

        except UnexpectedResponse as e:
            logger.error(f"Failed to create collection: {e}")
            raise

    def _get_dense_embedding(self, text: str) -> list[float]:
        """Get dense embedding using BGE-M3 (multilingual) or PubMedBERT"""
        if self.use_multilingual:
            return self.bgem3_encoder.encode_dense(text)[0]
        return self.dense_encoder.encode(text)

    def _get_dense_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Get dense embeddings for batch"""
        if self.use_multilingual:
            return self.bgem3_encoder.encode_dense(texts)
        return self.dense_encoder.encode_batch(texts)

    def _get_sparse_vector(self, text: str, is_query: bool = False) -> tuple[list[int], list[float]]:
        """Get sparse vector using BGE-M3 or SPLADE"""
        if self.use_multilingual:
            result = self.bgem3_encoder.encode_sparse(text)
            return result[0] if result else ([], [])
        return self.sparse_encoder.encode(text, is_query=is_query)

    def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add single document with both dense and sparse vectors"""
        # Get dense embedding
        dense_vector = self._get_dense_embedding(content)

        # Get sparse vector
        sparse_indices, sparse_values = self._get_sparse_vector(content)

        # Upsert to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=self._hash_id(doc_id),
                    vector={
                        "dense": dense_vector,
                        "sparse": models.SparseVector(
                            indices=sparse_indices,
                            values=sparse_values,
                        ),
                    },
                    payload={
                        "doc_id": doc_id,
                        "content": content,
                        **(metadata or {}),
                    },
                )
            ],
        )

    def add_documents_batch(
        self,
        documents: list[dict[str, Any]],
        batch_size: int = 100,
    ) -> int:
        """
        Add batch of documents.
        Each document should have: doc_id, content, and optional metadata.
        """
        total_added = 0

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            # Get contents for batch embedding
            contents = [doc.get("content", "") for doc in batch]
            dense_vectors = self._get_dense_embeddings_batch(contents)

            points = []
            for j, doc in enumerate(batch):
                doc_id = doc.get("doc_id", f"doc_{i+j}")
                content = doc.get("content", "")
                metadata = {k: v for k, v in doc.items() if k not in ["doc_id", "content"]}

                # Get sparse vector
                sparse_indices, sparse_values = self._get_sparse_vector(content)

                points.append(
                    models.PointStruct(
                        id=self._hash_id(doc_id),
                        vector={
                            "dense": dense_vectors[j],
                            "sparse": models.SparseVector(
                                indices=sparse_indices,
                                values=sparse_values,
                            ),
                        },
                        payload={
                            "doc_id": doc_id,
                            "content": content,
                            **metadata,
                        },
                    )
                )

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            total_added += len(points)
            logger.info(f"Added batch {i//batch_size + 1}: {len(points)} documents")

        return total_added

    def _hash_id(self, doc_id: str) -> int:
        """Convert string ID to integer for Qdrant"""
        return int(hashlib.md5(doc_id.encode()).hexdigest()[:16], 16)

    def search_dense(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[tuple[dict[str, Any], float]]:
        """Dense vector search (semantic)"""
        query_vector = self._get_dense_embedding(query)

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using="dense",
            limit=top_k,
            with_payload=True,
        )

        return [(hit.payload, hit.score) for hit in results.points]

    def search_sparse(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[tuple[dict[str, Any], float]]:
        """Sparse vector search (lexical)"""
        sparse_indices, sparse_values = self._get_sparse_vector(query, is_query=True)

        if not sparse_indices:
            return []

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=models.SparseVector(
                indices=sparse_indices,
                values=sparse_values,
            ),
            using="sparse",
            limit=top_k,
            with_payload=True,
        )

        return [(hit.payload, hit.score) for hit in results.points]

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        dense_weight: float | None = None,
        sparse_weight: float | None = None,
        rrf_k: int | None = None,
    ) -> HybridSearchResponse:
        """
        Hybrid search combining dense and sparse vectors with RRF fusion.

        Args:
            query: Search query
            top_k: Number of final results
            dense_weight: Weight for dense search (0-1)
            sparse_weight: Weight for sparse search (0-1)
            rrf_k: RRF constant (higher = smoother fusion)

        Returns:
            HybridSearchResponse with results and score visualization data
        """
        dense_weight = dense_weight or self.dense_weight
        sparse_weight = sparse_weight or self.sparse_weight
        rrf_k = rrf_k or self.rrf_k

        # Get more candidates for fusion
        candidate_k = min(top_k * 3, 100)

        # Dense search
        dense_results = self.search_dense(query, top_k=candidate_k)

        # Sparse search
        sparse_results = self.search_sparse(query, top_k=candidate_k)

        # Build lookup maps
        dense_map: dict[str, tuple[float, int]] = {}  # doc_id -> (score, rank)
        for rank, (payload, score) in enumerate(dense_results, 1):
            doc_id = payload.get("doc_id", "")
            dense_map[doc_id] = (score, rank)

        sparse_map: dict[str, tuple[float, int]] = {}  # doc_id -> (score, rank)
        for rank, (payload, score) in enumerate(sparse_results, 1):
            doc_id = payload.get("doc_id", "")
            sparse_map[doc_id] = (score, rank)

        # Collect all unique documents
        all_docs: dict[str, dict[str, Any]] = {}
        for payload, _ in dense_results:
            doc_id = payload.get("doc_id", "")
            all_docs[doc_id] = payload
        for payload, _ in sparse_results:
            doc_id = payload.get("doc_id", "")
            if doc_id not in all_docs:
                all_docs[doc_id] = payload

        # Calculate RRF scores
        results: list[HybridSearchResult] = []

        for doc_id, payload in all_docs.items():
            dense_score, dense_rank = dense_map.get(doc_id, (0.0, candidate_k + 1))
            sparse_score, sparse_rank = sparse_map.get(doc_id, (0.0, candidate_k + 1))

            # RRF formula: weight * (1 / (k + rank))
            rrf_dense = dense_weight * (1.0 / (rrf_k + dense_rank))
            rrf_sparse = sparse_weight * (1.0 / (rrf_k + sparse_rank))
            rrf_score = rrf_dense + rrf_sparse

            results.append(HybridSearchResult(
                doc_id=doc_id,
                content=payload.get("content", ""),
                metadata={k: v for k, v in payload.items() if k not in ["doc_id", "content"]},
                dense_score=dense_score,
                sparse_score=sparse_score,
                rrf_score=rrf_score,
                dense_rank=dense_rank if dense_rank <= candidate_k else 0,
                sparse_rank=sparse_rank if sparse_rank <= candidate_k else 0,
            ))

        # Sort by RRF score
        results.sort(key=lambda x: x.rrf_score, reverse=True)
        results = results[:top_k]

        # Calculate score distribution for visualization
        score_distribution = self._calculate_score_distribution(
            results, dense_weight, sparse_weight
        )

        return HybridSearchResponse(
            results=results,
            query=query,
            total_results=len(results),
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            rrf_k=rrf_k,
            score_distribution=score_distribution,
        )

    def _calculate_score_distribution(
        self,
        results: list[HybridSearchResult],
        dense_weight: float,
        sparse_weight: float,
    ) -> dict[str, Any]:
        """Calculate score distribution for visualization"""
        if not results:
            return {}

        dense_scores = [r.dense_score for r in results]
        sparse_scores = [r.sparse_score for r in results]
        rrf_scores = [r.rrf_score for r in results]

        # Calculate contribution ratios
        contributions = []
        for r in results:
            total = r.dense_score + r.sparse_score
            if total > 0:
                dense_contrib = (r.dense_score / total) * 100
                sparse_contrib = (r.sparse_score / total) * 100
            else:
                dense_contrib = 50.0
                sparse_contrib = 50.0
            contributions.append({
                "doc_id": r.doc_id,
                "dense_contribution": round(dense_contrib, 1),
                "sparse_contribution": round(sparse_contrib, 1),
            })

        return {
            "dense_scores": {
                "min": round(min(dense_scores), 4) if dense_scores else 0,
                "max": round(max(dense_scores), 4) if dense_scores else 0,
                "avg": round(sum(dense_scores) / len(dense_scores), 4) if dense_scores else 0,
            },
            "sparse_scores": {
                "min": round(min(sparse_scores), 4) if sparse_scores else 0,
                "max": round(max(sparse_scores), 4) if sparse_scores else 0,
                "avg": round(sum(sparse_scores) / len(sparse_scores), 4) if sparse_scores else 0,
            },
            "rrf_scores": {
                "min": round(min(rrf_scores), 4) if rrf_scores else 0,
                "max": round(max(rrf_scores), 4) if rrf_scores else 0,
                "avg": round(sum(rrf_scores) / len(rrf_scores), 4) if rrf_scores else 0,
            },
            "contributions": contributions,
            "config": {
                "dense_weight": dense_weight,
                "sparse_weight": sparse_weight,
            },
        }

    def count(self) -> int:
        """Get document count"""
        info = self.client.get_collection(self.collection_name)
        return info.points_count

    def delete_collection(self) -> None:
        """Delete collection"""
        self.client.delete_collection(self.collection_name)
        logger.info(f"Deleted collection: {self.collection_name}")

    def clear(self) -> None:
        """Clear all documents (recreate collection)"""
        try:
            self.delete_collection()
        except Exception:
            pass
        self._ensure_collection()


# Singleton instance
qdrant_store = QdrantHybridStore()
