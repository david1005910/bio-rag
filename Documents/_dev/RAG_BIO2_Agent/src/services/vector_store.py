import os
import uuid
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class SearchResult:
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]

# Singleton Qdrant client
_qdrant_client = None
_qdrant_lock = None

def _get_qdrant_client():
    """Get or create singleton Qdrant client."""
    global _qdrant_client
    if _qdrant_client is None:
        try:
            from qdrant_client import QdrantClient

            qdrant_url = os.environ.get("QDRANT_URL")
            qdrant_api_key = os.environ.get("QDRANT_API_KEY")

            if qdrant_url:
                _qdrant_client = QdrantClient(
                    url=qdrant_url,
                    api_key=qdrant_api_key
                )
            else:
                _qdrant_client = QdrantClient(path="./qdrant_data")
        except Exception as e:
            print(f"Error initializing Qdrant: {e}")
            raise
    return _qdrant_client

class QdrantVectorStore:
    def __init__(self, collection_name: str = None, dimension: int = 768):
        if collection_name is None:
            self.collection_name = f"biomedical_papers_{dimension}d"
        else:
            self.collection_name = collection_name
        self.dimension = dimension
        self._client = _get_qdrant_client()
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Ensure the collection exists."""
        try:
            from qdrant_client.http import models

            try:
                self._client.get_collection(self.collection_name)
            except Exception:
                self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.dimension,
                        distance=models.Distance.COSINE
                    )
                )
        except Exception as e:
            print(f"Error ensuring collection: {e}")
            raise
    
    def add_documents(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        from qdrant_client.http import models
        
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        
        points = []
        for i, (text, embedding, metadata) in enumerate(zip(texts, embeddings, metadatas)):
            payload = {
                "text": text,
                **metadata
            }
            points.append(models.PointStruct(
                id=ids[i],
                vector=embedding.tolist(),
                payload=payload
            ))
        
        self._client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        return ids
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        from qdrant_client.http import models

        query_filter = None
        if filter_dict:
            conditions = []
            for key, value in filter_dict.items():
                if isinstance(value, list):
                    conditions.append(models.FieldCondition(
                        key=key,
                        match=models.MatchAny(any=value)
                    ))
                else:
                    conditions.append(models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    ))
            query_filter = models.Filter(must=conditions)

        # Use query_points for newer qdrant-client versions
        results = self._client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            limit=top_k,
            query_filter=query_filter
        )

        search_results = []
        for point in results.points:
            payload = point.payload or {}
            search_results.append(SearchResult(
                id=str(point.id),
                text=payload.get("text", ""),
                score=point.score if point.score is not None else 0.0,
                metadata={k: v for k, v in payload.items() if k != "text"}
            ))

        return search_results
    
    def delete_by_pmid(self, pmid: str):
        from qdrant_client.http import models
        
        self._client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="pmid",
                            match=models.MatchValue(value=pmid)
                        )
                    ]
                )
            )
        )
    
    def get_collection_info(self) -> Dict[str, Any]:
        info = self._client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "points_count": info.points_count,
            "vectors_count": getattr(info, 'vectors_count', info.points_count)
        }
