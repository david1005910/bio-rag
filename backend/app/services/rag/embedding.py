"""Embedding service for RAG
- OpenAI 임베딩
- HuggingFace 모델 (PubMedBERT, BioBERT 등)
- 의학/과학 특화 모델 지원
"""

import logging
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)


# 지원되는 임베딩 모델 설정
EMBEDDING_MODELS = {
    # PubMed/의학 특화 모델 (HuggingFace)
    'pubmedbert': {
        'name': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
        'description': 'PubMed 논문 특화 (PubMedBERT Full)',
        'dimension': 768,
        'type': 'huggingface'
    },
    'pubmedbert-abs': {
        'name': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
        'description': 'PubMed 초록 특화 (PubMedBERT Abstract)',
        'dimension': 768,
        'type': 'huggingface'
    },
    'biobert': {
        'name': 'dmis-lab/biobert-base-cased-v1.2',
        'description': '의학/생물학 특화 (BioBERT v1.2)',
        'dimension': 768,
        'type': 'huggingface'
    },
    'scibert': {
        'name': 'allenai/scibert_scivocab_uncased',
        'description': '과학 논문 특화 (SciBERT)',
        'dimension': 768,
        'type': 'huggingface'
    },
    'biolinkbert': {
        'name': 'michiyasunaga/BioLinkBERT-base',
        'description': '의학 문헌 링크 학습 (BioLinkBERT)',
        'dimension': 768,
        'type': 'huggingface'
    },
    # 일반 모델
    'bert-base': {
        'name': 'bert-base-uncased',
        'description': 'BERT 기본 모델',
        'dimension': 768,
        'type': 'huggingface'
    },
    'minilm': {
        'name': 'sentence-transformers/all-MiniLM-L6-v2',
        'description': '일반 목적 (가볍고 빠름)',
        'dimension': 384,
        'type': 'huggingface'
    },
    # OpenAI API 모델
    'openai-small': {
        'name': 'text-embedding-3-small',
        'description': 'OpenAI 임베딩 (빠름, 저렴)',
        'dimension': 1536,
        'type': 'openai'
    },
    'openai-large': {
        'name': 'text-embedding-3-large',
        'description': 'OpenAI 임베딩 (고성능)',
        'dimension': 3072,
        'type': 'openai'
    },
    'openai-ada': {
        'name': 'text-embedding-ada-002',
        'description': 'OpenAI Ada (레거시)',
        'dimension': 1536,
        'type': 'openai'
    }
}


class BaseEmbedding(ABC):
    """임베딩 베이스 클래스"""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """단일 텍스트 임베딩"""
        pass

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """배치 텍스트 임베딩"""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """임베딩 차원"""
        pass


class OpenAIEmbedding(BaseEmbedding):
    """OpenAI API를 사용한 임베딩"""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None
    ) -> None:
        self.model = model
        self._api_key = api_key or settings.OPENAI_API_KEY
        self._client = None
        self._dimension = EMBEDDING_MODELS.get(
            'openai-small' if 'small' in model else 'openai-large',
            {'dimension': 1536}
        )['dimension']

    @property
    def client(self):
        """OpenAI 클라이언트 lazy initialization"""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key)
            logger.info(f"OpenAI embedding client initialized: {self.model}")
        return self._client

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        """단일 텍스트 임베딩"""
        if not text.strip():
            return []

        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model,
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """배치 텍스트 임베딩"""
        if not texts:
            return []

        filtered_texts = [t for t in texts if t.strip()]
        if not filtered_texts:
            return []

        try:
            response = self.client.embeddings.create(
                input=filtered_texts,
                model=self.model,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"OpenAI batch embedding error: {e}")
            raise


class HuggingFaceEmbedding(BaseEmbedding):
    """HuggingFace Transformers를 사용한 임베딩"""

    def __init__(
        self,
        model_name: str = "dmis-lab/biobert-base-cased-v1.2",
        device: str = 'cpu'
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None
        self._dimension = 768  # BERT 기본

    def _load_model(self):
        """모델 lazy loading"""
        if self._model is None:
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer

                logger.info(f"Loading HuggingFace model: {self.model_name}")
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModel.from_pretrained(self.model_name)
                self._model.to(self.device)
                self._model.eval()

                # 차원 확인
                self._dimension = self._model.config.hidden_size
                logger.info(f"Model loaded: dim={self._dimension}")

            except Exception as e:
                logger.error(f"Failed to load model {self.model_name}: {e}")
                raise

    @property
    def dimension(self) -> int:
        return self._dimension

    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling - attention mask를 고려한 평균"""
        import torch
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def embed(self, text: str) -> list[float]:
        """단일 텍스트 임베딩"""
        if not text.strip():
            return []

        self._load_model()

        import torch
        import torch.nn.functional as F

        try:
            encoded_input = self._tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

            with torch.no_grad():
                model_output = self._model(**encoded_input)

            embedding = self._mean_pooling(model_output, encoded_input['attention_mask'])
            embedding = F.normalize(embedding, p=2, dim=1)

            return embedding.cpu().numpy().tolist()[0]

        except Exception as e:
            logger.error(f"HuggingFace embedding error: {e}")
            raise

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """배치 텍스트 임베딩"""
        if not texts:
            return []

        filtered_texts = [t for t in texts if t.strip()]
        if not filtered_texts:
            return []

        self._load_model()

        import torch
        import torch.nn.functional as F

        try:
            all_embeddings = []
            batch_size = 8

            for i in range(0, len(filtered_texts), batch_size):
                batch = filtered_texts[i:i + batch_size]

                encoded_input = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
                encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

                with torch.no_grad():
                    model_output = self._model(**encoded_input)

                embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
                embeddings = F.normalize(embeddings, p=2, dim=1)
                all_embeddings.extend(embeddings.cpu().numpy().tolist())

            return all_embeddings

        except Exception as e:
            logger.error(f"HuggingFace batch embedding error: {e}")
            raise


class EmbeddingModelFactory:
    """임베딩 모델 팩토리"""

    @classmethod
    def get_available_models(cls) -> dict[str, dict[str, Any]]:
        """사용 가능한 모델 목록 반환"""
        return EMBEDDING_MODELS.copy()

    @classmethod
    def create(
        cls,
        model_type: str = 'openai-small',
        device: str = 'cpu',
        api_key: str | None = None
    ) -> BaseEmbedding:
        """
        임베딩 모델 생성

        Args:
            model_type: 모델 타입 (EMBEDDING_MODELS 키)
            device: 디바이스 ('cpu', 'cuda', 'mps')
            api_key: OpenAI API 키 (OpenAI 모델 사용 시)

        Returns:
            BaseEmbedding 인스턴스
        """
        if model_type not in EMBEDDING_MODELS:
            logger.warning(f"Unknown model: {model_type}. Using openai-small")
            model_type = 'openai-small'

        model_info = EMBEDDING_MODELS[model_type]
        model_name = model_info['name']

        logger.info(f"Creating embedding model: {model_type}")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Description: {model_info['description']}")
        logger.info(f"  Dimension: {model_info['dimension']}")

        if model_info['type'] == 'openai':
            if not api_key and not settings.OPENAI_API_KEY:
                logger.warning("OpenAI API key not found. Falling back to HuggingFace model.")
                return cls.create('biobert', device, api_key)
            return OpenAIEmbedding(model=model_name, api_key=api_key)
        else:
            try:
                return HuggingFaceEmbedding(model_name=model_name, device=device)
            except Exception as e:
                logger.error(f"Failed to load {model_type}: {e}")
                if model_type != 'bert-base':
                    logger.info("Falling back to bert-base")
                    return cls.create('bert-base', device, api_key)
                raise


# 호환성을 위한 기존 클래스 유지
class EmbeddingService(OpenAIEmbedding):
    """OpenAI Embedding service (기존 호환성)"""

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        super().__init__(model=model)


# Singleton instance
embedding_service = EmbeddingService()


@lru_cache(maxsize=1000)
def get_embedding(text: str) -> tuple[float, ...]:
    """Cached embedding function (returns tuple for hashability)"""
    embedding = embedding_service.embed(text)
    return tuple(embedding)


def embed_text(text: str) -> list[float]:
    """Embedding function to be used by retriever"""
    return list(get_embedding(text))


def create_embedding_service(
    model_type: str = 'openai-small',
    device: str = 'cpu',
    api_key: str | None = None
) -> BaseEmbedding:
    """임베딩 서비스 팩토리 함수"""
    return EmbeddingModelFactory.create(model_type, device, api_key)
