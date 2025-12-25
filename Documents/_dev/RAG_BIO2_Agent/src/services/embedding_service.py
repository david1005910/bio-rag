import os
import hashlib
import numpy as np
from typing import List, Optional, Literal
from abc import ABC, abstractmethod

EmbeddingModelType = Literal["pubmedbert", "openai", "simple"]

class BaseEmbeddingGenerator(ABC):
    @abstractmethod
    def encode(self, text: str) -> np.ndarray:
        pass
    
    @abstractmethod
    def batch_encode(self, texts: List[str]) -> np.ndarray:
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

class SimpleEmbedding(BaseEmbeddingGenerator):
    """Simple hash-based embedding generator that works without external dependencies.
    Uses word hashing and n-gram features to create semantic-ish embeddings.
    Not as accurate as neural embeddings but works for basic functionality.
    """
    def __init__(self, dimension: int = 768):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _hash_word(self, word: str, seed: int = 0) -> np.ndarray:
        """Hash a word to a vector using multiple hash functions."""
        h = hashlib.sha256((str(seed) + word.lower()).encode()).hexdigest()
        # Convert hex to floats
        values = []
        for i in range(0, min(len(h), 64), 2):
            values.append((int(h[i:i+2], 16) - 127.5) / 127.5)
        return np.array(values)

    def encode(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            return np.zeros(self._dimension)

        # Tokenize
        words = text.lower().split()
        if not words:
            return np.zeros(self._dimension)

        # Initialize embedding
        embedding = np.zeros(self._dimension)

        # Add word embeddings
        for i, word in enumerate(words):
            word_vec = np.zeros(self._dimension)
            for seed in range(24):  # Use multiple hash functions
                h = self._hash_word(word, seed)
                start = seed * 32
                word_vec[start:start + len(h)] = h
            # Weight by position (words at beginning are more important)
            weight = 1.0 / (1 + i * 0.1)
            embedding += word_vec * weight

        # Add n-gram features
        for n in [2, 3]:
            for i in range(len(words) - n + 1):
                ngram = ' '.join(words[i:i+n])
                ngram_vec = np.zeros(self._dimension)
                for seed in range(8):
                    h = self._hash_word(ngram, seed + 100)
                    start = seed * 32 + 512
                    end = min(start + len(h), self._dimension)
                    ngram_vec[start:end] = h[:end-start]
                embedding += ngram_vec * 0.5

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def batch_encode(self, texts: List[str]) -> np.ndarray:
        return np.array([self.encode(text) for text in texts])


class PubMedBERTEmbedding(BaseEmbeddingGenerator):
    _model = None
    _tokenizer = None
    
    def __init__(self, model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"):
        self.model_name = model_name
        self._dimension = 768
        self._load_model()
    
    def _load_model(self):
        if PubMedBERTEmbedding._model is None:
            try:
                from transformers import AutoTokenizer, AutoModel
                import torch
                
                PubMedBERTEmbedding._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                PubMedBERTEmbedding._model = AutoModel.from_pretrained(self.model_name)
                PubMedBERTEmbedding._model.eval()
                
                self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                PubMedBERTEmbedding._model.to(self._device)
            except Exception as e:
                print(f"Error loading PubMedBERT model: {e}")
                raise
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    def encode(self, text: str, max_length: int = 512) -> np.ndarray:
        import torch
        
        if not text or not text.strip():
            return np.zeros(self._dimension)
        
        inputs = PubMedBERTEmbedding._tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        device = next(PubMedBERTEmbedding._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = PubMedBERTEmbedding._model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embedding.squeeze()
    
    def batch_encode(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = [self.encode(text) for text in batch]
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)

class OpenAIEmbedding(BaseEmbeddingGenerator):
    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self._dimension = 1536
        self._client = None
        self._init_client()
    
    def _init_client(self):
        try:
            from openai import OpenAI
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                self._client = OpenAI(api_key=api_key)
            else:
                base_url = os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL")
                ai_key = os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY")
                if base_url and ai_key:
                    self._client = OpenAI(api_key=ai_key, base_url=base_url)
        except Exception as e:
            print(f"Error initializing OpenAI client for embeddings: {e}")
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    def encode(self, text: str) -> np.ndarray:
        if not self._client:
            raise ValueError("OpenAI client not initialized. Please set OPENAI_API_KEY.")
        
        if not text or not text.strip():
            return np.zeros(self._dimension)
        
        response = self._client.embeddings.create(
            input=text,
            model=self.model
        )
        
        return np.array(response.data[0].embedding)
    
    def batch_encode(self, texts: List[str]) -> np.ndarray:
        if not self._client:
            raise ValueError("OpenAI client not initialized. Please set OPENAI_API_KEY.")
        
        embeddings = []
        for text in texts:
            if text and text.strip():
                embedding = self.encode(text)
            else:
                embedding = np.zeros(self._dimension)
            embeddings.append(embedding)
        
        return np.array(embeddings)

class EmbeddingService:
    def __init__(self, model_type: EmbeddingModelType = "pubmedbert"):
        self.model_type = model_type
        self._generator: Optional[BaseEmbeddingGenerator] = None
        self._initialize()

    def _initialize(self):
        if self.model_type == "simple":
            self._generator = SimpleEmbedding()
            print("Using simple hash-based embeddings")
            return

        if self.model_type == "pubmedbert":
            try:
                self._generator = PubMedBERTEmbedding()
                print("Using PubMedBERT embeddings")
                return
            except Exception as e:
                print(f"Failed to load PubMedBERT: {e}")

        if self.model_type in ["pubmedbert", "openai"]:
            try:
                self._generator = OpenAIEmbedding()
                self.model_type = "openai"
                print("Using OpenAI embeddings")
                return
            except Exception as e:
                print(f"Failed to load OpenAI embeddings: {e}")

        # Final fallback to simple embeddings
        print("Falling back to simple hash-based embeddings")
        self._generator = SimpleEmbedding()
        self.model_type = "simple"
    
    @property
    def dimension(self) -> int:
        return self._generator.dimension
    
    def encode(self, text: str) -> np.ndarray:
        return self._generator.encode(text)
    
    def batch_encode(self, texts: List[str]) -> np.ndarray:
        return self._generator.batch_encode(texts)
    
    def switch_model(self, model_type: EmbeddingModelType):
        self.model_type = model_type
        self._initialize()
