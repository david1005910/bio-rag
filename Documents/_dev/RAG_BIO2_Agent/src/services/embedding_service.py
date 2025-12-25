import os
import numpy as np
from typing import List, Optional, Literal
from abc import ABC, abstractmethod

EmbeddingModelType = Literal["pubmedbert", "openai"]

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
        if self.model_type == "pubmedbert":
            try:
                self._generator = PubMedBERTEmbedding()
            except Exception as e:
                print(f"Failed to load PubMedBERT, falling back to OpenAI: {e}")
                self._generator = OpenAIEmbedding()
                self.model_type = "openai"
        else:
            self._generator = OpenAIEmbedding()
    
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
