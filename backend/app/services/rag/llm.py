"""LLM service for RAG generation"""

import logging
from typing import AsyncIterator

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """OpenAI LLM service for RAG generation"""

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 1000,
    ) -> None:
        self.model = model or settings.OPENAI_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client: ChatOpenAI | None = None

    @property
    def client(self) -> ChatOpenAI:
        """Get LLM client (lazy initialization)"""
        if self._client is None:
            self._client = ChatOpenAI(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=settings.OPENAI_API_KEY,
            )
            logger.info(f"LLM client initialized: {self.model}")
        return self._client

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """
        Generate response from LLM

        Args:
            system_prompt: System message
            user_prompt: User message

        Returns:
            Generated response text
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = await self.client.ainvoke(messages)
        return response.content

    async def generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> AsyncIterator[str]:
        """
        Generate streaming response from LLM

        Args:
            system_prompt: System message
            user_prompt: User message

        Yields:
            Response chunks
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        async for chunk in self.client.astream(messages):
            if chunk.content:
                yield chunk.content

    def count_tokens(self, text: str) -> int:
        """Estimate token count"""
        # Rough estimate: ~4 characters per token
        return len(text) // 4


# Singleton instance
llm_service = LLMService()
