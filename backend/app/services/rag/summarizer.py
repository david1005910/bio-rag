"""
Paper Summarizer Service
- OpenAI API를 사용한 논문 요약
- 다국어 지원 (영어/한국어)
"""

import asyncio
import logging
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class PaperSummary(BaseModel):
    """논문 요약 결과 모델"""
    paper_id: str
    title: str
    summary: str
    language: str
    success: bool = True
    error: str | None = None


class PaperSummarizer:
    """OpenAI API를 사용한 논문 요약 서비스"""

    SYSTEM_PROMPTS = {
        'ko': """당신은 의학/과학 논문을 요약하는 전문가입니다.
논문의 제목, 저자, 초록, 본문 내용을 바탕으로 핵심 내용을 한국어로 간결하게 요약해주세요.
요약은 다음 형식을 따르세요:
- 연구 목적
- 주요 방법
- 핵심 결과
- 결론 및 의의""",
        'en': """You are an expert at summarizing medical/scientific papers.
Based on the title, authors, abstract, and content, provide a concise summary.
Follow this format:
- Research Objective
- Key Methods
- Main Results
- Conclusion & Significance"""
    }

    def __init__(self, api_key: str | None = None, language: str = 'en'):
        """
        Args:
            api_key: OpenAI API 키
            language: 응답 언어 ('en' 또는 'ko')
        """
        self.api_key = api_key
        self.language = language
        self._client = None

    async def _get_client(self):
        """OpenAI 클라이언트 lazy initialization"""
        if self._client is None and self.api_key:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(api_key=self.api_key)
            except ImportError:
                logger.error("OpenAI package not installed")
                raise ImportError("Please install openai: pip install openai")
        return self._client

    async def summarize_paper(
        self,
        paper: dict[str, Any],
        content: str | None = None,
        max_content_length: int = 3000
    ) -> PaperSummary:
        """
        단일 논문 요약

        Args:
            paper: 논문 메타데이터 (title, authors, abstract, id 등)
            content: 논문 본문 텍스트 (선택)
            max_content_length: 본문 최대 길이

        Returns:
            PaperSummary 객체
        """
        paper_id = paper.get('id') or paper.get('pmid', 'unknown')
        title = paper.get('title', 'Unknown Title')

        if not self.api_key:
            # API 키가 없으면 초록으로 대체
            abstract = paper.get('abstract', '')
            return PaperSummary(
                paper_id=str(paper_id),
                title=title,
                summary=f"[초록 원문]\n{abstract}" if abstract else "요약 불가능 (API 키 필요)",
                language=self.language,
                success=False,
                error="OpenAI API key not configured"
            )

        try:
            client = await self._get_client()
            if client is None:
                raise ValueError("Failed to initialize OpenAI client")

            # 요약할 내용 구성
            authors = paper.get('authors', [])
            if isinstance(authors, list):
                author_str = ', '.join(authors[:5])
            else:
                author_str = str(authors)

            content_text = content[:max_content_length] if content else ""

            content_to_summarize = f"""
Title: {title}
Authors: {author_str}
Published: {paper.get('published', paper.get('publication_date', 'N/A'))}
Source: {paper.get('source', 'Unknown')}

Abstract:
{paper.get('abstract', 'No abstract available')}

Content:
{content_text}
"""

            system_prompt = self.SYSTEM_PROMPTS.get(self.language, self.SYSTEM_PROMPTS['en'])

            response = await client.chat.completions.create(
                model="gpt-4o-mini",  # Better quality for scientific content
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content_to_summarize}
                ],
                max_tokens=800,
                temperature=0.3
            )

            summary = response.choices[0].message.content.strip()

            return PaperSummary(
                paper_id=str(paper_id),
                title=title,
                summary=summary,
                language=self.language,
                success=True
            )

        except Exception as e:
            logger.error(f"Paper summarization failed for {paper_id}: {e}")
            abstract = paper.get('abstract', '')
            return PaperSummary(
                paper_id=str(paper_id),
                title=title,
                summary=f"[초록 원문]\n{abstract}" if abstract else "요약 실패",
                language=self.language,
                success=False,
                error=str(e)
            )

    async def summarize_papers(
        self,
        papers: list[dict[str, Any]],
        documents: list[dict[str, Any]] | None = None,
        concurrency: int = 3,
        delay: float = 0.5
    ) -> list[PaperSummary]:
        """
        여러 논문 일괄 요약

        Args:
            papers: 논문 메타데이터 리스트
            documents: 논문 본문 텍스트 리스트 (paper['id']와 매핑)
            concurrency: 동시 처리 수
            delay: API 호출 간 지연 (초)

        Returns:
            PaperSummary 리스트
        """
        if not papers:
            return []

        # 문서 내용 매핑
        doc_map: dict[str, str] = {}
        if documents:
            for doc in documents:
                source = doc.get('source', '')
                text = doc.get('text', '')
                # paper id로 매핑 시도
                for paper in papers:
                    paper_id = paper.get('id') or paper.get('pmid', '')
                    if str(paper_id) in source:
                        doc_map[str(paper_id)] = text
                        break

        results: list[PaperSummary] = []
        semaphore = asyncio.Semaphore(concurrency)

        async def summarize_with_limit(paper: dict[str, Any]) -> PaperSummary:
            async with semaphore:
                paper_id = str(paper.get('id') or paper.get('pmid', ''))
                content = doc_map.get(paper_id)
                result = await self.summarize_paper(paper, content)
                await asyncio.sleep(delay)
                return result

        tasks = [summarize_with_limit(paper) for paper in papers]
        results = await asyncio.gather(*tasks)

        return list(results)

    async def answer_question(
        self,
        question: str,
        contexts: list[dict[str, Any]],
        language: str | None = None
    ) -> str:
        """
        컨텍스트 기반 질문 답변

        Args:
            question: 사용자 질문
            contexts: 관련 문맥 리스트 (content, source 포함)
            language: 응답 언어 (None이면 self.language 사용)

        Returns:
            답변 텍스트
        """
        if not self.api_key:
            return "OpenAI API 키가 설정되지 않았습니다."

        try:
            client = await self._get_client()
            if client is None:
                return "OpenAI 클라이언트 초기화 실패"

            lang = language or self.language
            context_text = "\n\n".join([ctx.get('content', '') for ctx in contexts])

            if lang == 'ko':
                system_msg = "당신은 의학/과학 논문을 기반으로 질문에 답변하는 전문가입니다. 제공된 문맥을 바탕으로 한국어로 답변해주세요."
            else:
                system_msg = "You are an expert answering questions based on medical/scientific papers. Answer based on the provided context."

            response = await client.chat.completions.create(
                model="gpt-4o-mini",  # Better quality for scientific Q&A
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"}
                ],
                max_tokens=800,
                temperature=0.3
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Question answering failed: {e}")
            return f"답변 생성 실패: {str(e)}"


# Factory function
def create_paper_summarizer(
    api_key: str | None = None,
    language: str = 'en'
) -> PaperSummarizer:
    """PaperSummarizer 인스턴스 생성"""
    return PaperSummarizer(api_key=api_key, language=language)
