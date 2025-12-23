"""
Document Extraction Service
- PDF 다운로드
- 텍스트 추출 (PDF, TXT)
- 논문 초록 저장
"""

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class DocumentContent(BaseModel):
    """추출된 문서 내용"""
    source: str
    filepath: str
    text: str
    text_length: int
    success: bool = True
    error: str | None = None


class PDFDownloader:
    """PDF 다운로드 서비스"""

    def __init__(self, save_dir: str = "./papers"):
        """
        Args:
            save_dir: PDF 저장 디렉토리
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _safe_filename(self, title: str, paper_id: str, max_length: int = 50) -> str:
        """안전한 파일명 생성"""
        safe_title = re.sub(r'[^\w\s-]', '', title)[:max_length]
        return f"{paper_id}_{safe_title}".strip()

    async def download(
        self,
        paper: dict[str, Any],
        timeout: float = 60.0
    ) -> str | None:
        """
        논문 PDF 다운로드

        Args:
            paper: 논문 정보 (id, title, pdf_url, abstract 등)
            timeout: 다운로드 타임아웃

        Returns:
            저장된 파일 경로 또는 None
        """
        paper_id = paper.get('id') or paper.get('pmid', 'unknown')
        title = paper.get('title', 'untitled')
        pdf_url = paper.get('pdf_url') or paper.get('pmc_url')

        filename = self._safe_filename(title, str(paper_id))
        filepath = self.save_dir / f"{filename}.pdf"
        txt_filepath = self.save_dir / f"{filename}.txt"

        # 이미 존재하는 파일 확인
        if txt_filepath.exists():
            return str(txt_filepath)
        if filepath.exists():
            return str(filepath)

        # PDF URL이 없으면 초록만 저장
        if not pdf_url:
            if paper.get('source') == 'PubMed' or not pdf_url:
                return await self._save_abstract_as_text(paper, filename)
            return None

        try:
            headers = {'User-Agent': 'Mozilla/5.0 (compatible; ResearchBot/1.0)'}

            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(pdf_url, headers=headers, follow_redirects=True)
                response.raise_for_status()

                # Content-Type 확인
                content_type = response.headers.get('content-type', '')
                if 'html' in content_type or 'text' in content_type:
                    return await self._save_abstract_as_text(paper, filename)

                # 처음 몇 바이트로 HTML 여부 확인
                content = response.content
                if content[:5] in [b'<html', b'<!DOC']:
                    return await self._save_abstract_as_text(paper, filename)

                # PDF 저장
                filepath.write_bytes(content)
                logger.info(f"Downloaded PDF: {filename}")
                return str(filepath)

        except Exception as e:
            logger.warning(f"PDF download failed for {paper_id}: {e}")
            return await self._save_abstract_as_text(paper, filename)

    async def _save_abstract_as_text(
        self,
        paper: dict[str, Any],
        filename: str
    ) -> str | None:
        """초록을 텍스트 파일로 저장"""
        txt_filepath = self.save_dir / f"{filename}.txt"

        authors = paper.get('authors', [])
        if isinstance(authors, list):
            author_str = ', '.join(authors[:10])
        else:
            author_str = str(authors)

        content = f"""Title: {paper.get('title', 'N/A')}

Authors: {author_str}

Source: {paper.get('source', 'Unknown')}

Published: {paper.get('published', paper.get('publication_date', 'N/A'))}

Abstract:
{paper.get('abstract', 'No abstract available')}

URL: {paper.get('pubmed_url', paper.get('pdf_url', 'N/A'))}
"""

        try:
            txt_filepath.write_text(content, encoding='utf-8')
            logger.info(f"Saved abstract: {filename}.txt")
            return str(txt_filepath)
        except Exception as e:
            logger.error(f"Failed to save abstract: {e}")
            return None

    async def download_all(
        self,
        papers: list[dict[str, Any]],
        concurrency: int = 3,
        delay: float = 0.2
    ) -> list[str]:
        """
        여러 논문 일괄 다운로드

        Args:
            papers: 논문 리스트
            concurrency: 동시 다운로드 수
            delay: 다운로드 간 지연 (초)

        Returns:
            다운로드된 파일 경로 리스트
        """
        if not papers:
            return []

        downloaded: list[str] = []
        semaphore = asyncio.Semaphore(concurrency)

        async def download_with_limit(paper: dict[str, Any]) -> str | None:
            async with semaphore:
                result = await self.download(paper)
                await asyncio.sleep(delay)
                return result

        tasks = [download_with_limit(paper) for paper in papers]
        results = await asyncio.gather(*tasks)

        for result in results:
            if result:
                downloaded.append(result)

        return downloaded


class TextExtractor:
    """텍스트 추출 서비스"""

    @staticmethod
    def extract_from_txt(filepath: str) -> str:
        """TXT 파일에서 텍스트 추출"""
        try:
            return Path(filepath).read_text(encoding='utf-8')
        except Exception as e:
            logger.error(f"TXT extraction failed: {e}")
            return ""

    @staticmethod
    def extract_from_pdf(filepath: str) -> str:
        """PDF 파일에서 텍스트 추출"""
        text = ""

        # PyPDF2 시도
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(filepath)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

            if len(text) >= 100:
                return text.strip()
        except ImportError:
            logger.warning("PyPDF2 not installed")
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {e}")

        # pdfplumber 폴백
        try:
            import pdfplumber
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except ImportError:
            logger.warning("pdfplumber not installed")
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")

        return text.strip()

    @classmethod
    def extract(cls, filepath: str) -> str:
        """파일 유형에 따라 텍스트 추출"""
        filepath_lower = filepath.lower()

        if filepath_lower.endswith('.txt'):
            return cls.extract_from_txt(filepath)
        elif filepath_lower.endswith('.pdf'):
            return cls.extract_from_pdf(filepath)
        else:
            logger.warning(f"Unsupported file type: {filepath}")
            return ""

    @classmethod
    def extract_all(cls, filepaths: list[str]) -> list[DocumentContent]:
        """
        여러 파일에서 텍스트 추출

        Args:
            filepaths: 파일 경로 리스트

        Returns:
            DocumentContent 리스트
        """
        documents: list[DocumentContent] = []

        for filepath in filepaths:
            filename = os.path.basename(filepath)

            try:
                text = cls.extract(filepath)

                if text:
                    documents.append(DocumentContent(
                        source=filename,
                        filepath=filepath,
                        text=text,
                        text_length=len(text),
                        success=True
                    ))
                else:
                    documents.append(DocumentContent(
                        source=filename,
                        filepath=filepath,
                        text="",
                        text_length=0,
                        success=False,
                        error="No text extracted"
                    ))

            except Exception as e:
                logger.error(f"Extraction failed for {filepath}: {e}")
                documents.append(DocumentContent(
                    source=filename,
                    filepath=filepath,
                    text="",
                    text_length=0,
                    success=False,
                    error=str(e)
                ))

        return documents


# Factory functions
def create_pdf_downloader(save_dir: str = "./papers") -> PDFDownloader:
    """PDFDownloader 인스턴스 생성"""
    return PDFDownloader(save_dir=save_dir)


def create_text_extractor() -> TextExtractor:
    """TextExtractor 인스턴스 생성"""
    return TextExtractor()
