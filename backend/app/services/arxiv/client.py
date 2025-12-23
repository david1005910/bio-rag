"""
ArXiv API Client
- arXiv 논문 검색
- 메타데이터 조회
- PDF URL 제공
"""

import asyncio
import logging
from datetime import datetime
from typing import Any

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ArXivPaper(BaseModel):
    """arXiv 논문 메타데이터"""
    arxiv_id: str
    title: str
    abstract: str | None = None
    authors: list[str] | None = None
    published: datetime | None = None
    updated: datetime | None = None
    pdf_url: str | None = None
    categories: list[str] | None = None
    doi: str | None = None


class ArXivAPIError(Exception):
    """arXiv API 에러"""
    pass


class ArXivClient:
    """arXiv API 클라이언트 (비동기)"""

    BASE_URL = "https://export.arxiv.org/api/query"

    def __init__(self, rate_limit: int = 3):
        """
        Args:
            rate_limit: 동시 요청 제한
        """
        self.rate_limit = rate_limit
        self._semaphore = asyncio.Semaphore(rate_limit)

    async def search(
        self,
        query: str,
        max_results: int = 10,
        sort_by: str = "relevance",
        sort_order: str = "descending",
        start: int = 0
    ) -> list[ArXivPaper]:
        """
        arXiv 논문 검색

        Args:
            query: 검색 쿼리 (e.g., "machine learning", "cat:cs.AI")
            max_results: 최대 결과 수
            sort_by: 정렬 기준 ("relevance", "lastUpdatedDate", "submittedDate")
            sort_order: 정렬 순서 ("ascending", "descending")
            start: 시작 인덱스

        Returns:
            ArXivPaper 리스트
        """
        async with self._semaphore:
            params = {
                "search_query": f"all:{query}",
                "start": start,
                "max_results": max_results,
                "sortBy": sort_by,
                "sortOrder": sort_order,
            }

            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(self.BASE_URL, params=params)
                    response.raise_for_status()

                    return self._parse_response(response.text)

            except httpx.HTTPError as e:
                logger.error(f"arXiv API request failed: {e}")
                raise ArXivAPIError(f"API request failed: {e}") from e
            except Exception as e:
                logger.error(f"arXiv search error: {e}")
                raise ArXivAPIError(f"Search failed: {e}") from e

    def _parse_response(self, xml_content: str) -> list[ArXivPaper]:
        """XML 응답 파싱"""
        import xml.etree.ElementTree as ET

        papers: list[ArXivPaper] = []

        # Atom namespace
        ns = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }

        try:
            root = ET.fromstring(xml_content)

            for entry in root.findall('atom:entry', ns):
                try:
                    paper = self._parse_entry(entry, ns)
                    papers.append(paper)
                except Exception as e:
                    logger.warning(f"Failed to parse entry: {e}")
                    continue

        except ET.ParseError as e:
            logger.error(f"XML parse error: {e}")
            raise ArXivAPIError(f"Failed to parse response: {e}") from e

        return papers

    def _parse_entry(self, entry, ns: dict) -> ArXivPaper:
        """단일 엔트리 파싱"""
        # ID (URL에서 arXiv ID 추출)
        id_elem = entry.find('atom:id', ns)
        arxiv_url = id_elem.text if id_elem is not None else ""
        arxiv_id = arxiv_url.split('/abs/')[-1] if '/abs/' in arxiv_url else arxiv_url

        # Title
        title_elem = entry.find('atom:title', ns)
        title = title_elem.text.strip().replace('\n', ' ') if title_elem is not None else ""

        # Abstract
        summary_elem = entry.find('atom:summary', ns)
        abstract = summary_elem.text.strip() if summary_elem is not None else None

        # Authors
        authors: list[str] = []
        for author in entry.findall('atom:author', ns):
            name_elem = author.find('atom:name', ns)
            if name_elem is not None and name_elem.text:
                authors.append(name_elem.text)

        # Published date
        published_elem = entry.find('atom:published', ns)
        published = None
        if published_elem is not None and published_elem.text:
            try:
                published = datetime.fromisoformat(published_elem.text.replace('Z', '+00:00'))
            except ValueError:
                pass

        # Updated date
        updated_elem = entry.find('atom:updated', ns)
        updated = None
        if updated_elem is not None and updated_elem.text:
            try:
                updated = datetime.fromisoformat(updated_elem.text.replace('Z', '+00:00'))
            except ValueError:
                pass

        # PDF URL
        pdf_url = None
        for link in entry.findall('atom:link', ns):
            if link.get('title') == 'pdf':
                pdf_url = link.get('href')
                break

        # Categories
        categories: list[str] = []
        for category in entry.findall('atom:category', ns):
            term = category.get('term')
            if term:
                categories.append(term)

        # DOI
        doi_elem = entry.find('arxiv:doi', ns)
        doi = doi_elem.text if doi_elem is not None else None

        return ArXivPaper(
            arxiv_id=arxiv_id,
            title=title,
            abstract=abstract,
            authors=authors if authors else None,
            published=published,
            updated=updated,
            pdf_url=pdf_url,
            categories=categories if categories else None,
            doi=doi
        )

    async def fetch_paper(self, arxiv_id: str) -> ArXivPaper | None:
        """
        특정 논문 조회

        Args:
            arxiv_id: arXiv ID (e.g., "2301.00001" or "cs.AI/0101001")

        Returns:
            ArXivPaper 또는 None
        """
        # Clean ID
        arxiv_id = arxiv_id.replace('arXiv:', '')

        async with self._semaphore:
            params = {"id_list": arxiv_id}

            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(self.BASE_URL, params=params)
                    response.raise_for_status()

                    papers = self._parse_response(response.text)
                    return papers[0] if papers else None

            except Exception as e:
                logger.error(f"Failed to fetch paper {arxiv_id}: {e}")
                return None

    async def batch_fetch(
        self,
        arxiv_ids: list[str],
        batch_size: int = 20
    ) -> list[ArXivPaper]:
        """
        여러 논문 일괄 조회

        Args:
            arxiv_ids: arXiv ID 리스트
            batch_size: 배치 크기

        Returns:
            ArXivPaper 리스트
        """
        all_papers: list[ArXivPaper] = []

        for i in range(0, len(arxiv_ids), batch_size):
            batch = arxiv_ids[i:i + batch_size]
            id_list = ",".join(batch)

            async with self._semaphore:
                params = {"id_list": id_list, "max_results": len(batch)}

                try:
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        response = await client.get(self.BASE_URL, params=params)
                        response.raise_for_status()

                        papers = self._parse_response(response.text)
                        all_papers.extend(papers)

                except Exception as e:
                    logger.error(f"Batch fetch failed: {e}")
                    continue

                # Rate limiting
                if i + batch_size < len(arxiv_ids):
                    await asyncio.sleep(0.5)

        return all_papers

    def to_common_format(self, paper: ArXivPaper) -> dict[str, Any]:
        """논문을 공통 포맷으로 변환 (PubMed와 호환)"""
        return {
            'id': paper.arxiv_id,
            'source': 'arXiv',
            'title': paper.title,
            'abstract': paper.abstract,
            'authors': paper.authors or [],
            'published': paper.published.strftime('%Y-%m-%d') if paper.published else None,
            'pdf_url': paper.pdf_url,
            'doi': paper.doi,
            'categories': paper.categories,
        }


# Singleton instance
arxiv_client = ArXivClient()
