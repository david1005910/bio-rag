import asyncio
import json
import logging
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.schemas.paper import PaperMetadata

logger = logging.getLogger(__name__)


class PubMedAPIError(Exception):
    """PubMed API error"""

    pass


class PubMedClient:
    """PubMed E-utilities API client"""

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self) -> None:
        self.api_key = settings.PUBMED_API_KEY
        self.rate_limit = settings.PUBMED_RATE_LIMIT
        self._semaphore: asyncio.Semaphore | None = None

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Get or create semaphore in current event loop"""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.rate_limit)
        return self._semaphore

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def _request(self, endpoint: str, params: dict[str, Any]) -> str:
        """Make API request with rate limiting"""
        async with self._get_semaphore():
            if self.api_key and not self.api_key.startswith("your-"):
                params["api_key"] = self.api_key

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.BASE_URL}/{endpoint}",
                    params=params,
                )
                response.raise_for_status()
                return response.text

    async def search(
        self,
        query: str,
        max_results: int = 100,
        date_range: tuple[str, str] | None = None,
    ) -> list[str]:
        """
        Search papers and return PMID list

        Args:
            query: Search query (e.g., "cancer immunotherapy[Title/Abstract]")
            max_results: Maximum results
            date_range: (start_date, end_date) in YYYY/MM/DD format

        Returns:
            List of PMIDs
        """
        params: dict[str, Any] = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
        }

        if date_range:
            params["mindate"] = date_range[0]
            params["maxdate"] = date_range[1]
            params["datetype"] = "pdat"

        try:
            response = await self._request("esearch.fcgi", params)
            data = json.loads(response)
            return data.get("esearchresult", {}).get("idlist", [])
        except Exception as e:
            logger.error(f"PubMed search error: {e}")
            raise PubMedAPIError(f"Search failed: {e}") from e

    async def fetch_papers(self, pmid_list: list[str]) -> list[PaperMetadata]:
        """
        Fetch paper metadata by PMID list

        Args:
            pmid_list: List of PMIDs (max 200 recommended)

        Returns:
            List of PaperMetadata
        """
        if not pmid_list:
            return []

        params = {
            "db": "pubmed",
            "id": ",".join(pmid_list),
            "retmode": "xml",
        }

        try:
            response = await self._request("efetch.fcgi", params)
            return self._parse_xml(response)
        except Exception as e:
            logger.error(f"PubMed fetch error: {e}")
            raise PubMedAPIError(f"Fetch failed: {e}") from e

    def _parse_xml(self, xml_content: str) -> list[PaperMetadata]:
        """Parse XML response"""
        papers: list[PaperMetadata] = []
        root = ET.fromstring(xml_content)

        for article in root.findall(".//PubmedArticle"):
            try:
                paper = self._parse_article(article)
                papers.append(paper)
            except Exception as e:
                logger.warning(f"Error parsing article: {e}")
                continue

        return papers

    def _parse_article(self, article: ET.Element) -> PaperMetadata:
        """Parse single article"""
        medline = article.find(".//MedlineCitation")
        if medline is None:
            raise ValueError("MedlineCitation not found")

        article_elem = medline.find(".//Article")
        if article_elem is None:
            raise ValueError("Article not found")

        # PMID
        pmid_elem = medline.find(".//PMID")
        pmid = pmid_elem.text if pmid_elem is not None else ""

        # Title
        title_elem = article_elem.find(".//ArticleTitle")
        title = title_elem.text if title_elem is not None else ""

        # Abstract
        abstract_elem = article_elem.find(".//Abstract/AbstractText")
        abstract = abstract_elem.text if abstract_elem is not None else None

        # Authors
        authors: list[str] = []
        for author in article_elem.findall(".//Author"):
            last_name = author.find("LastName")
            first_name = author.find("ForeName")
            if last_name is not None and last_name.text:
                name = last_name.text
                if first_name is not None and first_name.text:
                    name = f"{first_name.text} {name}"
                authors.append(name)

        # Journal
        journal_elem = article_elem.find(".//Journal/Title")
        journal = journal_elem.text if journal_elem is not None else None

        # Publication Date
        pub_date_elem = article_elem.find(".//PubDate")
        pub_date = self._parse_date(pub_date_elem)

        # DOI
        doi: str | None = None
        for id_elem in article.findall(".//ArticleId"):
            if id_elem.get("IdType") == "doi":
                doi = id_elem.text
                break

        # Keywords
        keywords = [
            kw.text for kw in medline.findall(".//Keyword") if kw.text
        ]

        # MeSH Terms
        mesh_terms = [
            mesh.find(".//DescriptorName").text  # type: ignore
            for mesh in medline.findall(".//MeshHeading")
            if mesh.find(".//DescriptorName") is not None
            and mesh.find(".//DescriptorName").text  # type: ignore
        ]

        return PaperMetadata(
            pmid=pmid,
            title=title,
            abstract=abstract,
            authors=authors if authors else None,
            journal=journal,
            publication_date=pub_date,
            doi=doi,
            keywords=keywords if keywords else None,
            mesh_terms=mesh_terms if mesh_terms else None,
        )

    def _parse_date(self, pub_date_elem: ET.Element | None) -> datetime | None:
        """Parse publication date"""
        if pub_date_elem is None:
            return None

        year = pub_date_elem.find("Year")
        month = pub_date_elem.find("Month")
        day = pub_date_elem.find("Day")

        if year is None or year.text is None:
            return None

        try:
            year_val = int(year.text)
            month_val = self._parse_month(month.text if month is not None else None)
            day_val = int(day.text) if day is not None and day.text else 1
            return datetime(year_val, month_val, day_val)
        except (ValueError, TypeError):
            return None

    def _parse_month(self, month_str: str | None) -> int:
        """Parse month string to int"""
        if month_str is None:
            return 1

        month_map = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4,
            "may": 5, "jun": 6, "jul": 7, "aug": 8,
            "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        }

        try:
            return int(month_str)
        except ValueError:
            return month_map.get(month_str.lower()[:3], 1)

    async def batch_fetch(
        self,
        pmid_list: list[str],
        batch_size: int = 200,
    ) -> list[PaperMetadata]:
        """Fetch papers in batches"""
        all_papers: list[PaperMetadata] = []

        for i in range(0, len(pmid_list), batch_size):
            batch = pmid_list[i : i + batch_size]
            papers = await self.fetch_papers(batch)
            all_papers.extend(papers)

            # Rate limit compliance
            if i + batch_size < len(pmid_list):
                await asyncio.sleep(0.1)

        return all_papers


# Singleton instance
pubmed_client = PubMedClient()
