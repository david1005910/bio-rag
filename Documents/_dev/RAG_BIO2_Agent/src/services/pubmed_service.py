import httpx
import asyncio
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Optional, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential
from dataclasses import dataclass

@dataclass
class PaperMetadata:
    pmid: str
    title: str
    abstract: str
    authors: List[str]
    journal: str
    publication_date: Optional[datetime]
    doi: Optional[str]
    keywords: List[str]
    mesh_terms: List[str]
    pdf_url: Optional[str] = None

class PubMedService:
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.rate_limit = 10 if api_key else 3
        self._translator = None
    
    def _get_translator(self):
        if self._translator is None:
            try:
                from .translation_service import TranslationService
                self._translator = TranslationService()
            except Exception:
                pass
        return self._translator
    
    def _build_url(self, endpoint: str, params: Dict[str, Any]) -> str:
        if self.api_key:
            params['api_key'] = self.api_key
        params['db'] = 'pubmed'
        query_string = '&'.join(f"{k}={v}" for k, v in params.items())
        return f"{self.BASE_URL}{endpoint}?{query_string}"
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def search_papers(
        self,
        query: str,
        max_results: int = 20,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> List[str]:
        translator = self._get_translator()
        search_query = query
        if translator and translator.is_korean(query):
            search_query = translator.translate_to_english(query)
            print(f"Translated query: {query} -> {search_query}")
        
        params = {
            'term': search_query.replace(' ', '+'),
            'retmax': max_results,
            'retmode': 'json',
            'sort': 'relevance'
        }
        
        if date_from:
            params['mindate'] = date_from
        if date_to:
            params['maxdate'] = date_to
        if date_from or date_to:
            params['datetype'] = 'pdat'
        
        url = self._build_url('esearch.fcgi', params)
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            
        return data.get('esearchresult', {}).get('idlist', [])
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def fetch_paper_details(self, pmids: List[str]) -> List[PaperMetadata]:
        if not pmids:
            return []
        
        params = {
            'id': ','.join(pmids),
            'retmode': 'xml'
        }
        
        url = self._build_url('efetch.fcgi', params)
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            
        return self._parse_xml_response(response.text)
    
    def _parse_xml_response(self, xml_text: str) -> List[PaperMetadata]:
        papers = []
        
        try:
            root = ET.fromstring(xml_text)
            
            for article in root.findall('.//PubmedArticle'):
                try:
                    medline = article.find('.//MedlineCitation')
                    if medline is None:
                        continue
                    
                    pmid_elem = medline.find('.//PMID')
                    pmid = pmid_elem.text if pmid_elem is not None else ''
                    
                    article_elem = medline.find('.//Article')
                    if article_elem is None:
                        continue
                    
                    title_elem = article_elem.find('.//ArticleTitle')
                    title = self._get_text_content(title_elem) if title_elem is not None else ''
                    
                    abstract_elem = article_elem.find('.//Abstract')
                    abstract = ''
                    if abstract_elem is not None:
                        abstract_texts = []
                        for abstract_text in abstract_elem.findall('.//AbstractText'):
                            label = abstract_text.get('Label', '')
                            text = self._get_text_content(abstract_text)
                            if label:
                                abstract_texts.append(f"{label}: {text}")
                            else:
                                abstract_texts.append(text)
                        abstract = ' '.join(abstract_texts)
                    
                    authors = []
                    author_list = article_elem.find('.//AuthorList')
                    if author_list is not None:
                        for author in author_list.findall('.//Author'):
                            lastname = author.find('LastName')
                            forename = author.find('ForeName')
                            if lastname is not None:
                                name = lastname.text or ''
                                if forename is not None and forename.text:
                                    name = f"{forename.text} {name}"
                                authors.append(name)
                    
                    journal_elem = article_elem.find('.//Journal/Title')
                    journal = journal_elem.text if journal_elem is not None else ''
                    
                    pub_date = None
                    pub_date_elem = article_elem.find('.//ArticleDate') or medline.find('.//DateCompleted')
                    if pub_date_elem is not None:
                        year = pub_date_elem.find('Year')
                        month = pub_date_elem.find('Month')
                        day = pub_date_elem.find('Day')
                        if year is not None:
                            try:
                                pub_date = datetime(
                                    int(year.text or '2000'),
                                    int(month.text or '1') if month is not None else 1,
                                    int(day.text or '1') if day is not None else 1
                                )
                            except ValueError:
                                pass
                    
                    doi = None
                    for elocation in article_elem.findall('.//ELocationID'):
                        if elocation.get('EIdType') == 'doi':
                            doi = elocation.text
                            break
                    
                    keywords = []
                    keyword_list = medline.find('.//KeywordList')
                    if keyword_list is not None:
                        for kw in keyword_list.findall('.//Keyword'):
                            if kw.text:
                                keywords.append(kw.text)
                    
                    mesh_terms = []
                    mesh_heading_list = medline.find('.//MeshHeadingList')
                    if mesh_heading_list is not None:
                        for mesh in mesh_heading_list.findall('.//MeshHeading/DescriptorName'):
                            if mesh.text:
                                mesh_terms.append(mesh.text)
                    
                    papers.append(PaperMetadata(
                        pmid=pmid,
                        title=title,
                        abstract=abstract,
                        authors=authors,
                        journal=journal,
                        publication_date=pub_date,
                        doi=doi,
                        keywords=keywords,
                        mesh_terms=mesh_terms
                    ))
                    
                except Exception as e:
                    print(f"Error parsing article: {e}")
                    continue
                    
        except ET.ParseError as e:
            print(f"XML parse error: {e}")
            
        return papers
    
    def _get_text_content(self, element) -> str:
        if element is None:
            return ''
        texts = []
        if element.text:
            texts.append(element.text)
        for child in element:
            if child.text:
                texts.append(child.text)
            if child.tail:
                texts.append(child.tail)
        return ''.join(texts).strip()
    
    async def search_and_fetch(
        self,
        query: str,
        max_results: int = 20
    ) -> List[PaperMetadata]:
        pmids = await self.search_papers(query, max_results)
        if not pmids:
            return []
        
        await asyncio.sleep(0.1)
        return await self.fetch_paper_details(pmids)
