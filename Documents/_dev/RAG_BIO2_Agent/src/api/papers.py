from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional
import asyncio

from src.models import get_db, Paper, Author, Keyword
from src.services import PubMedService, RAGService
from .schemas import SearchRequest, PaperResponse, IndexPaperRequest
from .auth import get_current_user, get_optional_user

router = APIRouter(prefix="/papers", tags=["Papers"])

pubmed_service = PubMedService()
_rag_service = None

def get_rag_service():
    global _rag_service
    if _rag_service is None:
        try:
            _rag_service = RAGService(
                embedding_model="pubmedbert", 
                vector_dimension=768,
                collection_name="biomedical_papers_768d"
            )
        except Exception as e:
            print(f"PubMedBERT init failed, using OpenAI: {e}")
            _rag_service = RAGService(
                embedding_model="openai", 
                vector_dimension=1536,
                collection_name="biomedical_papers_1536d"
            )
    return _rag_service

@router.get("/search", response_model=List[PaperResponse])
async def search_papers(
    query: str = Query(..., min_length=2, description="Search query"),
    limit: int = Query(10, ge=1, le=50),
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    db: Session = Depends(get_db)
):
    try:
        papers = await pubmed_service.search_and_fetch(query, limit)
        
        results = []
        for paper in papers:
            pub_date = None
            if paper.publication_date:
                pub_date = paper.publication_date.strftime("%Y-%m-%d")
            
            results.append(PaperResponse(
                pmid=paper.pmid,
                title=paper.title,
                abstract=paper.abstract,
                authors=paper.authors,
                journal=paper.journal,
                publication_date=pub_date,
                doi=paper.doi,
                keywords=paper.keywords + paper.mesh_terms
            ))
        
        return results
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )

@router.get("/semantic-search", response_model=List[PaperResponse])
async def semantic_search(
    query: str = Query(..., min_length=2, description="Semantic search query"),
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db)
):
    try:
        rag_service = get_rag_service()
        results = rag_service.semantic_search(query, top_k=limit)
        
        papers = []
        for result in results:
            papers.append(PaperResponse(
                pmid=result["pmid"],
                title=result["title"],
                abstract=None,
                authors=[],
                journal=result.get("journal"),
                publication_date=result.get("publication_date"),
                doi=None,
                keywords=[],
                relevance=result.get("relevance"),
                excerpt=result.get("excerpt")
            ))
        
        return papers
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Semantic search failed: {str(e)}"
        )

@router.post("/index")
async def index_paper(
    paper_data: IndexPaperRequest,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        rag_service = get_rag_service()
        
        chunk_ids = rag_service.index_paper(
            pmid=paper_data.pmid,
            title=paper_data.title,
            abstract=paper_data.abstract,
            authors=paper_data.authors,
            journal=paper_data.journal,
            publication_date=paper_data.publication_date
        )
        
        existing_paper = db.query(Paper).filter(Paper.pmid == paper_data.pmid).first()
        if not existing_paper:
            new_paper = Paper(
                pmid=paper_data.pmid,
                title=paper_data.title,
                abstract=paper_data.abstract,
                journal=paper_data.journal
            )
            db.add(new_paper)
            db.commit()
        
        return {
            "status": "success",
            "pmid": paper_data.pmid,
            "chunks_indexed": len(chunk_ids)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indexing failed: {str(e)}"
        )

@router.post("/index-from-pubmed")
async def index_from_pubmed(
    query: str = Query(..., description="PubMed search query"),
    limit: int = Query(10, ge=1, le=50),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        papers = await pubmed_service.search_and_fetch(query, limit)
        rag_service = get_rag_service()
        
        indexed_count = 0
        for paper in papers:
            try:
                pub_date = None
                if paper.publication_date:
                    pub_date = paper.publication_date.strftime("%Y-%m-%d")
                
                rag_service.index_paper(
                    pmid=paper.pmid,
                    title=paper.title,
                    abstract=paper.abstract,
                    authors=paper.authors,
                    journal=paper.journal,
                    publication_date=pub_date
                )
                
                existing_paper = db.query(Paper).filter(Paper.pmid == paper.pmid).first()
                if not existing_paper:
                    new_paper = Paper(
                        pmid=paper.pmid,
                        title=paper.title,
                        abstract=paper.abstract,
                        journal=paper.journal,
                        publication_date=paper.publication_date
                    )
                    db.add(new_paper)
                
                indexed_count += 1
            except Exception as e:
                print(f"Error indexing paper {paper.pmid}: {e}")
                continue
        
        db.commit()
        
        return {
            "status": "success",
            "query": query,
            "papers_found": len(papers),
            "papers_indexed": indexed_count
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indexing failed: {str(e)}"
        )

@router.get("/{pmid}", response_model=PaperResponse)
async def get_paper(pmid: str, db: Session = Depends(get_db)):
    paper = db.query(Paper).filter(Paper.pmid == pmid).first()
    
    if not paper:
        papers = await pubmed_service.fetch_paper_details([pmid])
        if not papers:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Paper not found"
            )
        
        p = papers[0]
        pub_date = None
        if p.publication_date:
            pub_date = p.publication_date.strftime("%Y-%m-%d")
        
        return PaperResponse(
            pmid=p.pmid,
            title=p.title,
            abstract=p.abstract,
            authors=p.authors,
            journal=p.journal,
            publication_date=pub_date,
            doi=p.doi,
            keywords=p.keywords + p.mesh_terms
        )
    
    pub_date = None
    if paper.publication_date:
        pub_date = paper.publication_date.strftime("%Y-%m-%d")
    
    return PaperResponse(
        pmid=paper.pmid,
        title=paper.title,
        abstract=paper.abstract,
        authors=[a.name for a in paper.authors],
        journal=paper.journal,
        publication_date=pub_date,
        doi=paper.doi,
        keywords=[k.term for k in paper.keywords]
    )

@router.get("/{pmid}/similar", response_model=List[PaperResponse])
async def get_similar_papers(
    pmid: str,
    limit: int = Query(5, ge=1, le=20),
    db: Session = Depends(get_db)
):
    try:
        paper = db.query(Paper).filter(Paper.pmid == pmid).first()
        
        if paper and paper.abstract:
            search_text = f"{paper.title} {paper.abstract}"
        else:
            papers = await pubmed_service.fetch_paper_details([pmid])
            if papers:
                search_text = f"{papers[0].title} {papers[0].abstract}"
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Paper not found"
                )
        
        rag_service = get_rag_service()
        similar = rag_service.semantic_search(search_text, top_k=limit + 1)
        
        similar = [p for p in similar if p["pmid"] != pmid][:limit]
        
        return [
            PaperResponse(
                pmid=p["pmid"],
                title=p["title"],
                abstract=None,
                authors=[],
                journal=p.get("journal"),
                publication_date=p.get("publication_date"),
                doi=None,
                keywords=[],
                relevance=p.get("relevance"),
                excerpt=p.get("excerpt")
            )
            for p in similar
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to find similar papers: {str(e)}"
        )
