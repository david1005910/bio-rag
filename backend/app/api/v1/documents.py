"""Documents API endpoints - PDF download and text extraction"""

import os
import tempfile
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from app.services.document import (
    DocumentContent,
    PDFDownloader,
    TextExtractor,
    create_pdf_downloader,
)
from app.services.rag.summarizer import PaperSummarizer
from app.core.config import settings

router = APIRouter(prefix="/documents", tags=["Documents"])


# ==================== Request/Response Models ====================
class DownloadRequest(BaseModel):
    """PDF 다운로드 요청"""
    papers: list[dict[str, Any]] = Field(
        ...,
        description="논문 목록 (id, title, pdf_url 필수)"
    )


class DownloadResult(BaseModel):
    """다운로드 결과"""
    paper_id: str
    title: str
    filepath: str | None = None
    success: bool
    error: str | None = None


class DownloadResponse(BaseModel):
    """다운로드 응답"""
    downloaded: list[DownloadResult]
    total: int
    success_count: int


class ExtractRequest(BaseModel):
    """텍스트 추출 요청"""
    filepaths: list[str] = Field(..., description="파일 경로 목록")


class ExtractedDocument(BaseModel):
    """추출된 문서"""
    source: str
    filepath: str
    text: str
    text_length: int
    success: bool
    error: str | None = None


class ExtractResponse(BaseModel):
    """추출 응답"""
    documents: list[ExtractedDocument]
    total: int
    success_count: int


class SummarizeRequest(BaseModel):
    """요약 요청"""
    paper: dict[str, Any] = Field(
        ...,
        description="논문 정보 (id, title, abstract 등)"
    )
    content: str | None = Field(default=None, description="논문 본문 (선택)")
    language: str = Field(default="en", description="응답 언어 (en, ko)")


class SummaryResponse(BaseModel):
    """요약 응답"""
    paper_id: str
    title: str
    summary: str
    language: str
    success: bool
    error: str | None = None


# ==================== Endpoints ====================
@router.post("/download", response_model=DownloadResponse)
async def download_papers(request: DownloadRequest) -> DownloadResponse:
    """
    논문 PDF 다운로드

    - **papers**: 다운로드할 논문 목록
      - id: 논문 ID
      - title: 논문 제목
      - pdf_url: PDF URL (없으면 초록만 저장)
      - abstract: 초록 (pdf_url 없을 때 사용)

    PDF가 없거나 다운로드 실패 시 초록을 텍스트 파일로 저장합니다.
    """
    if not request.papers:
        raise HTTPException(status_code=400, detail="No papers provided")

    if len(request.papers) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 papers per request")

    # 임시 디렉토리에 다운로드
    download_dir = os.path.join(tempfile.gettempdir(), "biorag_papers")
    downloader = create_pdf_downloader(save_dir=download_dir)

    results: list[DownloadResult] = []
    success_count = 0

    for paper in request.papers:
        paper_id = paper.get('id') or paper.get('pmid', 'unknown')
        title = paper.get('title', 'Untitled')

        try:
            filepath = await downloader.download(paper)

            if filepath:
                results.append(DownloadResult(
                    paper_id=str(paper_id),
                    title=title,
                    filepath=filepath,
                    success=True,
                ))
                success_count += 1
            else:
                results.append(DownloadResult(
                    paper_id=str(paper_id),
                    title=title,
                    success=False,
                    error="Download failed - no PDF URL and no abstract",
                ))

        except Exception as e:
            results.append(DownloadResult(
                paper_id=str(paper_id),
                title=title,
                success=False,
                error=str(e),
            ))

    return DownloadResponse(
        downloaded=results,
        total=len(results),
        success_count=success_count,
    )


@router.post("/extract", response_model=ExtractResponse)
async def extract_text(request: ExtractRequest) -> ExtractResponse:
    """
    파일에서 텍스트 추출

    - **filepaths**: 추출할 파일 경로 목록 (PDF, TXT 지원)

    PDF는 PyPDF2 또는 pdfplumber를 사용하여 추출합니다.
    """
    if not request.filepaths:
        raise HTTPException(status_code=400, detail="No filepaths provided")

    if len(request.filepaths) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 files per request")

    # 파일 존재 여부 확인
    valid_paths = []
    for path in request.filepaths:
        if os.path.exists(path):
            valid_paths.append(path)

    if not valid_paths:
        raise HTTPException(status_code=404, detail="No valid files found")

    # 텍스트 추출
    documents = TextExtractor.extract_all(valid_paths)

    results = [
        ExtractedDocument(
            source=doc.source,
            filepath=doc.filepath,
            text=doc.text[:10000] if doc.text else "",  # 응답 크기 제한
            text_length=doc.text_length,
            success=doc.success,
            error=doc.error,
        )
        for doc in documents
    ]

    success_count = sum(1 for r in results if r.success)

    return ExtractResponse(
        documents=results,
        total=len(results),
        success_count=success_count,
    )


@router.post("/extract-upload")
async def extract_from_upload(file: UploadFile = File(...)) -> ExtractedDocument:
    """
    업로드된 파일에서 텍스트 추출

    PDF 또는 TXT 파일을 업로드하면 텍스트를 추출합니다.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename")

    # 파일 확장자 확인
    ext = file.filename.lower().split('.')[-1]
    if ext not in ['pdf', 'txt']:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Only PDF and TXT are supported."
        )

    # 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}') as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # 텍스트 추출
        text = TextExtractor.extract(tmp_path)

        return ExtractedDocument(
            source=file.filename,
            filepath=tmp_path,
            text=text[:10000] if text else "",
            text_length=len(text),
            success=bool(text),
            error=None if text else "No text extracted",
        )

    finally:
        # 임시 파일 정리
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@router.post("/summarize", response_model=SummaryResponse)
async def summarize_paper(request: SummarizeRequest) -> SummaryResponse:
    """
    논문 요약

    - **paper**: 논문 정보 (id, title, abstract 필수)
    - **content**: 논문 본문 (선택, 더 나은 요약을 위해)
    - **language**: 응답 언어 (en, ko)

    OpenAI API를 사용하여 논문을 요약합니다.
    API 키가 없으면 초록을 그대로 반환합니다.
    """
    api_key = settings.OPENAI_API_KEY
    if api_key and api_key.startswith("your-"):
        api_key = None

    summarizer = PaperSummarizer(api_key=api_key, language=request.language)

    try:
        result = await summarizer.summarize_paper(
            paper=request.paper,
            content=request.content,
        )

        return SummaryResponse(
            paper_id=result.paper_id,
            title=result.title,
            summary=result.summary,
            language=result.language,
            success=result.success,
            error=result.error,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")


@router.post("/summarize-batch", response_model=list[SummaryResponse])
async def summarize_papers_batch(
    papers: list[dict[str, Any]],
    language: str = "en",
) -> list[SummaryResponse]:
    """
    여러 논문 일괄 요약

    - **papers**: 요약할 논문 목록 (최대 10개)
    - **language**: 응답 언어

    병렬로 요약을 수행하여 빠르게 처리합니다.
    """
    if not papers:
        raise HTTPException(status_code=400, detail="No papers provided")

    if len(papers) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 papers per batch")

    api_key = settings.OPENAI_API_KEY
    if api_key and api_key.startswith("your-"):
        api_key = None

    summarizer = PaperSummarizer(api_key=api_key, language=language)

    try:
        results = await summarizer.summarize_papers(papers, concurrency=3)

        return [
            SummaryResponse(
                paper_id=r.paper_id,
                title=r.title,
                summary=r.summary,
                language=r.language,
                success=r.success,
                error=r.error,
            )
            for r in results
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch summarization failed: {str(e)}")
