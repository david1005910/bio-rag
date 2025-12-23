"""Internationalization (i18n) API endpoints"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.i18n import (
    detect_language,
    translate_medical_terms,
    translate_to_english,
    MultilingualSupport,
    MEDICAL_TERMS_KO_EN,
)

router = APIRouter(prefix="/i18n", tags=["Internationalization"])


# ==================== Request/Response Models ====================
class DetectLanguageRequest(BaseModel):
    """언어 감지 요청"""
    text: str = Field(..., min_length=1, max_length=5000, description="감지할 텍스트")


class DetectLanguageResponse(BaseModel):
    """언어 감지 응답"""
    text: str
    language: str
    language_name: str


class TranslateRequest(BaseModel):
    """번역 요청"""
    text: str = Field(..., min_length=1, max_length=2000, description="번역할 텍스트")
    use_ai: bool = Field(default=False, description="AI 번역 사용 여부")


class TranslateResponse(BaseModel):
    """번역 응답"""
    original: str
    translated: str
    source_language: str
    target_language: str
    method: str  # 'dictionary' or 'ai'


class MedicalTermResponse(BaseModel):
    """의학 용어 응답"""
    korean: str
    english: str


class QueryTranslationRequest(BaseModel):
    """검색 쿼리 번역 요청"""
    query: str = Field(..., min_length=1, max_length=500, description="검색 쿼리")


class QueryTranslationResponse(BaseModel):
    """검색 쿼리 번역 응답"""
    original: str
    translated: str
    language: str
    is_translated: bool


# ==================== Endpoints ====================
@router.post("/detect", response_model=DetectLanguageResponse)
async def detect_text_language(request: DetectLanguageRequest) -> DetectLanguageResponse:
    """
    텍스트 언어 감지

    - **text**: 언어를 감지할 텍스트

    한국어와 영어를 구분합니다.
    한글이 30% 이상이면 한국어로 판단합니다.
    """
    lang = detect_language(request.text)
    lang_name = "Korean" if lang == "ko" else "English"

    return DetectLanguageResponse(
        text=request.text,
        language=lang,
        language_name=lang_name,
    )


@router.get("/detect", response_model=DetectLanguageResponse)
async def detect_text_language_get(
    text: str = Query(..., min_length=1, max_length=5000, description="감지할 텍스트"),
) -> DetectLanguageResponse:
    """
    텍스트 언어 감지 (GET)
    """
    return await detect_text_language(DetectLanguageRequest(text=text))


@router.post("/translate", response_model=TranslateResponse)
async def translate_text(request: TranslateRequest) -> TranslateResponse:
    """
    한국어 → 영어 번역 (의학/과학 용어)

    - **text**: 번역할 한국어 텍스트
    - **use_ai**: AI 번역 사용 여부 (OpenAI API 필요)

    의학 용어 사전 기반 번역을 수행합니다.
    use_ai=true이고 OpenAI API 키가 있으면 AI 번역을 사용합니다.
    """
    source_lang = detect_language(request.text)

    if source_lang == 'en':
        # 영어는 그대로 반환
        return TranslateResponse(
            original=request.text,
            translated=request.text,
            source_language="en",
            target_language="en",
            method="none",
        )

    # 한국어 → 영어 번역
    if request.use_ai:
        api_key = settings.OPENAI_API_KEY
        if api_key and not api_key.startswith("your-"):
            try:
                translated = await translate_to_english(request.text, api_key)
                return TranslateResponse(
                    original=request.text,
                    translated=translated,
                    source_language="ko",
                    target_language="en",
                    method="ai",
                )
            except Exception:
                pass  # AI 실패 시 사전 기반으로 폴백

    # 사전 기반 번역
    translated = translate_medical_terms(request.text, 'ko_to_en')

    return TranslateResponse(
        original=request.text,
        translated=translated,
        source_language="ko",
        target_language="en",
        method="dictionary",
    )


@router.get("/translate", response_model=TranslateResponse)
async def translate_text_get(
    text: str = Query(..., min_length=1, max_length=2000, description="번역할 텍스트"),
    use_ai: bool = Query(default=False, description="AI 번역 사용 여부"),
) -> TranslateResponse:
    """
    한국어 → 영어 번역 (GET)
    """
    return await translate_text(TranslateRequest(text=text, use_ai=use_ai))


@router.post("/translate-query", response_model=QueryTranslationResponse)
async def translate_search_query(request: QueryTranslationRequest) -> QueryTranslationResponse:
    """
    검색 쿼리 번역

    - **query**: 검색 쿼리

    한국어 쿼리를 영어로 번역하여 검색에 사용할 수 있도록 합니다.
    영어 쿼리는 그대로 반환합니다.
    """
    api_key = settings.OPENAI_API_KEY
    if api_key and api_key.startswith("your-"):
        api_key = None

    support = MultilingualSupport(openai_api_key=api_key)
    result = await support.translate_query(request.query)

    return QueryTranslationResponse(
        original=result['original'],
        translated=result['translated'],
        language=result['language'],
        is_translated=result['original'] != result['translated'],
    )


@router.get("/medical-terms", response_model=list[MedicalTermResponse])
async def get_medical_terms(
    search: str | None = Query(default=None, description="검색어 (한글 또는 영어)"),
    limit: int = Query(default=50, ge=1, le=200, description="최대 결과 수"),
) -> list[MedicalTermResponse]:
    """
    의학 용어 사전 조회

    - **search**: 검색어 (선택, 한글 또는 영어로 필터링)
    - **limit**: 최대 결과 수

    내장된 의학 용어 한영 사전을 조회합니다.
    """
    results: list[MedicalTermResponse] = []

    for ko, en in MEDICAL_TERMS_KO_EN.items():
        if search:
            search_lower = search.lower()
            if search_lower not in ko and search_lower not in en.lower():
                continue

        results.append(MedicalTermResponse(korean=ko, english=en))

        if len(results) >= limit:
            break

    return results


@router.get("/supported-languages")
async def get_supported_languages() -> dict:
    """
    지원 언어 목록

    현재 지원하는 언어 목록을 반환합니다.
    """
    return {
        "languages": [
            {"code": "ko", "name": "Korean", "name_native": "한국어"},
            {"code": "en", "name": "English", "name_native": "English"},
        ],
        "translation_direction": "ko → en",
        "medical_terms_count": len(MEDICAL_TERMS_KO_EN),
    }
