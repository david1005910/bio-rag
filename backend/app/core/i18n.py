"""
Internationalization (i18n) Utilities
- 언어 감지
- 한영 번역 (의학 용어 포함)
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


# 의학 용어 한영 매핑
MEDICAL_TERMS_KO_EN = {
    # 질병
    '당뇨병': 'diabetes mellitus',
    '당뇨': 'diabetes',
    '암': 'cancer',
    '폐암': 'lung cancer',
    '유방암': 'breast cancer',
    '위암': 'gastric cancer stomach cancer',
    '간암': 'liver cancer hepatocellular carcinoma',
    '대장암': 'colon cancer colorectal cancer',
    '전립선암': 'prostate cancer',
    '췌장암': 'pancreatic cancer',
    '백혈병': 'leukemia',
    '림프종': 'lymphoma',
    '고혈압': 'hypertension',
    '심장병': 'heart disease cardiovascular disease',
    '뇌졸중': 'stroke cerebrovascular accident',
    '치매': 'dementia alzheimer',
    '파킨슨': 'parkinson',
    '우울증': 'depression',
    '불안장애': 'anxiety disorder',
    '비만': 'obesity',
    '골다공증': 'osteoporosis',
    '관절염': 'arthritis',
    '류마티스': 'rheumatoid',
    '천식': 'asthma',
    '알레르기': 'allergy',
    '감염': 'infection',
    '바이러스': 'virus viral',
    '세균': 'bacteria bacterial',
    '코로나': 'COVID-19 coronavirus',

    # 치료/의료
    '치료': 'treatment',
    '치료법': 'treatment therapy',
    '백신': 'vaccine vaccination',
    '항생제': 'antibiotic',
    '항암제': 'anticancer chemotherapy',
    '면역': 'immunity immune',
    '면역치료': 'immunotherapy',
    '진단': 'diagnosis diagnostic',
    '예방': 'prevention preventive',
    '증상': 'symptoms',
    '부작용': 'side effects adverse effects',
    '임상시험': 'clinical trial',
    '약물': 'drug medication',
    '수술': 'surgery surgical',
    '방사선': 'radiation radiotherapy',
    '화학요법': 'chemotherapy',
    '유전자': 'gene genetic',
    '단백질': 'protein',
    '세포': 'cell cellular',
    '줄기세포': 'stem cell',
    '바이오마커': 'biomarker',

    # 연구 관련
    '연구': 'research study',
    '효과': 'effect efficacy',
    '메커니즘': 'mechanism',
    '경로': 'pathway',
    '분석': 'analysis',
    '결과': 'results outcome',
}


def detect_language(text: str) -> str:
    """
    텍스트의 언어를 감지 (한국어 또는 영어)

    Args:
        text: 분석할 텍스트

    Returns:
        'ko' (한국어) 또는 'en' (영어)
    """
    if not text:
        return 'en'

    korean_chars = len(re.findall(r'[가-힣]', text))
    total_chars = len(re.findall(r'[a-zA-Z가-힣]', text))

    if total_chars == 0:
        return 'en'

    korean_ratio = korean_chars / total_chars
    return 'ko' if korean_ratio > 0.3 else 'en'


def translate_medical_terms(text: str, direction: str = 'ko_to_en') -> str:
    """
    의학 용어 번역 (사전 기반)

    Args:
        text: 번역할 텍스트
        direction: 'ko_to_en' 또는 'en_to_ko'

    Returns:
        번역된 텍스트
    """
    translated = text

    if direction == 'ko_to_en':
        # 더 긴 용어를 먼저 매치하도록 정렬
        sorted_terms = sorted(MEDICAL_TERMS_KO_EN.items(), key=lambda x: len(x[0]), reverse=True)
        for ko, en in sorted_terms:
            if ko in translated:
                translated = translated.replace(ko, en)
        # 남은 한글 제거
        translated = re.sub(r'[가-힣]+', '', translated).strip()
    else:
        # 역방향 매핑 (영어 -> 한국어)
        en_to_ko = {en.split()[0]: ko for ko, en in MEDICAL_TERMS_KO_EN.items()}
        for en, ko in en_to_ko.items():
            if en.lower() in translated.lower():
                translated = re.sub(re.escape(en), ko, translated, flags=re.IGNORECASE)

    return translated if translated else text


async def translate_to_english(
    text: str,
    openai_api_key: str | None = None
) -> str:
    """
    한국어 텍스트를 영어로 번역 (검색용)

    Args:
        text: 한국어 텍스트
        openai_api_key: OpenAI API 키 (선택)

    Returns:
        영어 번역 텍스트
    """
    # API 키가 없으면 사전 기반 번역 사용
    if not openai_api_key:
        return translate_medical_terms(text, 'ko_to_en')

    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=openai_api_key)

        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical/scientific translator. Translate the following Korean medical query to English. Only output the English translation, nothing else."
                },
                {"role": "user", "content": text}
            ],
            max_tokens=100,
            temperature=0.1
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.warning(f"Translation failed, using dictionary fallback: {e}")
        return translate_medical_terms(text, 'ko_to_en')


def translate_to_english_sync(
    text: str,
    openai_api_key: str | None = None
) -> str:
    """
    한국어 텍스트를 영어로 번역 (동기 버전)

    Args:
        text: 한국어 텍스트
        openai_api_key: OpenAI API 키 (선택)

    Returns:
        영어 번역 텍스트
    """
    # API 키가 없으면 사전 기반 번역 사용
    if not openai_api_key:
        return translate_medical_terms(text, 'ko_to_en')

    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_api_key)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical/scientific translator. Translate the following Korean medical query to English. Only output the English translation, nothing else."
                },
                {"role": "user", "content": text}
            ],
            max_tokens=100,
            temperature=0.1
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.warning(f"Translation failed, using dictionary fallback: {e}")
        return translate_medical_terms(text, 'ko_to_en')


class MultilingualSupport:
    """다국어 지원 유틸리티 클래스"""

    def __init__(self, openai_api_key: str | None = None):
        self.openai_api_key = openai_api_key

    def detect_language(self, text: str) -> str:
        """언어 감지"""
        return detect_language(text)

    async def translate_query(self, query: str) -> dict[str, Any]:
        """
        검색 쿼리 번역 및 언어 정보 반환

        Args:
            query: 원본 쿼리

        Returns:
            {'original': str, 'translated': str, 'language': str}
        """
        language = detect_language(query)

        if language == 'ko':
            translated = await translate_to_english(query, self.openai_api_key)
        else:
            translated = query

        return {
            'original': query,
            'translated': translated,
            'language': language
        }

    def get_response_language(self, detected_language: str) -> str:
        """감지된 언어에 따른 응답 언어 반환"""
        return detected_language


# Convenience functions
def create_multilingual_support(openai_api_key: str | None = None) -> MultilingualSupport:
    """MultilingualSupport 인스턴스 생성"""
    return MultilingualSupport(openai_api_key=openai_api_key)
