"""
Research Trend Analyzer Service
- 키워드 기반 연구 트렌드 분석
- 연도별 출판 동향
- 핵심 키워드 추출
- 급부상 주제 식별
- AI 기반 연구 내용 요약
"""

import asyncio
import logging
import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TrendData(BaseModel):
    """트렌드 분석 결과 데이터 모델"""
    keyword: str
    total_papers: int
    year_trend: dict[str, Any] | None = None
    key_terms: dict[str, Any] | None = None
    emerging_topics: list[dict[str, Any]] | None = None
    content_summary: dict[str, Any] | None = None
    search_summary: dict[str, Any] | None = None


class TrendAnalyzer:
    """키워드 기반 연구 트렌드 분석기"""

    # 영어 불용어
    STOPWORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
        'those', 'it', 'its', 'they', 'them', 'their', 'we', 'our', 'you',
        'your', 'i', 'me', 'my', 'he', 'she', 'his', 'her', 'which', 'who',
        'whom', 'what', 'when', 'where', 'why', 'how', 'all', 'each', 'every',
        'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
        'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
        'also', 'now', 'here', 'there', 'then', 'once', 'if', 'because',
        'while', 'although', 'though', 'after', 'before', 'since', 'until',
        'about', 'into', 'through', 'during', 'above', 'below', 'between',
        'under', 'again', 'further', 'then', 'once', 'study', 'studies',
        'results', 'conclusion', 'methods', 'background', 'objective',
        'aim', 'purpose', 'introduction', 'discussion', 'however', 'thus',
        'therefore', 'moreover', 'furthermore', 'additionally', 'including',
        'included', 'using', 'used', 'based', 'associated', 'related',
        'compared', 'among', 'within', 'without', 'between', 'across'
    }

    # 연구 주제 키워드 매핑
    THEME_KEYWORDS = {
        'diagnosis': ['diagnosis', 'diagnostic', 'detection', 'screening', 'biomarker'],
        'treatment': ['treatment', 'therapy', 'therapeutic', 'drug', 'intervention', 'medication'],
        'mechanism': ['mechanism', 'pathway', 'molecular', 'cellular', 'signaling'],
        'prevention': ['prevention', 'preventive', 'risk factor', 'protective'],
        'clinical': ['clinical', 'trial', 'patient', 'outcome', 'efficacy'],
        'review': ['review', 'overview', 'comprehensive', 'systematic', 'meta-analysis'],
        'epidemiology': ['epidemiology', 'prevalence', 'incidence', 'population'],
        'genetics': ['genetic', 'gene', 'mutation', 'hereditary', 'genomic'],
    }

    THEME_NAMES = {
        'diagnosis': '진단/검출',
        'treatment': '치료/약물',
        'mechanism': '메커니즘/경로',
        'prevention': '예방/위험요인',
        'clinical': '임상연구',
        'review': '종합리뷰',
        'epidemiology': '역학연구',
        'genetics': '유전학연구'
    }

    def __init__(self, openai_api_key: str | None = None):
        """
        Args:
            openai_api_key: OpenAI API 키 (AI 요약 기능용)
        """
        self.openai_api_key = openai_api_key
        self.papers: list[dict[str, Any]] = []
        self.trend_data: dict[str, Any] = {}

    def set_papers(self, papers: list[dict[str, Any]]) -> None:
        """분석할 논문 데이터 설정"""
        self.papers = papers
        self.trend_data = {}

    def analyze_publication_trend(self) -> dict[str, Any]:
        """연도별 출판 트렌드 분석"""
        if not self.papers:
            return {}

        years: list[int] = []
        for paper in self.papers:
            pub_date = paper.get('publication_date') or paper.get('published', '')
            if pub_date:
                try:
                    if isinstance(pub_date, datetime):
                        year = pub_date.year
                    else:
                        year = int(str(pub_date)[:4])
                    if 2000 <= year <= 2030:
                        years.append(year)
                except (ValueError, TypeError):
                    pass

        year_counts = Counter(years)
        sorted_years = sorted(year_counts.items())

        self.trend_data['year_trend'] = {
            'years': [y[0] for y in sorted_years],
            'counts': [y[1] for y in sorted_years],
            'total': len(years)
        }

        return self.trend_data['year_trend']

    def extract_key_terms(self, top_n: int = 20) -> dict[str, Any]:
        """논문 초록에서 핵심 키워드 추출"""
        if not self.papers:
            return {}

        all_words: list[str] = []
        for paper in self.papers:
            abstract = paper.get('abstract') or ''
            if abstract:
                words = re.findall(r'\b[a-zA-Z]{3,}\b', abstract.lower())
                words = [w for w in words if w not in self.STOPWORDS]
                all_words.extend(words)

        word_counts = Counter(all_words)
        top_terms = word_counts.most_common(top_n)

        self.trend_data['key_terms'] = {
            'terms': [t[0] for t in top_terms],
            'counts': [t[1] for t in top_terms]
        }

        return self.trend_data['key_terms']

    def identify_emerging_topics(self) -> list[dict[str, Any]]:
        """최근 급부상하는 주제 식별"""
        if not self.papers or len(self.papers) < 5:
            return []

        current_year = datetime.now().year
        recent_papers: list[dict] = []
        older_papers: list[dict] = []

        for paper in self.papers:
            pub_date = paper.get('publication_date') or paper.get('published', '')
            try:
                if isinstance(pub_date, datetime):
                    year = pub_date.year
                else:
                    year = int(str(pub_date)[:4])

                if year >= current_year - 1:
                    recent_papers.append(paper)
                else:
                    older_papers.append(paper)
            except (ValueError, TypeError):
                pass

        if not recent_papers or not older_papers:
            return []

        def extract_terms(papers: list[dict]) -> Counter:
            words: list[str] = []
            for paper in papers:
                abstract = paper.get('abstract') or ''
                if abstract:
                    w = re.findall(r'\b[a-zA-Z]{4,}\b', abstract.lower())
                    words.extend([x for x in w if x not in self.STOPWORDS])
            return Counter(words)

        recent_terms = extract_terms(recent_papers)
        older_terms = extract_terms(older_papers)

        emerging: list[dict[str, Any]] = []
        for term, recent_count in recent_terms.most_common(50):
            older_count = older_terms.get(term, 0)
            if older_count == 0:
                growth = 999.0
            else:
                growth = (recent_count / len(recent_papers)) / (older_count / len(older_papers))

            if growth > 1.5 and recent_count >= 3:
                emerging.append({
                    'term': term,
                    'recent_count': recent_count,
                    'older_count': older_count,
                    'growth_rate': growth if growth != 999 else 999.0
                })

        emerging.sort(key=lambda x: x['growth_rate'], reverse=True)
        self.trend_data['emerging_topics'] = emerging[:10]

        return self.trend_data['emerging_topics']

    def summarize_research_content(self, keyword: str) -> dict[str, Any]:
        """연구 내용 요약 생성"""
        if not self.papers:
            return {}

        # 주요 연구 주제 추출
        research_themes: dict[str, int] = {k: 0 for k in self.THEME_KEYWORDS}

        for paper in self.papers:
            title = paper.get('title') or ''
            abstract = paper.get('abstract') or ''
            text = (title + ' ' + abstract).lower()
            for theme, keywords in self.THEME_KEYWORDS.items():
                for kw in keywords:
                    if kw in text:
                        research_themes[theme] += 1
                        break

        sorted_themes = sorted(research_themes.items(), key=lambda x: x[1], reverse=True)
        top_themes = [(t, c) for t, c in sorted_themes if c > 0][:5]

        representative_papers = self.papers[:10]

        summary_text = self._generate_basic_content_summary(keyword, top_themes)

        self.trend_data['content_summary'] = {
            'themes': top_themes,
            'representative_papers': representative_papers,
            'summary_text': summary_text
        }

        return self.trend_data['content_summary']

    def _generate_basic_content_summary(
        self,
        keyword: str,
        themes: list[tuple[str, int]]
    ) -> str:
        """기본 연구 내용 요약 텍스트 생성"""
        summary_parts = []

        if themes:
            main_themes = [self.THEME_NAMES.get(t[0], t[0]) for t in themes[:3]]
            summary_parts.append(
                f"'{keyword}' 연구는 주로 {', '.join(main_themes)} 분야에 집중되어 있습니다."
            )

        if self.papers:
            recent_titles = [(p.get('title') or '')[:80] for p in self.papers[:3]]
            summary_parts.append("\n최근 주요 연구:")
            for i, title in enumerate(recent_titles, 1):
                summary_parts.append(f"  {i}. {title}...")

        return "\n".join(summary_parts)

    async def generate_ai_summary(self, keyword: str) -> str | None:
        """OpenAI를 사용한 AI 요약 생성"""
        if not self.openai_api_key or not self.papers:
            return None

        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=self.openai_api_key)

            abstracts = []
            for paper in self.papers[:10]:
                title = paper.get('title') or ''
                abstract = (paper.get('abstract') or '')[:600]
                pub_date = paper.get('publication_date') or paper.get('published', '')
                year = ''
                if pub_date:
                    if isinstance(pub_date, datetime):
                        year = str(pub_date.year)
                    else:
                        year = str(pub_date)[:4]
                authors = paper.get('authors', [])
                author_str = ', '.join(authors[:3]) if isinstance(authors, list) else str(authors)[:50]
                abstracts.append(f"[{year}] {title}\n저자: {author_str}\n초록: {abstract}")

            combined_text = "\n\n---\n\n".join(abstracts)

            year_stats = ""
            if 'year_trend' in self.trend_data:
                years = self.trend_data['year_trend']['years'][-5:]
                counts = self.trend_data['year_trend']['counts'][-5:]
                year_stats = "\n\n연도별 논문 수: " + ", ".join(
                    [f"{y}년: {c}편" for y, c in zip(years, counts)]
                )

            prompt = f"""당신은 의학/과학 연구 동향 분석 전문가입니다. 다음은 '{keyword}'에 관한 최근 연구 논문들입니다.

총 분석 논문 수: {len(self.papers)}편{year_stats}

=== 논문 목록 ===
{combined_text}

위 연구들을 종합적으로 분석하여 '{keyword}' 분야의 연구 동향을 다음 형식으로 요약해주세요:

📌 **연구 개요**: (이 분야가 무엇을 다루는지 1-2문장)

🔬 **주요 연구 방향**:
- (최근 연구들이 집중하는 핵심 주제 2-3가지)

💡 **핵심 발견/결과**:
- (주요 연구 결과나 발견 2-3가지)

🔮 **향후 전망**:
- (연구 트렌드 및 향후 방향 1-2문장)

한국어로 명확하고 구체적으로 작성해주세요."""

            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 의학/과학 연구 동향을 분석하는 전문가입니다."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.3
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.warning(f"AI summary generation failed: {e}")
            return None

    async def analyze(self, keyword: str) -> TrendData:
        """전체 트렌드 분석 실행"""
        if not self.papers:
            return TrendData(keyword=keyword, total_papers=0)

        # 분석 수행
        self.analyze_publication_trend()
        self.extract_key_terms()
        self.identify_emerging_topics()
        self.summarize_research_content(keyword)

        # AI 요약 생성 (API 키가 있는 경우)
        if self.openai_api_key:
            ai_summary = await self.generate_ai_summary(keyword)
            if ai_summary and 'content_summary' in self.trend_data:
                self.trend_data['content_summary']['ai_summary'] = ai_summary

        return TrendData(
            keyword=keyword,
            total_papers=len(self.papers),
            year_trend=self.trend_data.get('year_trend'),
            key_terms=self.trend_data.get('key_terms'),
            emerging_topics=self.trend_data.get('emerging_topics'),
            content_summary=self.trend_data.get('content_summary'),
            search_summary=self.trend_data.get('search_summary')
        )

    def generate_report(self, keyword: str) -> str:
        """트렌드 분석 리포트 생성"""
        report = []
        report.append(f"\n{'=' * 60}")
        report.append(f"📈 '{keyword}' 연구 트렌드 분석 리포트")
        report.append("=" * 60)
        report.append(f"\n📊 분석 대상: {len(self.papers)}개 논문\n")

        # 연도별 트렌드
        if 'year_trend' in self.trend_data:
            years = self.trend_data['year_trend']['years']
            counts = self.trend_data['year_trend']['counts']
            if years:
                report.append("📅 연도별 출판 동향:")
                for year, count in list(zip(years, counts))[-5:]:
                    bar = "█" * (count * 2)
                    report.append(f"   {year}: {bar} ({count})")

                if len(counts) >= 2:
                    recent_avg = sum(counts[-2:]) / 2
                    older_avg = sum(counts[:-2]) / max(len(counts) - 2, 1) if len(counts) > 2 else counts[0]
                    if recent_avg > older_avg * 1.2:
                        report.append("\n   📈 트렌드: 상승세 (최근 연구 증가)")
                    elif recent_avg < older_avg * 0.8:
                        report.append("\n   📉 트렌드: 하락세 (연구 관심 감소)")
                    else:
                        report.append("\n   ➡️ 트렌드: 안정적 (꾸준한 연구)")

        # 핵심 키워드
        if 'key_terms' in self.trend_data:
            report.append("\n🔑 핵심 키워드 (Top 10):")
            terms = self.trend_data['key_terms']['terms'][:10]
            counts = self.trend_data['key_terms']['counts'][:10]
            for i, (term, count) in enumerate(zip(terms, counts), 1):
                report.append(f"   {i:2d}. {term} ({count}회)")

        # 급부상 주제
        if 'emerging_topics' in self.trend_data and self.trend_data['emerging_topics']:
            report.append("\n🚀 급부상 주제 (최근 vs 과거):")
            for i, e in enumerate(self.trend_data['emerging_topics'][:5], 1):
                growth = e['growth_rate']
                growth_str = "NEW!" if growth >= 999 else f"{growth:.1f}x"
                report.append(f"   {i}. {e['term']} - {growth_str} 성장")

        # 연구 내용 요약
        if 'content_summary' in self.trend_data:
            summary_data = self.trend_data['content_summary']
            report.append(f"\n{'-' * 60}")
            report.append("📋 연구 내용 요약")
            report.append("-" * 60)

            if 'themes' in summary_data and summary_data['themes']:
                report.append("\n🔬 주요 연구 분야:")
                for theme, count in summary_data['themes'][:5]:
                    theme_name = self.THEME_NAMES.get(theme, theme)
                    bar = "▓" * min(count, 20)
                    report.append(f"   • {theme_name}: {bar} ({count}편)")

            if 'ai_summary' in summary_data:
                report.append("\n📝 AI 연구 동향 요약:")
                for line in summary_data['ai_summary'].split('\n'):
                    if line.strip():
                        report.append(f"   {line}")

        report.append(f"\n{'=' * 60}")

        return "\n".join(report)


# Singleton instance factory
def create_trend_analyzer(openai_api_key: str | None = None) -> TrendAnalyzer:
    """TrendAnalyzer 인스턴스 생성"""
    return TrendAnalyzer(openai_api_key=openai_api_key)
