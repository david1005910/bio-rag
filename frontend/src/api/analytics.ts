import api from './client';
import type {
  TrendAnalysisRequest,
  TrendAnalysisResponse,
  QuickTrendResponse,
} from '../types';

export const analyticsApi = {
  /**
   * 연구 트렌드 분석
   */
  async analyzeTrends(request: TrendAnalysisRequest): Promise<TrendAnalysisResponse> {
    const response = await api.post<TrendAnalysisResponse>('/analytics/trends', request);
    return response.data;
  },

  /**
   * 빠른 트렌드 분석 (AI 요약 제외)
   */
  async quickTrend(keyword: string, maxPapers: number = 30): Promise<QuickTrendResponse> {
    const response = await api.get<QuickTrendResponse>('/analytics/trends/quick', {
      params: {
        keyword,
        max_papers: maxPapers,
      },
    });
    return response.data;
  },

  /**
   * 제공된 논문 데이터로 트렌드 분석
   */
  async analyzeTrendsFromPapers(
    papers: Array<{ title: string; abstract: string; publication_date?: string }>,
    keyword: string
  ): Promise<TrendAnalysisResponse> {
    const response = await api.post<TrendAnalysisResponse>(
      '/analytics/trends/from-papers',
      papers,
      { params: { keyword } }
    );
    return response.data;
  },
};
