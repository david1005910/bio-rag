import { useQuery, useMutation } from '@tanstack/react-query';
import { analyticsApi } from '../api/analytics';
import type { TrendAnalysisRequest } from '../types';

export const analyticsKeys = {
  all: ['analytics'] as const,
  trends: () => [...analyticsKeys.all, 'trends'] as const,
  quickTrend: (keyword: string) => [...analyticsKeys.all, 'quick', keyword] as const,
};

/**
 * 빠른 트렌드 분석 훅
 */
export function useQuickTrend(keyword: string, options?: { enabled?: boolean; maxPapers?: number }) {
  return useQuery({
    queryKey: analyticsKeys.quickTrend(keyword),
    queryFn: () => analyticsApi.quickTrend(keyword, options?.maxPapers ?? 30),
    enabled: options?.enabled !== false && keyword.length > 0,
  });
}

/**
 * 연구 트렌드 분석 (mutation)
 */
export function useAnalyzeTrends() {
  return useMutation({
    mutationFn: (request: TrendAnalysisRequest) => analyticsApi.analyzeTrends(request),
  });
}

/**
 * 제공된 논문으로 트렌드 분석 (mutation)
 */
export function useAnalyzeTrendsFromPapers() {
  return useMutation({
    mutationFn: ({
      papers,
      keyword,
    }: {
      papers: Array<{ title: string; abstract: string; publication_date?: string }>;
      keyword: string;
    }) => analyticsApi.analyzeTrendsFromPapers(papers, keyword),
  });
}
