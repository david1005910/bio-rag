import { useQuery, useMutation } from '@tanstack/react-query';
import { arxivApi } from '../api/arxiv';
import type { ArXivSearchQuery } from '../types';

export const arxivKeys = {
  all: ['arxiv'] as const,
  search: (query: string) => [...arxivKeys.all, 'search', query] as const,
  paper: (id: string) => [...arxivKeys.all, 'paper', id] as const,
};

/**
 * ArXiv 논문 검색 훅
 */
export function useArxivSearch(query: string, options?: { enabled?: boolean; maxResults?: number }) {
  return useQuery({
    queryKey: arxivKeys.search(query),
    queryFn: () => arxivApi.searchGet(query, options?.maxResults ?? 10),
    enabled: options?.enabled !== false && query.length > 0,
  });
}

/**
 * ArXiv 논문 검색 (mutation)
 */
export function useArxivSearchMutation() {
  return useMutation({
    mutationFn: (searchQuery: ArXivSearchQuery) => arxivApi.search(searchQuery),
  });
}

/**
 * ArXiv 논문 상세 조회 훅
 */
export function useArxivPaper(arxivId: string, options?: { enabled?: boolean }) {
  return useQuery({
    queryKey: arxivKeys.paper(arxivId),
    queryFn: () => arxivApi.getPaper(arxivId),
    enabled: options?.enabled !== false && arxivId.length > 0,
  });
}
