import { useQuery, useMutation } from '@tanstack/react-query';
import { pubmedApi } from '../api/pubmed';

export const pubmedKeys = {
  all: ['pubmed'] as const,
  search: (query: string) => [...pubmedKeys.all, 'search', query] as const,
  paper: (pmid: string) => [...pubmedKeys.all, 'paper', pmid] as const,
};

/**
 * PubMed 논문 검색 훅
 */
export function usePubMedSearch(query: string, options?: { enabled?: boolean; maxResults?: number }) {
  return useQuery({
    queryKey: pubmedKeys.search(query),
    queryFn: () => pubmedApi.search(query, options?.maxResults ?? 20),
    enabled: options?.enabled !== false && query.length > 0,
  });
}

/**
 * PubMed 논문 검색 (mutation)
 */
export function usePubMedSearchMutation() {
  return useMutation({
    mutationFn: ({ query, maxResults }: { query: string; maxResults?: number }) =>
      pubmedApi.search(query, maxResults),
  });
}

/**
 * PubMed 논문 상세 조회 훅
 */
export function usePubMedPaper(pmid: string, options?: { enabled?: boolean }) {
  return useQuery({
    queryKey: pubmedKeys.paper(pmid),
    queryFn: () => pubmedApi.getPaper(pmid),
    enabled: options?.enabled !== false && pmid.length > 0,
  });
}
