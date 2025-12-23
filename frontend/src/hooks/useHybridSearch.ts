/**
 * React Query hooks for Hybrid Search API
 */

import { useMutation, useQuery } from '@tanstack/react-query';
import { hybridApi, IndexPapersRequest } from '../api/hybrid';

// Query keys
export const hybridKeys = {
  all: ['hybrid'] as const,
  stats: () => [...hybridKeys.all, 'stats'] as const,
  search: (query: string) => [...hybridKeys.all, 'search', query] as const,
};

/**
 * Hook to get index statistics
 */
export function useHybridStats() {
  return useQuery({
    queryKey: hybridKeys.stats(),
    queryFn: () => hybridApi.getStats(),
    staleTime: 30000, // 30 seconds
  });
}

/**
 * Hook to index papers from PubMed
 */
export function useIndexPapers() {
  return useMutation({
    mutationFn: (request: IndexPapersRequest) => hybridApi.indexPapers(request),
  });
}

/**
 * Hook to perform hybrid search
 */
export function useHybridSearch(
  query: string,
  options?: {
    enabled?: boolean;
    topK?: number;
    denseWeight?: number;
    sparseWeight?: number;
    rrfK?: number;
  }
) {
  return useQuery({
    queryKey: hybridKeys.search(query),
    queryFn: () =>
      hybridApi.search(query, {
        topK: options?.topK,
        denseWeight: options?.denseWeight,
        sparseWeight: options?.sparseWeight,
        rrfK: options?.rrfK,
      }),
    enabled: options?.enabled !== false && query.length > 0,
    staleTime: 60000, // 1 minute
  });
}

/**
 * Hook to perform hybrid search on demand (mutation)
 */
export function useHybridSearchMutation() {
  return useMutation({
    mutationFn: ({
      query,
      topK,
      denseWeight,
      sparseWeight,
      rrfK,
    }: {
      query: string;
      topK?: number;
      denseWeight?: number;
      sparseWeight?: number;
      rrfK?: number;
    }) =>
      hybridApi.search(query, { topK, denseWeight, sparseWeight, rrfK }),
  });
}

/**
 * Hook to clear index
 */
export function useClearIndex() {
  return useMutation({
    mutationFn: () => hybridApi.clearIndex(),
  });
}
