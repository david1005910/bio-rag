import api from './client';
import type { ArXivPaper, ArXivSearchQuery, ArXivSearchResult } from '../types';

export const arxivApi = {
  /**
   * ArXiv 논문 검색
   */
  async search(query: ArXivSearchQuery): Promise<ArXivSearchResult> {
    const response = await api.post<ArXivSearchResult>('/arxiv/search', query);
    return response.data;
  },

  /**
   * ArXiv 논문 검색 (GET)
   */
  async searchGet(
    query: string,
    maxResults: number = 10,
    sortBy?: string,
    sortOrder?: string
  ): Promise<ArXivSearchResult> {
    const response = await api.get<ArXivSearchResult>('/arxiv/search', {
      params: {
        query,
        max_results: maxResults,
        sort_by: sortBy,
        sort_order: sortOrder,
      },
    });
    return response.data;
  },

  /**
   * ArXiv 논문 상세 조회
   */
  async getPaper(arxivId: string): Promise<ArXivPaper> {
    const response = await api.get<ArXivPaper>(`/arxiv/paper/${arxivId}`);
    return response.data;
  },
};
