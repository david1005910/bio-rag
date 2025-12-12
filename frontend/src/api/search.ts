import api from './client';
import type { SearchQuery, SearchResult, PaperSummary } from '../types';

export const searchApi = {
  async search(query: SearchQuery): Promise<SearchResult> {
    const response = await api.post<SearchResult>('/search', query);
    return response.data;
  },

  async getSimilarPapers(pmid: string, limit: number = 5): Promise<PaperSummary[]> {
    const response = await api.get<PaperSummary[]>(`/search/similar/${pmid}`, {
      params: { limit },
    });
    return response.data;
  },

  async getPapersByPmids(pmids: string[]): Promise<PaperSummary[]> {
    const response = await api.post<PaperSummary[]>('/search/by-pmids', pmids);
    return response.data;
  },
};
