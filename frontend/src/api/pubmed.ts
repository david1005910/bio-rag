import api from './client';

export interface PubMedPaper {
  pmid: string;
  title: string;
  abstract: string | null;
  authors: string[] | null;
  journal: string | null;
  publication_date: string | null;
  doi: string | null;
  keywords: string[] | null;
  mesh_terms: string[] | null;
  pdf_url: string | null;
}

export interface PubMedSearchResult {
  results: PubMedPaper[];
  total: number;
  query: string;
}

export const pubmedApi = {
  /**
   * PubMed 실시간 검색
   */
  async search(query: string, maxResults: number = 20): Promise<PubMedSearchResult> {
    const response = await api.get<PubMedSearchResult>('/pubmed/search', {
      params: { query, max_results: maxResults },
    });
    return response.data;
  },

  /**
   * PubMed 논문 상세 조회
   */
  async getPaper(pmid: string): Promise<PubMedPaper> {
    const response = await api.get<PubMedPaper>(`/pubmed/paper/${pmid}`);
    return response.data;
  },
};
