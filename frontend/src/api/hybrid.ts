/**
 * Hybrid Search API - Dense (PubMedBERT) + Sparse (SPLADE) with RRF
 */

import api from './client';

// Types
export interface ScoreBreakdown {
  dense_score: number;
  sparse_score: number;
  rrf_score: number;
  dense_rank: number;
  sparse_rank: number;
  dense_contribution: number;
  sparse_contribution: number;
}

export interface HybridSearchResultItem {
  doc_id: string;
  pmid: string;
  title: string;
  content: string;
  authors: string[];
  journal: string;
  publication_date: string;
  scores: ScoreBreakdown;
}

export interface ScoreDistribution {
  min: number;
  max: number;
  avg: number;
}

export interface ScoreVisualization {
  dense_scores: ScoreDistribution;
  sparse_scores: ScoreDistribution;
  rrf_scores: ScoreDistribution;
  contributions: Array<{
    doc_id: string;
    dense_contribution: number;
    sparse_contribution: number;
  }>;
  config: {
    dense_weight: number;
    sparse_weight: number;
  };
}

export interface HybridSearchResponse {
  query: string;
  total_results: number;
  dense_weight: number;
  sparse_weight: number;
  rrf_k: number;
  results: HybridSearchResultItem[];
  score_visualization: ScoreVisualization;
}

export interface IndexPapersRequest {
  query: string;
  max_papers?: number;
}

export interface IndexPapersResponse {
  indexed_count: number;
  collection_name: string;
  total_in_collection: number;
}

export interface IndexStats {
  collection_name: string;
  document_count: number;
  dense_dimension: number;
  config: {
    dense_weight: number;
    sparse_weight: number;
    rrf_k: number;
  };
}

// API functions
export const hybridApi = {
  /**
   * Index papers from PubMed to Qdrant
   */
  async indexPapers(request: IndexPapersRequest): Promise<IndexPapersResponse> {
    const response = await api.post<IndexPapersResponse>('/hybrid/index', request);
    return response.data;
  },

  /**
   * Perform hybrid search with RRF fusion
   */
  async search(
    query: string,
    options?: {
      topK?: number;
      denseWeight?: number;
      sparseWeight?: number;
      rrfK?: number;
    }
  ): Promise<HybridSearchResponse> {
    const params = new URLSearchParams();
    params.append('query', query);
    if (options?.topK) params.append('top_k', options.topK.toString());
    if (options?.denseWeight !== undefined) params.append('dense_weight', options.denseWeight.toString());
    if (options?.sparseWeight !== undefined) params.append('sparse_weight', options.sparseWeight.toString());
    if (options?.rrfK) params.append('rrf_k', options.rrfK.toString());

    const response = await api.get<HybridSearchResponse>(`/hybrid/search?${params.toString()}`);
    return response.data;
  },

  /**
   * Get index statistics
   */
  async getStats(): Promise<IndexStats> {
    const response = await api.get<IndexStats>('/hybrid/stats');
    return response.data;
  },

  /**
   * Clear all indexed documents
   */
  async clearIndex(): Promise<{ message: string }> {
    const response = await api.delete<{ message: string }>('/hybrid/clear');
    return response.data;
  },
};
