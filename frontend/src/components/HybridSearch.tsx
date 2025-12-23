/**
 * Hybrid Search Component with RRF Score Visualization
 * - Dense: BGE-M3 (multilingual semantic, 1024 dim)
 * - Sparse: BGE-M3 sparse (lexical)
 * - Fusion: RRF (Reciprocal Rank Fusion)
 * - Supports Korean and 100+ languages
 */

import { useState } from 'react';
import {
  useHybridStats,
  useIndexPapers,
  useHybridSearchMutation,
} from '../hooks/useHybridSearch';
import { HybridSearchResultItem, ScoreVisualization } from '../api/hybrid';

// Score bar component
function ScoreBar({
  dense,
  sparse,
  label,
}: {
  dense: number;
  sparse: number;
  label: string;
}) {
  const total = dense + sparse;
  const densePercent = total > 0 ? (dense / total) * 100 : 50;
  const sparsePercent = total > 0 ? (sparse / total) * 100 : 50;

  return (
    <div className="mb-2">
      <div className="flex justify-between text-xs text-gray-400 mb-1">
        <span>{label}</span>
        <span>
          Dense: {densePercent.toFixed(1)}% | Sparse: {sparsePercent.toFixed(1)}%
        </span>
      </div>
      <div className="flex h-2 rounded-full overflow-hidden bg-gray-700">
        <div
          className="bg-blue-500 transition-all duration-300"
          style={{ width: `${densePercent}%` }}
          title={`Dense: ${dense.toFixed(4)}`}
        />
        <div
          className="bg-green-500 transition-all duration-300"
          style={{ width: `${sparsePercent}%` }}
          title={`Sparse: ${sparse.toFixed(4)}`}
        />
      </div>
    </div>
  );
}

// RRF Score Card
function RRFScoreCard({ result, rank }: { result: HybridSearchResultItem; rank: number }) {
  const { scores } = result;

  return (
    <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-4 border border-gray-700/50 hover:border-blue-500/50 transition-all">
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="bg-gradient-to-r from-blue-500 to-purple-500 text-white text-xs font-bold px-2 py-1 rounded">
            #{rank}
          </span>
          <span className="text-xs text-gray-400">
            RRF: {(scores.rrf_score * 1000).toFixed(3)}
          </span>
        </div>
        {result.pmid && (
          <a
            href={`https://pubmed.ncbi.nlm.nih.gov/${result.pmid}`}
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-blue-400 hover:text-blue-300"
          >
            PMID: {result.pmid}
          </a>
        )}
      </div>

      {/* Title */}
      <h3 className="text-white font-medium mb-2 line-clamp-2">{result.title}</h3>

      {/* Score Visualization */}
      <div className="bg-gray-900/50 rounded-lg p-3 mb-3">
        <ScoreBar
          dense={scores.dense_score}
          sparse={scores.sparse_score}
          label="Score Contribution"
        />

        <div className="grid grid-cols-2 gap-2 text-xs mt-2">
          <div className="bg-blue-500/10 rounded p-2">
            <div className="text-blue-400 font-medium">Dense (BGE-M3)</div>
            <div className="text-white">
              Score: {scores.dense_score.toFixed(4)}
            </div>
            <div className="text-gray-400">
              Rank: {scores.dense_rank > 0 ? `#${scores.dense_rank}` : 'N/A'}
            </div>
          </div>
          <div className="bg-green-500/10 rounded p-2">
            <div className="text-green-400 font-medium">Sparse (Lexical)</div>
            <div className="text-white">
              Score: {scores.sparse_score.toFixed(4)}
            </div>
            <div className="text-gray-400">
              Rank: {scores.sparse_rank > 0 ? `#${scores.sparse_rank}` : 'N/A'}
            </div>
          </div>
        </div>
      </div>

      {/* Content Preview */}
      <p className="text-sm text-gray-300 line-clamp-3">{result.content}</p>

      {/* Metadata */}
      {(result.authors.length > 0 || result.journal) && (
        <div className="mt-2 text-xs text-gray-400">
          {result.authors.length > 0 && (
            <span>{result.authors.slice(0, 3).join(', ')}{result.authors.length > 3 ? ' et al.' : ''}</span>
          )}
          {result.journal && <span> | {result.journal}</span>}
          {result.publication_date && <span> ({result.publication_date.slice(0, 4)})</span>}
        </div>
      )}
    </div>
  );
}

// Score Distribution Chart
function ScoreDistributionChart({ visualization }: { visualization: ScoreVisualization }) {
  return (
    <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-4 border border-gray-700/50">
      <h3 className="text-white font-medium mb-4">Score Distribution</h3>

      <div className="grid grid-cols-3 gap-4 text-sm">
        {/* Dense Scores */}
        <div className="text-center">
          <div className="text-blue-400 font-medium mb-2">Dense</div>
          <div className="space-y-1">
            <div className="flex justify-between">
              <span className="text-gray-400">Min:</span>
              <span className="text-white">{visualization.dense_scores.min.toFixed(4)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Avg:</span>
              <span className="text-white">{visualization.dense_scores.avg.toFixed(4)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Max:</span>
              <span className="text-white">{visualization.dense_scores.max.toFixed(4)}</span>
            </div>
          </div>
        </div>

        {/* Sparse Scores */}
        <div className="text-center">
          <div className="text-green-400 font-medium mb-2">Sparse</div>
          <div className="space-y-1">
            <div className="flex justify-between">
              <span className="text-gray-400">Min:</span>
              <span className="text-white">{visualization.sparse_scores.min.toFixed(4)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Avg:</span>
              <span className="text-white">{visualization.sparse_scores.avg.toFixed(4)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Max:</span>
              <span className="text-white">{visualization.sparse_scores.max.toFixed(4)}</span>
            </div>
          </div>
        </div>

        {/* RRF Scores */}
        <div className="text-center">
          <div className="text-purple-400 font-medium mb-2">RRF</div>
          <div className="space-y-1">
            <div className="flex justify-between">
              <span className="text-gray-400">Min:</span>
              <span className="text-white">{(visualization.rrf_scores.min * 1000).toFixed(3)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Avg:</span>
              <span className="text-white">{(visualization.rrf_scores.avg * 1000).toFixed(3)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Max:</span>
              <span className="text-white">{(visualization.rrf_scores.max * 1000).toFixed(3)}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Weight Config */}
      <div className="mt-4 pt-4 border-t border-gray-700">
        <div className="flex items-center justify-center gap-4 text-xs">
          <span className="text-gray-400">Weights:</span>
          <span className="bg-blue-500/20 text-blue-400 px-2 py-1 rounded">
            Dense: {(visualization.config.dense_weight * 100).toFixed(0)}%
          </span>
          <span className="bg-green-500/20 text-green-400 px-2 py-1 rounded">
            Sparse: {(visualization.config.sparse_weight * 100).toFixed(0)}%
          </span>
        </div>
      </div>
    </div>
  );
}

// Main Component
export default function HybridSearch() {
  const [indexQuery, setIndexQuery] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [denseWeight, setDenseWeight] = useState(0.7);
  const [sparseWeight, setSparseWeight] = useState(0.3);
  const [maxPapers, setMaxPapers] = useState(50);

  const { data: stats, refetch: refetchStats } = useHybridStats();
  const indexMutation = useIndexPapers();
  const searchMutation = useHybridSearchMutation();

  const handleIndex = async () => {
    if (!indexQuery.trim()) return;
    await indexMutation.mutateAsync({ query: indexQuery, max_papers: maxPapers });
    refetchStats();
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    await searchMutation.mutateAsync({
      query: searchQuery,
      topK: 10,
      denseWeight,
      sparseWeight,
    });
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center">
        <h2 className="text-2xl font-bold text-white mb-2">
          Hybrid Search
        </h2>
        <p className="text-gray-400">
          BGE-M3 (Dense + Sparse) with RRF Fusion | 한글 검색 지원
        </p>
      </div>

      {/* Stats */}
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-4 border border-gray-700/50">
        <div className="flex items-center justify-between">
          <div className="text-sm text-gray-400">
            Collection: <span className="text-white">{stats?.collection_name || 'N/A'}</span>
          </div>
          <div className="text-sm text-gray-400">
            Documents: <span className="text-white">{stats?.document_count || 0}</span>
          </div>
          <div className="text-sm text-gray-400">
            Dense Dim: <span className="text-white">{stats?.dense_dimension || 768}</span>
          </div>
        </div>
      </div>

      {/* Indexing Section */}
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-4 border border-gray-700/50">
        <h3 className="text-white font-medium mb-4">1. Index Papers from PubMed</h3>
        <div className="flex gap-4">
          <input
            type="text"
            placeholder="Search query (e.g., cancer immunotherapy)"
            value={indexQuery}
            onChange={(e) => setIndexQuery(e.target.value)}
            className="flex-1 bg-gray-900/50 border border-gray-600 rounded-lg px-4 py-2 text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
          />
          <input
            type="number"
            placeholder="Max papers"
            value={maxPapers}
            onChange={(e) => setMaxPapers(Number(e.target.value))}
            className="w-24 bg-gray-900/50 border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
          />
          <button
            onClick={handleIndex}
            disabled={indexMutation.isPending || !indexQuery.trim()}
            className="bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700 disabled:opacity-50 text-white px-6 py-2 rounded-lg font-medium transition-all"
          >
            {indexMutation.isPending ? 'Indexing...' : 'Index'}
          </button>
        </div>
        {indexMutation.isSuccess && (
          <div className="mt-2 text-sm text-green-400">
            Indexed {indexMutation.data?.indexed_count} papers. Total: {indexMutation.data?.total_in_collection}
          </div>
        )}
      </div>

      {/* Search Section */}
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-4 border border-gray-700/50">
        <h3 className="text-white font-medium mb-4">2. Hybrid Search</h3>

        {/* Weight Sliders */}
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div>
            <label className="text-sm text-gray-400 flex justify-between mb-1">
              <span>Dense Weight (Semantic)</span>
              <span className="text-blue-400">{(denseWeight * 100).toFixed(0)}%</span>
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={denseWeight}
              onChange={(e) => {
                const value = Number(e.target.value);
                setDenseWeight(value);
                setSparseWeight(1 - value);
              }}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
            />
          </div>
          <div>
            <label className="text-sm text-gray-400 flex justify-between mb-1">
              <span>Sparse Weight (Lexical)</span>
              <span className="text-green-400">{(sparseWeight * 100).toFixed(0)}%</span>
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={sparseWeight}
              onChange={(e) => {
                const value = Number(e.target.value);
                setSparseWeight(value);
                setDenseWeight(1 - value);
              }}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-green-500"
            />
          </div>
        </div>

        {/* Search Input */}
        <div className="flex gap-4">
          <input
            type="text"
            placeholder="검색어 입력 (한글/영어 모두 지원) e.g., 암 면역치료"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
            className="flex-1 bg-gray-900/50 border border-gray-600 rounded-lg px-4 py-2 text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
          />
          <button
            onClick={handleSearch}
            disabled={searchMutation.isPending || !searchQuery.trim()}
            className="bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 disabled:opacity-50 text-white px-6 py-2 rounded-lg font-medium transition-all"
          >
            {searchMutation.isPending ? 'Searching...' : 'Search'}
          </button>
        </div>
      </div>

      {/* Error */}
      {searchMutation.isError && (
        <div className="bg-red-500/10 border border-red-500/50 rounded-xl p-4 text-red-400">
          {(searchMutation.error as Error).message}
        </div>
      )}

      {/* Results */}
      {searchMutation.data && (
        <div className="space-y-4">
          {/* Score Distribution */}
          <ScoreDistributionChart visualization={searchMutation.data.score_visualization} />

          {/* Results Header */}
          <div className="flex items-center justify-between">
            <h3 className="text-white font-medium">
              Results ({searchMutation.data.total_results})
            </h3>
            <div className="text-sm text-gray-400">
              Query: "{searchMutation.data.query}" | RRF k={searchMutation.data.rrf_k}
            </div>
          </div>

          {/* Result Cards */}
          <div className="grid gap-4">
            {searchMutation.data.results.map((result, index) => (
              <RRFScoreCard key={result.doc_id} result={result} rank={index + 1} />
            ))}
          </div>

          {searchMutation.data.results.length === 0 && (
            <div className="text-center text-gray-400 py-8">
              No results found. Try indexing more papers first.
            </div>
          )}
        </div>
      )}
    </div>
  );
}
