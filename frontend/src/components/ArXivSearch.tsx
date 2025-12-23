import { useState, FormEvent } from 'react';
import { useArxivSearch } from '../hooks/useArxiv';
import type { ArXivPaper } from '../types';

const glassPanel = {
  background: 'rgba(255, 255, 255, 0.18)',
  backdropFilter: 'blur(16px)',
  WebkitBackdropFilter: 'blur(16px)',
  border: '1px solid rgba(255, 255, 255, 0.4)',
  boxShadow: '0px 4px 24px rgba(0, 0, 0, 0.15), inset 0px 0px 12px rgba(255, 255, 255, 0.25)',
};

const glassInput = {
  background: 'rgba(255, 255, 255, 0.15)',
  border: '1px solid rgba(255, 255, 255, 0.3)',
  backdropFilter: 'blur(8px)',
  textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)',
};

const gradientButton = {
  background: 'linear-gradient(135deg, #06b6d4, #8b5cf6)',
  boxShadow: '0px 4px 16px rgba(139, 92, 246, 0.4)',
  textShadow: '0px 1px 2px rgba(0, 0, 0, 0.3)',
};

export default function ArXivSearch() {
  const [query, setQuery] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [maxResults, setMaxResults] = useState(10);

  const { data, isLoading, error } = useArxivSearch(searchQuery, {
    enabled: searchQuery.length > 0,
    maxResults,
  });

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      setSearchQuery(query.trim());
    }
  };

  return (
    <div className="space-y-6">
      {/* Search Form */}
      <div className="rounded-2xl p-6" style={glassPanel}>
        <h2
          className="text-xl font-bold text-white mb-4"
          style={{ textShadow: '0px 2px 4px rgba(0, 0, 0, 0.3)' }}
        >
          ArXiv Paper Search
        </h2>
        <form onSubmit={handleSubmit}>
          <div className="flex gap-4 mb-4">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search arXiv papers (e.g., machine learning, CRISPR)..."
              className="flex-1 rounded-xl px-4 py-3 text-white placeholder-white/60 focus:outline-none focus:ring-2 focus:ring-cyan-400/50 transition-all"
              style={glassInput}
            />
            <select
              value={maxResults}
              onChange={(e) => setMaxResults(Number(e.target.value))}
              className="rounded-xl px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-cyan-400/50"
              style={glassInput}
            >
              <option value={5}>5 results</option>
              <option value={10}>10 results</option>
              <option value={20}>20 results</option>
              <option value={50}>50 results</option>
            </select>
            <button
              type="submit"
              disabled={isLoading || !query.trim()}
              className="px-6 py-3 rounded-xl text-white font-medium focus:outline-none focus:ring-2 focus:ring-cyan-400/50 disabled:opacity-50 disabled:cursor-not-allowed transition-all hover:scale-105 active:scale-95"
              style={gradientButton}
            >
              {isLoading ? 'Searching...' : 'Search'}
            </button>
          </div>
        </form>
      </div>

      {/* Error */}
      {error && (
        <div
          className="rounded-2xl p-4"
          style={{
            background: 'rgba(239, 68, 68, 0.3)',
            backdropFilter: 'blur(16px)',
            border: '1px solid rgba(239, 68, 68, 0.5)',
          }}
        >
          <p className="text-white" style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.3)' }}>
            Search failed. Please try again.
          </p>
        </div>
      )}

      {/* Results */}
      {data && (
        <div>
          <p
            className="text-sm text-white/80 mb-4"
            style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)' }}
          >
            Found {data.total} results for "{data.query}"
          </p>
          <div className="space-y-4">
            {data.results.map((paper) => (
              <ArXivPaperCard key={paper.arxiv_id} paper={paper} />
            ))}
          </div>
        </div>
      )}

      {/* Empty State */}
      {!searchQuery && !error && (
        <div className="text-center py-12 rounded-2xl" style={glassPanel}>
          <p
            className="text-white/80"
            style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)' }}
          >
            Enter a search query to find papers from arXiv.
          </p>
        </div>
      )}
    </div>
  );
}

function ArXivPaperCard({ paper }: { paper: ArXivPaper }) {
  return (
    <div
      className="rounded-2xl p-6 transition-all hover:scale-[1.01]"
      style={glassPanel}
    >
      <h3 className="text-lg font-semibold mb-2">
        <a
          href={`https://arxiv.org/abs/${paper.arxiv_id}`}
          target="_blank"
          rel="noopener noreferrer"
          className="text-white hover:text-cyan-300 transition-colors"
          style={{ textShadow: '0px 1px 3px rgba(0, 0, 0, 0.25)' }}
        >
          {paper.title}
        </a>
      </h3>

      <div
        className="text-sm text-white/70 mb-2 flex flex-wrap gap-2"
        style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)' }}
      >
        <span className="font-medium">arXiv:{paper.arxiv_id}</span>
        <span>|</span>
        <span>{new Date(paper.published).toLocaleDateString()}</span>
        {paper.categories.length > 0 && (
          <>
            <span>|</span>
            <span>{paper.categories.slice(0, 3).join(', ')}</span>
          </>
        )}
      </div>

      {paper.authors.length > 0 && (
        <div
          className="text-sm text-white/60 mb-3"
          style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)' }}
        >
          {paper.authors.slice(0, 5).join(', ')}
          {paper.authors.length > 5 && ` et al. (+${paper.authors.length - 5})`}
        </div>
      )}

      <p
        className="text-sm text-white/80 line-clamp-3 mb-4"
        style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)' }}
      >
        {paper.abstract}
      </p>

      <div className="flex gap-2">
        <a
          href={paper.pdf_url}
          target="_blank"
          rel="noopener noreferrer"
          className="px-3 py-1.5 rounded-lg text-xs text-white font-medium transition-all hover:scale-105"
          style={{
            background: 'rgba(6, 182, 212, 0.5)',
            border: '1px solid rgba(6, 182, 212, 0.7)',
          }}
        >
          PDF
        </a>
        <a
          href={`https://arxiv.org/abs/${paper.arxiv_id}`}
          target="_blank"
          rel="noopener noreferrer"
          className="px-3 py-1.5 rounded-lg text-xs text-white font-medium transition-all hover:scale-105"
          style={{
            background: 'rgba(139, 92, 246, 0.5)',
            border: '1px solid rgba(139, 92, 246, 0.7)',
          }}
        >
          Abstract
        </a>
      </div>
    </div>
  );
}
