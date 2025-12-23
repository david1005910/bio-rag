import { useState, FormEvent } from 'react';
import { useQuery } from '@tanstack/react-query';
import { searchApi } from '../api/search';
import type { SearchResult, PaperSummary } from '../types';

interface ApiError {
  response?: {
    data?: {
      detail?: string;
    };
  };
  message?: string;
}

export default function Search() {
  const [query, setQuery] = useState('');
  const [searchQuery, setSearchQuery] = useState('');

  const { data, isLoading, error } = useQuery<SearchResult, ApiError>({
    queryKey: ['search', searchQuery],
    queryFn: () => searchApi.search({ query: searchQuery, limit: 20 }),
    enabled: !!searchQuery,
    retry: false,
  });

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      setSearchQuery(query.trim());
    }
  };

  const getErrorMessage = (err: ApiError): string => {
    return err?.response?.data?.detail || err?.message || 'An error occurred while searching.';
  };

  return (
    <div className="py-8 px-4">
      <div className="max-w-4xl mx-auto">
        {/* Search Form - Glass Panel */}
        <div
          className="rounded-2xl p-6 mb-8"
          style={{
            background: 'rgba(255, 255, 255, 0.18)',
            backdropFilter: 'blur(16px)',
            WebkitBackdropFilter: 'blur(16px)',
            border: '1px solid rgba(255, 255, 255, 0.4)',
            boxShadow: '0px 4px 24px rgba(0, 0, 0, 0.15), inset 0px 0px 12px rgba(255, 255, 255, 0.25)',
          }}
        >
          <h1
            className="text-2xl font-bold text-white mb-4"
            style={{ textShadow: '0px 2px 4px rgba(0, 0, 0, 0.3)' }}
          >
            Paper Search
          </h1>
          <form onSubmit={handleSubmit}>
            <div className="flex gap-4">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search for biomedical papers..."
                className="flex-1 rounded-xl px-4 py-3 text-white placeholder-white/60 focus:outline-none focus:ring-2 focus:ring-cyan-400/50 transition-all"
                style={{
                  background: 'rgba(255, 255, 255, 0.15)',
                  border: '1px solid rgba(255, 255, 255, 0.3)',
                  backdropFilter: 'blur(8px)',
                  textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)',
                }}
              />
              <button
                type="submit"
                disabled={isLoading}
                className="px-6 py-3 rounded-xl text-white font-medium focus:outline-none focus:ring-2 focus:ring-cyan-400/50 disabled:opacity-50 disabled:cursor-not-allowed transition-all hover:scale-105 active:scale-95"
                style={{
                  background: 'linear-gradient(135deg, #06b6d4, #8b5cf6)',
                  boxShadow: '0px 4px 16px rgba(139, 92, 246, 0.4)',
                  textShadow: '0px 1px 2px rgba(0, 0, 0, 0.3)',
                }}
              >
                {isLoading ? 'Searching...' : 'Search'}
              </button>
            </div>
          </form>
        </div>

        {/* Error Message */}
        {error && (
          <div
            className="rounded-2xl p-4 mb-6"
            style={{
              background: 'rgba(239, 68, 68, 0.3)',
              backdropFilter: 'blur(16px)',
              WebkitBackdropFilter: 'blur(16px)',
              border: '1px solid rgba(239, 68, 68, 0.5)',
              boxShadow: '0px 4px 24px rgba(0, 0, 0, 0.15)',
            }}
          >
            <p
              className="text-white"
              style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.3)' }}
            >
              {getErrorMessage(error)}
            </p>
          </div>
        )}

        {/* Results */}
        {data && (
          <div>
            <div className="mb-4">
              <p
                className="text-sm text-white/80"
                style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)' }}
              >
                Found {data.total} results in {data.query_time_ms}ms
              </p>
            </div>

            <div className="space-y-4">
              {data.results.map((paper) => (
                <PaperCard key={paper.pmid} paper={paper} />
              ))}
            </div>

            {data.results.length === 0 && (
              <div
                className="text-center py-12 rounded-2xl"
                style={{
                  background: 'rgba(255, 255, 255, 0.18)',
                  backdropFilter: 'blur(16px)',
                  border: '1px solid rgba(255, 255, 255, 0.4)',
                }}
              >
                <p
                  className="text-white/80"
                  style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)' }}
                >
                  No papers found for your search query.
                </p>
              </div>
            )}
          </div>
        )}

        {!searchQuery && !error && (
          <div
            className="text-center py-12 rounded-2xl"
            style={{
              background: 'rgba(255, 255, 255, 0.18)',
              backdropFilter: 'blur(16px)',
              WebkitBackdropFilter: 'blur(16px)',
              border: '1px solid rgba(255, 255, 255, 0.4)',
              boxShadow: '0px 4px 24px rgba(0, 0, 0, 0.15), inset 0px 0px 12px rgba(255, 255, 255, 0.25)',
            }}
          >
            <p
              className="text-white/80"
              style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)' }}
            >
              Enter a search query to find biomedical papers.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

function PaperCard({ paper }: { paper: PaperSummary }) {
  return (
    <div
      className="rounded-2xl p-6 transition-all hover:scale-[1.01]"
      style={{
        background: 'rgba(255, 255, 255, 0.18)',
        backdropFilter: 'blur(16px)',
        WebkitBackdropFilter: 'blur(16px)',
        border: '1px solid rgba(255, 255, 255, 0.4)',
        boxShadow: '0px 4px 24px rgba(0, 0, 0, 0.15), inset 0px 0px 12px rgba(255, 255, 255, 0.25)',
      }}
    >
      <h3 className="text-lg font-semibold mb-2">
        <a
          href={`https://pubmed.ncbi.nlm.nih.gov/${paper.pmid}/`}
          target="_blank"
          rel="noopener noreferrer"
          className="text-white hover:text-cyan-300 transition-colors"
          style={{ textShadow: '0px 1px 3px rgba(0, 0, 0, 0.25)' }}
        >
          {paper.title}
        </a>
      </h3>

      <div
        className="text-sm text-white/70 mb-2"
        style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)' }}
      >
        <span className="font-medium">PMID:</span> {paper.pmid}
        {paper.journal && (
          <>
            <span className="mx-2">|</span>
            <span>{paper.journal}</span>
          </>
        )}
        {paper.publication_date && (
          <>
            <span className="mx-2">|</span>
            <span>{new Date(paper.publication_date).toLocaleDateString()}</span>
          </>
        )}
      </div>

      {paper.authors && paper.authors.length > 0 && (
        <div
          className="text-sm text-white/60 mb-3"
          style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)' }}
        >
          {paper.authors.join(', ')}
          {paper.authors.length >= 3 && ' et al.'}
        </div>
      )}

      {paper.abstract && (
        <p
          className="text-sm text-white/80 line-clamp-3"
          style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)' }}
        >
          {paper.abstract}
        </p>
      )}

      {paper.relevance_score && (
        <div className="mt-3 flex items-center gap-2">
          <div className="h-2 w-24 rounded-full bg-white/20 overflow-hidden">
            <div
              className="h-full rounded-full"
              style={{
                width: `${paper.relevance_score}%`,
                background: 'linear-gradient(90deg, #06b6d4, #8b5cf6)',
              }}
            />
          </div>
          <span
            className="text-xs text-white/60"
            style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)' }}
          >
            {paper.relevance_score.toFixed(1)}% match
          </span>
        </div>
      )}
    </div>
  );
}
