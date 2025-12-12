import { useState, FormEvent } from 'react';
import { useQuery } from '@tanstack/react-query';
import { searchApi } from '../api/search';
import type { SearchResult, PaperSummary } from '../types';

export default function Search() {
  const [query, setQuery] = useState('');
  const [searchQuery, setSearchQuery] = useState('');

  const { data, isLoading, error } = useQuery<SearchResult>({
    queryKey: ['search', searchQuery],
    queryFn: () => searchApi.search({ query: searchQuery, limit: 20 }),
    enabled: !!searchQuery,
  });

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      setSearchQuery(query.trim());
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <header className="bg-white dark:bg-gray-800 shadow">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            Paper Search
          </h1>
        </div>
      </header>

      {/* Search Form */}
      <div className="max-w-7xl mx-auto px-4 py-8">
        <form onSubmit={handleSubmit} className="mb-8">
          <div className="flex gap-4">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search for biomedical papers..."
              className="flex-1 rounded-lg border border-gray-300 dark:border-gray-700 px-4 py-3 bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
            />
            <button
              type="submit"
              disabled={isLoading}
              className="px-6 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
            >
              {isLoading ? 'Searching...' : 'Search'}
            </button>
          </div>
        </form>

        {/* Results */}
        {error && (
          <div className="bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800 text-red-600 dark:text-red-400 rounded-lg p-4 mb-6">
            An error occurred while searching. Please try again.
          </div>
        )}

        {data && (
          <div>
            <div className="flex justify-between items-center mb-4">
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Found {data.total} results in {data.query_time_ms}ms
              </p>
            </div>

            <div className="space-y-4">
              {data.results.map((paper) => (
                <PaperCard key={paper.pmid} paper={paper} />
              ))}
            </div>

            {data.results.length === 0 && (
              <div className="text-center py-12 text-gray-500 dark:text-gray-400">
                No papers found for your search query.
              </div>
            )}
          </div>
        )}

        {!searchQuery && (
          <div className="text-center py-12 text-gray-500 dark:text-gray-400">
            Enter a search query to find biomedical papers.
          </div>
        )}
      </div>
    </div>
  );
}

function PaperCard({ paper }: { paper: PaperSummary }) {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
        <a
          href={`https://pubmed.ncbi.nlm.nih.gov/${paper.pmid}/`}
          target="_blank"
          rel="noopener noreferrer"
          className="hover:text-indigo-600 dark:hover:text-indigo-400"
        >
          {paper.title}
        </a>
      </h3>

      <div className="text-sm text-gray-600 dark:text-gray-400 mb-2">
        <span className="font-medium">PMID:</span> {paper.pmid}
        {paper.journal && (
          <>
            <span className="mx-2">|</span>
            <span>{paper.journal}</span>
          </>
        )}
        {paper.pub_date && (
          <>
            <span className="mx-2">|</span>
            <span>{paper.pub_date}</span>
          </>
        )}
      </div>

      {paper.authors.length > 0 && (
        <div className="text-sm text-gray-500 dark:text-gray-500 mb-3">
          {paper.authors.join(', ')}
          {paper.authors.length >= 3 && ' et al.'}
        </div>
      )}

      {paper.abstract_snippet && (
        <p className="text-sm text-gray-700 dark:text-gray-300 line-clamp-3">
          {paper.abstract_snippet}...
        </p>
      )}
    </div>
  );
}
