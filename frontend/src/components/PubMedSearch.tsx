import { useState, FormEvent } from 'react';
import { usePubMedSearch } from '../hooks/usePubMed';
import type { PubMedPaper } from '../api/pubmed';

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

export default function PubMedSearch() {
  const [query, setQuery] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [maxResults, setMaxResults] = useState(20);

  const { data, isLoading, error } = usePubMedSearch(searchQuery, {
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
          PubMed Search
        </h2>
        <form onSubmit={handleSubmit}>
          <div className="flex gap-4 mb-4">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search PubMed (e.g., diabetes treatment, cancer immunotherapy)..."
              className="flex-1 rounded-xl px-4 py-3 text-white placeholder-white/60 focus:outline-none focus:ring-2 focus:ring-cyan-400/50 transition-all"
              style={glassInput}
            />
            <select
              value={maxResults}
              onChange={(e) => setMaxResults(Number(e.target.value))}
              className="rounded-xl px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-cyan-400/50"
              style={glassInput}
            >
              <option value={10}>10 results</option>
              <option value={20}>20 results</option>
              <option value={50}>50 results</option>
              <option value={100}>100 results</option>
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

      {/* Loading */}
      {isLoading && (
        <div className="text-center py-12 rounded-2xl" style={glassPanel}>
          <div className="animate-spin w-12 h-12 border-4 border-white/30 border-t-cyan-400 rounded-full mx-auto mb-4" />
          <p className="text-white/80" style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)' }}>
            Searching PubMed...
          </p>
        </div>
      )}

      {/* Results */}
      {data && !isLoading && (
        <div>
          <p
            className="text-sm text-white/80 mb-4"
            style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)' }}
          >
            Found {data.total} results for "{data.query}"
          </p>
          <div className="space-y-4">
            {data.results.map((paper) => (
              <PubMedPaperCard key={paper.pmid} paper={paper} />
            ))}
          </div>
        </div>
      )}

      {/* Empty State */}
      {!searchQuery && !error && !isLoading && (
        <div className="text-center py-12 rounded-2xl" style={glassPanel}>
          <p
            className="text-white/80"
            style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)' }}
          >
            Enter a search query to find papers from PubMed.
          </p>
        </div>
      )}
    </div>
  );
}

function PubMedPaperCard({ paper }: { paper: PubMedPaper }) {
  return (
    <div
      className="rounded-2xl p-6 transition-all hover:scale-[1.01]"
      style={glassPanel}
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
        className="text-sm text-white/70 mb-2 flex flex-wrap gap-2"
        style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)' }}
      >
        <span className="font-medium">PMID: {paper.pmid}</span>
        {paper.journal && (
          <>
            <span>|</span>
            <span>{paper.journal}</span>
          </>
        )}
        {paper.publication_date && (
          <>
            <span>|</span>
            <span>{new Date(paper.publication_date).toLocaleDateString()}</span>
          </>
        )}
      </div>

      {paper.authors && paper.authors.length > 0 && (
        <div
          className="text-sm text-white/60 mb-3"
          style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)' }}
        >
          {paper.authors.slice(0, 5).join(', ')}
          {paper.authors.length > 5 && ` et al. (+${paper.authors.length - 5})`}
        </div>
      )}

      {paper.abstract && (
        <p
          className="text-sm text-white/80 line-clamp-3 mb-4"
          style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)' }}
        >
          {paper.abstract}
        </p>
      )}

      {/* Keywords/MeSH Terms */}
      {(paper.keywords?.length || paper.mesh_terms?.length) && (
        <div className="flex flex-wrap gap-1 mb-4">
          {(paper.keywords || paper.mesh_terms || []).slice(0, 5).map((term, i) => (
            <span
              key={i}
              className="px-2 py-0.5 rounded text-xs text-white/80"
              style={{
                background: 'rgba(139, 92, 246, 0.3)',
                border: '1px solid rgba(139, 92, 246, 0.5)',
              }}
            >
              {term}
            </span>
          ))}
        </div>
      )}

      <div className="flex gap-2">
        <a
          href={`https://pubmed.ncbi.nlm.nih.gov/${paper.pmid}/`}
          target="_blank"
          rel="noopener noreferrer"
          className="px-3 py-1.5 rounded-lg text-xs text-white font-medium transition-all hover:scale-105"
          style={{
            background: 'rgba(6, 182, 212, 0.5)',
            border: '1px solid rgba(6, 182, 212, 0.7)',
          }}
        >
          PubMed
        </a>
        {paper.doi && (
          <a
            href={`https://doi.org/${paper.doi}`}
            target="_blank"
            rel="noopener noreferrer"
            className="px-3 py-1.5 rounded-lg text-xs text-white font-medium transition-all hover:scale-105"
            style={{
              background: 'rgba(139, 92, 246, 0.5)',
              border: '1px solid rgba(139, 92, 246, 0.7)',
            }}
          >
            DOI
          </a>
        )}
      </div>
    </div>
  );
}
