import { useState, FormEvent } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { useAnalyzeTrends } from '../hooks/useAnalytics';
import type { TrendAnalysisResponse } from '../types';

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

export default function TrendAnalysis() {
  const [keyword, setKeyword] = useState('');
  const [maxPapers, setMaxPapers] = useState(50);
  const [source, setSource] = useState<'pubmed' | 'arxiv' | 'both'>('pubmed');

  const analyzeMutation = useAnalyzeTrends();

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (keyword.trim()) {
      analyzeMutation.mutate({
        keyword: keyword.trim(),
        max_papers: maxPapers,
        source,
        include_ai_summary: true,
      });
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
          Research Trend Analysis
        </h2>
        <form onSubmit={handleSubmit}>
          <div className="flex flex-wrap gap-4 mb-4">
            <input
              type="text"
              value={keyword}
              onChange={(e) => setKeyword(e.target.value)}
              placeholder="Enter research keyword (Korean OK: 당뇨병, 암 치료)..."
              className="flex-1 min-w-[200px] rounded-xl px-4 py-3 text-white placeholder-white/60 focus:outline-none focus:ring-2 focus:ring-cyan-400/50 transition-all"
              style={glassInput}
            />
            <select
              value={source}
              onChange={(e) => setSource(e.target.value as 'pubmed' | 'arxiv' | 'both')}
              className="rounded-xl px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-cyan-400/50"
              style={glassInput}
            >
              <option value="pubmed">PubMed</option>
              <option value="arxiv">arXiv</option>
              <option value="both">Both</option>
            </select>
            <select
              value={maxPapers}
              onChange={(e) => setMaxPapers(Number(e.target.value))}
              className="rounded-xl px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-cyan-400/50"
              style={glassInput}
            >
              <option value={30}>30 papers</option>
              <option value={50}>50 papers</option>
              <option value={100}>100 papers</option>
            </select>
            <button
              type="submit"
              disabled={analyzeMutation.isPending || !keyword.trim()}
              className="px-6 py-3 rounded-xl text-white font-medium focus:outline-none focus:ring-2 focus:ring-cyan-400/50 disabled:opacity-50 disabled:cursor-not-allowed transition-all hover:scale-105 active:scale-95"
              style={gradientButton}
            >
              {analyzeMutation.isPending ? 'Analyzing...' : 'Analyze'}
            </button>
          </div>
        </form>
      </div>

      {/* Loading */}
      {analyzeMutation.isPending && (
        <div className="text-center py-12 rounded-2xl" style={glassPanel}>
          <div className="animate-spin w-12 h-12 border-4 border-white/30 border-t-cyan-400 rounded-full mx-auto mb-4" />
          <p className="text-white/80" style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)' }}>
            Analyzing research trends...
          </p>
        </div>
      )}

      {/* Error */}
      {analyzeMutation.isError && (
        <div
          className="rounded-2xl p-4"
          style={{
            background: 'rgba(239, 68, 68, 0.3)',
            backdropFilter: 'blur(16px)',
            border: '1px solid rgba(239, 68, 68, 0.5)',
          }}
        >
          <p className="text-white" style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.3)' }}>
            Analysis failed. Please try again.
          </p>
        </div>
      )}

      {/* Results */}
      {analyzeMutation.isSuccess && analyzeMutation.data && (
        <TrendResults data={analyzeMutation.data} />
      )}

      {/* Empty State */}
      {!analyzeMutation.isPending && !analyzeMutation.data && !analyzeMutation.isError && (
        <div className="text-center py-12 rounded-2xl" style={glassPanel}>
          <p className="text-white/80" style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)' }}>
            Enter a keyword to analyze research trends.
          </p>
        </div>
      )}
    </div>
  );
}

function TrendResults({ data }: { data: TrendAnalysisResponse }) {
  const yearData = data.year_trend
    ? data.year_trend.years.map((year, i) => ({
        year,
        count: data.year_trend!.counts[i],
      }))
    : [];

  const termData = data.key_terms
    ? data.key_terms.terms.slice(0, 10).map((term, i) => ({
        term,
        frequency: data.key_terms!.frequencies[i],
      }))
    : [];

  return (
    <div className="space-y-6">
      {/* Summary */}
      <div className="rounded-2xl p-6" style={glassPanel}>
        <h3
          className="text-lg font-semibold text-white mb-4"
          style={{ textShadow: '0px 1px 3px rgba(0, 0, 0, 0.25)' }}
        >
          Analysis Summary
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard label="Keyword" value={data.keyword} />
          {data.translated_keyword && (
            <StatCard label="Translated" value={data.translated_keyword} />
          )}
          <StatCard label="Total Papers" value={data.total_papers.toString()} />
          <StatCard label="Language" value={data.original_language === 'ko' ? 'Korean' : 'English'} />
        </div>
      </div>

      {/* Year Trend Chart */}
      {yearData.length > 0 && (
        <div className="rounded-2xl p-6" style={glassPanel}>
          <h3
            className="text-lg font-semibold text-white mb-4"
            style={{ textShadow: '0px 1px 3px rgba(0, 0, 0, 0.25)' }}
          >
            Publication Trend by Year
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={yearData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.2)" />
                <XAxis dataKey="year" stroke="rgba(255,255,255,0.7)" />
                <YAxis stroke="rgba(255,255,255,0.7)" />
                <Tooltip
                  contentStyle={{
                    background: 'rgba(0,0,0,0.8)',
                    border: 'none',
                    borderRadius: '8px',
                    color: '#fff',
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="count"
                  stroke="#06b6d4"
                  strokeWidth={3}
                  dot={{ fill: '#8b5cf6', strokeWidth: 2 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Key Terms Chart */}
      {termData.length > 0 && (
        <div className="rounded-2xl p-6" style={glassPanel}>
          <h3
            className="text-lg font-semibold text-white mb-4"
            style={{ textShadow: '0px 1px 3px rgba(0, 0, 0, 0.25)' }}
          >
            Top Keywords
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={termData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.2)" />
                <XAxis type="number" stroke="rgba(255,255,255,0.7)" />
                <YAxis dataKey="term" type="category" stroke="rgba(255,255,255,0.7)" width={100} />
                <Tooltip
                  contentStyle={{
                    background: 'rgba(0,0,0,0.8)',
                    border: 'none',
                    borderRadius: '8px',
                    color: '#fff',
                  }}
                />
                <Bar dataKey="frequency" fill="url(#gradient)" radius={[0, 4, 4, 0]} />
                <defs>
                  <linearGradient id="gradient" x1="0" y1="0" x2="1" y2="0">
                    <stop offset="0%" stopColor="#06b6d4" />
                    <stop offset="100%" stopColor="#8b5cf6" />
                  </linearGradient>
                </defs>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Emerging Topics */}
      {data.emerging_topics && data.emerging_topics.length > 0 && (
        <div className="rounded-2xl p-6" style={glassPanel}>
          <h3
            className="text-lg font-semibold text-white mb-4"
            style={{ textShadow: '0px 1px 3px rgba(0, 0, 0, 0.25)' }}
          >
            Emerging Topics
          </h3>
          <div className="flex flex-wrap gap-2">
            {data.emerging_topics.map((topic, i) => (
              <span
                key={i}
                className="px-3 py-1.5 rounded-full text-sm text-white"
                style={{
                  background: 'linear-gradient(135deg, rgba(6, 182, 212, 0.5), rgba(139, 92, 246, 0.5))',
                  border: '1px solid rgba(255, 255, 255, 0.3)',
                }}
              >
                {topic.topic}
                {topic.growth_rate > 0 && (
                  <span className="ml-1 text-green-300">↑{(topic.growth_rate * 100).toFixed(0)}%</span>
                )}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* AI Report */}
      {data.report && (
        <div className="rounded-2xl p-6" style={glassPanel}>
          <h3
            className="text-lg font-semibold text-white mb-4"
            style={{ textShadow: '0px 1px 3px rgba(0, 0, 0, 0.25)' }}
          >
            AI Analysis Report
          </h3>
          <div
            className="text-white/80 whitespace-pre-wrap text-sm leading-relaxed"
            style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)' }}
          >
            {data.report}
          </div>
        </div>
      )}
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div
      className="rounded-xl p-4 text-center"
      style={{
        background: 'rgba(255, 255, 255, 0.1)',
        border: '1px solid rgba(255, 255, 255, 0.2)',
      }}
    >
      <div
        className="text-xs text-white/60 mb-1"
        style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)' }}
      >
        {label}
      </div>
      <div
        className="text-lg font-semibold text-white truncate"
        style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)' }}
      >
        {value}
      </div>
    </div>
  );
}
