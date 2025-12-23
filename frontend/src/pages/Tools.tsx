import { useState } from 'react';
import PubMedSearch from '../components/PubMedSearch';
import ArXivSearch from '../components/ArXivSearch';
import TrendAnalysis from '../components/TrendAnalysis';
import DocumentSummarizer from '../components/DocumentSummarizer';
import Translator from '../components/Translator';
import HybridSearch from '../components/HybridSearch';

const glassPanel = {
  background: 'rgba(255, 255, 255, 0.18)',
  backdropFilter: 'blur(16px)',
  WebkitBackdropFilter: 'blur(16px)',
  border: '1px solid rgba(255, 255, 255, 0.4)',
  boxShadow: '0px 4px 24px rgba(0, 0, 0, 0.15), inset 0px 0px 12px rgba(255, 255, 255, 0.25)',
};

type TabType = 'hybrid' | 'pubmed' | 'arxiv' | 'trends' | 'summarizer' | 'translator';

const tabs: { id: TabType; label: string; icon: string }[] = [
  { id: 'hybrid', label: 'Hybrid Search', icon: '🔀' },
  { id: 'pubmed', label: 'PubMed', icon: '🔬' },
  { id: 'arxiv', label: 'ArXiv', icon: '📄' },
  { id: 'trends', label: 'Trends', icon: '📊' },
  { id: 'summarizer', label: 'Summarizer', icon: '📝' },
  { id: 'translator', label: 'Translator', icon: '🌐' },
];

export default function Tools() {
  const [activeTab, setActiveTab] = useState<TabType>('pubmed');

  return (
    <div className="py-8 px-4">
      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <div className="mb-6">
          <h1
            className="text-3xl font-bold text-white mb-2"
            style={{ textShadow: '0px 2px 4px rgba(0, 0, 0, 0.3)' }}
          >
            Research Tools
          </h1>
          <p
            className="text-white/70"
            style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)' }}
          >
            Powerful tools for biomedical research
          </p>
        </div>

        {/* Tabs */}
        <div className="rounded-2xl p-2 mb-6" style={glassPanel}>
          <div className="flex flex-wrap gap-2">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex-1 min-w-[120px] px-4 py-3 rounded-xl text-white font-medium transition-all hover:scale-105 ${
                  activeTab === tab.id ? 'scale-105' : 'opacity-70 hover:opacity-100'
                }`}
                style={
                  activeTab === tab.id
                    ? {
                        background: 'linear-gradient(135deg, #06b6d4, #8b5cf6)',
                        boxShadow: '0px 4px 16px rgba(139, 92, 246, 0.4)',
                        textShadow: '0px 1px 2px rgba(0, 0, 0, 0.3)',
                      }
                    : {
                        background: 'rgba(255, 255, 255, 0.1)',
                        textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)',
                      }
                }
              >
                <span className="mr-2">{tab.icon}</span>
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        {/* Content */}
        {activeTab === 'hybrid' && <HybridSearch />}
        {activeTab === 'pubmed' && <PubMedSearch />}
        {activeTab === 'arxiv' && <ArXivSearch />}
        {activeTab === 'trends' && <TrendAnalysis />}
        {activeTab === 'summarizer' && <DocumentSummarizer />}
        {activeTab === 'translator' && <Translator />}
      </div>
    </div>
  );
}
