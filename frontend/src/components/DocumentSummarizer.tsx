import { useState, FormEvent } from 'react';
import { useSummarizePaper, useExtractFromUpload } from '../hooks/useDocuments';

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

export default function DocumentSummarizer() {
  const [title, setTitle] = useState('');
  const [abstract, setAbstract] = useState('');
  const [language, setLanguage] = useState<'en' | 'ko'>('en');
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [extractedText, setExtractedText] = useState('');

  const summarizeMutation = useSummarizePaper();
  const extractMutation = useExtractFromUpload();

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (title.trim() && abstract.trim()) {
      summarizeMutation.mutate({
        paper: {
          id: `manual-${Date.now()}`,
          title: title.trim(),
          abstract: abstract.trim(),
        },
        content: extractedText || undefined,
        language,
      });
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploadedFile(file);
    extractMutation.mutate(file, {
      onSuccess: (data) => {
        setExtractedText(data.text);
      },
    });
  };

  return (
    <div className="space-y-6">
      {/* Input Form */}
      <div className="rounded-2xl p-6" style={glassPanel}>
        <h2
          className="text-xl font-bold text-white mb-4"
          style={{ textShadow: '0px 2px 4px rgba(0, 0, 0, 0.3)' }}
        >
          Paper Summarizer
        </h2>

        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Title */}
          <div>
            <label
              className="block text-sm text-white/80 mb-2"
              style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)' }}
            >
              Paper Title
            </label>
            <input
              type="text"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="Enter paper title..."
              className="w-full rounded-xl px-4 py-3 text-white placeholder-white/60 focus:outline-none focus:ring-2 focus:ring-cyan-400/50 transition-all"
              style={glassInput}
            />
          </div>

          {/* Abstract */}
          <div>
            <label
              className="block text-sm text-white/80 mb-2"
              style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)' }}
            >
              Abstract
            </label>
            <textarea
              value={abstract}
              onChange={(e) => setAbstract(e.target.value)}
              placeholder="Paste the paper abstract here..."
              rows={5}
              className="w-full rounded-xl px-4 py-3 text-white placeholder-white/60 focus:outline-none focus:ring-2 focus:ring-cyan-400/50 transition-all resize-none"
              style={glassInput}
            />
          </div>

          {/* File Upload */}
          <div>
            <label
              className="block text-sm text-white/80 mb-2"
              style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)' }}
            >
              Upload PDF (Optional - for better summary)
            </label>
            <div
              className="rounded-xl p-4 text-center cursor-pointer transition-all hover:scale-[1.01]"
              style={{
                ...glassInput,
                border: '2px dashed rgba(255, 255, 255, 0.3)',
              }}
            >
              <input
                type="file"
                accept=".pdf,.txt"
                onChange={handleFileUpload}
                className="hidden"
                id="file-upload"
              />
              <label htmlFor="file-upload" className="cursor-pointer">
                {extractMutation.isPending ? (
                  <span className="text-white/60">Extracting text...</span>
                ) : uploadedFile ? (
                  <span className="text-cyan-300">{uploadedFile.name}</span>
                ) : (
                  <span className="text-white/60">Click to upload PDF or TXT file</span>
                )}
              </label>
            </div>
            {extractedText && (
              <p className="text-xs text-green-300 mt-2">
                Extracted {extractedText.length.toLocaleString()} characters from file
              </p>
            )}
          </div>

          {/* Language & Submit */}
          <div className="flex gap-4">
            <select
              value={language}
              onChange={(e) => setLanguage(e.target.value as 'en' | 'ko')}
              className="rounded-xl px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-cyan-400/50"
              style={glassInput}
            >
              <option value="en">English</option>
              <option value="ko">한국어</option>
            </select>
            <button
              type="submit"
              disabled={summarizeMutation.isPending || !title.trim() || !abstract.trim()}
              className="flex-1 px-6 py-3 rounded-xl text-white font-medium focus:outline-none focus:ring-2 focus:ring-cyan-400/50 disabled:opacity-50 disabled:cursor-not-allowed transition-all hover:scale-105 active:scale-95"
              style={gradientButton}
            >
              {summarizeMutation.isPending ? 'Summarizing...' : 'Generate Summary'}
            </button>
          </div>
        </form>
      </div>

      {/* Error */}
      {summarizeMutation.isError && (
        <div
          className="rounded-2xl p-4"
          style={{
            background: 'rgba(239, 68, 68, 0.3)',
            backdropFilter: 'blur(16px)',
            border: '1px solid rgba(239, 68, 68, 0.5)',
          }}
        >
          <p className="text-white" style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.3)' }}>
            Summarization failed. Please try again.
          </p>
        </div>
      )}

      {/* Result */}
      {summarizeMutation.isSuccess && summarizeMutation.data && (
        <div className="rounded-2xl p-6" style={glassPanel}>
          <h3
            className="text-lg font-semibold text-white mb-2"
            style={{ textShadow: '0px 1px 3px rgba(0, 0, 0, 0.25)' }}
          >
            {summarizeMutation.data.title}
          </h3>
          <div className="flex items-center gap-2 mb-4">
            <span
              className="px-2 py-1 rounded text-xs text-white"
              style={{
                background: summarizeMutation.data.language === 'ko'
                  ? 'rgba(236, 72, 153, 0.5)'
                  : 'rgba(6, 182, 212, 0.5)',
                border: '1px solid rgba(255, 255, 255, 0.3)',
              }}
            >
              {summarizeMutation.data.language === 'ko' ? '한국어' : 'English'}
            </span>
            {summarizeMutation.data.success && (
              <span className="text-green-300 text-xs">✓ Success</span>
            )}
          </div>
          <div
            className="text-white/80 whitespace-pre-wrap leading-relaxed"
            style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)' }}
          >
            {summarizeMutation.data.summary}
          </div>
        </div>
      )}
    </div>
  );
}
