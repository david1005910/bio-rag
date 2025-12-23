import { useState, useEffect } from 'react';
import { useDetectLanguage, useTranslate, useMedicalTerms } from '../hooks/useI18n';

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

export default function Translator() {
  const [inputText, setInputText] = useState('');
  const [debouncedText, setDebouncedText] = useState('');
  const [termSearch, setTermSearch] = useState('');

  // Debounce input
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedText(inputText);
    }, 500);
    return () => clearTimeout(timer);
  }, [inputText]);

  const { data: langData } = useDetectLanguage(debouncedText, {
    enabled: debouncedText.length > 0,
  });

  const { data: translateData, isLoading: isTranslating } = useTranslate(debouncedText, {
    enabled: debouncedText.length > 0 && langData?.language === 'ko',
  });

  const { data: medicalTerms } = useMedicalTerms(termSearch || undefined, {
    limit: 20,
  });

  return (
    <div className="space-y-6">
      {/* Translation Panel */}
      <div className="rounded-2xl p-6" style={glassPanel}>
        <h2
          className="text-xl font-bold text-white mb-4"
          style={{ textShadow: '0px 2px 4px rgba(0, 0, 0, 0.3)' }}
        >
          Medical Term Translator
        </h2>

        <div className="grid md:grid-cols-2 gap-4">
          {/* Input */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label
                className="text-sm text-white/80"
                style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)' }}
              >
                Input (Korean/English)
              </label>
              {langData && (
                <span
                  className="px-2 py-0.5 rounded text-xs"
                  style={{
                    background: langData.language === 'ko'
                      ? 'rgba(236, 72, 153, 0.5)'
                      : 'rgba(6, 182, 212, 0.5)',
                    color: 'white',
                  }}
                >
                  {langData.language_name}
                </span>
              )}
            </div>
            <textarea
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="Enter text to translate (e.g., 당뇨병 치료, 암 면역치료)..."
              rows={5}
              className="w-full rounded-xl px-4 py-3 text-white placeholder-white/60 focus:outline-none focus:ring-2 focus:ring-cyan-400/50 transition-all resize-none"
              style={glassInput}
            />
          </div>

          {/* Output */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label
                className="text-sm text-white/80"
                style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)' }}
              >
                Translation (English)
              </label>
              {translateData?.method && translateData.method !== 'none' && (
                <span
                  className="px-2 py-0.5 rounded text-xs"
                  style={{
                    background: 'rgba(139, 92, 246, 0.5)',
                    color: 'white',
                  }}
                >
                  {translateData.method === 'dictionary' ? 'Dictionary' : 'AI'}
                </span>
              )}
            </div>
            <div
              className="w-full rounded-xl px-4 py-3 min-h-[140px] text-white"
              style={{
                ...glassInput,
                background: 'rgba(255, 255, 255, 0.1)',
              }}
            >
              {isTranslating ? (
                <span className="text-white/50">Translating...</span>
              ) : translateData ? (
                translateData.translated
              ) : langData?.language === 'en' ? (
                <span className="text-white/70">{inputText || 'English text (no translation needed)'}</span>
              ) : (
                <span className="text-white/50">Translation will appear here...</span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Medical Terms Dictionary */}
      <div className="rounded-2xl p-6" style={glassPanel}>
        <h3
          className="text-lg font-semibold text-white mb-4"
          style={{ textShadow: '0px 1px 3px rgba(0, 0, 0, 0.25)' }}
        >
          Medical Terms Dictionary
        </h3>

        <input
          type="text"
          value={termSearch}
          onChange={(e) => setTermSearch(e.target.value)}
          placeholder="Search medical terms (Korean or English)..."
          className="w-full rounded-xl px-4 py-3 text-white placeholder-white/60 focus:outline-none focus:ring-2 focus:ring-cyan-400/50 transition-all mb-4"
          style={glassInput}
        />

        {medicalTerms && medicalTerms.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {medicalTerms.map((term, i) => (
              <div
                key={i}
                className="rounded-lg px-4 py-2 flex justify-between items-center cursor-pointer transition-all hover:scale-[1.02]"
                style={{
                  background: 'rgba(255, 255, 255, 0.1)',
                  border: '1px solid rgba(255, 255, 255, 0.2)',
                }}
                onClick={() => setInputText(term.korean)}
              >
                <span
                  className="text-white font-medium"
                  style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)' }}
                >
                  {term.korean}
                </span>
                <span
                  className="text-white/70 text-sm"
                  style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)' }}
                >
                  {term.english}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <p
            className="text-center text-white/60 py-4"
            style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)' }}
          >
            {termSearch ? 'No matching terms found.' : 'Type to search medical terms.'}
          </p>
        )}
      </div>
    </div>
  );
}
