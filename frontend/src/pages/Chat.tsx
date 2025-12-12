import { useState, useRef, useEffect, FormEvent } from 'react';
import { useMutation } from '@tanstack/react-query';
import { chatApi } from '../api/chat';
import type { ChatMessage, Citation, ChatResponse } from '../types';

export default function Chat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const queryMutation = useMutation({
    mutationFn: (query: string) => chatApi.query({
      session_id: sessionId || undefined,
      query,
    }),
    onSuccess: (data: ChatResponse) => {
      if (!sessionId) {
        setSessionId(data.session_id);
      }

      setMessages((prev) => [
        ...prev,
        {
          message_id: data.message_id,
          role: 'assistant',
          content: data.answer,
          citations: data.citations,
          timestamp: new Date().toISOString(),
        },
      ]);
    },
  });

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!input.trim() || queryMutation.isPending) return;

    const userMessage: ChatMessage = {
      message_id: crypto.randomUUID(),
      role: 'user',
      content: input,
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');

    queryMutation.mutate(input);
  };

  // Note: Streaming query handler available in chatApi.queryStream
  // Can be implemented when real-time streaming is needed

  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-violet-600 via-purple-500 to-fuchsia-500">
      {/* Animated background orbs for depth */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-pink-400 rounded-full mix-blend-multiply filter blur-3xl opacity-70 animate-pulse" />
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-cyan-400 rounded-full mix-blend-multiply filter blur-3xl opacity-70 animate-pulse [animation-delay:2s]" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-purple-400 rounded-full mix-blend-multiply filter blur-3xl opacity-50 animate-pulse [animation-delay:4s]" />
      </div>

      {/* Header - Glass Panel */}
      <header
        className="relative px-6 py-5 mx-4 mt-4 rounded-2xl"
        style={{
          background: 'rgba(255, 255, 255, 0.22)',
          backdropFilter: 'blur(16px)',
          WebkitBackdropFilter: 'blur(16px)',
          border: '1px solid rgba(255, 255, 255, 0.4)',
          boxShadow: '0px 4px 24px rgba(0, 0, 0, 0.15), inset 0px 0px 12px rgba(255, 255, 255, 0.25)',
        }}
      >
        <h1
          className="text-xl font-semibold text-white"
          style={{ textShadow: '0px 1px 3px rgba(0, 0, 0, 0.25)' }}
        >
          Research Assistant
        </h1>
        <p
          className="text-sm text-white/80"
          style={{ textShadow: '0px 1px 3px rgba(0, 0, 0, 0.2)' }}
        >
          Ask questions about biomedical research
        </p>
      </header>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-3xl mx-auto space-y-6">
          {messages.length === 0 && (
            <div
              className="text-center py-12 px-6 rounded-2xl"
              style={{
                background: 'rgba(255, 255, 255, 0.18)',
                backdropFilter: 'blur(16px)',
                WebkitBackdropFilter: 'blur(16px)',
                border: '1px solid rgba(255, 255, 255, 0.4)',
                boxShadow: '0px 4px 24px rgba(0, 0, 0, 0.15), inset 0px 0px 12px rgba(255, 255, 255, 0.25)',
              }}
            >
              <h2
                className="text-xl font-medium text-white mb-2"
                style={{ textShadow: '0px 1px 3px rgba(0, 0, 0, 0.25)' }}
              >
                Welcome to Bio-RAG
              </h2>
              <p
                className="text-white/80 mb-6"
                style={{ textShadow: '0px 1px 3px rgba(0, 0, 0, 0.2)' }}
              >
                Ask any question about biomedical research and I'll search through
                scientific papers to provide evidence-based answers.
              </p>
              <div className="space-y-3">
                <SuggestionButton
                  text="What are the latest treatments for Type 2 diabetes?"
                  onClick={() => setInput('What are the latest treatments for Type 2 diabetes?')}
                />
                <SuggestionButton
                  text="How does CRISPR-Cas9 gene editing work?"
                  onClick={() => setInput('How does CRISPR-Cas9 gene editing work?')}
                />
                <SuggestionButton
                  text="What are the mechanisms of CAR-T cell therapy?"
                  onClick={() => setInput('What are the mechanisms of CAR-T cell therapy?')}
                />
              </div>
            </div>
          )}

          {messages.map((message) => (
            <MessageBubble key={message.message_id} message={message} />
          ))}

          {queryMutation.isPending && (
            <div className="flex items-start gap-3">
              <div
                className="w-10 h-10 rounded-full flex items-center justify-center"
                style={{
                  background: 'linear-gradient(135deg, #06b6d4, #8b5cf6)',
                  boxShadow: '0px 4px 12px rgba(139, 92, 246, 0.4)',
                }}
              >
                <span
                  className="text-white text-sm font-medium"
                  style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.3)' }}
                >
                  AI
                </span>
              </div>
              <div
                className="flex-1 rounded-2xl p-4"
                style={{
                  background: 'rgba(255, 255, 255, 0.22)',
                  backdropFilter: 'blur(16px)',
                  WebkitBackdropFilter: 'blur(16px)',
                  border: '1px solid rgba(255, 255, 255, 0.4)',
                  boxShadow: '0px 4px 24px rgba(0, 0, 0, 0.15), inset 0px 0px 12px rgba(255, 255, 255, 0.25)',
                }}
              >
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce" style={{ boxShadow: '0 0 8px rgba(34, 211, 238, 0.6)' }} />
                  <div className="w-2 h-2 bg-violet-400 rounded-full animate-bounce [animation-delay:0.1s]" style={{ boxShadow: '0 0 8px rgba(167, 139, 250, 0.6)' }} />
                  <div className="w-2 h-2 bg-fuchsia-400 rounded-full animate-bounce [animation-delay:0.2s]" style={{ boxShadow: '0 0 8px rgba(232, 121, 249, 0.6)' }} />
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input - Glass Panel */}
      <div
        className="mx-4 mb-4 px-6 py-4 rounded-2xl"
        style={{
          background: 'rgba(255, 255, 255, 0.22)',
          backdropFilter: 'blur(16px)',
          WebkitBackdropFilter: 'blur(16px)',
          border: '1px solid rgba(255, 255, 255, 0.4)',
          boxShadow: '0px 4px 24px rgba(0, 0, 0, 0.15), inset 0px 0px 12px rgba(255, 255, 255, 0.25)',
        }}
      >
        <form onSubmit={handleSubmit} className="max-w-3xl mx-auto">
          <div className="flex gap-4">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question about biomedical research..."
              disabled={queryMutation.isPending}
              className="flex-1 rounded-xl px-4 py-3 text-white placeholder-white/60 focus:outline-none focus:ring-2 focus:ring-cyan-400/50 disabled:opacity-50 transition-all"
              style={{
                background: 'rgba(255, 255, 255, 0.15)',
                backdropFilter: 'blur(12px)',
                WebkitBackdropFilter: 'blur(12px)',
                border: '1px solid rgba(255, 255, 255, 0.3)',
                boxShadow: 'inset 0px 0px 8px rgba(255, 255, 255, 0.1)',
                textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)',
              }}
            />
            <button
              type="submit"
              disabled={queryMutation.isPending || !input.trim()}
              className="px-6 py-3 rounded-xl text-white font-medium focus:outline-none focus:ring-2 focus:ring-cyan-400/50 disabled:opacity-50 disabled:cursor-not-allowed transition-all hover:scale-105 active:scale-95"
              style={{
                background: 'linear-gradient(135deg, #06b6d4, #8b5cf6)',
                boxShadow: '0px 4px 16px rgba(139, 92, 246, 0.4)',
                textShadow: '0px 1px 2px rgba(0, 0, 0, 0.3)',
              }}
            >
              Send
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === 'user';

  return (
    <div className={`flex items-start gap-3 ${isUser ? 'flex-row-reverse' : ''}`}>
      <div
        className="w-10 h-10 rounded-full flex items-center justify-center"
        style={{
          background: isUser
            ? 'linear-gradient(135deg, #ec4899, #f97316)'
            : 'linear-gradient(135deg, #06b6d4, #8b5cf6)',
          boxShadow: isUser
            ? '0px 4px 12px rgba(236, 72, 153, 0.4)'
            : '0px 4px 12px rgba(139, 92, 246, 0.4)',
        }}
      >
        <span
          className="text-white text-sm font-medium"
          style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.3)' }}
        >
          {isUser ? 'U' : 'AI'}
        </span>
      </div>

      <div className={`flex-1 max-w-[80%] ${isUser ? 'text-right' : ''}`}>
        <div
          className="rounded-2xl p-4 inline-block text-left"
          style={{
            background: isUser
              ? 'linear-gradient(135deg, rgba(236, 72, 153, 0.6), rgba(249, 115, 22, 0.6))'
              : 'rgba(255, 255, 255, 0.22)',
            backdropFilter: 'blur(16px)',
            WebkitBackdropFilter: 'blur(16px)',
            border: '1px solid rgba(255, 255, 255, 0.4)',
            boxShadow: '0px 4px 24px rgba(0, 0, 0, 0.15), inset 0px 0px 12px rgba(255, 255, 255, 0.25)',
          }}
        >
          <p
            className="whitespace-pre-wrap text-white"
            style={{ textShadow: '0px 1px 3px rgba(0, 0, 0, 0.25)' }}
          >
            {message.content}
          </p>
        </div>

        {message.citations && message.citations.length > 0 && (
          <div className="mt-3 space-y-2">
            <p
              className="text-xs text-white/70 font-medium"
              style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)' }}
            >
              Sources:
            </p>
            {message.citations.map((citation) => (
              <CitationCard key={citation.pmid} citation={citation} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function CitationCard({ citation }: { citation: Citation }) {
  return (
    <a
      href={`https://pubmed.ncbi.nlm.nih.gov/${citation.pmid}/`}
      target="_blank"
      rel="noopener noreferrer"
      className="block rounded-xl p-3 transition-all hover:scale-[1.02] active:scale-[0.98]"
      style={{
        background: 'rgba(255, 255, 255, 0.15)',
        backdropFilter: 'blur(12px)',
        WebkitBackdropFilter: 'blur(12px)',
        border: '1px solid rgba(255, 255, 255, 0.3)',
        boxShadow: '0px 4px 16px rgba(0, 0, 0, 0.1)',
      }}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <p
            className="text-sm font-medium text-white truncate"
            style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.25)' }}
          >
            {citation.title}
          </p>
          <p
            className="text-xs text-white/60"
            style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)' }}
          >
            PMID: {citation.pmid}
          </p>
        </div>
        <div
          className="text-xs font-semibold px-2 py-1 rounded-lg"
          style={{
            background: 'linear-gradient(135deg, rgba(6, 182, 212, 0.6), rgba(139, 92, 246, 0.6))',
            color: 'white',
            textShadow: '0px 1px 2px rgba(0, 0, 0, 0.3)',
          }}
        >
          {Math.round(citation.relevance_score * 100)}%
        </div>
      </div>
    </a>
  );
}

function SuggestionButton({ text, onClick }: { text: string; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="block w-full text-left px-4 py-3 rounded-xl transition-all hover:scale-[1.02] active:scale-[0.98]"
      style={{
        background: 'rgba(255, 255, 255, 0.12)',
        backdropFilter: 'blur(12px)',
        WebkitBackdropFilter: 'blur(12px)',
        border: '1px solid rgba(255, 255, 255, 0.3)',
        boxShadow: '0px 4px 16px rgba(0, 0, 0, 0.1)',
        color: 'white',
        textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)',
      }}
    >
      {text}
    </button>
  );
}
