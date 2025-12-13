import { useState, useRef, useEffect, FormEvent } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { chatApi } from '../api/chat';
import type { ChatMessage, Citation, ChatResponse, ChatSession } from '../types';

export default function Chat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const queryClient = useQueryClient();

  // Fetch sessions
  const { data: sessions = [], isLoading: sessionsLoading } = useQuery({
    queryKey: ['chatSessions'],
    queryFn: () => chatApi.listSessions(50, 0),
  });

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Load session messages when sessionId changes
  useEffect(() => {
    if (sessionId) {
      chatApi.getSession(sessionId).then((session) => {
        setMessages(session.messages || []);
      }).catch(() => {
        // Session might not exist yet or be invalid
        setMessages([]);
      });
    }
  }, [sessionId]);

  const queryMutation = useMutation({
    mutationFn: (query: string) => chatApi.query({
      session_id: sessionId || undefined,
      query,
    }),
    onSuccess: (data: ChatResponse) => {
      setError(null);
      if (!sessionId) {
        setSessionId(data.session_id);
        // Refresh sessions list
        queryClient.invalidateQueries({ queryKey: ['chatSessions'] });
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
    onError: (err: Error & { response?: { data?: { detail?: string } } }) => {
      const errorMessage = err.response?.data?.detail || err.message || 'An error occurred';
      setError(errorMessage);
      setMessages((prev) => prev.slice(0, -1));
    },
  });

  const deleteSessionMutation = useMutation({
    mutationFn: (id: string) => chatApi.deleteSession(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['chatSessions'] });
      if (sessionId) {
        handleNewChat();
      }
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

  const handleNewChat = () => {
    setSessionId(null);
    setMessages([]);
    setError(null);
  };

  const handleSelectSession = (session: ChatSession) => {
    setSessionId(session.session_id);
    setError(null);
  };

  const handleDeleteSession = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    if (confirm('Delete this conversation?')) {
      deleteSessionMutation.mutate(id);
    }
  };

  return (
    <div className="flex h-screen bg-gradient-to-br from-violet-600 via-purple-500 to-fuchsia-500">
      {/* Animated background orbs */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-pink-400 rounded-full mix-blend-multiply filter blur-3xl opacity-70 animate-pulse" />
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-cyan-400 rounded-full mix-blend-multiply filter blur-3xl opacity-70 animate-pulse [animation-delay:2s]" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-purple-400 rounded-full mix-blend-multiply filter blur-3xl opacity-50 animate-pulse [animation-delay:4s]" />
      </div>

      {/* Sidebar */}
      <aside
        className={`relative z-10 flex flex-col transition-all duration-300 ${
          sidebarOpen ? 'w-72' : 'w-0'
        } overflow-hidden`}
      >
        <div
          className="flex flex-col h-full m-4 mr-0 rounded-2xl"
          style={{
            background: 'rgba(255, 255, 255, 0.18)',
            backdropFilter: 'blur(16px)',
            WebkitBackdropFilter: 'blur(16px)',
            border: '1px solid rgba(255, 255, 255, 0.4)',
            boxShadow: '0px 4px 24px rgba(0, 0, 0, 0.15), inset 0px 0px 12px rgba(255, 255, 255, 0.25)',
          }}
        >
          {/* New Chat Button */}
          <div className="p-4">
            <button
              onClick={handleNewChat}
              className="w-full px-4 py-3 rounded-xl text-white font-medium transition-all hover:scale-[1.02] active:scale-[0.98]"
              style={{
                background: 'linear-gradient(135deg, #06b6d4, #8b5cf6)',
                boxShadow: '0px 4px 16px rgba(139, 92, 246, 0.4)',
                textShadow: '0px 1px 2px rgba(0, 0, 0, 0.3)',
              }}
            >
              + New Chat
            </button>
          </div>

          {/* Sessions List */}
          <div className="flex-1 overflow-y-auto px-4 pb-4">
            <p
              className="text-xs text-white/60 uppercase tracking-wider mb-3"
              style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)' }}
            >
              Recent Conversations
            </p>

            {sessionsLoading ? (
              <div className="text-white/60 text-sm text-center py-4">Loading...</div>
            ) : sessions.length === 0 ? (
              <div className="text-white/60 text-sm text-center py-4">No conversations yet</div>
            ) : (
              <div className="space-y-2">
                {sessions.map((session) => (
                  <SessionItem
                    key={session.session_id}
                    session={session}
                    isActive={sessionId === session.session_id}
                    onSelect={() => handleSelectSession(session)}
                    onDelete={(e) => handleDeleteSession(e, session.session_id)}
                  />
                ))}
              </div>
            )}
          </div>

          {/* Home Link */}
          <div className="p-4 border-t border-white/20">
            <a
              href="/"
              className="flex items-center gap-2 text-white/80 hover:text-white transition-colors"
              style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)' }}
            >
              <span>←</span>
              <span>Back to Home</span>
            </a>
          </div>
        </div>
      </aside>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col relative z-10">
        {/* Header */}
        <header
          className="px-6 py-4 mx-4 mt-4 rounded-2xl flex items-center justify-between"
          style={{
            background: 'rgba(255, 255, 255, 0.22)',
            backdropFilter: 'blur(16px)',
            WebkitBackdropFilter: 'blur(16px)',
            border: '1px solid rgba(255, 255, 255, 0.4)',
            boxShadow: '0px 4px 24px rgba(0, 0, 0, 0.15), inset 0px 0px 12px rgba(255, 255, 255, 0.25)',
          }}
        >
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-2 rounded-lg hover:bg-white/10 transition-colors"
            title={sidebarOpen ? 'Hide sidebar' : 'Show sidebar'}
          >
            <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>

          <div className="text-center">
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
              {sessionId ? 'Continuing conversation' : 'New conversation'}
            </p>
          </div>

          <div className="w-10" /> {/* Spacer for centering */}
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-4 py-6">
          <div className="max-w-3xl mx-auto space-y-6">
            {messages.length === 0 && (
              <WelcomePanel onSuggestionClick={setInput} />
            )}

            {messages.map((message) => (
              <MessageBubble key={message.message_id} message={message} />
            ))}

            {error && (
              <ErrorMessage error={error} onDismiss={() => setError(null)} />
            )}

            {queryMutation.isPending && <LoadingIndicator />}

            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input */}
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
    </div>
  );
}

function SessionItem({
  session,
  isActive,
  onSelect,
  onDelete,
}: {
  session: ChatSession;
  isActive: boolean;
  onSelect: () => void;
  onDelete: (e: React.MouseEvent) => void;
}) {
  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    const now = new Date();
    const diffDays = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60 * 24));

    if (diffDays === 0) return 'Today';
    if (diffDays === 1) return 'Yesterday';
    if (diffDays < 7) return `${diffDays} days ago`;
    return date.toLocaleDateString();
  };

  return (
    <button
      onClick={onSelect}
      className={`w-full text-left p-3 rounded-xl transition-all group ${
        isActive ? 'scale-[1.02]' : 'hover:scale-[1.01]'
      }`}
      style={{
        background: isActive
          ? 'linear-gradient(135deg, rgba(6, 182, 212, 0.4), rgba(139, 92, 246, 0.4))'
          : 'rgba(255, 255, 255, 0.1)',
        border: `1px solid ${isActive ? 'rgba(255, 255, 255, 0.5)' : 'rgba(255, 255, 255, 0.2)'}`,
      }}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <p
            className="text-sm font-medium text-white truncate"
            style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.25)' }}
          >
            {session.title || 'New conversation'}
          </p>
          <p
            className="text-xs text-white/60 mt-1"
            style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)' }}
          >
            {session.message_count} messages · {formatDate(session.updated_at)}
          </p>
        </div>
        <button
          onClick={onDelete}
          className="opacity-0 group-hover:opacity-100 p-1 hover:bg-white/20 rounded transition-all"
          title="Delete conversation"
        >
          <svg className="w-4 h-4 text-white/70" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
          </svg>
        </button>
      </div>
    </button>
  );
}

function WelcomePanel({ onSuggestionClick }: { onSuggestionClick: (text: string) => void }) {
  return (
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
          onClick={() => onSuggestionClick('What are the latest treatments for Type 2 diabetes?')}
        />
        <SuggestionButton
          text="How does CRISPR-Cas9 gene editing work?"
          onClick={() => onSuggestionClick('How does CRISPR-Cas9 gene editing work?')}
        />
        <SuggestionButton
          text="What are the mechanisms of CAR-T cell therapy?"
          onClick={() => onSuggestionClick('What are the mechanisms of CAR-T cell therapy?')}
        />
      </div>
    </div>
  );
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === 'user';

  return (
    <div className={`flex items-start gap-3 ${isUser ? 'flex-row-reverse' : ''}`}>
      <div
        className="w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0"
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
          className="text-xs font-semibold px-2 py-1 rounded-lg flex-shrink-0"
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

function ErrorMessage({ error, onDismiss }: { error: string; onDismiss: () => void }) {
  return (
    <div
      className="rounded-2xl p-4"
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
        {error}
      </p>
      <button
        onClick={onDismiss}
        className="mt-2 text-sm text-white/80 hover:text-white underline"
      >
        Dismiss
      </button>
    </div>
  );
}

function LoadingIndicator() {
  return (
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
  );
}
