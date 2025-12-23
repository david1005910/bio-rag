import { Routes, Route } from 'react-router-dom';
import { AuthProvider } from './contexts/AuthContext';
import Layout from './components/Layout';
import ProtectedRoute from './components/ProtectedRoute';
import { Login, Register, Search, Chat, Tools } from './pages';

function App() {
  return (
    <AuthProvider>
      <Routes>
        {/* Public routes - Full screen Glassmorphism */}
        <Route path="/" element={<Home />} />
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />

        {/* Chat - Full screen Glassmorphism */}
        <Route
          path="/chat"
          element={
            <ProtectedRoute>
              <Chat />
            </ProtectedRoute>
          }
        />

        {/* Protected routes with Layout navigation */}
        <Route element={<Layout />}>
          <Route
            path="/search"
            element={
              <ProtectedRoute>
                <Search />
              </ProtectedRoute>
            }
          />
          <Route
            path="/tools"
            element={
              <ProtectedRoute>
                <Tools />
              </ProtectedRoute>
            }
          />
        </Route>
      </Routes>
    </AuthProvider>
  );
}

function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-violet-600 via-purple-500 to-fuchsia-500">
      {/* Animated background orbs */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-pink-400 rounded-full mix-blend-multiply filter blur-3xl opacity-70 animate-pulse" />
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-cyan-400 rounded-full mix-blend-multiply filter blur-3xl opacity-70 animate-pulse [animation-delay:2s]" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-purple-400 rounded-full mix-blend-multiply filter blur-3xl opacity-50 animate-pulse [animation-delay:4s]" />
      </div>

      {/* Navigation */}
      <nav
        className="relative mx-4 mt-4 px-6 py-4 rounded-2xl"
        style={{
          background: 'rgba(255, 255, 255, 0.22)',
          backdropFilter: 'blur(16px)',
          WebkitBackdropFilter: 'blur(16px)',
          border: '1px solid rgba(255, 255, 255, 0.4)',
          boxShadow: '0px 4px 24px rgba(0, 0, 0, 0.15), inset 0px 0px 12px rgba(255, 255, 255, 0.25)',
        }}
      >
        <div className="flex justify-between items-center">
          <a
            href="/"
            className="text-xl font-bold text-white"
            style={{ textShadow: '0px 1px 3px rgba(0, 0, 0, 0.3)' }}
          >
            Bio-RAG
          </a>
          <div className="flex gap-4">
            <a
              href="/login"
              className="text-white/80 hover:text-white transition-colors"
              style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)' }}
            >
              Login
            </a>
            <a
              href="/register"
              className="px-4 py-2 rounded-xl text-white font-medium transition-all hover:scale-105"
              style={{
                background: 'linear-gradient(135deg, #06b6d4, #8b5cf6)',
                boxShadow: '0px 4px 12px rgba(139, 92, 246, 0.4)',
                textShadow: '0px 1px 2px rgba(0, 0, 0, 0.3)',
              }}
            >
              Register
            </a>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <div className="relative max-w-7xl mx-auto px-4 py-20">
        <div className="text-center">
          <h1
            className="text-5xl font-bold text-white mb-6"
            style={{ textShadow: '0px 2px 4px rgba(0, 0, 0, 0.3)' }}
          >
            Bio-RAG
          </h1>
          <p
            className="text-2xl text-white/90 mb-4"
            style={{ textShadow: '0px 1px 3px rgba(0, 0, 0, 0.25)' }}
          >
            Biomedical Research AI-Guided Analytics Platform
          </p>
          <p
            className="text-lg text-white/70 max-w-2xl mx-auto mb-12"
            style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)' }}
          >
            Discover and understand biomedical research with AI-powered semantic search
            and intelligent Q&A. Get evidence-based answers with citations from
            peer-reviewed scientific papers.
          </p>

          <div className="flex justify-center gap-4">
            <a
              href="/search"
              className="px-8 py-4 rounded-xl text-white font-medium transition-all hover:scale-105 active:scale-95"
              style={{
                background: 'linear-gradient(135deg, #06b6d4, #8b5cf6)',
                boxShadow: '0px 4px 20px rgba(139, 92, 246, 0.5)',
                textShadow: '0px 1px 2px rgba(0, 0, 0, 0.3)',
              }}
            >
              Start Searching
            </a>
            <a
              href="/chat"
              className="px-8 py-4 rounded-xl text-white font-medium transition-all hover:scale-105 active:scale-95"
              style={{
                background: 'rgba(255, 255, 255, 0.2)',
                backdropFilter: 'blur(12px)',
                WebkitBackdropFilter: 'blur(12px)',
                border: '1px solid rgba(255, 255, 255, 0.4)',
                boxShadow: '0px 4px 16px rgba(0, 0, 0, 0.15)',
                textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)',
              }}
            >
              Ask Questions
            </a>
          </div>
        </div>

        {/* Features */}
        <div className="mt-20 grid md:grid-cols-2 lg:grid-cols-4 gap-6">
          <FeatureCard
            title="Semantic Search"
            description="Find relevant papers using natural language queries with AI-powered semantic understanding."
            icon="🔍"
            href="/search"
            buttonText="Try Search"
          />
          <FeatureCard
            title="AI-Powered Q&A"
            description="Get evidence-based answers to your biomedical questions with citations from scientific papers."
            icon="🤖"
            href="/chat"
            buttonText="Ask Now"
          />
          <FeatureCard
            title="Research Tools"
            description="ArXiv search, trend analysis, paper summarization, and Korean-English medical translation."
            icon="🛠️"
            href="/tools"
            buttonText="Open Tools"
          />
          <FeatureCard
            title="Citation Verification"
            description="Every answer includes verified citations to ensure accuracy and traceability."
            icon="✅"
            href="/chat"
            buttonText="See Citations"
          />
        </div>
      </div>
    </div>
  );
}

function FeatureCard({
  title,
  description,
  icon,
  href,
  buttonText
}: {
  title: string;
  description: string;
  icon: string;
  href: string;
  buttonText: string;
}) {
  return (
    <div
      className="rounded-2xl p-6 transition-all hover:scale-[1.02] flex flex-col"
      style={{
        background: 'rgba(255, 255, 255, 0.18)',
        backdropFilter: 'blur(16px)',
        WebkitBackdropFilter: 'blur(16px)',
        border: '1px solid rgba(255, 255, 255, 0.4)',
        boxShadow: '0px 4px 24px rgba(0, 0, 0, 0.15), inset 0px 0px 12px rgba(255, 255, 255, 0.25)',
      }}
    >
      <div className="text-4xl mb-4">{icon}</div>
      <h3
        className="text-lg font-semibold text-white mb-2"
        style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.25)' }}
      >
        {title}
      </h3>
      <p
        className="text-white/70 flex-1"
        style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)' }}
      >
        {description}
      </p>
      <a
        href={href}
        className="mt-4 inline-block px-4 py-2 rounded-xl text-white text-sm font-medium text-center transition-all hover:scale-105 active:scale-95"
        style={{
          background: 'linear-gradient(135deg, rgba(6, 182, 212, 0.6), rgba(139, 92, 246, 0.6))',
          border: '1px solid rgba(255, 255, 255, 0.3)',
          textShadow: '0px 1px 2px rgba(0, 0, 0, 0.3)',
        }}
      >
        {buttonText} →
      </a>
    </div>
  );
}

export default App;
