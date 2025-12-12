import { Routes, Route } from 'react-router-dom';
import { AuthProvider } from './contexts/AuthContext';
import Layout from './components/Layout';
import ProtectedRoute from './components/ProtectedRoute';
import { Login, Register, Search, Chat } from './pages';

function App() {
  return (
    <AuthProvider>
      <Routes>
        {/* Public routes */}
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />

        {/* Protected routes with layout */}
        <Route element={<Layout />}>
          <Route path="/" element={<Home />} />
          <Route
            path="/search"
            element={
              <ProtectedRoute>
                <Search />
              </ProtectedRoute>
            }
          />
          <Route
            path="/chat"
            element={
              <ProtectedRoute>
                <Chat />
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
    <div className="max-w-7xl mx-auto px-4 py-12">
      <div className="text-center">
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
          Bio-RAG
        </h1>
        <p className="text-xl text-gray-600 dark:text-gray-300 mb-8">
          Biomedical Research AI-Guided Analytics Platform
        </p>
        <p className="text-gray-500 dark:text-gray-400 max-w-2xl mx-auto mb-8">
          Discover and understand biomedical research with AI-powered semantic search
          and intelligent Q&A. Get evidence-based answers with citations from
          peer-reviewed scientific papers.
        </p>

        <div className="flex justify-center gap-4">
          <a
            href="/search"
            className="px-6 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 font-medium"
          >
            Start Searching
          </a>
          <a
            href="/chat"
            className="px-6 py-3 border border-indigo-600 text-indigo-600 dark:text-indigo-400 rounded-lg hover:bg-indigo-50 dark:hover:bg-indigo-900/20 font-medium"
          >
            Ask Questions
          </a>
        </div>
      </div>

      {/* Features */}
      <div className="mt-16 grid md:grid-cols-3 gap-8">
        <FeatureCard
          title="Semantic Search"
          description="Find relevant papers using natural language queries with AI-powered semantic understanding."
        />
        <FeatureCard
          title="AI-Powered Q&A"
          description="Get evidence-based answers to your biomedical questions with citations from scientific papers."
        />
        <FeatureCard
          title="Citation Verification"
          description="Every answer includes verified citations to ensure accuracy and traceability."
        />
      </div>
    </div>
  );
}

function FeatureCard({ title, description }: { title: string; description: string }) {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
        {title}
      </h3>
      <p className="text-gray-600 dark:text-gray-400">{description}</p>
    </div>
  );
}

export default App;
