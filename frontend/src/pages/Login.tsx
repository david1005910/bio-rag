import { useState, FormEvent } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';

export default function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);

    try {
      await login({ email, password });
      navigate('/');
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Login failed. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-violet-600 via-purple-500 to-fuchsia-500 py-12 px-4">
      {/* Animated background orbs */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-pink-400 rounded-full mix-blend-multiply filter blur-3xl opacity-70 animate-pulse" />
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-cyan-400 rounded-full mix-blend-multiply filter blur-3xl opacity-70 animate-pulse [animation-delay:2s]" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-purple-400 rounded-full mix-blend-multiply filter blur-3xl opacity-50 animate-pulse [animation-delay:4s]" />
      </div>

      <div
        className="relative max-w-md w-full space-y-8 p-8 rounded-2xl"
        style={{
          background: 'rgba(255, 255, 255, 0.22)',
          backdropFilter: 'blur(16px)',
          WebkitBackdropFilter: 'blur(16px)',
          border: '1px solid rgba(255, 255, 255, 0.4)',
          boxShadow: '0px 4px 24px rgba(0, 0, 0, 0.2), inset 0px 0px 12px rgba(255, 255, 255, 0.25)',
        }}
      >
        <div>
          <h1
            className="text-center text-3xl font-bold text-white"
            style={{ textShadow: '0px 2px 4px rgba(0, 0, 0, 0.3)' }}
          >
            Bio-RAG
          </h1>
          <h2
            className="mt-4 text-center text-xl font-semibold text-white/90"
            style={{ textShadow: '0px 1px 3px rgba(0, 0, 0, 0.25)' }}
          >
            Sign in to your account
          </h2>
        </div>

        <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
          {error && (
            <div
              className="rounded-xl p-4 text-sm text-white"
              style={{
                background: 'rgba(239, 68, 68, 0.3)',
                border: '1px solid rgba(239, 68, 68, 0.5)',
                backdropFilter: 'blur(8px)',
              }}
            >
              {error}
            </div>
          )}

          <div className="space-y-4">
            <div>
              <label
                htmlFor="email"
                className="block text-sm font-medium text-white/90 mb-1"
                style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)' }}
              >
                Email address
              </label>
              <input
                id="email"
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="block w-full rounded-xl px-4 py-3 text-white placeholder-white/50 focus:outline-none focus:ring-2 focus:ring-cyan-400/50 transition-all"
                style={{
                  background: 'rgba(255, 255, 255, 0.15)',
                  border: '1px solid rgba(255, 255, 255, 0.3)',
                  backdropFilter: 'blur(8px)',
                }}
                placeholder="you@example.com"
              />
            </div>

            <div>
              <label
                htmlFor="password"
                className="block text-sm font-medium text-white/90 mb-1"
                style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)' }}
              >
                Password
              </label>
              <input
                id="password"
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="block w-full rounded-xl px-4 py-3 text-white placeholder-white/50 focus:outline-none focus:ring-2 focus:ring-cyan-400/50 transition-all"
                style={{
                  background: 'rgba(255, 255, 255, 0.15)',
                  border: '1px solid rgba(255, 255, 255, 0.3)',
                  backdropFilter: 'blur(8px)',
                }}
                placeholder="Enter your password"
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={isLoading}
            className="w-full py-3 px-4 rounded-xl text-white font-medium focus:outline-none focus:ring-2 focus:ring-cyan-400/50 disabled:opacity-50 disabled:cursor-not-allowed transition-all hover:scale-[1.02] active:scale-[0.98]"
            style={{
              background: 'linear-gradient(135deg, #06b6d4, #8b5cf6)',
              boxShadow: '0px 4px 16px rgba(139, 92, 246, 0.4)',
              textShadow: '0px 1px 2px rgba(0, 0, 0, 0.3)',
            }}
          >
            {isLoading ? 'Signing in...' : 'Sign in'}
          </button>

          <div
            className="text-center text-sm text-white/80"
            style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)' }}
          >
            Don't have an account?{' '}
            <Link
              to="/register"
              className="font-medium text-cyan-300 hover:text-cyan-200 transition-colors"
            >
              Register here
            </Link>
          </div>

          <div className="text-center">
            <Link
              to="/"
              className="text-sm text-white/60 hover:text-white/80 transition-colors"
              style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)' }}
            >
              ← Back to Home
            </Link>
          </div>
        </form>
      </div>
    </div>
  );
}
