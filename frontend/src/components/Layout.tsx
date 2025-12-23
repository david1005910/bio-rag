import { Link, useNavigate, Outlet } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';

export default function Layout() {
  const { user, isAuthenticated, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-violet-600 via-purple-500 to-fuchsia-500">
      {/* Animated background orbs */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-pink-400 rounded-full mix-blend-multiply filter blur-3xl opacity-70 animate-pulse" />
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-cyan-400 rounded-full mix-blend-multiply filter blur-3xl opacity-70 animate-pulse [animation-delay:2s]" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-purple-400 rounded-full mix-blend-multiply filter blur-3xl opacity-50 animate-pulse [animation-delay:4s]" />
      </div>

      {/* Navigation - Glass Panel */}
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
          <div className="flex items-center gap-8">
            <Link
              to="/"
              className="text-xl font-bold text-white"
              style={{ textShadow: '0px 1px 3px rgba(0, 0, 0, 0.3)' }}
            >
              Bio-RAG
            </Link>

            {isAuthenticated && (
              <div className="flex gap-6">
                <Link
                  to="/search"
                  className="text-white/80 hover:text-white transition-colors"
                  style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)' }}
                >
                  Search
                </Link>
                <Link
                  to="/chat"
                  className="text-white/80 hover:text-white transition-colors"
                  style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)' }}
                >
                  Chat
                </Link>
                <Link
                  to="/tools"
                  className="text-white/80 hover:text-white transition-colors"
                  style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)' }}
                >
                  Tools
                </Link>
              </div>
            )}
          </div>

          <div className="flex items-center gap-4">
            {isAuthenticated ? (
              <>
                <span
                  className="text-sm text-white/80"
                  style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)' }}
                >
                  {user?.name}
                </span>
                <button
                  onClick={handleLogout}
                  className="text-sm px-4 py-2 rounded-xl text-white/80 hover:text-white transition-all hover:scale-105"
                  style={{
                    background: 'rgba(255, 255, 255, 0.15)',
                    border: '1px solid rgba(255, 255, 255, 0.3)',
                    textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)',
                  }}
                >
                  Logout
                </button>
              </>
            ) : (
              <>
                <Link
                  to="/login"
                  className="text-white/80 hover:text-white transition-colors"
                  style={{ textShadow: '0px 1px 2px rgba(0, 0, 0, 0.2)' }}
                >
                  Login
                </Link>
                <Link
                  to="/register"
                  className="px-4 py-2 rounded-xl text-white font-medium transition-all hover:scale-105"
                  style={{
                    background: 'linear-gradient(135deg, #06b6d4, #8b5cf6)',
                    boxShadow: '0px 4px 12px rgba(139, 92, 246, 0.4)',
                    textShadow: '0px 1px 2px rgba(0, 0, 0, 0.3)',
                  }}
                >
                  Register
                </Link>
              </>
            )}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="relative">
        <Outlet />
      </main>
    </div>
  );
}
