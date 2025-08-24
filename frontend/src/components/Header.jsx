import React, { useEffect, useState } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';

function Header() {
  const navigate = useNavigate();
  const location = useLocation();
  const [user, setUser] = useState(null);

  useEffect(() => {
    try {
      const raw = localStorage.getItem('tm_user');
      const token = localStorage.getItem('tm_token');
      const parsed = raw ? JSON.parse(raw) : null;
      setUser(token ? parsed : null);
    } catch {
      setUser(null);
    }
  }, []);

  const handleSignOut = () => {
    try {
      localStorage.removeItem('tm_token');
      localStorage.removeItem('tm_user');
    } finally {
      setUser(null);
      navigate('/');
    }
  };

  const displayName = (user?.name || user?.username || '').trim();
  const initials = displayName
    ? displayName.split(/\s+/).map(p => p[0]).slice(0, 2).join('').toUpperCase()
    : 'TM';
  const isChat = (location?.pathname || '').startsWith('/chat');

  return (
    <header className="w-full bg-white border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center">
            <div className="flex items-center space-x-3">
              <Link to="/" className="text-xl font-semibold text-gray-900">
                TalentMind
              </Link>
            </div>
          </div>
          <nav className="flex items-center space-x-4">
            <Link to="#features" className="text-gray-600 hover:text-gray-900 px-3 py-2 rounded-md text-sm font-medium">
              Features
            </Link>
            <Link to="#pricing" className="text-gray-600 hover:text-gray-900 px-3 py-2 rounded-md text-sm font-medium">
              Pricing
            </Link>
            <Link to="#business" className="text-gray-600 hover:text-gray-900 px-3 py-2 rounded-md text-sm font-medium">
              For Business
            </Link>
            {user ? (
              <div className="flex items-center space-x-3">
                {isChat ? (
                  <Link to="/chat" className="hidden sm:block text-gray-700">
                    <span className="mr-2">Welcome,</span>
                    <span className="font-semibold">{displayName || 'Professional'}</span>
                  </Link>
                ) : (
                  <Link to="/chat" className="hidden sm:block text-gray-700">
                    <span className="font-semibold">Chat</span>
                  </Link>
                )}
                <div className="w-8 h-8 rounded-full bg-blue-600 text-white flex items-center justify-center text-xs font-bold" title={displayName || 'User'}>
                  {initials}
                </div>
                <button onClick={handleSignOut} className="text-gray-600 hover:text-gray-900 px-3 py-2 rounded-md text-sm font-medium border border-gray-200 hover:border-gray-300">
                  Sign Out
                </button>
              </div>
            ) : (
              <>
                {isChat ? (
                  <>
                    <span className="hidden sm:block text-gray-700">Hello, <span className="font-semibold">Professional</span></span>
                    <Link to="/signin" className="bg-blue-500 text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-blue-600 transition-colors">
                      Sign In
                    </Link>
                  </>
                ) : (
                  <Link to="/signin" className="bg-blue-500 text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-blue-600 transition-colors">
                    Sign In
                  </Link>
                )}
              </>
            )}
          </nav>
        </div>
      </div>
    </header>
  );
}

export default Header;
