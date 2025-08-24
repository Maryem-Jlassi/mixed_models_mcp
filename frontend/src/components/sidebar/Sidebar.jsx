import { useContext, useState, useEffect } from "react";
import { Link } from 'react-router-dom';
import { Context } from "../../context/Context";
import { Menu, Plus, MessageSquare, HelpCircle, Settings, X, User } from 'lucide-react';

const Sidebar = () => {
  const [sidebarExtended, setSidebarExtended] = useState(false);
  const [showProfileMenu, setShowProfileMenu] = useState(false);
  const [showLogoutConfirm, setShowLogoutConfirm] = useState(false);
  const { sessions, refreshSessions, openSession, newChat } = useContext(Context);

  // derive user info from localStorage
  let user = null;
  try {
    const s = localStorage.getItem('tm_user');
    user = s ? JSON.parse(s) : null;
  } catch {}
  const userName = user?.name || '';
  const userEmail = user?.email || '';
  const initials = (userName || userEmail || 'U')
    .split(' ')
    .map(p => p[0])
    .join('')
    .slice(0, 2)
    .toUpperCase();

  const handleLogout = () => {
    setShowLogoutConfirm(true);
    setShowProfileMenu(false);
  };

  const confirmLogout = () => {
    try {
      localStorage.removeItem('tm_token');
      localStorage.removeItem('tm_user');
    } catch {}
    // Reload same page to switch to visitor mode without routing away
    window.location.reload();
  };
  
  useEffect(() => {
    // keep sessions fresh when sidebar toggles
    refreshSessions();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className={`flex flex-col justify-between h-screen bg-gray-50 p-4 transition-all duration-300 ease-in-out ${sidebarExtended ? 'w-64' : 'w-20'}`}>
      <div className="space-y-4">
        <div className="flex items-center justify-between mb-6">
          <button 
            onClick={() => setSidebarExtended(prev => !prev)}
            className="p-2 rounded-md hover:bg-gray-200 transition-colors duration-200"
            aria-label={sidebarExtended ? 'Collapse sidebar' : 'Expand sidebar'}
          >
            {sidebarExtended ? (
              <X className="w-5 h-5 text-gray-700" />
            ) : (
              <Menu className="w-5 h-5 text-gray-700" />
            )}
          </button>
        </div>

        <button
          onClick={() => newChat()}
          className={`flex items-center gap-3 w-full p-3 rounded-lg bg-white hover:bg-gray-100 transition-colors duration-200 shadow-sm ${
            sidebarExtended ? 'justify-start' : 'justify-center'
          }`}
        >
          <Plus className="w-5 h-5 text-blue-600" />
          {sidebarExtended && <span className="text-sm font-medium text-gray-700">New Chat</span>}
        </button>

        {/* Visitor prompt when not authenticated */}
        {sidebarExtended && !Boolean(typeof window !== 'undefined' && localStorage.getItem('tm_token')) && (
          <div className="mt-6 p-3 bg-white rounded-lg shadow-sm border border-gray-100">
            <h3 className="text-sm font-semibold text-gray-800">Sign in to start saving your chats</h3>
            <p className="mt-2 text-xs text-gray-600">Once you're signed in, you can access your recent chats here.</p>
            <Link
              to="/signin"
              className="mt-3 inline-flex items-center justify-center px-3 py-1.5 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700"
            >
              Sign in
            </Link>
          </div>
        )}

        {sidebarExtended && sessions.length > 0 && (
          <div className="mt-6">
            <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3 px-2">
              Recent Chats
            </h3>
            <div className="space-y-1 max-h-[calc(100vh-300px)] overflow-y-auto">
              {sessions.map((s) => (
                <button
                  key={s.session_id}
                  onClick={() => openSession(s.session_id)}
                  className="flex items-center gap-3 w-full p-2 rounded-lg hover:bg-gray-100 transition-colors duration-200 text-left"
                >
                  <MessageSquare className="w-4 h-4 text-gray-500 flex-shrink-0" />
                  <span className="text-sm text-gray-700 truncate">
                    {s.last_message_preview && s.last_message_preview.length > 24
                      ? `${s.last_message_preview.substring(0, 24)}...`
                      : (s.last_message_preview || s.session_id)}
                  </span>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="space-y-1 relative">
        {/* Help button only for visitors (no token) */}
        {!Boolean(typeof window !== 'undefined' && localStorage.getItem('tm_token')) && (
          <button
            className={`flex items-center gap-3 w-full p-2 rounded-lg hover:bg-gray-100 transition-colors duration-200 ${
              sidebarExtended ? 'justify-start' : 'justify-center'
            }`}
          >
            <HelpCircle className="w-5 h-5 text-gray-600" />
            {sidebarExtended && <span className="text-sm text-gray-700">Help</span>}
          </button>
        )}

        {/* Profile section at the bottom - only if authenticated */}
        {Boolean(typeof window !== 'undefined' && localStorage.getItem('tm_token')) && (
        <div className="mt-3">
          <button
            onClick={() => setShowProfileMenu(v => !v)}
            className={`flex items-center gap-3 w-full p-2 rounded-lg bg-white hover:bg-gray-100 transition-colors duration-200 border border-gray-100 shadow-sm ${
              sidebarExtended ? 'justify-start' : 'justify-center'
            }`}
            aria-label="Profile menu"
          >
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-600 to-purple-600 text-white flex items-center justify-center font-semibold text-sm">
              {initials}
            </div>
            {sidebarExtended && (
              <div className="flex flex-col items-start min-w-0">
                <span className="text-sm font-medium text-gray-800 truncate">{userName || 'Guest'}</span>
                <span className="text-xs text-gray-500 truncate">{userEmail || 'Not signed in'}</span>
              </div>
            )}
          </button>

          {showProfileMenu && (
            <div className={`absolute left-4 right-4 bottom-16 z-20 bg-white rounded-xl shadow-xl border border-gray-100 p-2 ${sidebarExtended ? '' : 'mx-6'}`}>
              <button className="flex w-full items-center gap-3 px-3 py-2 rounded-lg hover:bg-gray-50">
                <Settings className="w-4 h-4 text-gray-600" />
                <span className="text-sm text-gray-700">Settings</span>
              </button>
              <button className="flex w-full items-center gap-3 px-3 py-2 rounded-lg hover:bg-gray-50">
                <HelpCircle className="w-4 h-4 text-gray-600" />
                <span className="text-sm text-gray-700">Help</span>
              </button>
              <div className="h-px bg-gray-100 my-1" />
              <button onClick={handleLogout} className="flex w-full items-center gap-3 px-3 py-2 rounded-lg hover:bg-red-50">
                <User className="w-4 h-4 text-red-600" />
                <span className="text-sm text-red-600">Logout</span>
              </button>
            </div>
          )}
        </div>
        )}
      </div>

      {/* Logout confirmation modal */}
      {showLogoutConfirm && (
        <div className="fixed inset-0 z-30 flex items-center justify-center">
          <div className="absolute inset-0 bg-black/30" onClick={() => setShowLogoutConfirm(false)} />
          <div className="relative bg-white rounded-2xl shadow-2xl w-full max-w-md mx-4 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Confirmation</h3>
            <p className="text-sm text-gray-700 mb-4">
              Souhaitez-vous vraiment vous déconnecter ?<br />
              Se déconnecter de TalentMind en tant que {userEmail || 'utilisateur'} ?
            </p>
            <div className="flex justify-end gap-3">
              <button
                onClick={() => setShowLogoutConfirm(false)}
                className="px-4 py-2 rounded-lg border border-gray-300 text-gray-700 hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                onClick={confirmLogout}
                className="px-4 py-2 rounded-lg text-white bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500"
              >
                Logout
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Sidebar;