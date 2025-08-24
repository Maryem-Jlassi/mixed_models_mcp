import React from 'react';
import Main from '../components/Main/Main';
import Sidebar from '../components/sidebar/Sidebar';

function ChatPage() {
  return (
    <div className="flex h-screen bg-gray-50">
      <Main />
      <Sidebar />
    </div>
  );
}

export default ChatPage;
