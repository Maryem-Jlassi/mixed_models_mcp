import React from 'react';
import { useNavigate } from 'react-router-dom';
import Header from '../components/Header';
import Hero from '../components/Hero';
import Features from '../components/Features';
import RecentConversations from '../components/RecentConversations';
import Footer from '../components/Footer';

function HomePage() {
  const navigate = useNavigate();

  const handleNewChat = () => {
    navigate('/chat');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      <Header />
      <div className="bg-gradient-to-b from-white to-gray-50">
        <Hero onNewChatClick={handleNewChat} />
      </div>
      <div className="bg-white">
        <Features />
        <RecentConversations />
        <Footer />
      </div>
    </div>
  );
}

export default HomePage;
