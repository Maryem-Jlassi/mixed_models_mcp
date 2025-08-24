import React from 'react';

function Hero({ onNewChatClick }) {
  return (
    <section className="w-full bg-gradient-to-br from-gray-50 via-white to-gray-100 py-20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center">
          <h1 className="text-5xl font-bold text-gray-900 mb-6">
            Welcome to <span className="text-blue-500">TalentMind</span>
          </h1>
          <p className="text-2xl text-gray-700 mb-8">
            Your AI-Powered Career Assistant
          </p>
          <p className="text-lg text-gray-600 mb-12 max-w-2xl mx-auto">
            TalentMind helps you optimize your CV, prepare for interviews, and advance your career with AI-powered insights and recommendations.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
            <button 
              onClick={onNewChatClick}
              className="bg-blue-500 text-white px-8 py-4 rounded-lg text-lg font-semibold hover:bg-blue-600 transition-colors duration-200 shadow-md"
            >
              Start New Chat
            </button>
            <button className="bg-white text-blue-500 border-2 border-blue-500 px-8 py-4 rounded-lg text-lg font-semibold hover:bg-blue-50 transition-colors duration-200">
              Learn More
            </button>
          </div>
          <div className="mt-8 text-sm text-gray-500">
            <p>TalentMind uses AI to assist with your career development. Always review and verify the information provided.</p>
          </div>
        </div>
      </div>
    </section>
  );
}

export default Hero;
