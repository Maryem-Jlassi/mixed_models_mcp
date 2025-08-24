import React from 'react';

function Features() {
  const features = [
    {
      title: "AI-Powered CV Review",
      description: "Get instant feedback and suggestions to improve your CV and make it stand out to employers",
      icon: "📝"
    },
    {
      title: "Interview Preparation",
      description: "Practice with common interview questions and receive AI-powered feedback on your responses",
      icon: "💼"
    },
    {
      title: "Career Pathing",
      description: "Discover potential career paths based on your skills, experience, and interests",
      icon: "🛣️"
    },
    {
      title: "Skill Gap Analysis",
      description: "Identify skills you need to develop to advance in your desired career",
      icon: "📊"
    }
  ];

  return (
    <section className="w-full py-20 bg-white" id="features">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">Powerful Features</h2>
          <p className="text-lg text-gray-600">Discover what TalentMind can do for your career</p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {features.map((feature, index) => (
            <div key={index} className="bg-blue-50 rounded-xl p-6 text-center hover:shadow-lg transition-shadow duration-300 border border-blue-100">
              <div className="text-4xl mb-4">{feature.icon}</div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">{feature.title}</h3>
              <p className="text-gray-600">{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

export default Features;
