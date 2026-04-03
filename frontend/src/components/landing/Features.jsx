import React from 'react';

const Features = ({ isDark }) => {
  const features = [
    {
      icon: "⚡",
      title: "Real-Time Fraud Detection",
      description: "Instant fraud detection with quantum-enhanced ML models. Get results in milliseconds with high precision.",
      color: "blue"
    },
    {
      icon: "🧠",
      title: "AI + Quantum Hybrid Model",
      description: "Hybrid Quantum-Classical Neural Network (HQNN) combined with XGBoost for superior pattern recognition.",
      color: "purple"
    },
    {
      icon: "📊",
      title: "99% Accuracy",
      description: "Exceptional detection accuracy with minimal false positives. Trained on real-world credit card fraud data.",
      color: "green"
    },
    {
      icon: "🔔",
      title: "Instant Alert System",
      description: "Real-time notifications and alerts for suspicious transactions. WebSocket-based live monitoring.",
      color: "yellow"
    },
    {
      icon: "🔒",
      title: "Secure & Privacy Focused",
      description: "Enterprise-grade security with encrypted data transmission. GDPR and PCI-DSS compliant architecture.",
      color: "red"
    },
    {
      icon: "📈",
      title: "Advanced Analytics Dashboard",
      description: "Comprehensive analytics with interactive charts, fraud trends, and risk distribution insights.",
      color: "cyan"
    }
  ];

  const colorClasses = {
    blue: "from-blue-500 to-blue-600 border-blue-500/30 bg-blue-500/10",
    purple: "from-purple-500 to-purple-600 border-purple-500/30 bg-purple-500/10",
    green: "from-green-500 to-green-600 border-green-500/30 bg-green-500/10",
    yellow: "from-yellow-500 to-yellow-600 border-yellow-500/30 bg-yellow-500/10",
    red: "from-red-500 to-red-600 border-red-500/30 bg-red-500/10",
    cyan: "from-cyan-500 to-cyan-600 border-cyan-500/30 bg-cyan-500/10"
  };

  return (
    <section id="features" className={`py-20 px-6 ${isDark ? 'bg-gray-900/50' : 'bg-gray-50'}`}>
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-16">
          <h2 className={`text-4xl md:text-5xl font-bold mb-4 ${isDark ? 'text-white' : 'text-gray-900'}`}>Powerful Features</h2>
          <p className={`text-xl max-w-3xl mx-auto ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
            Advanced quantum-enhanced fraud detection capabilities designed for modern financial platforms
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <div 
              key={index}
              className={`backdrop-blur-sm border rounded-2xl p-8 hover:shadow-xl hover:scale-105 transition-all duration-300 ${
                isDark 
                  ? 'bg-gray-800/50 border-gray-700 hover:border-blue-500/50' 
                  : 'bg-white border-gray-200 hover:border-blue-400 shadow-md'
              }`}
            >
              <div className={`w-16 h-16 bg-gradient-to-br ${colorClasses[feature.color]} rounded-xl flex items-center justify-center mb-6 text-4xl`}>
                {feature.icon}
              </div>
              <h3 className={`text-2xl font-semibold mb-3 ${isDark ? 'text-white' : 'text-gray-900'}`}>{feature.title}</h3>
              <p className={`leading-relaxed ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Features;
