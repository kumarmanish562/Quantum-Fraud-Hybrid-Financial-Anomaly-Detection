import React from 'react';

const HowItWorks = ({ isDark }) => {
  const steps = [
    {
      number: "01",
      title: "Transaction Submission",
      description: "User initiates a financial transaction through your platform or API",
      icon: "💳"
    },
    {
      number: "02",
      title: "Feature Extraction",
      description: "System extracts key features: Time, Amount, Location, Merchant, and PCA components",
      icon: "🔍"
    },
    {
      number: "03",
      title: "HQNN Model Processing",
      description: "Hybrid Quantum-Classical Neural Network analyzes patterns using quantum entanglement",
      icon: "⚛️"
    },
    {
      number: "04",
      title: "Fraud Probability Generated",
      description: "Model calculates fraud probability score with confidence level and risk factors",
      icon: "📊"
    },
    {
      number: "05",
      title: "Alert & Response",
      description: "If fraud detected, instant alert triggered with recommended actions",
      icon: "🚨"
    }
  ];

  return (
    <section id="how-it-works" className={`py-20 px-6 ${isDark ? '' : 'bg-gray-50'}`}>
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-16">
          <h2 className={`text-4xl md:text-5xl font-bold mb-4 ${isDark ? 'text-white' : 'text-gray-900'}`}>How It Works</h2>
          <p className={`text-xl max-w-3xl mx-auto ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
            Our quantum-enhanced fraud detection system processes transactions in 5 simple steps
          </p>
        </div>

        <div className="relative">
          {/* Connection Line */}
          <div className={`hidden lg:block absolute top-1/2 left-0 right-0 h-1 transform -translate-y-1/2 z-0 ${isDark ? 'bg-gradient-to-r from-blue-500 via-purple-500 to-cyan-500' : 'bg-gradient-to-r from-blue-300 via-purple-300 to-cyan-300'}`}></div>

          <div className="grid md:grid-cols-2 lg:grid-cols-5 gap-8 relative z-10">
            {steps.map((step, index) => (
              <div key={index} className="relative">
                <div className={`backdrop-blur-sm border rounded-2xl p-6 transition-all duration-300 ${isDark ? 'bg-gray-800/80 border-gray-700 hover:border-blue-500/50 hover:shadow-xl' : 'bg-white border-gray-200 hover:border-blue-400 shadow-md hover:shadow-xl'}`}>
                  {/* Step Number */}
                  <div className="absolute -top-4 -left-4 w-12 h-12 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-full flex items-center justify-center text-white font-bold text-lg shadow-lg">
                    {step.number}
                  </div>

                  {/* Icon */}
                  <div className="text-6xl mb-4 text-center">{step.icon}</div>

                  {/* Content */}
                  <h3 className={`text-xl font-semibold mb-3 text-center ${isDark ? 'text-white' : 'text-gray-900'}`}>{step.title}</h3>
                  <p className={`text-sm text-center leading-relaxed ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>{step.description}</p>
                </div>

                {/* Arrow for mobile */}
                {index < steps.length - 1 && (
                  <div className="lg:hidden flex justify-center my-4">
                    <svg className="w-6 h-6 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
                    </svg>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Technical Details */}
        <div className={`mt-16 rounded-2xl p-8 border ${isDark ? 'bg-gradient-to-br from-blue-500/10 to-purple-500/10 border-blue-500/30' : 'bg-gradient-to-br from-blue-50 to-purple-50 border-blue-200'}`}>
          <h3 className={`text-2xl font-bold mb-6 text-center ${isDark ? 'text-white' : 'text-gray-900'}`}>Technical Architecture</h3>
          <div className="grid md:grid-cols-3 gap-6">
            <div className="text-center">
              <div className="text-4xl mb-3">🔬</div>
              <h4 className={`font-semibold mb-2 ${isDark ? 'text-white' : 'text-gray-900'}`}>Quantum Layer</h4>
              <p className={`text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>PennyLane quantum circuits with 4 qubits for pattern recognition</p>
            </div>
            <div className="text-center">
              <div className="text-4xl mb-3">🤖</div>
              <h4 className={`font-semibold mb-2 ${isDark ? 'text-white' : 'text-gray-900'}`}>Classical Layer</h4>
              <p className={`text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>XGBoost ensemble model for traditional feature analysis</p>
            </div>
            <div className="text-center">
              <div className="text-4xl mb-3">⚡</div>
              <h4 className={`font-semibold mb-2 ${isDark ? 'text-white' : 'text-gray-900'}`}>Hybrid Fusion</h4>
              <p className={`text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>Neural network combines quantum and classical outputs</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default HowItWorks;
