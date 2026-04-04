import React from 'react';

const About = ({ isDark }) => {
  return (
    <section id="about" className={`py-20 px-6 ${isDark ? 'bg-gray-50/50' : 'bg-white'}`}>
      <div className="max-w-7xl mx-auto">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          {/* Left Content */}
          <div>
            <div className={`inline-block mb-4 px-4 py-2 rounded-full ${isDark ? 'bg-purple-500/10 border border-purple-500/30' : 'bg-purple-50 border border-purple-200'}`}>
              <span className={`text-sm font-medium ${isDark ? 'text-purple-400' : 'text-purple-600'}`}>About Our System</span>
            </div>
            
            <h2 className={`text-4xl md:text-5xl font-bold mb-6 ${isDark ? 'text-gray-900' : 'text-gray-900'}`}>
              Next-Generation Fraud Detection Technology
            </h2>
            
            <p className={`text-lg mb-6 leading-relaxed ${isDark ? 'text-gray-600' : 'text-gray-700'}`}>
              Our Quantum-Enhanced Financial Fraud Detection System represents a breakthrough in 
              cybersecurity and financial protection. By combining quantum computing with classical 
              machine learning, we've created a solution that detects fraud patterns invisible to 
              traditional systems.
            </p>

            {/* Problem Statement */}
            <div className={`rounded-xl p-6 mb-6 ${isDark ? 'bg-red-500/10 border border-red-500/30' : 'bg-red-50 border border-red-200'}`}>
              <h3 className={`text-xl font-semibold mb-3 flex items-center ${isDark ? 'text-gray-900' : 'text-gray-900'}`}>
                <span className="text-2xl mr-2">⚠️</span>
                The Problem
              </h3>
              <p className={`leading-relaxed ${isDark ? 'text-gray-600' : 'text-gray-700'}`}>
                Financial fraud costs billions annually. Traditional rule-based systems and basic ML 
                models struggle with sophisticated fraud patterns, resulting in high false positives 
                and missed fraudulent transactions.
              </p>
            </div>

            {/* Solution */}
            <div className={`rounded-xl p-6 mb-6 ${isDark ? 'bg-green-500/10 border border-green-500/30' : 'bg-green-50 border border-green-200'}`}>
              <h3 className={`text-xl font-semibold mb-3 flex items-center ${isDark ? 'text-gray-900' : 'text-gray-900'}`}>
                <span className="text-2xl mr-2">✅</span>
                Our Solution
              </h3>
              <p className={`leading-relaxed mb-4 ${isDark ? 'text-gray-600' : 'text-gray-700'}`}>
                Hybrid Quantum-Classical Neural Network (HQNN) combined with XGBoost creates a 
                powerful fraud detection system that leverages quantum computing advantages:
              </p>
              <ul className="space-y-2">
                <li className="flex items-start text-gray-600">
                  <span className="text-green-400 mr-2">▸</span>
                  <span>Quantum entanglement for complex pattern recognition</span>
                </li>
                <li className="flex items-start text-gray-600">
                  <span className="text-green-400 mr-2">▸</span>
                  <span>Superposition enables parallel processing of multiple fraud scenarios</span>
                </li>
                <li className="flex items-start text-gray-600">
                  <span className="text-green-400 mr-2">▸</span>
                  <span>Classical ML provides interpretability and reliability</span>
                </li>
              </ul>
            </div>

            {/* Benefits */}
            <div className="space-y-3">
              <h3 className={`text-xl font-semibold mb-4 ${isDark ? 'text-gray-900' : 'text-gray-900'}`}>Key Benefits</h3>
              {[
                { icon: "🎯", text: "Detects complex fraud patterns invisible to traditional systems" },
                { icon: "💰", text: "Reduces financial losses by up to 95%" },
                { icon: "⚡", text: "Real-time processing with <50ms response time" },
                { icon: "📊", text: "Supports data-driven decision making with detailed analytics" },
                { icon: "🔄", text: "Continuous learning and model improvement" }
              ].map((benefit, index) => (
                <div key={index} className={`flex items-center space-x-3 rounded-lg p-3 ${isDark ? 'bg-gray-100' : 'bg-gray-50 border border-gray-200'}`}>
                  <div className="text-2xl">{benefit.icon}</div>
                  <span className={isDark ? 'text-gray-600' : 'text-gray-700'}>{benefit.text}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Right Content - Stats & Metrics */}
          <div className="space-y-6">
            {/* Main Accuracy Card */}
            <div className={`rounded-2xl p-8 border ${isDark ? 'bg-gradient-to-br from-blue-500/20 to-cyan-500/20 border-blue-500/30' : 'bg-gradient-to-br from-blue-50 to-cyan-50 border-blue-200'}`}>
              <div className="text-7xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent mb-4">
                99.45%
              </div>
              <div className="text-2xl text-gray-900 font-semibold mb-6">Detection Accuracy</div>
              <p className="text-gray-600 mb-6">
                Our Hybrid Quantum-Classical Neural Network achieved exceptional performance on 
                credit card fraud detection dataset.
              </p>
              
              {/* Performance Metrics */}
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-sm text-gray-600 mb-2">
                    <span className="font-medium">Precision</span>
                    <span className="font-bold">96.2%</span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-3">
                    <div className="bg-gradient-to-r from-blue-500 to-cyan-400 h-3 rounded-full" style={{width: '96.2%'}}></div>
                  </div>
                </div>
                
                <div>
                  <div className="flex justify-between text-sm text-gray-600 mb-2">
                    <span className="font-medium">Recall</span>
                    <span className="font-bold">94.8%</span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-3">
                    <div className="bg-gradient-to-r from-green-500 to-emerald-400 h-3 rounded-full" style={{width: '94.8%'}}></div>
                  </div>
                </div>
                
                <div>
                  <div className="flex justify-between text-sm text-gray-600 mb-2">
                    <span className="font-medium">F1-Score</span>
                    <span className="font-bold">95.5%</span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-3">
                    <div className="bg-gradient-to-r from-purple-500 to-pink-400 h-3 rounded-full" style={{width: '95.5%'}}></div>
                  </div>
                </div>
              </div>
            </div>

            {/* Technology Stack */}
            <div className={`border rounded-2xl p-6 ${isDark ? 'bg-gray-100 border-gray-300' : 'bg-white border-gray-200 shadow-md'}`}>
              <h3 className={`text-xl font-semibold mb-4 ${isDark ? 'text-gray-900' : 'text-gray-900'}`}>Technology Stack</h3>
              <div className="grid grid-cols-2 gap-4">
                {[
                  { name: "PennyLane", desc: "Quantum ML" },
                  { name: "XGBoost", desc: "Classical ML" },
                  { name: "PyTorch", desc: "Deep Learning" },
                  { name: "FastAPI", desc: "Backend API" },
                  { name: "React", desc: "Frontend UI" },
                  { name: "PostgreSQL", desc: "Database" }
                ].map((tech, index) => (
                  <div key={index} className={`rounded-lg p-3 text-center ${isDark ? 'bg-gray-50/50' : 'bg-gray-50 border border-gray-200'}`}>
                    <div className={`font-semibold text-sm ${isDark ? 'text-gray-900' : 'text-gray-900'}`}>{tech.name}</div>
                    <div className={`text-xs ${isDark ? 'text-gray-500' : 'text-gray-600'}`}>{tech.desc}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Dataset Info */}
            <div className={`rounded-2xl p-6 border ${isDark ? 'bg-gradient-to-br from-purple-500/10 to-pink-500/10 border-purple-500/30' : 'bg-gradient-to-br from-purple-50 to-pink-50 border-purple-200'}`}>
              <h3 className={`text-xl font-semibold mb-3 ${isDark ? 'text-gray-900' : 'text-gray-900'}`}>Dataset</h3>
              <p className={`text-sm mb-3 ${isDark ? 'text-gray-600' : 'text-gray-700'}`}>
                Trained on Kaggle Credit Card Fraud Detection dataset with 284,807 transactions
              </p>
              <div className="grid grid-cols-2 gap-3 text-center">
                <div className={`rounded-lg p-3 ${isDark ? 'bg-gray-50/50' : 'bg-white border border-gray-200'}`}>
                  <div className={`text-2xl font-bold ${isDark ? 'text-gray-900' : 'text-gray-900'}`}>284K+</div>
                  <div className={`text-xs ${isDark ? 'text-gray-500' : 'text-gray-600'}`}>Transactions</div>
                </div>
                <div className={`rounded-lg p-3 ${isDark ? 'bg-gray-50/50' : 'bg-white border border-gray-200'}`}>
                  <div className={`text-2xl font-bold ${isDark ? 'text-gray-900' : 'text-gray-900'}`}>30</div>
                  <div className={`text-xs ${isDark ? 'text-gray-500' : 'text-gray-600'}`}>Features</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default About;
