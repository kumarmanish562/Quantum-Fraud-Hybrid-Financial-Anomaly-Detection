import React from 'react';
import { useNavigate } from 'react-router-dom';

const Hero = ({ isDark }) => {
  const navigate = useNavigate();

  return (
    <section id="home" className="pt-32 pb-20 px-6 min-h-screen flex items-center">
      <div className="max-w-7xl mx-auto w-full">
        <div className="text-center">
          <div className="inline-block mb-4 px-4 py-2 bg-blue-500/10 border border-blue-500/30 rounded-full">
            <span className="text-blue-400 text-sm font-medium">🚀 AI + Quantum Computing</span>
          </div>
          
          <h1 className={`text-5xl md:text-7xl font-bold mb-6 leading-tight ${isDark ? 'text-gray-900' : 'text-gray-900'}`}>
            Quantum-Enhanced
            <br />
            <span className="bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
              Financial Fraud Detection
            </span>
          </h1>
          
          <p className={`text-xl md:text-2xl mb-4 max-w-4xl mx-auto ${isDark ? 'text-gray-600' : 'text-gray-700'}`}>
            Detect fraud in real-time using Hybrid Quantum-Classical Neural Networks
          </p>
          
          <p className={`text-lg mb-10 max-w-3xl mx-auto ${isDark ? 'text-gray-500' : 'text-gray-600'}`}>
            Our system uses hybrid quantum-classical models (HQNN + XGBoost) to identify fraudulent 
            transactions with exceptional accuracy and speed. Protect your financial platform with 
            cutting-edge quantum computing technology.
          </p>
          
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-16">
            <button 
              onClick={() => navigate('/register')} 
              className="w-full sm:w-auto bg-gradient-to-r from-blue-600 to-cyan-500 text-white px-8 py-4 rounded-xl text-lg font-semibold hover:shadow-2xl hover:scale-105 transition-all"
            >
              🚀 Get Started Free
            </button>
            <button 
              onClick={() => navigate('/dashboard')} 
              className={`w-full sm:w-auto px-8 py-4 rounded-xl text-lg font-semibold transition ${isDark ? 'bg-white text-gray-900 hover:bg-gray-700 border border-gray-300' : 'bg-white text-gray-900 hover:bg-gray-50 border border-gray-300 shadow-md'}`}
            >
              🔍 View Dashboard
            </button>
          </div>
          
          {/* Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8 max-w-4xl mx-auto">
            <div className={`backdrop-blur-sm rounded-xl p-6 ${isDark ? 'bg-gray-100 border border-gray-300' : 'bg-white border border-gray-200 shadow-lg'}`}>
              <div className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent mb-2">99%+</div>
              <div className={`text-sm ${isDark ? 'text-gray-500' : 'text-gray-600'}`}>Detection Accuracy</div>
            </div>
            <div className={`backdrop-blur-sm rounded-xl p-6 ${isDark ? 'bg-gray-100 border border-gray-300' : 'bg-white border border-gray-200 shadow-lg'}`}>
              <div className="text-4xl font-bold bg-gradient-to-r from-green-400 to-emerald-400 bg-clip-text text-transparent mb-2">&lt;50ms</div>
              <div className={`text-sm ${isDark ? 'text-gray-500' : 'text-gray-600'}`}>Response Time</div>
            </div>
            <div className={`backdrop-blur-sm rounded-xl p-6 ${isDark ? 'bg-gray-100 border border-gray-300' : 'bg-white border border-gray-200 shadow-lg'}`}>
              <div className="text-4xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent mb-2">Real-Time</div>
              <div className={`text-sm ${isDark ? 'text-gray-500' : 'text-gray-600'}`}>Fraud Detection</div>
            </div>
            <div className={`backdrop-blur-sm rounded-xl p-6 ${isDark ? 'bg-gray-100 border border-gray-300' : 'bg-white border border-gray-200 shadow-lg'}`}>
              <div className="text-4xl font-bold bg-gradient-to-r from-orange-400 to-red-400 bg-clip-text text-transparent mb-2">24/7</div>
              <div className={`text-sm ${isDark ? 'text-gray-500' : 'text-gray-600'}`}>Monitoring</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Hero;
