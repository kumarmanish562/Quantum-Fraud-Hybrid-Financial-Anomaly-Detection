import React from 'react';
import { useNavigate } from 'react-router-dom';

const DashboardPreview = ({ isDark }) => {
  const navigate = useNavigate();

  return (
    <section id="dashboard-preview" className={`py-20 px-6 ${isDark ? '' : 'bg-gray-50'}`}>
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-16">
          <h2 className={`text-4xl md:text-5xl font-bold mb-4 ${isDark ? 'text-white' : 'text-gray-900'}`}>Powerful Analytics Dashboard</h2>
          <p className={`text-xl max-w-3xl mx-auto ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
            Monitor transactions, fraud alerts, and analytics in real-time with our intuitive dashboard
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8 items-center">
          {/* Dashboard Screenshot Placeholder */}
          <div className={`rounded-2xl p-8 shadow-2xl ${isDark ? 'bg-gradient-to-br from-gray-800 to-gray-900 border border-gray-700' : 'bg-white border border-gray-200'}`}>
            <div className={`rounded-xl overflow-hidden ${isDark ? 'bg-gray-900 border border-gray-700' : 'bg-gray-50 border border-gray-200'}`}>
              {/* Mock Dashboard UI */}
              <div className={`p-4 flex items-center justify-between ${isDark ? 'bg-gray-800 border-b border-gray-700' : 'bg-gray-100 border-b border-gray-200'}`}>
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                  <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                  <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                </div>
                <span className={`text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>Dashboard</span>
              </div>
              
              <div className="p-6 space-y-4">
                {/* Stats Cards */}
                <div className="grid grid-cols-3 gap-4">
                  <div className={`rounded-lg p-3 ${isDark ? 'bg-blue-500/10 border border-blue-500/30' : 'bg-blue-50 border border-blue-200'}`}>
                    <div className={`text-xs mb-1 ${isDark ? 'text-blue-400' : 'text-blue-600'}`}>Total Transactions</div>
                    <div className={`text-xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>1,234</div>
                  </div>
                  <div className={`rounded-lg p-3 ${isDark ? 'bg-red-500/10 border border-red-500/30' : 'bg-red-50 border border-red-200'}`}>
                    <div className={`text-xs mb-1 ${isDark ? 'text-red-400' : 'text-red-600'}`}>Fraud Detected</div>
                    <div className={`text-xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>23</div>
                  </div>
                  <div className={`rounded-lg p-3 ${isDark ? 'bg-green-500/10 border border-green-500/30' : 'bg-green-50 border border-green-200'}`}>
                    <div className={`text-xs mb-1 ${isDark ? 'text-green-400' : 'text-green-600'}`}>Safe</div>
                    <div className={`text-xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>1,211</div>
                  </div>
                </div>

                {/* Chart Placeholder */}
                <div className={`rounded-lg p-4 ${isDark ? 'bg-gray-800 border border-gray-700' : 'bg-white border border-gray-200'}`}>
                  <div className={`text-sm mb-3 ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>Fraud Detection Trends</div>
                  <div className="flex items-end space-x-2 h-32">
                    {[40, 65, 45, 80, 55, 90, 70, 85].map((height, i) => (
                      <div key={i} className="flex-1 bg-gradient-to-t from-blue-500 to-cyan-400 rounded-t" style={{height: `${height}%`}}></div>
                    ))}
                  </div>
                </div>

                {/* Recent Transactions */}
                <div className={`rounded-lg p-4 ${isDark ? 'bg-gray-800 border border-gray-700' : 'bg-white border border-gray-200'}`}>
                  <div className={`text-sm mb-3 ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>Recent Transactions</div>
                  <div className="space-y-2">
                    {[
                      { amount: '₹45,000', status: 'fraud', prob: '87%' },
                      { amount: '₹1,200', status: 'safe', prob: '12%' },
                      { amount: '₹78,500', status: 'fraud', prob: '92%' }
                    ].map((tx, i) => (
                      <div key={i} className="flex items-center justify-between text-xs">
                        <span className={isDark ? 'text-white' : 'text-gray-900'}>{tx.amount}</span>
                        <span className={`px-2 py-1 rounded ${tx.status === 'fraud' ? 'bg-red-500/20 text-red-400' : 'bg-green-500/20 text-green-400'}`}>
                          {tx.status}
                        </span>
                        <span className={isDark ? 'text-gray-400' : 'text-gray-600'}>{tx.prob}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Features List */}
          <div className="space-y-6">
            <h3 className={`text-3xl font-bold mb-6 ${isDark ? 'text-white' : 'text-gray-900'}`}>Dashboard Features</h3>
            
            {[
              {
                icon: "📊",
                title: "Real-Time Analytics",
                description: "Live transaction monitoring with instant fraud detection and risk scoring"
              },
              {
                icon: "📈",
                title: "Interactive Charts",
                description: "Visualize fraud trends, patterns, and risk distribution with dynamic charts"
              },
              {
                icon: "🔔",
                title: "Instant Alerts",
                description: "Get notified immediately when suspicious transactions are detected"
              },
              {
                icon: "📋",
                title: "Transaction History",
                description: "Complete audit trail with detailed fraud analysis for each transaction"
              },
              {
                icon: "⚙️",
                title: "Customizable Settings",
                description: "Configure detection thresholds, quantum model settings, and alert preferences"
              },
              {
                icon: "🔐",
                title: "API Key Management",
                description: "Secure API keys with usage tracking and regeneration capabilities"
              }
            ].map((feature, index) => (
              <div key={index} className={`flex items-start space-x-4 rounded-xl p-4 transition ${isDark ? 'bg-gray-800/50 border border-gray-700 hover:border-blue-500/50' : 'bg-white border border-gray-200 hover:border-blue-400 shadow-sm'}`}>
                <div className="text-4xl">{feature.icon}</div>
                <div>
                  <h4 className={`font-semibold mb-1 ${isDark ? 'text-white' : 'text-gray-900'}`}>{feature.title}</h4>
                  <p className={`text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>{feature.description}</p>
                </div>
              </div>
            ))}

            <button 
              onClick={() => navigate('/dashboard')}
              className="w-full bg-gradient-to-r from-blue-600 to-cyan-500 text-white px-8 py-4 rounded-xl text-lg font-semibold hover:shadow-2xl hover:scale-105 transition-all"
            >
              Explore Dashboard →
            </button>
          </div>
        </div>
      </div>
    </section>
  );
};

export default DashboardPreview;
