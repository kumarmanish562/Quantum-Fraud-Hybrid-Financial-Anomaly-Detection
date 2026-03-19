import React, { useState, useEffect, useCallback } from 'react';
import { analyticsAPI, transactionsAPI } from '../services/api';

const Analytics = () => {
  const [dateRange, setDateRange] = useState('7D');
  const [animationComplete, setAnimationComplete] = useState(false);
  const [fraudTrends, setFraudTrends] = useState(null);
  const [riskDistribution, setRiskDistribution] = useState(null);
  const [transactionPatterns, setTransactionPatterns] = useState(null);
  const [recentHighRisk, setRecentHighRisk] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Trigger animations after component mounts
    const timer = setTimeout(() => setAnimationComplete(true), 100);
    return () => clearTimeout(timer);
  }, []);

  // Fetch analytics data
  const fetchAnalyticsData = useCallback(async () => {
    try {
      setLoading(true);
      
      // Map date range to API period
      const periodMap = {
        '7D': '7d',
        '30D': '30d',
        '1Y': '365d'
      };
      
      const [trends, risk, patterns, highRiskTxns] = await Promise.all([
        analyticsAPI.getFraudTrends(periodMap[dateRange]),
        analyticsAPI.getRiskDistribution(),
        analyticsAPI.getTransactionPatterns('amount', dateRange === '7D' ? 7 : dateRange === '30D' ? 30 : 365),
        transactionsAPI.getAll({ is_fraud: true, limit: 3 })
      ]);
      
      setFraudTrends(trends);
      setRiskDistribution(risk);
      setTransactionPatterns(patterns);
      setRecentHighRisk(highRiskTxns);
    } catch (error) {
      console.error('Failed to fetch analytics data:', error);
    } finally {
      setLoading(false);
    }
  }, [dateRange]);

  useEffect(() => {
    fetchAnalyticsData();
  }, [fetchAnalyticsData]);

  // Auto-refresh every 30 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      fetchAnalyticsData();
    }, 30000);

    return () => clearInterval(interval);
  }, [fetchAnalyticsData]);

  // Sample data for charts
  const fraudOverTimeData = fraudTrends?.data || [];

  const transactionDistribution = transactionPatterns?.buckets 
    ? Object.entries(transactionPatterns.buckets).map(([range, data]) => ({
        range: `₹${range}`,
        count: data.count,
        percentage: Math.round((data.count / Object.values(transactionPatterns.buckets).reduce((sum, b) => sum + b.count, 0)) * 100)
      }))
    : [];

  const fraudVsLegit = riskDistribution 
    ? {
        safe: Math.round((riskDistribution.distribution.very_low?.percentage || 0) + (riskDistribution.distribution.low?.percentage || 0)),
        fraud: Math.round((riskDistribution.distribution.high?.percentage || 0) + (riskDistribution.distribution.very_high?.percentage || 0)),
        total: riskDistribution.total_transactions || 0,
        fraudCount: (riskDistribution.distribution.high?.count || 0) + (riskDistribution.distribution.very_high?.count || 0),
        safeCount: (riskDistribution.distribution.very_low?.count || 0) + (riskDistribution.distribution.low?.count || 0)
      }
    : { safe: 0, fraud: 0, total: 0, fraudCount: 0, safeCount: 0 };

  // Format chart data
  const currentData = fraudOverTimeData.length > 0 
    ? fraudOverTimeData.slice(0, dateRange === '7D' ? 7 : dateRange === '30D' ? 4 : 4).map((point, index) => {
        const date = new Date(point.timestamp);
        let day;
        if (dateRange === '7D') {
          day = date.toLocaleDateString('en-US', { weekday: 'short' });
        } else if (dateRange === '30D') {
          day = `W${Math.floor(index / 7) + 1}`;
        } else {
          day = `Q${Math.floor(index / 3) + 1}`;
        }
        
        return {
          day,
          fraud: point.fraud_count || 0,
          baseline: Math.max((point.fraud_count || 0) - 5, 0)
        };
      })
    : [];

  if (loading && !fraudTrends) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <div className="text-white text-xl mb-2">Loading analytics...</div>
          <div className="text-gray-400 text-sm">Fetching real-time data</div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white mb-2">System Analytics</h1>
          <p className="text-gray-400">Deep behavioral analysis and threat intelligence metrics across all operational nodes.</p>
        </div>
        
        {/* Date Range Filters */}
        <div className="flex items-center space-x-2 bg-[#111827]/60 backdrop-blur-sm border border-gray-800/50 rounded-2xl p-2">
          {['7D', '30D', '1Y'].map((range) => (
            <button
              key={range}
              onClick={() => setDateRange(range)}
              className={`px-4 py-2 rounded-xl text-sm font-medium transition-all duration-200 ${
                dateRange === range
                  ? 'bg-blue-500 text-white shadow-lg'
                  : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
              }`}
            >
              {range}
            </button>
          ))}
          <button className="px-4 py-2 text-gray-400 hover:text-white rounded-xl text-sm font-medium border border-gray-700/50 hover:border-gray-600/50 transition-all duration-200">
            Custom Range
          </button>
        </div>
      </div>
      {/* Main Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Threat Vector Analysis - Line Chart */}
        <div className="bg-[#111827]/60 backdrop-blur-sm border border-gray-800/50 rounded-2xl p-6">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h3 className="text-xl font-semibold text-white">Threat Vector Analysis</h3>
              <p className="text-gray-400 text-sm">Fraud Over Time</p>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-blue-400 rounded-full"></div>
                <span className="text-gray-400 text-sm">Detected Incidents</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-gray-500 rounded-full"></div>
                <span className="text-gray-400 text-sm">Baseline</span>
              </div>
            </div>
          </div>
          
          {/* Line Chart */}
          <div className="relative h-64">
            {currentData.length > 0 ? (
              <svg className="w-full h-full" viewBox="0 0 400 200">
                {/* Grid Lines */}
                <defs>
                  <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
                    <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#374151" strokeWidth="0.5" opacity="0.3"/>
                  </pattern>
                  <linearGradient id="fraudGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" stopColor="#3B82F6" stopOpacity="0.3"/>
                    <stop offset="100%" stopColor="#3B82F6" stopOpacity="0"/>
                  </linearGradient>
                </defs>
                <rect width="100%" height="100%" fill="url(#grid)" />
                
                {/* Baseline Line */}
                <path
                  d={`M 50 ${180 - (currentData[0].baseline * 4)} ${currentData.map((point, index) => 
                    `L ${50 + (index * 50)} ${180 - (point.baseline * 4)}`
                  ).join(' ')}`}
                  stroke="#6B7280"
                  strokeWidth="2"
                  fill="none"
                  className={`transition-all duration-1000 ${animationComplete ? 'opacity-100' : 'opacity-0'}`}
                  strokeDasharray={animationComplete ? "0" : "1000"}
                  strokeDashoffset={animationComplete ? "0" : "1000"}
                />
                
                {/* Fraud Line with Area */}
                <path
                  d={`M 50 200 L 50 ${180 - (currentData[0].fraud * 4)} ${currentData.map((point, index) => 
                    `L ${50 + (index * 50)} ${180 - (point.fraud * 4)}`
                  ).join(' ')} L ${50 + ((currentData.length - 1) * 50)} 200 Z`}
                  fill="url(#fraudGradient)"
                  className={`transition-all duration-1000 ${animationComplete ? 'opacity-100' : 'opacity-0'}`}
                />
                
                <path
                  d={`M 50 ${180 - (currentData[0].fraud * 4)} ${currentData.map((point, index) => 
                    `L ${50 + (index * 50)} ${180 - (point.fraud * 4)}`
                  ).join(' ')}`}
                  stroke="#3B82F6"
                  strokeWidth="3"
                  fill="none"
                  className={`transition-all duration-1000 ${animationComplete ? 'opacity-100' : 'opacity-0'}`}
                  strokeDasharray={animationComplete ? "0" : "1000"}
                  strokeDashoffset={animationComplete ? "0" : "1000"}
                />
                
                {/* Data Points */}
                {currentData.map((point, index) => (
                  <circle
                    key={index}
                    cx={50 + (index * 50)}
                    cy={180 - (point.fraud * 4)}
                    r="4"
                    fill="#3B82F6"
                    className={`transition-all duration-1000 delay-${index * 100} ${animationComplete ? 'opacity-100' : 'opacity-0'}`}
                  />
                ))}
                
                {/* X-axis Labels */}
                {currentData.map((point, index) => (
                  <text
                    key={index}
                    x={50 + (index * 50)}
                    y="195"
                    textAnchor="middle"
                    className="fill-gray-400 text-xs"
                  >
                    {point.day}
                  </text>
                ))}
              </svg>
            ) : (
              <div className="flex flex-col items-center justify-center h-full text-gray-400">
                <svg className="w-16 h-16 mb-4 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                <div className="text-lg mb-1">No fraud trend data</div>
                <div className="text-sm text-gray-500">Process transactions to see trends</div>
              </div>
            )}
          </div>
        </div>

        {/* Traffic Integrity - Pie Chart */}
        <div className="bg-[#111827]/60 backdrop-blur-sm border border-gray-800/50 rounded-2xl p-6">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h3 className="text-xl font-semibold text-white">Traffic Integrity</h3>
              <p className="text-gray-400 text-sm">Fraud Rate</p>
            </div>
          </div>
          
          {/* Pie Chart */}
          <div className="flex items-center justify-center mb-6">
            <div className="relative w-40 h-40">
              <svg className="w-40 h-40 transform -rotate-90" viewBox="0 0 100 100">
                {/* Background circle */}
                <circle
                  cx="50"
                  cy="50"
                  r="35"
                  stroke="#374151"
                  strokeWidth="8"
                  fill="none"
                />
                {/* Safe transactions arc */}
                <circle
                  cx="50"
                  cy="50"
                  r="35"
                  stroke="#10B981"
                  strokeWidth="8"
                  fill="none"
                  strokeDasharray={`${fraudVsLegit.safe * 2.2} ${(100 - fraudVsLegit.safe) * 2.2}`}
                  strokeLinecap="round"
                  className={`transition-all duration-1500 ${animationComplete ? 'opacity-100' : 'opacity-0'}`}
                  style={{
                    strokeDashoffset: animationComplete ? 0 : 220
                  }}
                />
                {/* Fraud transactions arc */}
                <circle
                  cx="50"
                  cy="50"
                  r="35"
                  stroke="#EF4444"
                  strokeWidth="8"
                  fill="none"
                  strokeDasharray={`${fraudVsLegit.fraud * 2.2} ${(100 - fraudVsLegit.fraud) * 2.2}`}
                  strokeDashoffset={`${-fraudVsLegit.safe * 2.2}`}
                  strokeLinecap="round"
                  className={`transition-all duration-1500 delay-500 ${animationComplete ? 'opacity-100' : 'opacity-0'}`}
                />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center">
                  <div className="text-2xl font-bold text-white">{fraudVsLegit.safe}%</div>
                  <div className="text-xs text-gray-400 uppercase tracking-wider">Safe Traffic</div>
                </div>
              </div>
            </div>
          </div>

          {/* Legend */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-green-400 rounded-full"></div>
                <span className="text-gray-300 text-sm">Validated Transactions</span>
              </div>
              <span className="text-white font-medium">{fraudVsLegit.safeCount.toLocaleString()}</span>
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-red-400 rounded-full"></div>
                <span className="text-gray-300 text-sm">Flagged Anomalies</span>
              </div>
              <span className="text-white font-medium">{fraudVsLegit.fraudCount}</span>
            </div>
          </div>
        </div>
      </div>
      {/* Bottom Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Volume Segmentation - Bar Chart */}
        <div className="bg-[#111827]/60 backdrop-blur-sm border border-gray-800/50 rounded-2xl p-6">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h3 className="text-xl font-semibold text-white">Volume Segmentation</h3>
              <p className="text-gray-400 text-sm">Amount Distribution</p>
            </div>
          </div>
          
          {/* Bar Chart */}
          <div className="space-y-4">
            {transactionDistribution.map((item, index) => (
              <div key={index} className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-gray-300 text-sm font-medium">{item.range}</span>
                  <div className="flex items-center space-x-2">
                    <span className="text-white font-bold">{item.count}</span>
                    <span className="text-gray-400 text-sm">{item.percentage}%</span>
                  </div>
                </div>
                <div className="w-full bg-gray-800 rounded-full h-3">
                  <div
                    className={`h-3 rounded-full bg-gradient-to-r transition-all duration-1000 delay-${index * 200} ${
                      index === 0 ? 'from-blue-500 to-blue-400' :
                      index === 1 ? 'from-cyan-500 to-cyan-400' :
                      index === 2 ? 'from-purple-500 to-purple-400' :
                      'from-pink-500 to-pink-400'
                    }`}
                    style={{ 
                      width: animationComplete ? `${item.percentage}%` : '0%'
                    }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Threat Categorization */}
        <div className="bg-[#111827]/60 backdrop-blur-sm border border-gray-800/50 rounded-2xl p-6">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h3 className="text-xl font-semibold text-white">Threat Categorization</h3>
              <p className="text-gray-400 text-sm">Risk Level Exposure</p>
            </div>
            <span className="px-3 py-1 bg-red-500/20 text-red-400 rounded-full text-xs font-medium uppercase tracking-wider">
              Live Monitoring
            </span>
          </div>
          
          {/* Risk Level Cards */}
          <div className="grid grid-cols-3 gap-4">
            <div className="bg-gray-800/30 rounded-xl p-4 text-center">
              <div className="text-2xl font-bold text-green-400 mb-1">842</div>
              <div className="text-gray-400 text-xs uppercase tracking-wider">Low Risk</div>
              <div className="text-green-400 text-xs">-2.4%</div>
            </div>
            <div className="bg-gray-800/30 rounded-xl p-4 text-center">
              <div className="text-2xl font-bold text-yellow-400 mb-1">156</div>
              <div className="text-gray-400 text-xs uppercase tracking-wider">Medium</div>
              <div className="text-yellow-400 text-xs">+1.2%</div>
            </div>
            <div className="bg-gray-800/30 rounded-xl p-4 text-center">
              <div className="text-2xl font-bold text-red-400 mb-1">24</div>
              <div className="text-gray-400 text-xs uppercase tracking-wider">Critical</div>
              <div className="text-red-400 text-xs">+0.8%</div>
            </div>
          </div>
        </div>
      </div>

      {/* Recent High-Risk Hits Table */}
      <div className="bg-[#111827]/60 backdrop-blur-sm border border-gray-800/50 rounded-2xl p-6">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-xl font-semibold text-white">Recent High-Risk Hits</h3>
          <button className="text-blue-400 text-sm font-medium hover:text-blue-300 transition-colors">
            View All Intelligence
          </button>
        </div>

        {/* Table */}
        <div className="overflow-hidden">
          {/* Header */}
          <div className="grid grid-cols-5 gap-4 mb-4 pb-3 border-b border-gray-800/50">
            <div className="text-gray-400 text-xs font-medium uppercase tracking-wider">Incident ID</div>
            <div className="text-gray-400 text-xs font-medium uppercase tracking-wider">Time</div>
            <div className="text-gray-400 text-xs font-medium uppercase tracking-wider">Merchant</div>
            <div className="text-gray-400 text-xs font-medium uppercase tracking-wider">Amount</div>
            <div className="text-gray-400 text-xs font-medium uppercase tracking-wider">Risk Score</div>
          </div>

          {/* Rows */}
          <div className="space-y-3">
            {recentHighRisk.length > 0 ? (
              recentHighRisk.map((txn, index) => {
                const timeStr = new Date(txn.timestamp).toLocaleTimeString('en-US', { hour12: false });
                const risk = Math.round((txn.fraud_probability || 0) * 100);
                
                return (
                  <div 
                    key={txn.id} 
                    className={`grid grid-cols-5 gap-4 items-center p-3 bg-gray-800/20 rounded-xl hover:bg-gray-800/40 transition-all duration-300 delay-${index * 100} ${
                      animationComplete ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
                    }`}
                  >
                    <div className="text-blue-400 font-mono text-sm">#{txn.id.substring(0, 8).toUpperCase()}</div>
                    <div className="text-gray-300 text-sm">{timeStr}</div>
                    <div className="text-white text-sm font-medium">{txn.merchant_name || 'Unknown'}</div>
                    <div className="text-cyan-400 font-bold text-sm">₹{txn.amount.toFixed(2)}</div>
                    <div className="flex items-center space-x-2">
                      <div className="w-12 bg-gray-800 rounded-full h-2">
                        <div
                          className="h-2 rounded-full bg-gradient-to-r from-red-500 to-red-400"
                          style={{ width: `${risk}%` }}
                        ></div>
                      </div>
                      <span className="text-red-400 text-sm font-medium">{risk}%</span>
                    </div>
                  </div>
                );
              })
            ) : (
              <div className="text-center py-8 text-gray-400">
                No high-risk transactions detected
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Analytics;