import React, { useState, useEffect, useCallback } from 'react';
import { Card, StatCard, ModelBadge, AIStatusBadge, NeuralActivityBadge } from './ui';
import { analyticsAPI, transactionsAPI, WebSocketService } from '../services/api';

const Dashboard = () => {
  const [chartPeriod, setChartPeriod] = useState('daily');
  const [dashboardMetrics, setDashboardMetrics] = useState(null);
  const [fraudTrends, setFraudTrends] = useState(null);
  const [recentFraudTransactions, setRecentFraudTransactions] = useState([]);
  const [modelPerformance, setModelPerformance] = useState(null);
  const [realtimeMetrics, setRealtimeMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [wsConnected, setWsConnected] = useState(false);

  // Fetch dashboard metrics
  const fetchDashboardData = useCallback(async () => {
    try {
      setLoading(true);
      const [metrics, trends, fraudTxns, performance, realtime] = await Promise.all([
        analyticsAPI.getDashboard(),
        analyticsAPI.getFraudTrends(chartPeriod === 'daily' ? '7d' : '30d'),
        transactionsAPI.getAll({ is_fraud: true, limit: 10 }),
        analyticsAPI.getModelPerformance(),
        analyticsAPI.getRealTimeMetrics()
      ]);

      setDashboardMetrics(metrics);
      setFraudTrends(trends);
      setRecentFraudTransactions(fraudTxns);
      setModelPerformance(performance);
      setRealtimeMetrics(realtime);
    } catch (error) {
      console.error('Failed to fetch dashboard data:', error);
    } finally {
      setLoading(false);
    }
  }, [chartPeriod]);

  // Initial data fetch
  useEffect(() => {
    fetchDashboardData();
  }, [fetchDashboardData]);

  // Auto-refresh every 30 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      fetchDashboardData();
    }, 30000);

    return () => clearInterval(interval);
  }, [fetchDashboardData]);

  // WebSocket for real-time updates
  useEffect(() => {
    const ws = new WebSocketService(`dashboard_${Date.now()}`);
    
    ws.on('connected', () => {
      setWsConnected(true);
    });

    ws.on('disconnected', () => {
      setWsConnected(false);
    });

    ws.on('fraud_alert', (data) => {
      // Add new fraud alert to the list
      setRecentFraudTransactions(prev => [data, ...prev.slice(0, 9)]);
      // Refresh metrics
      fetchDashboardData();
    });

    ws.on('metrics_update', (data) => {
      setRealtimeMetrics(data);
    });

    ws.connect();

    return () => {
      ws.disconnect();
    };
  }, [fetchDashboardData]);

  // Handle chart period change
  const handlePeriodChange = (period) => {
    setChartPeriod(period);
  };

  if (loading && !dashboardMetrics) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <div className="text-white text-xl mb-2">Loading dashboard...</div>
          <div className="text-gray-400 text-sm">Fetching real-time data</div>
        </div>
      </div>
    );
  }

  // Handle empty state
  const hasData = dashboardMetrics && dashboardMetrics.total_transactions_today > 0;

  // Calculate percentage changes from API data
  const calculateChange = (current, previous) => {
    if (!previous || previous === 0) return 0;
    return (((current - previous) / previous) * 100).toFixed(1);
  };

  const stats = dashboardMetrics ? [
    {
      title: 'Total Transactions',
      value: dashboardMetrics.total_transactions_today?.toLocaleString() || '0',
      change: dashboardMetrics.transaction_change_percentage 
        ? `${dashboardMetrics.transaction_change_percentage > 0 ? '+' : ''}${dashboardMetrics.transaction_change_percentage.toFixed(1)}%`
        : '+0.0%',
      changeType: (dashboardMetrics.transaction_change_percentage || 0) >= 0 ? 'positive' : 'negative',
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
      ),
      bgColor: 'bg-blue-500/20',
      iconColor: 'text-blue-400'
    },
    {
      title: 'Fraud Detected',
      value: dashboardMetrics.fraud_detected_today?.toString() || '0',
      change: dashboardMetrics.fraud_change_percentage 
        ? `${dashboardMetrics.fraud_change_percentage > 0 ? '+' : ''}${dashboardMetrics.fraud_change_percentage.toFixed(1)}%`
        : '0.0%',
      changeType: (dashboardMetrics.fraud_change_percentage || 0) <= 0 ? 'positive' : 'negative',
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
        </svg>
      ),
      bgColor: 'bg-red-500/20',
      iconColor: 'text-red-400',
      priority: 'High Priority'
    },
    {
      title: 'Fraud Rate',
      value: `${dashboardMetrics.fraud_rate_today || 0}%`,
      change: dashboardMetrics.fraud_rate_change_percentage 
        ? `${dashboardMetrics.fraud_rate_change_percentage > 0 ? '+' : ''}${dashboardMetrics.fraud_rate_change_percentage.toFixed(1)}%`
        : '0.0%',
      changeType: (dashboardMetrics.fraud_rate_change_percentage || 0) <= 0 ? 'positive' : 'negative',
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
      ),
      bgColor: 'bg-orange-500/20',
      iconColor: 'text-orange-400'
    },
    {
      title: 'Safe Transactions',
      value: ((dashboardMetrics.total_transactions_today || 0) - (dashboardMetrics.fraud_detected_today || 0)).toLocaleString(),
      change: dashboardMetrics.safe_transaction_change_percentage 
        ? `${dashboardMetrics.safe_transaction_change_percentage > 0 ? '+' : ''}${dashboardMetrics.safe_transaction_change_percentage.toFixed(1)}%`
        : '+0.0%',
      changeType: (dashboardMetrics.safe_transaction_change_percentage || 0) >= 0 ? 'positive' : 'negative',
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
        </svg>
      ),
      bgColor: 'bg-green-500/20',
      iconColor: 'text-green-400'
    }
  ] : [
    // Fallback stats when no data
    {
      title: 'Total Transactions',
      value: '0',
      change: '+0.0%',
      changeType: 'positive',
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
      ),
      bgColor: 'bg-blue-500/20',
      iconColor: 'text-blue-400'
    },
    {
      title: 'Fraud Detected',
      value: '0',
      change: '0.0%',
      changeType: 'positive',
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
        </svg>
      ),
      bgColor: 'bg-red-500/20',
      iconColor: 'text-red-400'
    },
    {
      title: 'Fraud Rate',
      value: '0%',
      change: '0.0%',
      changeType: 'positive',
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
      ),
      bgColor: 'bg-orange-500/20',
      iconColor: 'text-orange-400'
    },
    {
      title: 'Safe Transactions',
      value: '0',
      change: '+0.0%',
      changeType: 'positive',
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
        </svg>
      ),
      bgColor: 'bg-green-500/20',
      iconColor: 'text-green-400'
    }
  ];

  const dailyData = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN'];
  const weeklyData = ['Week 1', 'Week 2', 'Week 3', 'Week 4'];
  
  const currentData = chartPeriod === 'daily' ? dailyData : weeklyData;
  const chartData = fraudTrends?.data || [];

  // Format recent fraud transactions for display
  const formattedAlerts = recentFraudTransactions.slice(0, 4).map((txn) => {
    const riskLevel = txn.fraud_probability > 0.8 ? 'CRITICAL' : 
                      txn.fraud_probability > 0.5 ? 'MEDIUM' : 'LOW';
    const icon = riskLevel === 'CRITICAL' ? '⚠️' : 
                 riskLevel === 'MEDIUM' ? '🔍' : '🛡️';
    
    // Format timestamp
    const txnDate = new Date(txn.timestamp);
    const now = new Date();
    const diffMs = now - txnDate;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);
    
    let timeAgo;
    if (diffMins < 1) {
      timeAgo = 'Just now';
    } else if (diffMins < 60) {
      timeAgo = `${diffMins} min${diffMins > 1 ? 's' : ''} ago`;
    } else if (diffHours < 24) {
      timeAgo = `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
    } else {
      timeAgo = `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
    }
    
    return {
      id: txn.id ? txn.id.substring(0, 8).toUpperCase() : 'UNKNOWN',
      entity: txn.merchant_name || 'Unknown Merchant',
      amount: `₹${txn.amount?.toFixed(2) || '0.00'}`,
      riskLevel,
      timestamp: timeAgo,
      icon
    };
  });

  return (
    <div className="space-y-6">
      {/* AI Model Badges */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <ModelBadge modelType="Quantum" size="md" animated={true} />
          <AIStatusBadge 
            status={wsConnected ? "online" : "offline"} 
            processingTime={realtimeMetrics?.avg_processing_time || 0} 
          />
          <NeuralActivityBadge 
            activity={realtimeMetrics?.transactions_per_second > 0 ? "active" : "idle"} 
            nodeCount={2048} 
          />
        </div>
        <div className="text-sm text-gray-400">
          Last updated: {dashboardMetrics?.last_updated ? new Date(dashboardMetrics.last_updated).toLocaleTimeString() : 'N/A'}
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat, index) => (
          <StatCard
            key={index}
            title={stat.title}
            value={stat.value}
            icon={stat.icon}
            change={stat.change}
            changeType={stat.changeType}
            className={stat.title === 'Fraud Detected' ? 'border-red-500/20' : ''}
          />
        ))}
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Fraud Trends Chart */}
        <div className="lg:col-span-2 bg-[#111827]/60 backdrop-blur-sm border border-gray-800/50 rounded-2xl p-6">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h3 className="text-white text-lg font-semibold">Fraud Trends</h3>
              <p className="text-gray-400 text-sm">Real-time surveillance overview</p>
            </div>
            <div className="flex items-center space-x-4">
              <button 
                onClick={() => handlePeriodChange('daily')}
                className={`px-3 py-1 rounded-lg text-sm font-medium transition-all duration-200 ${
                  chartPeriod === 'daily' 
                    ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30' 
                    : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
                }`}
              >
                Daily
              </button>
              <button 
                onClick={() => handlePeriodChange('weekly')}
                className={`px-3 py-1 rounded-lg text-sm font-medium transition-all duration-200 ${
                  chartPeriod === 'weekly' 
                    ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30' 
                    : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
                }`}
              >
                Weekly
              </button>
            </div>
          </div>
          
          {/* Simulated Chart */}
          <div className="relative h-64 bg-gray-900/30 rounded-xl p-4">
            {chartData.length > 0 && chartData.some(d => d.total_transactions > 0) ? (
              <>
                <div className="absolute inset-0 flex items-end justify-between px-4 pb-4">
                  {chartData.slice(0, chartPeriod === 'daily' ? 7 : 4).map((dataPoint, index) => {
                    const maxHeight = 180;
                    const maxFraudCount = Math.max(...chartData.map(d => d.fraud_count), 1);
                    const height = Math.max((dataPoint.fraud_count / maxFraudCount) * maxHeight, 5);
                    const date = new Date(dataPoint.timestamp);
                    const label = chartPeriod === 'daily' 
                      ? date.toLocaleDateString('en-US', { weekday: 'short' }).toUpperCase()
                      : `W${Math.floor(index / 7) + 1}`;
                    
                    return (
                      <div key={index} className="flex flex-col items-center space-y-2">
                        <div 
                          className="w-8 bg-gradient-to-t from-blue-500 to-cyan-400 rounded-t-lg transition-all duration-500"
                          style={{ height: `${height}px` }}
                          title={`${dataPoint.fraud_count} frauds, ${dataPoint.fraud_rate}% rate`}
                        ></div>
                        <span className="text-gray-400 text-xs">{label}</span>
                      </div>
                    );
                  })}
                </div>
                
                {/* Trend Line Overlay */}
                <svg className="absolute inset-0 w-full h-full" viewBox="0 0 400 200">
                  <path
                    d={chartPeriod === 'daily' 
                      ? "M 50 150 Q 100 120 150 130 T 250 110 T 350 100"
                      : "M 80 160 Q 150 100 220 120 T 320 90"
                    }
                    stroke="#06B6D4"
                    strokeWidth="3"
                    fill="none"
                    className="drop-shadow-lg transition-all duration-500"
                  />
                  <defs>
                    <linearGradient id="lineGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                      <stop offset="0%" stopColor="#2563EB" />
                      <stop offset="100%" stopColor="#06B6D4" />
                    </linearGradient>
                  </defs>
                </svg>
              </>
            ) : (
              <div className="flex flex-col items-center justify-center h-full text-gray-400">
                <svg className="w-16 h-16 mb-4 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                <div className="text-lg mb-1">No transaction data yet</div>
                <div className="text-sm text-gray-500">Process transactions to see fraud trends</div>
              </div>
            )}
          </div>
        </div>

        {/* System Health Pie Chart */}
        <div className="bg-[#111827]/60 backdrop-blur-sm border border-gray-800/50 rounded-2xl p-6">
          <h3 className="text-white text-lg font-semibold mb-6 uppercase tracking-wider">System Health</h3>
          
          {/* Simulated Pie Chart */}
          <div className="flex items-center justify-center mb-6">
            <div className="relative w-32 h-32">
              <svg className="w-32 h-32 transform -rotate-90" viewBox="0 0 100 100">
                {/* Background circle */}
                <circle
                  cx="50"
                  cy="50"
                  r="40"
                  stroke="#374151"
                  strokeWidth="8"
                  fill="none"
                />
                {/* Progress circle */}
                <circle
                  cx="50"
                  cy="50"
                  r="40"
                  stroke="url(#pieGradient)"
                  strokeWidth="8"
                  fill="none"
                  strokeDasharray={`${97 * 2.51} ${(100 - 97) * 2.51}`}
                  strokeLinecap="round"
                />
                <defs>
                  <linearGradient id="pieGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#06B6D4" />
                    <stop offset="100%" stopColor="#8B5CF6" />
                  </linearGradient>
                </defs>
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center">
                  <div className="text-2xl font-bold text-white">
                    {dashboardMetrics ? Math.round((dashboardMetrics.total_transactions_today - dashboardMetrics.fraud_detected_today) / dashboardMetrics.total_transactions_today * 100) : 97}%
                  </div>
                  <div className="text-xs text-gray-400 uppercase tracking-wider">Trust Score</div>
                </div>
              </div>
            </div>
          </div>

          {/* Legend */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-green-400 rounded-full"></div>
                <span className="text-gray-300 text-sm">Legit</span>
              </div>
              <span className="text-white font-medium">
                {dashboardMetrics ? (dashboardMetrics.total_transactions_today - dashboardMetrics.fraud_detected_today).toLocaleString() : '0'}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-red-400 rounded-full"></div>
                <span className="text-gray-300 text-sm">Fraud</span>
              </div>
              <span className="text-white font-medium">
                {dashboardMetrics?.fraud_detected_today?.toLocaleString() || '0'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Quantum Threat Intelligence */}
      <div className="bg-[#111827]/60 backdrop-blur-sm border border-gray-800/50 rounded-2xl p-6">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-white text-lg font-semibold">Quantum Threat Intelligence</h3>
          <button className="text-blue-400 text-sm font-medium hover:text-blue-300 transition-colors">
            View All Logs
          </button>
        </div>

        {/* Table Header */}
        <div className="grid grid-cols-6 gap-4 mb-4 pb-3 border-b border-gray-800/50">
          <div className="text-gray-400 text-xs font-medium uppercase tracking-wider">Transaction ID</div>
          <div className="text-gray-400 text-xs font-medium uppercase tracking-wider">Entity Name</div>
          <div className="text-gray-400 text-xs font-medium uppercase tracking-wider">Amount</div>
          <div className="text-gray-400 text-xs font-medium uppercase tracking-wider">Risk Level</div>
          <div className="text-gray-400 text-xs font-medium uppercase tracking-wider">Timestamp</div>
          <div className="text-gray-400 text-xs font-medium uppercase tracking-wider">Actions</div>
        </div>

        {/* Alert Rows */}
        <div className="space-y-3">
          {formattedAlerts.length > 0 ? (
            formattedAlerts.map((alert, index) => (
              <div key={alert.id || index} className="grid grid-cols-6 gap-4 items-center p-3 bg-gray-800/20 rounded-xl hover:bg-gray-800/40 transition-all duration-200">
                <div className="flex items-center space-x-2">
                  <span className="text-lg">{alert.icon}</span>
                  <span className="text-gray-300 text-sm font-mono">{alert.id}</span>
                </div>
                <div className="text-white text-sm font-medium truncate">{alert.entity}</div>
                <div className="text-cyan-400 text-sm font-bold">{alert.amount}</div>
                <div>
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                    alert.riskLevel === 'CRITICAL' ? 'bg-red-500/20 text-red-400' :
                    alert.riskLevel === 'MEDIUM' ? 'bg-yellow-500/20 text-yellow-400' :
                    'bg-green-500/20 text-green-400'
                  }`}>
                    {alert.riskLevel}
                  </span>
                </div>
                <div className="text-gray-400 text-sm">{alert.timestamp}</div>
                <div className="flex items-center space-x-2">
                  <button 
                    className="text-blue-400 hover:text-blue-300 text-xs"
                    title="View Details"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                    </svg>
                  </button>
                  <button 
                    className="text-red-400 hover:text-red-300 text-xs"
                    title="Block Transaction"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636" />
                    </svg>
                  </button>
                </div>
              </div>
            ))
          ) : (
            <div className="text-center py-12">
              <div className="text-gray-400 text-lg mb-2">No fraud alerts detected</div>
              <div className="text-gray-500 text-sm">
                {hasData ? 'All transactions are secure' : 'Start processing transactions to see fraud detection in action'}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;