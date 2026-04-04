import React, { useState, useEffect, useCallback } from 'react';
import { transactionsAPI, analyticsAPI } from '../services/api';

const Alerts = () => {
  const [filter, setFilter] = useState('All');
  const [sortBy, setSortBy] = useState('newest');
  const [selectedAlerts, setSelectedAlerts] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [alertsSummary, setAlertsSummary] = useState(null);

  // Fetch alerts from API
  const fetchAlerts = useCallback(async () => {
    try {
      setLoading(true);
      
      // Get fraud transactions (these are our alerts)
      const [fraudTxns, summary] = await Promise.all([
        transactionsAPI.getAll({ is_fraud: true, limit: 50 }),
        analyticsAPI.getAlertsSummary()
      ]);
      
      // Format transactions as alerts
      const formattedAlerts = fraudTxns.map(txn => {
        // Determine severity based on fraud probability
        let severity = 'Low';
        if (txn.fraud_probability > 0.8) {
          severity = 'High';
        } else if (txn.fraud_probability > 0.5) {
          severity = 'Medium';
        }
        
        // Determine alert type based on risk factors or amount
        let type = 'Suspicious Pattern';
        if (txn.amount > 100000) {
          type = 'Amount Anomaly';
        } else if (txn.amount < 100) {
          type = 'Micro Transaction';
        } else if (txn.fraud_probability > 0.9) {
          type = 'ML Model Alert';
        }
        
        // Determine status
        let status = 'Active';
        if (txn.fraud_probability > 0.9) {
          status = 'Blocked';
        } else if (txn.fraud_probability > 0.7) {
          status = 'Escalated';
        } else if (txn.fraud_probability > 0.5) {
          status = 'Under Review';
        }
        
        // Format time ago
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
          timeAgo = `${diffMins} minute${diffMins > 1 ? 's' : ''} ago`;
        } else if (diffHours < 24) {
          timeAgo = `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
        } else {
          timeAgo = `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
        }
        
        return {
          id: `ALT-${txn.id.substring(0, 8).toUpperCase()}`,
          transactionId: `#${txn.id.substring(0, 8).toUpperCase()}`,
          amount: `₹${txn.amount.toFixed(2)}`,
          amountValue: txn.amount,
          time: timeAgo,
          timestamp: new Date(txn.timestamp),
          severity,
          type,
          merchant: txn.merchant_name || 'Unknown Merchant',
          location: txn.location || 'Unknown Location',
          description: `Transaction flagged with ${Math.round(txn.fraud_probability * 100)}% fraud probability`,
          status,
          fraudProbability: txn.fraud_probability
        };
      });
      
      setAlerts(formattedAlerts);
      setAlertsSummary(summary);
    } catch (error) {
      console.error('Failed to fetch alerts:', error);
      setAlerts([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchAlerts();
  }, [fetchAlerts]);

  // Auto-refresh every 30 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      fetchAlerts();
    }, 30000);

    return () => clearInterval(interval);
  }, [fetchAlerts]);

  // Sample alerts data (fallback)
  const alertsData = alerts;

  // Filter alerts based on severity
  const filteredAlerts = alertsData.filter(alert => {
    if (filter === 'All') return true;
    return alert.severity === filter;
  });

  // Sort alerts
  const sortedAlerts = [...filteredAlerts].sort((a, b) => {
    if (sortBy === 'newest') {
      return b.timestamp - a.timestamp;
    } else if (sortBy === 'severity') {
      const severityOrder = { 'High': 3, 'Medium': 2, 'Low': 1 };
      return severityOrder[b.severity] - severityOrder[a.severity];
    } else if (sortBy === 'amount') {
      return (b.amountValue || 0) - (a.amountValue || 0);
    }
    return 0;
  });

  if (loading && alerts.length === 0) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <div className="text-gray-900 text-xl mb-2">Loading alerts...</div>
          <div className="text-gray-500 text-sm">Fetching real-time fraud alerts</div>
        </div>
      </div>
    );
  }

  const getSeverityConfig = (severity) => {
    switch (severity) {
      case 'High':
        return {
          bgColor: 'bg-red-500/10 border-red-500/30',
          textColor: 'text-red-400',
          badgeColor: 'bg-red-500/20 text-red-400',
          iconColor: 'text-red-400'
        };
      case 'Medium':
        return {
          bgColor: 'bg-yellow-500/10 border-yellow-500/30',
          textColor: 'text-yellow-400',
          badgeColor: 'bg-yellow-500/20 text-yellow-400',
          iconColor: 'text-yellow-400'
        };
      case 'Low':
        return {
          bgColor: 'bg-blue-500/10 border-blue-500/30',
          textColor: 'text-blue-400',
          badgeColor: 'bg-blue-500/20 text-blue-400',
          iconColor: 'text-blue-400'
        };
      default:
        return {
          bgColor: 'bg-gray-500/10 border-gray-500/30',
          textColor: 'text-gray-500',
          badgeColor: 'bg-gray-500/20 text-gray-500',
          iconColor: 'text-gray-500'
        };
    }
  };

  const getStatusConfig = (status) => {
    switch (status) {
      case 'Active':
        return 'bg-red-500/20 text-red-400';
      case 'Under Review':
        return 'bg-yellow-500/20 text-yellow-400';
      case 'Escalated':
        return 'bg-purple-500/20 text-purple-400';
      case 'Blocked':
        return 'bg-red-600/20 text-red-300';
      case 'Resolved':
      case 'Cleared':
        return 'bg-green-500/20 text-green-400';
      default:
        return 'bg-gray-500/20 text-gray-500';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Fraud Alerts</h1>
          <p className="text-gray-500">Real-time monitoring and threat detection alerts</p>
        </div>
        
        {/* Alert Summary */}
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2 px-4 py-2 bg-red-500/20 border border-red-500/30 rounded-full">
            <div className="w-2 h-2 bg-red-400 rounded-full animate-pulse"></div>
            <span className="text-red-400 text-sm font-medium">
              {alertsData.filter(a => a.severity === 'High').length} High Priority
            </span>
          </div>
          <div className="flex items-center space-x-2 px-4 py-2 bg-yellow-500/20 border border-yellow-500/30 rounded-full">
            <div className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse"></div>
            <span className="text-yellow-400 text-sm font-medium">
              {alertsData.filter(a => a.severity === 'Medium').length} Medium
            </span>
          </div>
        </div>
      </div>
      {/* Controls */}
      <div className="flex items-center justify-between bg-white/60 backdrop-blur-sm border border-gray-200 rounded-2xl p-6">
        <div className="flex items-center space-x-4">
          {/* Filter Buttons */}
          <div className="flex items-center space-x-2">
            <span className="text-gray-500 text-sm font-medium">Filter:</span>
            {['All', 'High', 'Medium', 'Low'].map((severity) => (
              <button
                key={severity}
                onClick={() => setFilter(severity)}
                className={`px-4 py-2 rounded-xl text-sm font-medium transition-all duration-200 ${
                  filter === severity
                    ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
                    : 'text-gray-500 hover:text-gray-900 hover:bg-gray-100'
                }`}
              >
                {severity}
              </button>
            ))}
          </div>
        </div>

        {/* Sort Options */}
        <div className="flex items-center space-x-4">
          <span className="text-gray-500 text-sm font-medium">Sort by:</span>
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            className="bg-gray-100 border border-gray-300 rounded-xl px-4 py-2 text-gray-600 focus:outline-none focus:border-blue-500/50 appearance-none"
          >
            <option value="newest">Newest First</option>
            <option value="severity">Severity</option>
            <option value="amount">Amount</option>
          </select>
        </div>
      </div>

      {/* Alerts List */}
      <div className="space-y-4">
        {sortedAlerts.map((alert, index) => {
          const severityConfig = getSeverityConfig(alert.severity);
          const statusConfig = getStatusConfig(alert.status);
          
          return (
            <div
              key={alert.id}
              className={`bg-white/60 backdrop-blur-sm border rounded-2xl p-6 hover:bg-white/80 transition-all duration-200 cursor-pointer ${severityConfig.bgColor}`}
            >
              <div className="flex items-start justify-between">
                {/* Left Section */}
                <div className="flex items-start space-x-4">
                  {/* Alert Icon */}
                  <div className={`p-3 rounded-xl ${severityConfig.badgeColor} flex-shrink-0`}>
                    <svg className={`w-6 h-6 ${severityConfig.iconColor}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                    </svg>
                  </div>

                  {/* Alert Details */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center space-x-3 mb-2">
                      <h3 className="text-gray-900 font-semibold text-lg">{alert.type}</h3>
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${severityConfig.badgeColor}`}>
                        {alert.severity}
                      </span>
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${statusConfig}`}>
                        {alert.status}
                      </span>
                    </div>
                    
                    <p className="text-gray-600 text-sm mb-3">{alert.description}</p>
                    
                    {/* Transaction Details */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                      <div>
                        <span className="text-gray-500">Transaction ID</span>
                        <p className="text-blue-400 font-mono font-medium">{alert.transactionId}</p>
                      </div>
                      <div>
                        <span className="text-gray-500">Amount</span>
                        <p className="text-cyan-400 font-bold">{alert.amount}</p>
                      </div>
                      <div>
                        <span className="text-gray-500">Merchant</span>
                        <p className="text-gray-900 font-medium">{alert.merchant}</p>
                      </div>
                      <div>
                        <span className="text-gray-500">Location</span>
                        <p className="text-gray-600">{alert.location}</p>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Right Section */}
                <div className="flex flex-col items-end space-y-2 flex-shrink-0">
                  {/* Time */}
                  <div className="flex items-center space-x-2 text-gray-500 text-sm">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <span>{alert.time}</span>
                  </div>
                  
                  {/* Alert ID */}
                  <div className="text-gray-500 text-xs font-mono">
                    {alert.id}
                  </div>
                  
                  {/* Action Buttons */}
                  <div className="flex items-center space-x-2 mt-4">
                    <button className="px-3 py-1 bg-blue-500/20 text-blue-400 rounded-lg text-xs font-medium hover:bg-blue-500/30 transition-colors duration-200">
                      Investigate
                    </button>
                    <button className="px-3 py-1 bg-gray-100 text-gray-600 rounded-lg text-xs font-medium hover:bg-gray-700/50 transition-colors duration-200">
                      Dismiss
                    </button>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Load More */}
      {sortedAlerts.length > 0 && (
        <div className="text-center">
          <button className="px-6 py-3 bg-gray-100 text-gray-600 rounded-2xl font-medium hover:bg-gray-700/50 transition-colors duration-200">
            Load More Alerts
          </button>
        </div>
      )}

      {/* Empty State */}
      {sortedAlerts.length === 0 && (
        <div className="text-center py-12">
          <svg className="w-16 h-16 text-gray-600 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <h3 className="text-gray-500 text-lg font-medium mb-2">No alerts found</h3>
          <p className="text-gray-500">No alerts match the current filter criteria.</p>
        </div>
      )}
    </div>
  );
};

export default Alerts;