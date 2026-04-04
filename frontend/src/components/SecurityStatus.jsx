import React, { useState, useEffect, useCallback } from 'react';
import { Card } from './ui';
import { analyticsAPI, transactionsAPI } from '../services/api';

const SecurityStatus = () => {
  const [stabilityData, setStabilityData] = useState([
    { value: 85, active: false },
    { value: 92, active: false },
    { value: 88, active: false },
    { value: 95, active: false },
    { value: 99, active: true },
    { value: 87, active: false },
    { value: 91, active: false }
  ]);

  const [realTimeFeed, setRealTimeFeed] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [animationComplete, setAnimationComplete] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setAnimationComplete(true), 500);
    return () => clearTimeout(timer);
  }, []);

  // Fetch security metrics
  const fetchSecurityData = useCallback(async () => {
    try {
      setLoading(true);
      
      const [realtimeMetrics, stats, recentFraud] = await Promise.all([
        analyticsAPI.getRealTimeMetrics(),
        transactionsAPI.getStats(),
        transactionsAPI.getAll({ is_fraud: true, limit: 3 })
      ]);
      
      setMetrics({
        ...realtimeMetrics,
        stats
      });
      
      // Format recent fraud as real-time feed
      const feed = recentFraud.map((txn, index) => {
        const txnDate = new Date(txn.timestamp);
        const now = new Date();
        const diffMins = Math.floor((now - txnDate) / 60000);
        
        let timeAgo;
        if (diffMins < 1) {
          timeAgo = 'Just now';
        } else if (diffMins < 60) {
          timeAgo = `${diffMins} minute${diffMins > 1 ? 's' : ''} ago`;
        } else {
          const diffHours = Math.floor(diffMins / 60);
          timeAgo = `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
        }
        
        let type = 'info';
        if (txn.fraud_probability > 0.8) {
          type = 'warning';
        } else if (txn.fraud_probability < 0.5) {
          type = 'success';
        }
        
        return {
          id: txn.id,
          type,
          message: `Transaction flagged at ${txn.merchant_name || 'Unknown Merchant'}`,
          time: timeAgo,
          icon: type === 'warning' ? '🟡' : type === 'success' ? '🟢' : '🔵'
        };
      });
      
      setRealTimeFeed(feed);
    } catch (error) {
      console.error('Failed to fetch security data:', error);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchSecurityData();
  }, [fetchSecurityData]);

  // Auto-refresh every 30 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      fetchSecurityData();
    }, 30000);

    return () => clearInterval(interval);
  }, [fetchSecurityData]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Security Status</h1>
        <p className="text-gray-500">Quantum node stability and threat monitoring overview</p>
      </div>

      {/* Main Status Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* System Status */}
        <Card className="lg:col-span-2">
          <div className="mb-6">
            <h2 className="text-gray-500 text-sm font-medium uppercase tracking-wider mb-2">System Status</h2>
            <h3 className="text-gray-900 text-2xl font-bold">Quantum Node Stability: 99.98%</h3>
          </div>

          {/* Stability Chart */}
          <div className="mb-8">
            <div className="flex items-end justify-between h-32 space-x-2">
              {stabilityData.map((bar, index) => (
                <div
                  key={index}
                  className="flex-1 flex flex-col items-center"
                >
                  <div
                    className={`w-full rounded-t-lg transition-all duration-1000 delay-${index * 100} ${
                      bar.active 
                        ? 'bg-gradient-to-t from-blue-500 to-cyan-400 shadow-lg shadow-blue-500/50' 
                        : 'bg-gradient-to-t from-gray-600 to-gray-500'
                    }`}
                    style={{ 
                      height: animationComplete ? `${bar.value}%` : '0%'
                    }}
                  ></div>
                </div>
              ))}
            </div>
          </div>

          {/* Metrics Row */}
          <div className="grid grid-cols-2 gap-8">
            <div>
              <h4 className="text-gray-500 text-sm font-medium uppercase tracking-wider mb-2">
                Detected Anomalies
              </h4>
              <div className="flex items-baseline space-x-2">
                <span className="text-gray-900 text-4xl font-bold">
                  {metrics?.stats?.fraud_count || 0}
                </span>
                <span className="text-red-400 text-sm font-medium">-24% from last hour</span>
              </div>
            </div>
            <div>
              <h4 className="text-gray-500 text-sm font-medium uppercase tracking-wider mb-2">
                Intercepted Capital
              </h4>
              <div className="flex items-baseline space-x-2">
                <span className="text-gray-900 text-4xl font-bold">
                  ₹{metrics?.stats?.total_amount ? (metrics.stats.total_amount / 1000000).toFixed(1) : '0.0'}M
                </span>
                <span className="text-green-400 text-sm font-medium">+12% total recovery</span>
              </div>
            </div>
          </div>
        </Card>

        {/* Real-time Feed */}
        <Card>
          <div className="mb-6">
            <h2 className="text-gray-500 text-sm font-medium uppercase tracking-wider mb-2">
              Real-time Feed
            </h2>
          </div>

          <div className="space-y-4">
            {realTimeFeed.length > 0 ? (
              realTimeFeed.map((item, index) => (
                <div
                  key={item.id}
                  className={`flex items-start space-x-3 p-3 rounded-xl bg-white/30 transition-all duration-500 delay-${index * 200} ${
                    animationComplete ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
                  }`}
                >
                  <div className="flex-shrink-0 mt-1">
                    <div className={`w-2 h-2 rounded-full ${
                      item.type === 'info' ? 'bg-blue-400' :
                      item.type === 'warning' ? 'bg-yellow-400' :
                      'bg-green-400'
                    } animate-pulse`}></div>
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-gray-900 text-sm font-medium mb-1">{item.message}</p>
                    <p className="text-gray-500 text-xs">{item.time}</p>
                  </div>
                </div>
              ))
            ) : (
              <div className="text-center py-8 text-gray-500">
                <p className="text-sm">No recent activity</p>
              </div>
            )}
          </div>

          {/* Additional Status Indicators */}
          <div className="mt-6 pt-6 border-t border-gray-200">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-gray-500 text-sm">Network Latency</span>
                <span className="text-green-400 text-sm font-medium">
                  {metrics?.avg_processing_time ? `${Math.round(metrics.avg_processing_time)}ms` : '12ms'}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-500 text-sm">Active Connections</span>
                <span className="text-gray-900 text-sm font-medium">
                  {metrics?.websocket_connections?.toLocaleString() || '2,847'}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-500 text-sm">Threat Level</span>
                <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                  metrics?.stats?.fraud_rate > 5 
                    ? 'bg-red-500/20 text-red-400' 
                    : metrics?.stats?.fraud_rate > 2 
                    ? 'bg-yellow-500/20 text-yellow-400' 
                    : 'bg-green-500/20 text-green-400'
                }`}>
                  {metrics?.stats?.fraud_rate > 5 ? 'HIGH' : metrics?.stats?.fraud_rate > 2 ? 'MEDIUM' : 'LOW'}
                </span>
              </div>
            </div>
          </div>
        </Card>
      </div>

      {/* Additional Security Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card>
          <div className="flex items-center space-x-3 mb-4">
            <div className="p-2 bg-blue-500/20 rounded-lg">
              <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
              </svg>
            </div>
            <div>
              <h3 className="text-gray-900 font-semibold">Firewall Status</h3>
              <p className="text-green-400 text-sm">Active & Protected</p>
            </div>
          </div>
          <div className="text-gray-500 text-sm">
            Last scan: 2 minutes ago
          </div>
        </Card>

        <Card>
          <div className="flex items-center space-x-3 mb-4">
            <div className="p-2 bg-purple-500/20 rounded-lg">
              <svg className="w-5 h-5 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <div>
              <h3 className="text-gray-900 font-semibold">Quantum Encryption</h3>
              <p className="text-purple-400 text-sm">256-bit Active</p>
            </div>
          </div>
          <div className="text-gray-500 text-sm">
            Key rotation: 4 hours ago
          </div>
        </Card>

        <Card>
          <div className="flex items-center space-x-3 mb-4">
            <div className="p-2 bg-cyan-500/20 rounded-lg">
              <svg className="w-5 h-5 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
              </svg>
            </div>
            <div>
              <h3 className="text-gray-900 font-semibold">Monitoring</h3>
              <p className="text-cyan-400 text-sm">24/7 Surveillance</p>
            </div>
          </div>
          <div className="text-gray-500 text-sm">
            Uptime: 99.98%
          </div>
        </Card>
      </div>
    </div>
  );
};

export default SecurityStatus;