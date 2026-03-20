import React, { useState, useEffect, useCallback } from 'react';
import { transactionsAPI, fraudAPI } from '../services/api';

const Transactions = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('All');
  const [currentPage, setCurrentPage] = useState(1);
  const [selectedTransaction, setSelectedTransaction] = useState(null);
  const [showModal, setShowModal] = useState(false);
  const [transactions, setTransactions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [totalCount, setTotalCount] = useState(0);
  
  const itemsPerPage = 10;

  // Fetch transactions from API
  const fetchTransactions = useCallback(async () => {
    try {
      setLoading(true);
      
      // Fetch more transactions to enable proper pagination
      const params = {
        skip: 0,
        limit: 1000 // Fetch more to enable client-side pagination
      };
      
      // Add status filter
      if (statusFilter !== 'All') {
        if (statusFilter === 'Fraud') {
          params.is_fraud = true;
        } else if (statusFilter === 'Legit') {
          params.is_fraud = false;
        }
        // Note: 'Suspicious' would need backend support for fraud_probability range
      }
      
      const data = await transactionsAPI.getAll(params);
      setTransactions(data);
      setTotalCount(data.length);
    } catch (error) {
      console.error('Failed to fetch transactions:', error);
      setTransactions([]);
    } finally {
      setLoading(false);
    }
  }, [statusFilter]);

  // Initial fetch and auto-refresh
  useEffect(() => {
    fetchTransactions();
  }, [fetchTransactions]);

  // Auto-refresh every 30 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      fetchTransactions();
    }, 30000);

    return () => clearInterval(interval);
  }, [fetchTransactions]);

  // Format transaction data for display
  const formatTransaction = (txn) => {
    // Determine status based on fraud probability
    let status = 'LEGIT';
    if (txn.is_fraud || txn.fraud_probability > 0.7) {
      status = 'FRAUD';
    } else if (txn.fraud_probability > 0.3) {
      status = 'SUSPICIOUS';
    }
    
    // Calculate fraud score percentage
    const fraudScore = Math.round((txn.fraud_probability || 0) * 100);
    
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
      timeAgo = `${diffMins} min${diffMins > 1 ? 's' : ''} ago`;
    } else if (diffHours < 24) {
      timeAgo = `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
    } else {
      timeAgo = `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
    }
    
    // Get initials for icon
    const merchantName = txn.merchant_name || 'Unknown';
    const initials = merchantName
      .split(' ')
      .map(word => word[0])
      .join('')
      .substring(0, 2)
      .toUpperCase();
    
    return {
      id: `#${txn.id.substring(0, 8).toUpperCase()}`,
      fullId: txn.id,
      entityName: merchantName,
      entityType: txn.merchant_category || 'Unknown',
      amount: `₹${txn.amount.toFixed(2)}`,
      time: timeAgo,
      status,
      fraudScore,
      icon: initials,
      location: txn.location,
      description: txn.description,
      userId: txn.user_id,
      timestamp: txn.timestamp
    };
  };

  // Sample transaction data (fallback for empty state)
  const allTransactions = transactions.length > 0 
    ? transactions.map(formatTransaction)
    : [];

  // Filter transactions based on search and status
  const filteredTransactions = allTransactions.filter(transaction => {
    const matchesSearch = transaction.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         transaction.entityName.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         transaction.amount.toLowerCase().includes(searchTerm.toLowerCase());
    
    const matchesStatus = statusFilter === 'All' || transaction.status === statusFilter.toUpperCase();
    
    return matchesSearch && matchesStatus;
  });

  // Pagination
  const totalPages = Math.ceil(filteredTransactions.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = Math.min(startIndex + itemsPerPage, filteredTransactions.length);
  const paginatedTransactions = filteredTransactions.slice(startIndex, endIndex);

  // Reset to page 1 if current page exceeds total pages
  useEffect(() => {
    if (currentPage > totalPages && totalPages > 0) {
      setCurrentPage(1);
    }
  }, [currentPage, totalPages]);

  // Loading state
  if (loading && transactions.length === 0) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <div className="text-white text-xl mb-2">Loading transactions...</div>
          <div className="text-gray-400 text-sm">Fetching real-time data</div>
        </div>
      </div>
    );
  }

  const getStatusBadge = (status, fraudScore) => {
    const baseClasses = "px-3 py-1 rounded-full text-xs font-medium uppercase tracking-wider";
    
    switch (status) {
      case 'LEGIT':
        return `${baseClasses} bg-green-500/20 text-green-400`;
      case 'SUSPICIOUS':
        return `${baseClasses} bg-yellow-500/20 text-yellow-400`;
      case 'FRAUD':
        return `${baseClasses} bg-red-500/20 text-red-400`;
      default:
        return `${baseClasses} bg-gray-500/20 text-gray-400`;
    }
  };

  const getFraudScoreBar = (score) => {
    let colorClass = 'from-green-500 to-green-400';
    if (score > 30 && score <= 70) colorClass = 'from-yellow-500 to-yellow-400';
    if (score > 70) colorClass = 'from-red-500 to-red-400';
    
    return (
      <div className="flex items-center space-x-2">
        <div className="w-16 bg-gray-800 rounded-full h-2">
          <div
            className={`h-2 rounded-full bg-gradient-to-r ${colorClass}`}
            style={{ width: `${score}%` }}
          ></div>
        </div>
        <span className="text-white text-sm font-medium">{score}%</span>
      </div>
    );
  };

  const handleRowClick = (transaction) => {
    setSelectedTransaction(transaction);
    setShowModal(true);
  };

  const closeModal = () => {
    setShowModal(false);
    setSelectedTransaction(null);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white mb-2">Transaction Ledger</h1>
          <p className="text-gray-400">Monitoring {filteredTransactions.length.toLocaleString()} operations in the last 24 hours.</p>
        </div>
      </div>

      {/* Controls */}
      <div className="flex items-center justify-between bg-[#111827]/60 backdrop-blur-sm border border-gray-800/50 rounded-2xl p-6">
        <div className="flex items-center space-x-4">
          {/* Dropdown */}
          <div className="relative">
            <select 
              className="bg-gray-800/50 border border-gray-700/50 rounded-xl px-4 py-2 text-gray-300 focus:outline-none focus:border-blue-500/50 appearance-none pr-8"
              value="All Transactions"
              onChange={() => {}}
            >
              <option>All Transactions</option>
            </select>
            <svg className="absolute right-2 top-2.5 w-4 h-4 text-gray-500 pointer-events-none" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </div>

          {/* Filter Buttons */}
          <div className="flex items-center space-x-2">
            {['All', 'Legit', 'Suspicious', 'Fraud'].map((filter) => (
              <button
                key={filter}
                onClick={() => {
                  setStatusFilter(filter);
                  setCurrentPage(1);
                }}
                className={`px-4 py-2 rounded-xl text-sm font-medium transition-all duration-200 ${
                  statusFilter === filter
                    ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
                    : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
                }`}
              >
                {filter}
              </button>
            ))}
          </div>
        </div>

        {/* Search Bar */}
        <div className="relative">
          <input
            type="text"
            placeholder="Search hash, entity or ID..."
            value={searchTerm}
            onChange={(e) => {
              setSearchTerm(e.target.value);
              setCurrentPage(1);
            }}
            className="bg-gray-800/50 border border-gray-700/50 rounded-xl px-4 py-2 pl-10 text-gray-300 placeholder-gray-500 focus:outline-none focus:border-blue-500/50 focus:bg-gray-800/80 transition-all duration-200 w-80"
          />
          <svg className="absolute left-3 top-2.5 w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
        </div>
      </div>

      {/* Transaction Table */}
      <div className="bg-[#111827]/60 backdrop-blur-sm border border-gray-800/50 rounded-2xl overflow-hidden">
        {/* Table Header */}
        <div className="grid grid-cols-7 gap-4 p-6 border-b border-gray-800/50 bg-gray-900/30">
          <div className="text-gray-400 text-xs font-medium uppercase tracking-wider">Transaction ID</div>
          <div className="text-gray-400 text-xs font-medium uppercase tracking-wider">Entity Name</div>
          <div className="text-gray-400 text-xs font-medium uppercase tracking-wider">Amount</div>
          <div className="text-gray-400 text-xs font-medium uppercase tracking-wider">Time</div>
          <div className="text-gray-400 text-xs font-medium uppercase tracking-wider">Status</div>
          <div className="text-gray-400 text-xs font-medium uppercase tracking-wider">Fraud Score</div>
          <div className="text-gray-400 text-xs font-medium uppercase tracking-wider">Action</div>
        </div>

        {/* Table Body */}
        <div className="divide-y divide-gray-800/30">
          {paginatedTransactions.length > 0 ? (
            paginatedTransactions.map((transaction, index) => (
              <div
                key={transaction.fullId || transaction.id}
                onClick={() => handleRowClick(transaction)}
                className="grid grid-cols-7 gap-4 p-6 hover:bg-gray-800/30 transition-all duration-200 cursor-pointer group"
              >
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 bg-gray-700 rounded-lg flex items-center justify-center text-xs font-medium text-gray-300">
                    {transaction.icon}
                  </div>
                  <span className="text-blue-400 font-mono text-sm">{transaction.id}</span>
                </div>
                
                <div>
                  <div className="text-white font-medium text-sm">{transaction.entityName}</div>
                  <div className="text-gray-400 text-xs capitalize">{transaction.entityType}</div>
                </div>
                
                <div className="text-cyan-400 font-bold text-sm">{transaction.amount}</div>
                
                <div className="text-gray-300 text-sm">{transaction.time}</div>
                
                <div>
                  <span className={getStatusBadge(transaction.status, transaction.fraudScore)}>
                    {transaction.status}
                  </span>
                </div>
                
                <div>
                  {getFraudScoreBar(transaction.fraudScore)}
                </div>
                
                <div className="flex items-center">
                  <svg className="w-5 h-5 text-gray-400 group-hover:text-white transition-colors duration-200" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </div>
              </div>
            ))
          ) : (
            <div className="p-12 text-center">
              <svg className="w-16 h-16 mx-auto mb-4 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <div className="text-gray-400 text-lg mb-2">No transactions found</div>
              <div className="text-gray-500 text-sm">
                {searchTerm || statusFilter !== 'All' 
                  ? 'Try adjusting your filters' 
                  : 'Start processing transactions to see them here'}
              </div>
            </div>
          )}
        </div>

        {/* Pagination */}
        <div className="flex items-center justify-between p-6 border-t border-gray-800/50">
          <div className="text-gray-400 text-sm">
            Showing {filteredTransactions.length > 0 ? startIndex + 1 : 0}-{endIndex} of {filteredTransactions.length.toLocaleString()}
          </div>
          
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
              disabled={currentPage === 1 || totalPages === 0}
              className="p-2 text-gray-400 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed transition-colors duration-200"
              title="Previous page"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
            </button>
            
            {totalPages > 0 && [...Array(Math.min(5, totalPages))].map((_, i) => {
              let pageNum;
              if (totalPages <= 5) {
                pageNum = i + 1;
              } else if (currentPage <= 3) {
                pageNum = i + 1;
              } else if (currentPage >= totalPages - 2) {
                pageNum = totalPages - 4 + i;
              } else {
                pageNum = currentPage - 2 + i;
              }
              
              return (
                <button
                  key={pageNum}
                  onClick={() => setCurrentPage(pageNum)}
                  className={`w-8 h-8 rounded-lg text-sm font-medium transition-all duration-200 ${
                    currentPage === pageNum
                      ? 'bg-blue-500 text-white'
                      : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
                  }`}
                >
                  {pageNum}
                </button>
              );
            })}
            
            {totalPages > 5 && currentPage < totalPages - 2 && (
              <>
                <span className="text-gray-400">...</span>
                <button
                  onClick={() => setCurrentPage(totalPages)}
                  className="w-8 h-8 rounded-lg text-sm font-medium text-gray-400 hover:text-white hover:bg-gray-800/50 transition-all duration-200"
                >
                  {totalPages}
                </button>
              </>
            )}
            
            <button
              onClick={() => setCurrentPage(prev => Math.min(totalPages, prev + 1))}
              disabled={currentPage === totalPages || totalPages === 0}
              className="p-2 text-gray-400 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed transition-colors duration-200"
              title="Next page"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </button>
          </div>
        </div>
      </div>

      {/* Transaction Detail Modal */}
      {showModal && selectedTransaction && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-[#111827] border border-gray-800/50 rounded-2xl p-6 max-w-2xl w-full max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-bold text-white">Transaction Details</h3>
              <button
                onClick={closeModal}
                className="p-2 text-gray-400 hover:text-white transition-colors duration-200"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            
            <div className="space-y-6">
              <div className="grid grid-cols-2 gap-6">
                <div>
                  <label className="text-gray-400 text-sm font-medium">Transaction ID</label>
                  <p className="text-white font-mono text-lg">{selectedTransaction.id}</p>
                  <p className="text-gray-500 text-xs mt-1">Full ID: {selectedTransaction.fullId}</p>
                </div>
                <div>
                  <label className="text-gray-400 text-sm font-medium">Amount</label>
                  <p className="text-cyan-400 font-bold text-lg">{selectedTransaction.amount}</p>
                </div>
              </div>
              
              <div className="grid grid-cols-2 gap-6">
                <div>
                  <label className="text-gray-400 text-sm font-medium">Entity Name</label>
                  <p className="text-white font-medium">{selectedTransaction.entityName}</p>
                  <p className="text-gray-400 text-sm capitalize">{selectedTransaction.entityType}</p>
                </div>
                <div>
                  <label className="text-gray-400 text-sm font-medium">Location</label>
                  <p className="text-white">{selectedTransaction.location || 'N/A'}</p>
                </div>
              </div>
              
              <div className="grid grid-cols-2 gap-6">
                <div>
                  <label className="text-gray-400 text-sm font-medium">Time</label>
                  <p className="text-white">{selectedTransaction.time}</p>
                  <p className="text-gray-500 text-xs mt-1">
                    {selectedTransaction.timestamp ? new Date(selectedTransaction.timestamp).toLocaleString() : 'N/A'}
                  </p>
                </div>
                <div>
                  <label className="text-gray-400 text-sm font-medium">User ID</label>
                  <p className="text-white font-mono text-sm">{selectedTransaction.userId || 'N/A'}</p>
                </div>
              </div>
              
              <div>
                <label className="text-gray-400 text-sm font-medium">Description</label>
                <p className="text-white mt-1">{selectedTransaction.description || 'No description available'}</p>
              </div>
              
              <div className="grid grid-cols-2 gap-6">
                <div>
                  <label className="text-gray-400 text-sm font-medium">Status</label>
                  <div className="mt-2">
                    <span className={getStatusBadge(selectedTransaction.status, selectedTransaction.fraudScore)}>
                      {selectedTransaction.status}
                    </span>
                  </div>
                </div>
                <div>
                  <label className="text-gray-400 text-sm font-medium">Fraud Score</label>
                  <div className="mt-2">
                    {getFraudScoreBar(selectedTransaction.fraudScore)}
                  </div>
                </div>
              </div>
              
              <div className="pt-4 border-t border-gray-800/50">
                <div className="flex items-center space-x-4">
                  <button className="px-4 py-2 bg-blue-500 text-white rounded-xl font-medium hover:bg-blue-600 transition-colors duration-200">
                    View Full Details
                  </button>
                  <button className="px-4 py-2 bg-gray-800 text-gray-300 rounded-xl font-medium hover:bg-gray-700 transition-colors duration-200">
                    Export Data
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Transactions;