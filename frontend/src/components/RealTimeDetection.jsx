import React, { useState, useEffect } from 'react';
import { fraudAPI } from '../services/api';

const RealTimeDetection = () => {
  const [formData, setFormData] = useState({
    amount: '',
    time: '',
    location: '',
    merchantType: '',
    cardType: '',
    frequency: ''
  });
  
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [modelStatus, setModelStatus] = useState(null);
  const [apiError, setApiError] = useState(null);

  // Check model status on component mount
  useEffect(() => {
    checkModelStatus();
  }, []);

  const checkModelStatus = async () => {
    try {
      const status = await fraudAPI.getModelStatus();
      setModelStatus(status);
    } catch (error) {
      console.error('Failed to get model status:', error);
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const performFraudCheck = async () => {
    setIsLoading(true);
    setResult(null);
    setApiError(null);

    try {
      // Prepare transaction data for API
      const amount = parseFloat(formData.amount) || 0;
      
      // Generate transaction ID
      const transactionId = `TXN-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
      
      // Format time as ISO datetime string (backend expects datetime object)
      let transactionTime;
      if (formData.time) {
        const today = new Date();
        const [hours, minutes] = formData.time.split(':');
        today.setHours(parseInt(hours), parseInt(minutes), 0, 0);
        transactionTime = today.toISOString();
      } else {
        transactionTime = new Date().toISOString();
      }

      // Generate random PCA features (V1-V28) for demonstration
      // In production, these would come from actual transaction data
      const pcaFeatures = {};
      for (let i = 1; i <= 28; i++) {
        pcaFeatures[`v${i}`] = parseFloat(((Math.random() - 0.5) * 2).toFixed(6));
      }

      const transactionData = {
        transaction_id: transactionId,
        time: transactionTime,
        amount: amount,
        ...pcaFeatures
      };

      // Call real API
      const startTime = Date.now();
      const apiResult = await fraudAPI.predictSingle(transactionData);
      const processingTime = Date.now() - startTime;

      // Transform API response to match UI expectations
      setResult({
        fraudScore: Math.round(apiResult.fraud_probability * 100),
        status: apiResult.is_fraud ? 'FRAUD' : 'SAFE',
        modelType: apiResult.model_used === 'hybrid_quantum' ? 'Quantum' : 
                   apiResult.model_used === 'classical_xgboost' ? 'Classical' : 'Rule-based',
        confidence: Math.round(apiResult.confidence_score * 100),
        processingTime: processingTime,
        riskFactors: apiResult.risk_factors || []
      });

    } catch (error) {
      console.error('Fraud check failed:', error);
      setApiError(error.message || 'Failed to connect to backend API');
      
      // Fallback to simulated result if API fails
      const amount = parseFloat(formData.amount) || 0;
      const isHighAmount = amount > 10000;
      let fraudScore = Math.random() * 100;
      if (isHighAmount) fraudScore += 20;
      fraudScore = Math.min(100, Math.max(0, fraudScore));
      
      setResult({
        fraudScore: Math.round(fraudScore),
        status: fraudScore > 60 ? 'FRAUD' : 'SAFE',
        modelType: 'Fallback',
        confidence: 50,
        processingTime: 0,
        riskFactors: ['API connection failed - using fallback']
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!formData.amount) return;
    performFraudCheck();
  };

  const resetForm = () => {
    setFormData({
      amount: '',
      time: '',
      location: '',
      merchantType: '',
      cardType: '',
      frequency: ''
    });
    setResult(null);
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-3xl font-bold text-white mb-4">Real-Time Fraud Detection</h1>
        <p className="text-gray-400 text-lg">AI-powered transaction analysis with quantum-enhanced algorithms</p>
        
        {/* API Error Alert */}
        {apiError && (
          <div className="mt-4 mx-auto max-w-2xl px-4 py-3 bg-red-500/20 border border-red-500/30 rounded-xl">
            <div className="flex items-center space-x-2 text-red-400">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
              <span className="text-sm font-medium">{apiError}</span>
            </div>
          </div>
        )}
        
        {/* AI Status Indicator */}
        <div className="flex items-center justify-center space-x-4 mt-6">
          <div className={`flex items-center space-x-2 px-4 py-2 bg-gradient-to-r ${
            modelStatus ? 'from-green-500/20 to-emerald-500/20 border-green-500/30' : 'from-gray-500/20 to-gray-500/20 border-gray-500/30'
          } border rounded-full`}>
            <div className={`w-2 h-2 ${modelStatus ? 'bg-green-400 animate-pulse' : 'bg-gray-400'} rounded-full`}></div>
            <span className={`${modelStatus ? 'text-green-400' : 'text-gray-400'} text-sm font-medium`}>
              {modelStatus ? 'AI System Online' : 'Connecting...'}
            </span>
          </div>
          {modelStatus && modelStatus.models && modelStatus.models.hybrid && modelStatus.models.hybrid.loaded && (
            <div className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-purple-500/20 to-blue-500/20 border border-purple-500/30 rounded-full">
              <svg className="w-4 h-4 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              <span className="text-purple-400 text-sm font-medium">Quantum Ready</span>
            </div>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Input Form */}
        <div className="bg-[#111827]/60 backdrop-blur-sm border border-gray-800/50 rounded-2xl p-8">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-semibold text-white">Transaction Details</h2>
            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="text-blue-400 text-sm font-medium hover:text-blue-300 transition-colors duration-200"
            >
              {showAdvanced ? 'Hide Advanced' : 'Show Advanced'}
            </button>
          </div>

          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Required Fields */}
            <div className="space-y-4">
              <div>
                <label className="block text-gray-300 text-sm font-medium mb-2">
                  Transaction Amount *
                </label>
                <div className="relative">
                  <span className="absolute left-3 top-3 text-gray-400">₹</span>
                  <input
                    type="number"
                    name="amount"
                    value={formData.amount}
                    onChange={handleInputChange}
                    placeholder="0.00"
                    step="0.01"
                    min="0"
                    required
                    className="w-full bg-gray-800/50 border border-gray-700/50 rounded-xl px-4 py-3 pl-8 text-white placeholder-gray-500 focus:outline-none focus:border-blue-500/50 focus:bg-gray-800/80 transition-all duration-200"
                  />
                </div>
              </div>

              <div>
                <label className="block text-gray-300 text-sm font-medium mb-2">
                  Transaction Time
                </label>
                <input
                  type="time"
                  name="time"
                  value={formData.time}
                  onChange={handleInputChange}
                  className="w-full bg-gray-800/50 border border-gray-700/50 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-blue-500/50 focus:bg-gray-800/80 transition-all duration-200"
                />
              </div>
            </div>

            {/* Advanced Fields */}
            {showAdvanced && (
              <div className="space-y-4 pt-4 border-t border-gray-800/50">
                <div>
                  <label className="block text-gray-300 text-sm font-medium mb-2">
                    Location
                  </label>
                  <input
                    type="text"
                    name="location"
                    value={formData.location}
                    onChange={handleInputChange}
                    placeholder="e.g., New York, USA"
                    className="w-full bg-gray-800/50 border border-gray-700/50 rounded-xl px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-blue-500/50 focus:bg-gray-800/80 transition-all duration-200"
                  />
                </div>

                <div>
                  <label className="block text-gray-300 text-sm font-medium mb-2">
                    Merchant Type
                  </label>
                  <select
                    name="merchantType"
                    value={formData.merchantType}
                    onChange={handleInputChange}
                    className="w-full bg-gray-800/50 border border-gray-700/50 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-blue-500/50 focus:bg-gray-800/80 transition-all duration-200"
                  >
                    <option value="">Select merchant type</option>
                    <option value="retail">Retail Store</option>
                    <option value="online">Online Store</option>
                    <option value="restaurant">Restaurant</option>
                    <option value="gas">Gas Station</option>
                    <option value="atm">ATM</option>
                    <option value="other">Other</option>
                  </select>
                </div>

                <div>
                  <label className="block text-gray-300 text-sm font-medium mb-2">
                    Card Type
                  </label>
                  <select
                    name="cardType"
                    value={formData.cardType}
                    onChange={handleInputChange}
                    className="w-full bg-gray-800/50 border border-gray-700/50 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-blue-500/50 focus:bg-gray-800/80 transition-all duration-200"
                  >
                    <option value="">Select card type</option>
                    <option value="credit">Credit Card</option>
                    <option value="debit">Debit Card</option>
                    <option value="prepaid">Prepaid Card</option>
                  </select>
                </div>

                <div>
                  <label className="block text-gray-300 text-sm font-medium mb-2">
                    Transaction Frequency (last 24h)
                  </label>
                  <input
                    type="number"
                    name="frequency"
                    value={formData.frequency}
                    onChange={handleInputChange}
                    placeholder="Number of transactions"
                    min="0"
                    className="w-full bg-gray-800/50 border border-gray-700/50 rounded-xl px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-blue-500/50 focus:bg-gray-800/80 transition-all duration-200"
                  />
                </div>
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex items-center space-x-4 pt-6">
              <button
                type="submit"
                disabled={!formData.amount || isLoading}
                className="flex-1 bg-gradient-to-r from-blue-600 to-cyan-500 text-white py-3 px-6 rounded-2xl font-semibold text-lg hover:from-blue-700 hover:to-cyan-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 shadow-lg hover:shadow-blue-500/25"
              >
                {isLoading ? (
                  <div className="flex items-center justify-center space-x-2">
                    <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                    <span>Analyzing...</span>
                  </div>
                ) : (
                  'Check Fraud'
                )}
              </button>
              
              <button
                type="button"
                onClick={resetForm}
                className="px-6 py-3 bg-gray-800 text-gray-300 rounded-2xl font-medium hover:bg-gray-700 transition-colors duration-200"
              >
                Reset
              </button>
            </div>
          </form>
        </div>

        {/* Results Section */}
        <div className="bg-[#111827]/60 backdrop-blur-sm border border-gray-800/50 rounded-2xl p-8">
          <h2 className="text-xl font-semibold text-white mb-6">Detection Results</h2>
          
          {!result && !isLoading && (
            <div className="flex items-center justify-center h-64 text-gray-400">
              <div className="text-center">
                <svg className="w-16 h-16 mx-auto mb-4 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                <p>Enter transaction details and click "Check Fraud" to analyze</p>
              </div>
            </div>
          )}

          {isLoading && (
            <div className="flex items-center justify-center h-64">
              <div className="text-center">
                <div className="w-16 h-16 border-4 border-blue-500/30 border-t-blue-500 rounded-full animate-spin mx-auto mb-4"></div>
                <p className="text-white font-medium">AI Analysis in Progress...</p>
                <p className="text-gray-400 text-sm mt-2">Processing transaction patterns</p>
              </div>
            </div>
          )}

          {result && (
            <div className="space-y-6 animate-fadeIn">
              {/* Fraud Score */}
              <div className="text-center">
                <div className="mb-4">
                  <div className={`text-6xl font-bold mb-2 ${
                    result.status === 'FRAUD' ? 'text-red-400' : 'text-green-400'
                  }`}>
                    {result.fraudScore}%
                  </div>
                  <p className="text-gray-400">Fraud Score</p>
                </div>
                
                {/* Status Badge */}
                <div className={`inline-flex items-center space-x-2 px-6 py-3 rounded-2xl font-semibold text-lg ${
                  result.status === 'FRAUD' 
                    ? 'bg-red-500/20 text-red-400 border border-red-500/30' 
                    : 'bg-green-500/20 text-green-400 border border-green-500/30'
                }`}>
                  <div className={`w-3 h-3 rounded-full ${
                    result.status === 'FRAUD' ? 'bg-red-400' : 'bg-green-400'
                  } animate-pulse`}></div>
                  <span>{result.status === 'FRAUD' ? 'FRAUD DETECTED' : 'TRANSACTION SAFE'}</span>
                </div>
              </div>

              {/* Model Indicator */}
              <div className="flex items-center justify-center">
                <div className={`flex items-center space-x-2 px-4 py-2 rounded-full ${
                  result.modelType === 'Quantum' 
                    ? 'bg-purple-500/20 text-purple-400 border border-purple-500/30' 
                    : 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
                }`}>
                  {result.modelType === 'Quantum' ? (
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                  ) : (
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                    </svg>
                  )}
                  <span className="text-sm font-medium">
                    {result.modelType === 'Quantum' ? 'Quantum Model Activated' : 'Classical Model'}
                  </span>
                </div>
              </div>

              {/* Additional Metrics */}
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-gray-800/30 rounded-xl p-4 text-center">
                  <div className="text-2xl font-bold text-white mb-1">{result.confidence}%</div>
                  <div className="text-gray-400 text-sm">Confidence</div>
                </div>
                <div className="bg-gray-800/30 rounded-xl p-4 text-center">
                  <div className="text-2xl font-bold text-white mb-1">{result.processingTime}ms</div>
                  <div className="text-gray-400 text-sm">Processing Time</div>
                </div>
              </div>

              {/* Risk Factors */}
              {result.riskFactors.length > 0 && (
                <div>
                  <h3 className="text-white font-medium mb-3">Risk Factors Detected:</h3>
                  <div className="space-y-2">
                    {result.riskFactors.map((factor, index) => (
                      <div key={index} className="flex items-center space-x-2 text-yellow-400 text-sm">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                        </svg>
                        <span>{factor}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default RealTimeDetection;