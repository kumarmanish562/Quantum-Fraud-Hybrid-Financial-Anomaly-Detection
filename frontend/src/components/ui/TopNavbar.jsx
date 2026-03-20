import React, { useState, useRef, useEffect } from 'react';
import { analyticsAPI, transactionsAPI } from '../../services/api';

const TopNavbar = ({ title = "Dashboard", notificationCount = 0, userName = "Admin User", userRole = "Security Analyst", onMenuClick }) => {
  const [isProfileOpen, setIsProfileOpen] = useState(false);
  const [isNotificationOpen, setIsNotificationOpen] = useState(false);
  const [notifications, setNotifications] = useState([]);
  const [isLoadingNotifications, setIsLoadingNotifications] = useState(false);
  const [isSimulating, setIsSimulating] = useState(false);
  const profileRef = useRef(null);
  const notificationRef = useRef(null);

  // Fetch real-time notifications
  const fetchNotifications = async () => {
    try {
      setIsLoadingNotifications(true);
      
      // Fetch recent fraud transactions as notifications
      const fraudTransactions = await transactionsAPI.getAll({ is_fraud: true, limit: 5 });
      
      // Transform to notification format
      const notifs = fraudTransactions.map((txn, index) => {
        const timeAgo = getTimeAgo(new Date(txn.timestamp));
        return {
          id: txn.id || index,
          title: `High-risk transaction detected: ₹${txn.amount?.toLocaleString()}`,
          time: timeAgo,
          type: 'alert',
          amount: txn.amount,
          timestamp: txn.timestamp
        };
      });
      
      setNotifications(notifs);
    } catch (error) {
      console.error('Failed to fetch notifications:', error);
      // Fallback to empty array
      setNotifications([]);
    } finally {
      setIsLoadingNotifications(false);
    }
  };

  // Calculate time ago
  const getTimeAgo = (date) => {
    const seconds = Math.floor((new Date() - date) / 1000);
    
    if (seconds < 60) return `${seconds} sec ago`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)} min ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)} hour${Math.floor(seconds / 3600) > 1 ? 's' : ''} ago`;
    return `${Math.floor(seconds / 86400)} day${Math.floor(seconds / 86400) > 1 ? 's' : ''} ago`;
  };

  // Fetch notifications when dropdown opens
  useEffect(() => {
    if (isNotificationOpen) {
      fetchNotifications();
    }
  }, [isNotificationOpen]);

  // Auto-refresh notifications every 30 seconds when dropdown is open
  useEffect(() => {
    if (isNotificationOpen) {
      const interval = setInterval(() => {
        fetchNotifications();
      }, 30000);
      
      return () => clearInterval(interval);
    }
  }, [isNotificationOpen]);

  // Close dropdowns when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (profileRef.current && !profileRef.current.contains(event.target)) {
        setIsProfileOpen(false);
      }
      if (notificationRef.current && !notificationRef.current.contains(event.target)) {
        setIsNotificationOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Handle View All Notifications click
  const handleViewAllNotifications = () => {
    setIsNotificationOpen(false);
    if (onMenuClick) {
      onMenuClick('Alerts');
    }
  };

  // Simulate Transaction
  const simulateTransaction = async () => {
    if (isSimulating) return;
    
    setIsSimulating(true);
    
    try {
      const merchants = [
        { name: "Flipkart", category: "online" },
        { name: "Amazon India", category: "online" },
        { name: "Swiggy", category: "food_delivery" },
        { name: "Zomato", category: "food_delivery" },
        { name: "Big Bazaar", category: "grocery" },
        { name: "Reliance Digital", category: "retail" },
        { name: "PVR Cinemas", category: "entertainment" },
        { name: "Tanishq Jewellers", category: "jewelry" },
        { name: "Cafe Coffee Day", category: "restaurant" },
        { name: "BookMyShow", category: "entertainment" }
      ];
      
      const locations = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Pune", "Kolkata"];
      
      // Random merchant and location
      const merchant = merchants[Math.floor(Math.random() * merchants.length)];
      const location = locations[Math.floor(Math.random() * locations.length)];
      
      // Generate amount with realistic distribution
      const rand = Math.random();
      let amount;
      let isFraudIntent = false;
      
      if (rand < 0.50) {
        // 50% normal transactions (₹100-5000)
        amount = Math.round((Math.random() * 4900 + 100) * 100) / 100;
        isFraudIntent = false;
      } else if (rand < 0.70) {
        // 20% medium transactions (₹5000-25000)
        amount = Math.round((Math.random() * 20000 + 5000) * 100) / 100;
        isFraudIntent = false;
      } else if (rand < 0.85) {
        // 15% high-value suspicious (₹25000-100000) - FRAUD
        amount = Math.round((Math.random() * 75000 + 25000) * 100) / 100;
        isFraudIntent = true;
      } else {
        // 15% very high-value fraud (₹100000-500000) - FRAUD
        amount = Math.round((Math.random() * 400000 + 100000) * 100) / 100;
        isFraudIntent = true;
      }
      
      const userId = `user_${Math.floor(Math.random() * 100) + 1}`;
      const description = `Purchase at ${merchant.name}, ${location}`;
      
      // Create transaction
      const transactionData = {
        user_id: userId,
        amount: amount,
        merchant_name: merchant.name,
        merchant_category: merchant.category,
        location: location,
        description: description
      };
      
      const response = await fetch('http://localhost:8000/api/v1/transactions/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(transactionData)
      });
      
      if (!response.ok) throw new Error('Failed to create transaction');
      
      const transaction = await response.json();
      
      // Generate features for fraud detection
      const currentHour = new Date().getHours();
      const features = generateFeatures(amount, currentHour, isFraudIntent);
      
      // Run fraud prediction
      const fraudData = {
        transaction_id: transaction.id,
        time: new Date().toISOString(),
        ...features,
        amount: amount
      };
      
      const fraudResponse = await fetch('http://localhost:8000/api/v1/fraud/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(fraudData)
      });
      
      if (fraudResponse.ok) {
        const fraudResult = await fraudResponse.json();
        
        // Update transaction with fraud status
        await fetch(`http://localhost:8000/api/v1/transactions/${transaction.id}/fraud-status?is_fraud=${fraudResult.is_fraud}&fraud_probability=${fraudResult.fraud_probability}&risk_score=${fraudResult.fraud_probability}`, {
          method: 'PATCH'
        });
        
        // Show notification
        const statusIcon = fraudResult.is_fraud ? '🚨' : '✅';
        const statusText = fraudResult.is_fraud ? 'FRAUD DETECTED' : 'Safe Transaction';
        const statusColor = fraudResult.is_fraud ? 'bg-red-500' : 'bg-green-500';
        
        // Create toast notification
        showToast(
          `${statusIcon} ${statusText}`,
          `₹${amount.toLocaleString()} at ${merchant.name} - ${(fraudResult.fraud_probability * 100).toFixed(0)}% risk`,
          statusColor
        );
        
        // Refresh page data after a short delay
        setTimeout(() => {
          window.location.reload();
        }, 2000);
      }
      
    } catch (error) {
      console.error('Simulation failed:', error);
      showToast('❌ Simulation Failed', 'Could not create transaction', 'bg-red-500');
    } finally {
      setIsSimulating(false);
    }
  };
  
  // Generate features for ML model
  const generateFeatures = (amount, timeHour, isFraudIntent) => {
    const features = {};
    
    for (let i = 1; i <= 28; i++) {
      // Start with normal distribution
      let baseValue = (Math.random() - 0.5) * 2 * 1.5; // Gaussian-like
      
      if (isFraudIntent) {
        // Strong fraud indicators - make features significantly different
        if (amount > 100000) {
          baseValue += Math.random() * 3 + 2.5; // Very strong signal
        } else if (amount > 50000) {
          baseValue += Math.random() * 2 + 1.5; // Strong signal
        } else if (amount > 25000) {
          baseValue += Math.random() * 1.5 + 1.0; // Medium signal
        }
        
        // Late night adds more suspicion
        if (timeHour < 5 || timeHour > 23) {
          baseValue += Math.random() * 1.5 + 0.5;
        }
        
        // Add some random spikes in certain features
        if (i % 3 === 0) {
          baseValue += Math.random() * 2;
        }
      } else {
        // Normal transaction - keep features close to zero
        baseValue = baseValue * 0.3; // Reduce variance
      }
      
      features[`v${i}`] = parseFloat(baseValue.toFixed(6));
    }
    
    return features;
  };
  
  // Toast notification helper
  const showToast = (title, message, bgColor) => {
    const toast = document.createElement('div');
    toast.className = `fixed top-20 right-6 ${bgColor} text-white px-6 py-4 rounded-xl shadow-2xl z-50 animate-slide-in-right max-w-md`;
    toast.innerHTML = `
      <div class="font-bold text-lg mb-1">${title}</div>
      <div class="text-sm opacity-90">${message}</div>
    `;
    document.body.appendChild(toast);
    
    setTimeout(() => {
      toast.style.animation = 'slide-out-right 0.3s ease-out';
      setTimeout(() => toast.remove(), 300);
    }, 3000);
  };

  return (
    <header className="bg-[#111827]/80 backdrop-blur-sm border-b border-gray-800/50 px-6 py-4 sticky top-0 z-40">
      <div className="flex items-center justify-between">
        {/* Left Section - Page Title */}
        <div className="flex items-center space-x-4">
          <h1 className="text-white text-2xl font-bold">{title}</h1>
          <div className="flex items-center space-x-2 px-3 py-1 bg-green-500/20 rounded-full">
            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
            <span className="text-green-400 text-xs font-medium">System Active</span>
          </div>
        </div>
        
        {/* Right Section */}
        <div className="flex items-center space-x-6">
          {/* Simulate Transaction Button */}
          <button
            onClick={simulateTransaction}
            disabled={isSimulating}
            className={`flex items-center space-x-2 px-4 py-2 rounded-xl font-medium transition-all duration-200 ${
              isSimulating 
                ? 'bg-gray-700 text-gray-400 cursor-not-allowed' 
                : 'bg-gradient-to-r from-blue-600 to-cyan-500 hover:from-blue-700 hover:to-cyan-600 text-white shadow-lg hover:shadow-xl'
            }`}
          >
            {isSimulating ? (
              <>
                <div className="w-4 h-4 border-2 border-gray-400 border-t-transparent rounded-full animate-spin"></div>
                <span className="text-sm">Simulating...</span>
              </>
            ) : (
              <>
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                <span className="text-sm">Simulate Transaction</span>
              </>
            )}
          </button>
          
          {/* Search Bar */}
          <div className="relative hidden md:block">
            <input
              type="text"
              placeholder="Search transactions..."
              className="bg-gray-800/50 border border-gray-700/50 rounded-xl px-4 py-2 pl-10 text-gray-300 placeholder-gray-500 focus:outline-none focus:border-blue-500/50 focus:bg-gray-800/80 transition-all duration-200 w-64"
            />
            <svg className="absolute left-3 top-2.5 w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </div>
          
          {/* Notifications */}
          <div className="relative" ref={notificationRef}>
            <button 
              onClick={() => setIsNotificationOpen(!isNotificationOpen)}
              className="relative p-2 text-gray-400 hover:text-white transition-colors duration-200 hover:bg-gray-800/50 rounded-xl"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-5 5v-5zM12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z" />
              </svg>
              {notificationCount > 0 && (
                <span className="absolute -top-1 -right-1 w-5 h-5 bg-red-500 text-white text-xs rounded-full flex items-center justify-center font-medium animate-pulse">
                  {notificationCount > 9 ? '9+' : notificationCount}
                </span>
              )}
            </button>

            {/* Notification Dropdown */}
            {isNotificationOpen && (
              <div className="absolute right-0 mt-2 w-80 bg-[#111827] border border-gray-800/50 rounded-2xl shadow-xl z-50 backdrop-blur-sm">
                <div className="p-4 border-b border-gray-800/50">
                  <h3 className="text-white font-semibold">Notifications</h3>
                </div>
                <div className="max-h-64 overflow-y-auto">
                  {isLoadingNotifications ? (
                    <div className="p-8 text-center">
                      <div className="w-8 h-8 border-2 border-blue-500/30 border-t-blue-500 rounded-full animate-spin mx-auto"></div>
                      <p className="text-gray-400 text-sm mt-2">Loading notifications...</p>
                    </div>
                  ) : notifications.length === 0 ? (
                    <div className="p-8 text-center">
                      <svg className="w-12 h-12 mx-auto text-gray-600 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
                      </svg>
                      <p className="text-gray-400 text-sm">No new notifications</p>
                    </div>
                  ) : (
                    notifications.map((notification) => (
                      <div key={notification.id} className="p-4 border-b border-gray-800/30 hover:bg-gray-800/30 transition-colors duration-200 cursor-pointer">
                        <div className="flex items-start space-x-3">
                          <div className={`w-2 h-2 rounded-full mt-2 ${
                            notification.type === 'alert' ? 'bg-red-400' :
                            notification.type === 'warning' ? 'bg-yellow-400' :
                            'bg-blue-400'
                          }`}></div>
                          <div className="flex-1">
                            <p className="text-white text-sm font-medium">{notification.title}</p>
                            <p className="text-gray-400 text-xs mt-1">{notification.time}</p>
                          </div>
                        </div>
                      </div>
                    ))
                  )}
                </div>
                <div className="p-4">
                  <button 
                    onClick={handleViewAllNotifications}
                    className="w-full text-blue-400 text-sm font-medium hover:text-blue-300 transition-colors duration-200"
                  >
                    View All Notifications
                  </button>
                </div>
              </div>
            )}
          </div>
          
          {/* User Profile */}
          <div className="relative" ref={profileRef}>
            <button 
              onClick={() => setIsProfileOpen(!isProfileOpen)}
              className="flex items-center space-x-3 hover:bg-gray-800/50 rounded-xl p-2 transition-colors duration-200"
            >
              <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-cyan-400 rounded-full flex items-center justify-center shadow-lg">
                <span className="text-white text-sm font-bold">{userName.charAt(0)}</span>
              </div>
              <div className="text-right hidden sm:block">
                <p className="text-white text-sm font-medium">{userName}</p>
                <p className="text-gray-400 text-xs">{userRole}</p>
              </div>
              <svg className={`w-4 h-4 text-gray-400 transition-transform duration-200 ${isProfileOpen ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>

            {/* Profile Dropdown */}
            {isProfileOpen && (
              <div className="absolute right-0 mt-2 w-64 bg-[#111827] border border-gray-800/50 rounded-2xl shadow-xl z-50 backdrop-blur-sm">
                {/* Profile Header */}
                <div className="p-4 border-b border-gray-800/50">
                  <div className="flex items-center space-x-3">
                    <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-cyan-400 rounded-full flex items-center justify-center">
                      <span className="text-white font-bold">{userName.charAt(0)}</span>
                    </div>
                    <div>
                      <p className="text-white font-semibold">{userName}</p>
                      <p className="text-gray-400 text-sm">{userRole}</p>
                    </div>
                  </div>
                </div>

                {/* Profile Menu Items */}
                <div className="py-2">
                  <button 
                    onClick={() => {
                      setIsProfileOpen(false);
                      onMenuClick && onMenuClick('My Profile');
                    }}
                    className="w-full flex items-center space-x-3 px-4 py-3 text-gray-300 hover:text-white hover:bg-gray-800/50 transition-colors duration-200"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                    </svg>
                    <span className="text-sm font-medium">My Profile</span>
                  </button>
                  
                  <button 
                    onClick={() => {
                      setIsProfileOpen(false);
                      onMenuClick && onMenuClick('Help & Support');
                    }}
                    className="w-full flex items-center space-x-3 px-4 py-3 text-gray-300 hover:text-white hover:bg-gray-800/50 transition-colors duration-200"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <span className="text-sm font-medium">Help & Support</span>
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  );
};

export default TopNavbar;