import React, { useState, useEffect, useCallback } from 'react';
import { Card } from './ui';
import { authAPI, analyticsAPI, transactionsAPI } from '../services/api';

const MyProfile = () => {
  const [profileData, setProfileData] = useState({
    firstName: '',
    lastName: '',
    email: '',
    phone: '',
    role: '',
    department: '',
    location: '',
    timezone: '',
    joinDate: '',
    lastLogin: ''
  });

  const [activityStats, setActivityStats] = useState({
    reportsGenerated: 0,
    transactionsReviewed: 0,
    alertsHandled: 0,
    daysActive: 0
  });

  const [isEditing, setIsEditing] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch user profile data
  const fetchProfileData = useCallback(async () => {
    // Check if user is authenticated before making the call
    const token = localStorage.getItem('auth_token');
    
    if (!token) {
      // No token, skip API call and use default data
      setProfileData({
        firstName: 'Security',
        lastName: 'Analyst',
        email: 'analyst@frauddetection.com',
        phone: '+91 (555) 123-4567',
        role: 'Security Analyst',
        department: 'Fraud Detection',
        location: 'India',
        timezone: 'IST (UTC+5:30)',
        joinDate: new Date().toISOString(),
        lastLogin: new Date().toISOString()
      });
      return;
    }

    try {
      setError(null);
      
      // Try to fetch current user info
      const userData = await authAPI.getCurrentUser();
      
      setProfileData({
        firstName: userData.first_name || userData.username?.split('.')[0] || 'User',
        lastName: userData.last_name || userData.username?.split('.')[1] || '',
        email: userData.email || 'user@example.com',
        phone: userData.phone || '+91 (000) 000-0000',
        role: userData.role || 'Security Analyst',
        department: userData.department || 'Fraud Detection',
        location: userData.location || 'India',
        timezone: userData.timezone || 'IST (UTC+5:30)',
        joinDate: userData.created_at || new Date().toISOString(),
        lastLogin: userData.last_login || new Date().toISOString()
      });
      
    } catch (err) {
      // Silently handle any errors and use default data
      setProfileData({
        firstName: 'Security',
        lastName: 'Analyst',
        email: 'analyst@frauddetection.com',
        phone: '+91 (555) 123-4567',
        role: 'Security Analyst',
        department: 'Fraud Detection',
        location: 'India',
        timezone: 'IST (UTC+5:30)',
        joinDate: new Date().toISOString(),
        lastLogin: new Date().toISOString()
      });
    }
  }, []);

  // Fetch activity statistics
  const fetchActivityStats = useCallback(async () => {
    try {
      // Fetch transaction stats
      const transactionStats = await transactionsAPI.getStats();
      
      // Fetch alerts summary
      const alertsData = await analyticsAPI.getDashboard();
      
      // Calculate days active (from join date to now)
      const joinDate = new Date(profileData.joinDate || Date.now());
      const today = new Date();
      const daysActive = Math.floor((today - joinDate) / (1000 * 60 * 60 * 24));
      
      setActivityStats({
        reportsGenerated: alertsData.total_reports || 0,
        transactionsReviewed: transactionStats.total_transactions || 0,
        alertsHandled: alertsData.total_alerts || 0,
        daysActive: daysActive > 0 ? daysActive : 1
      });
      
    } catch (err) {
      console.error('Failed to fetch activity stats:', err);
      // Keep existing stats or use zeros
    } finally {
      setIsLoading(false);
    }
  }, [profileData.joinDate]);

  // Initial data fetch
  useEffect(() => {
    fetchProfileData();
  }, [fetchProfileData]);

  // Fetch activity stats after profile is loaded
  useEffect(() => {
    if (profileData.joinDate) {
      fetchActivityStats();
    }
  }, [profileData.joinDate, fetchActivityStats]);

  // Auto-refresh every 30 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      fetchActivityStats();
    }, 30000);

    return () => clearInterval(interval);
  }, [fetchActivityStats]);

  const handleSave = async () => {
    try {
      // TODO: Implement profile update API call
      setIsEditing(false);
      // In production, call API to update profile
      // await authAPI.updateProfile(profileData);
    } catch (err) {
      setError('Failed to save profile changes');
    }
  };

  const handleInputChange = (field, value) => {
    setProfileData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-blue-500/30 border-t-blue-500 rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-900 font-medium">Loading profile...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Error Alert */}
      {error && (
        <div className="bg-red-500/20 border border-red-500/30 rounded-xl px-4 py-3">
          <div className="flex items-center space-x-2 text-red-400">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            <span className="text-sm font-medium">{error}</span>
          </div>
        </div>
      )}

      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 mb-2">My Profile</h1>
          <p className="text-gray-500">Manage your personal information and preferences</p>
        </div>
        <button
          onClick={() => setIsEditing(!isEditing)}
          className="px-6 py-3 bg-blue-600 text-white rounded-xl font-medium hover:bg-blue-700 transition-colors duration-200"
        >
          {isEditing ? 'Cancel' : 'Edit Profile'}
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Profile Picture & Basic Info */}
        <Card className="text-center">
          <div className="w-32 h-32 bg-gradient-to-br from-blue-500 to-cyan-400 rounded-full flex items-center justify-center mx-auto mb-6">
            <span className="text-gray-900 text-4xl font-bold">
              {profileData.firstName.charAt(0)}{profileData.lastName.charAt(0)}
            </span>
          </div>
          <h2 className="text-2xl font-bold text-gray-900 mb-2">
            {profileData.firstName} {profileData.lastName}
          </h2>
          <p className="text-blue-400 font-medium mb-1">{profileData.role}</p>
          <p className="text-gray-500 text-sm mb-6">{profileData.department}</p>
          
          <div className="space-y-3 text-sm">
            <div className="flex items-center justify-between">
              <span className="text-gray-500">Member since</span>
              <span className="text-gray-900">{new Date(profileData.joinDate).toLocaleDateString()}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-500">Last login</span>
              <span className="text-gray-900">{profileData.lastLogin}</span>
            </div>
          </div>
        </Card>

        {/* Personal Information */}
        <Card className="lg:col-span-2">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-xl font-semibold text-gray-900">Personal Information</h3>
            {isEditing && (
              <button
                onClick={handleSave}
                className="px-4 py-2 bg-green-600 text-white rounded-lg font-medium hover:bg-green-700 transition-colors duration-200"
              >
                Save Changes
              </button>
            )}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-gray-600 text-sm font-medium mb-2">First Name</label>
              {isEditing ? (
                <input
                  type="text"
                  value={profileData.firstName}
                  onChange={(e) => handleInputChange('firstName', e.target.value)}
                  className="w-full bg-gray-100 border border-gray-300 rounded-xl px-4 py-3 text-gray-900 focus:outline-none focus:border-blue-500/50"
                />
              ) : (
                <p className="text-gray-900 bg-white/30 px-4 py-3 rounded-xl">{profileData.firstName}</p>
              )}
            </div>

            <div>
              <label className="block text-gray-600 text-sm font-medium mb-2">Last Name</label>
              {isEditing ? (
                <input
                  type="text"
                  value={profileData.lastName}
                  onChange={(e) => handleInputChange('lastName', e.target.value)}
                  className="w-full bg-gray-100 border border-gray-300 rounded-xl px-4 py-3 text-gray-900 focus:outline-none focus:border-blue-500/50"
                />
              ) : (
                <p className="text-gray-900 bg-white/30 px-4 py-3 rounded-xl">{profileData.lastName}</p>
              )}
            </div>

            <div>
              <label className="block text-gray-600 text-sm font-medium mb-2">Email Address</label>
              {isEditing ? (
                <input
                  type="email"
                  value={profileData.email}
                  onChange={(e) => handleInputChange('email', e.target.value)}
                  className="w-full bg-gray-100 border border-gray-300 rounded-xl px-4 py-3 text-gray-900 focus:outline-none focus:border-blue-500/50"
                />
              ) : (
                <p className="text-gray-900 bg-white/30 px-4 py-3 rounded-xl">{profileData.email}</p>
              )}
            </div>

            <div>
              <label className="block text-gray-600 text-sm font-medium mb-2">Phone Number</label>
              {isEditing ? (
                <input
                  type="tel"
                  value={profileData.phone}
                  onChange={(e) => handleInputChange('phone', e.target.value)}
                  className="w-full bg-gray-100 border border-gray-300 rounded-xl px-4 py-3 text-gray-900 focus:outline-none focus:border-blue-500/50"
                />
              ) : (
                <p className="text-gray-900 bg-white/30 px-4 py-3 rounded-xl">{profileData.phone}</p>
              )}
            </div>

            <div>
              <label className="block text-gray-600 text-sm font-medium mb-2">Role</label>
              <p className="text-gray-900 bg-white/30 px-4 py-3 rounded-xl">{profileData.role}</p>
            </div>

            <div>
              <label className="block text-gray-600 text-sm font-medium mb-2">Department</label>
              <p className="text-gray-900 bg-white/30 px-4 py-3 rounded-xl">{profileData.department}</p>
            </div>

            <div>
              <label className="block text-gray-600 text-sm font-medium mb-2">Location</label>
              {isEditing ? (
                <input
                  type="text"
                  value={profileData.location}
                  onChange={(e) => handleInputChange('location', e.target.value)}
                  className="w-full bg-gray-100 border border-gray-300 rounded-xl px-4 py-3 text-gray-900 focus:outline-none focus:border-blue-500/50"
                />
              ) : (
                <p className="text-gray-900 bg-white/30 px-4 py-3 rounded-xl">{profileData.location}</p>
              )}
            </div>

            <div>
              <label className="block text-gray-600 text-sm font-medium mb-2">Timezone</label>
              {isEditing ? (
                <select
                  value={profileData.timezone}
                  onChange={(e) => handleInputChange('timezone', e.target.value)}
                  className="w-full bg-gray-100 border border-gray-300 rounded-xl px-4 py-3 text-gray-900 focus:outline-none focus:border-blue-500/50"
                >
                  <option value="IST (UTC+5:30)">IST (UTC+5:30)</option>
                  <option value="EST (UTC-5)">EST (UTC-5)</option>
                  <option value="PST (UTC-8)">PST (UTC-8)</option>
                  <option value="GMT (UTC+0)">GMT (UTC+0)</option>
                  <option value="CET (UTC+1)">CET (UTC+1)</option>
                </select>
              ) : (
                <p className="text-gray-900 bg-white/30 px-4 py-3 rounded-xl">{profileData.timezone}</p>
              )}
            </div>
          </div>
        </Card>
      </div>

      {/* Activity Summary */}
      <Card>
        <h3 className="text-xl font-semibold text-gray-900 mb-6">Activity Summary</h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="text-center">
            <div className="text-3xl font-bold text-blue-400 mb-2">
              {activityStats.reportsGenerated.toLocaleString()}
            </div>
            <div className="text-gray-500 text-sm">Reports Generated</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-green-400 mb-2">
              {activityStats.transactionsReviewed.toLocaleString()}
            </div>
            <div className="text-gray-500 text-sm">Transactions Reviewed</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-yellow-400 mb-2">
              {activityStats.alertsHandled.toLocaleString()}
            </div>
            <div className="text-gray-500 text-sm">Alerts Handled</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-purple-400 mb-2">
              {activityStats.daysActive.toLocaleString()}
            </div>
            <div className="text-gray-500 text-sm">Days Active</div>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default MyProfile;