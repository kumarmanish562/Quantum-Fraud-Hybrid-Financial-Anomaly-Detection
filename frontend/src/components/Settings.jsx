import React, { useState, useEffect, useCallback } from 'react';
import { fraudAPI, analyticsAPI } from '../services/api';

const Settings = () => {
  const [settings, setSettings] = useState({
    // Model Settings
    quantumModel: true,
    confidenceThreshold: 85,
    
    // API Settings
    apiKey: '',  // User should configure their own API key
    secretToken: '',  // User should configure their own secret token
    webhookUrl: 'https://api.example.com/webhook',
    
    // Notification Settings
    autoBlock: true,
    realTimeMonitoring: true,
    emailNotifications: true,
    smsAlerts: false
  });

  const [modelStatus, setModelStatus] = useState(null);
  const [systemMetrics, setSystemMetrics] = useState(null);
  const [hasChanges, setHasChanges] = useState(false);
  const [showApiKey, setShowApiKey] = useState(false);
  const [showSecretToken, setShowSecretToken] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState(null);

  // Fetch model status and system metrics
  const fetchSystemData = useCallback(async () => {
    try {
      // Fetch model status
      const status = await fraudAPI.getModelStatus();
      setModelStatus(status);
      
      // Update quantum model setting based on actual status
      if (status && status.models && status.models.hybrid) {
        setSettings(prev => ({
          ...prev,
          quantumModel: status.models.hybrid.loaded || false
        }));
      }

      // Fetch real-time metrics
      const metrics = await analyticsAPI.getRealTimeMetrics();
      setSystemMetrics(metrics);
      
    } catch (error) {
      console.error('Failed to fetch system data:', error);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Initial data fetch
  useEffect(() => {
    fetchSystemData();
  }, [fetchSystemData]);

  // Auto-refresh every 30 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      fetchSystemData();
    }, 30000);

    return () => clearInterval(interval);
  }, [fetchSystemData]);

  const handleSettingChange = (key, value) => {
    setSettings(prev => ({
      ...prev,
      [key]: value
    }));
    setHasChanges(true);
    setSaveMessage(null);
  };

  const handleSave = async () => {
    setIsSaving(true);
    setSaveMessage(null);
    
    try {
      // TODO: Implement settings save API endpoint
      console.log('Saving settings:', settings);
      
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setHasChanges(false);
      setSaveMessage({ type: 'success', text: 'Settings saved successfully!' });
      
      // Refresh system data after save
      fetchSystemData();
      
    } catch (error) {
      console.error('Failed to save settings:', error);
      setSaveMessage({ type: 'error', text: 'Failed to save settings. Please try again.' });
    } finally {
      setIsSaving(false);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-blue-500/30 border-t-blue-500 rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-900 font-medium">Loading settings...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Save Message */}
      {saveMessage && (
        <div className={`${
          saveMessage.type === 'success' 
            ? 'bg-green-500/20 border-green-500/30 text-green-400' 
            : 'bg-red-500/20 border-red-500/30 text-red-400'
        } border rounded-xl px-4 py-3`}>
          <div className="flex items-center space-x-2">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              {saveMessage.type === 'success' ? (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              ) : (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              )}
            </svg>
            <span className="text-sm font-medium">{saveMessage.text}</span>
          </div>
        </div>
      )}

      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 mb-2">System Settings</h1>
        <p className="text-gray-500">Configure your fraud detection system parameters and integrations</p>
      </div>

      {/* System Status */}
      {modelStatus && (
        <div className="bg-white/60 backdrop-blur-sm border border-gray-200 rounded-2xl p-6">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-gray-900 font-medium mb-1">System Status</h3>
              <p className="text-gray-500 text-sm">Real-time system health monitoring</p>
            </div>
            <div className="flex items-center space-x-4">
              {modelStatus.models && modelStatus.models.hybrid && modelStatus.models.hybrid.loaded && (
                <div className="flex items-center space-x-2 px-4 py-2 bg-purple-500/20 border border-purple-500/30 rounded-full">
                  <div className="w-2 h-2 bg-purple-400 rounded-full animate-pulse"></div>
                  <span className="text-purple-400 text-sm font-medium">Quantum Model Active</span>
                </div>
              )}
              <div className="flex items-center space-x-2 px-4 py-2 bg-green-500/20 border border-green-500/30 rounded-full">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                <span className="text-green-400 text-sm font-medium">System Online</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Quantum Model Settings */}
      <div className="bg-white/60 backdrop-blur-sm border border-gray-200 rounded-2xl p-8">
        <div className="flex items-start justify-between mb-6">
          <div>
            <h2 className="text-xl font-semibold text-gray-900 mb-2">Quantum Model</h2>
            <p className="text-gray-500 text-sm">Enable quantum-enhanced fraud detection algorithms</p>
          </div>
          <div className="flex items-center space-x-4">
            <span className={`text-sm font-medium ${settings.quantumModel ? 'text-green-400' : 'text-gray-500'}`}>
              {settings.quantumModel ? 'Enabled' : 'Disabled'}
            </span>
            <button
              onClick={() => handleSettingChange('quantumModel', !settings.quantumModel)}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors duration-200 ${
                settings.quantumModel ? 'bg-blue-500' : 'bg-gray-600'
              }`}
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform duration-200 ${
                  settings.quantumModel ? 'translate-x-6' : 'translate-x-1'
                }`}
              />
            </button>
          </div>
        </div>
      </div>

      {/* Confidence Threshold Settings */}
      <div className="bg-white/60 backdrop-blur-sm border border-gray-200 rounded-2xl p-8">
        <div className="flex items-start justify-between mb-6">
          <div>
            <h2 className="text-xl font-semibold text-gray-900 mb-2">Detection Threshold</h2>
            <p className="text-gray-500 text-sm">Set the confidence level for fraud detection</p>
          </div>
        </div>
        
        <div>
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-gray-900 font-medium">Confidence Level</h3>
            <span className="text-2xl font-bold text-gray-900">{settings.confidenceThreshold}%</span>
          </div>
          <p className="text-gray-500 text-sm mb-6">
            Transactions with fraud probability above this threshold will be flagged
          </p>
          
          <div className="relative">
            <input
              type="range"
              min="0"
              max="100"
              value={settings.confidenceThreshold}
              onChange={(e) => handleSettingChange('confidenceThreshold', parseInt(e.target.value))}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
              style={{
                background: `linear-gradient(to right, #3B82F6 0%, #3B82F6 ${settings.confidenceThreshold}%, #374151 ${settings.confidenceThreshold}%, #374151 100%)`
              }}
            />
            <div 
              className="absolute top-0 w-4 h-4 bg-blue-500 rounded-full transform -translate-y-1 -translate-x-2 shadow-lg"
              style={{ left: `${settings.confidenceThreshold}%` }}
            ></div>
          </div>
          
          <div className="flex justify-between text-xs text-gray-500 mt-2">
            <span>Low (0%)</span>
            <span>Medium (50%)</span>
            <span>High (100%)</span>
          </div>
        </div>
      </div>

      {/* System Preferences */}
      <div className="bg-white/60 backdrop-blur-sm border border-gray-200 rounded-2xl p-8">
        <div className="flex items-start justify-between mb-6">
          <div>
            <h2 className="text-xl font-semibold text-gray-900 mb-2">System Preferences</h2>
            <p className="text-gray-500 text-sm">Configure monitoring and notification settings</p>
          </div>
        </div>

        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-gray-900 font-medium">Auto-Block Suspicious Transactions</h3>
              <p className="text-gray-500 text-sm">Automatically block high-risk transactions</p>
            </div>
            <button
              onClick={() => handleSettingChange('autoBlock', !settings.autoBlock)}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors duration-200 ${
                settings.autoBlock ? 'bg-blue-500' : 'bg-gray-600'
              }`}
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform duration-200 ${
                  settings.autoBlock ? 'translate-x-6' : 'translate-x-1'
                }`}
              />
            </button>
          </div>

          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-gray-900 font-medium">Real-Time Monitoring</h3>
              <p className="text-gray-500 text-sm">Enable continuous transaction monitoring</p>
            </div>
            <button
              onClick={() => handleSettingChange('realTimeMonitoring', !settings.realTimeMonitoring)}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors duration-200 ${
                settings.realTimeMonitoring ? 'bg-blue-500' : 'bg-gray-600'
              }`}
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform duration-200 ${
                  settings.realTimeMonitoring ? 'translate-x-6' : 'translate-x-1'
                }`}
              />
            </button>
          </div>

          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-gray-900 font-medium">Email Notifications</h3>
              <p className="text-gray-500 text-sm">Receive fraud alerts via email</p>
            </div>
            <button
              onClick={() => handleSettingChange('emailNotifications', !settings.emailNotifications)}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors duration-200 ${
                settings.emailNotifications ? 'bg-blue-500' : 'bg-gray-600'
              }`}
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform duration-200 ${
                  settings.emailNotifications ? 'translate-x-6' : 'translate-x-1'
                }`}
              />
            </button>
          </div>

          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-gray-900 font-medium">SMS Alerts</h3>
              <p className="text-gray-500 text-sm">Receive critical alerts via SMS</p>
            </div>
            <button
              onClick={() => handleSettingChange('smsAlerts', !settings.smsAlerts)}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors duration-200 ${
                settings.smsAlerts ? 'bg-blue-500' : 'bg-gray-600'
              }`}
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform duration-200 ${
                  settings.smsAlerts ? 'translate-x-6' : 'translate-x-1'
                }`}
              />
            </button>
          </div>
        </div>
      </div>

      {/* Save Button */}
      <div className="flex items-center justify-end space-x-4">
        {hasChanges && (
          <span className="text-yellow-400 text-sm flex items-center space-x-2">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            <span>You have unsaved changes</span>
          </span>
        )}
        <button
          onClick={handleSave}
          disabled={!hasChanges || isSaving}
          className={`px-8 py-3 rounded-xl font-semibold transition-all duration-200 ${
            hasChanges && !isSaving
              ? 'bg-gradient-to-r from-blue-600 to-cyan-500 text-white hover:from-blue-700 hover:to-cyan-600 shadow-lg hover:shadow-blue-500/25'
              : 'bg-gray-700 text-gray-500 cursor-not-allowed'
          }`}
        >
          {isSaving ? (
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
              <span>Saving...</span>
            </div>
          ) : (
            'Save Changes'
          )}
        </button>
      </div>
    </div>
  );
};

export default Settings;
