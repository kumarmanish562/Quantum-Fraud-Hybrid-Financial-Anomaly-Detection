import React, { useState, useEffect } from 'react';

const APIKeys = () => {
  const [user, setUser] = useState(null);
  const [apiKeys, setApiKeys] = useState([]);
  const [loading, setLoading] = useState(true);
  const [copiedKey, setCopiedKey] = useState('');
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newKeyName, setNewKeyName] = useState('');
  const [newlyCreatedKey, setNewlyCreatedKey] = useState(null);
  const [showNewKeyModal, setShowNewKeyModal] = useState(false);

  useEffect(() => {
    fetchAPIKeys();
  }, []);

  const fetchAPIKeys = async () => {
    const token = localStorage.getItem('access_token');
    if (!token) return;

    try {
      const response = await fetch(`http://localhost:8000/api/v1/user-auth/profile?token=${token}`);
      const data = await response.json();

      if (response.ok) {
        setUser(data.user);
        setApiKeys(data.api_keys);
      }
    } catch (err) {
      console.error('Failed to fetch API keys', err);
    } finally {
      setLoading(false);
    }
  };

  const createNewKey = async () => {
    if (!newKeyName.trim()) {
      alert('Please enter a key name');
      return;
    }

    const token = localStorage.getItem('access_token');
    
    try {
      const response = await fetch(`http://localhost:8000/api/v1/user-auth/api-keys/create?token=${token}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ name: newKeyName })
      });

      const data = await response.json();

      if (response.ok) {
        setNewlyCreatedKey(data.api_key);
        setShowCreateModal(false);
        setShowNewKeyModal(true);
        setNewKeyName('');
        fetchAPIKeys();
      }
    } catch (err) {
      alert('Failed to create API key');
    }
  };

  const deleteKey = async (keyId) => {
    if (!confirm('Are you sure you want to delete this API key? This action cannot be undone.')) {
      return;
    }

    const token = localStorage.getItem('access_token');
    
    try {
      const response = await fetch(`http://localhost:8000/api/v1/user-auth/api-keys/delete/${keyId}?token=${token}`, {
        method: 'DELETE'
      });

      if (response.ok) {
        fetchAPIKeys();
        alert('API Key deleted successfully!');
      }
    } catch (err) {
      alert('Failed to delete key');
    }
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
    setCopiedKey(text);
    setTimeout(() => setCopiedKey(''), 2000);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-white">Loading...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-white mb-2">API Keys Management</h1>
        <p className="text-gray-400">Manage your API keys for fraud detection integration</p>
      </div>

      {/* User Info Card */}
      <div className="bg-[#111827]/60 backdrop-blur-sm border border-gray-800/50 rounded-2xl p-6">
        <div className="flex items-center space-x-4">
          <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-cyan-400 rounded-full flex items-center justify-center">
            <span className="text-white text-2xl font-bold">{user?.full_name?.charAt(0)}</span>
          </div>
          <div>
            <h2 className="text-xl font-bold text-white">{user?.full_name}</h2>
            <p className="text-gray-400">{user?.email}</p>
            {user?.company_name && <p className="text-gray-500 text-sm">{user?.company_name}</p>}
          </div>
        </div>
      </div>

      {/* API Keys List */}
      <div className="bg-[#111827]/60 backdrop-blur-sm border border-gray-800/50 rounded-2xl p-8">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold text-white">Your API Keys</h2>
          <button 
            onClick={() => setShowCreateModal(true)}
            className="bg-gradient-to-r from-blue-600 to-cyan-500 text-white px-4 py-2 rounded-lg hover:shadow-lg transition"
          >
            Create New Key
          </button>
        </div>

        <div className="space-y-4">
          {apiKeys.map((key) => (
            <div key={key.id} className="bg-gray-900/50 border border-gray-700 rounded-xl p-6">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h3 className="text-white font-semibold mb-1">{key.name}</h3>
                  <p className="text-gray-400 text-sm">Created: {new Date(key.created_at).toLocaleDateString()}</p>
                </div>
                <div className={`px-3 py-1 rounded-full text-sm ${key.is_active ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'}`}>
                  {key.is_active ? 'Active' : 'Inactive'}
                </div>
              </div>

              <div className="bg-gray-800 rounded-lg p-4 mb-4">
                <div className="flex items-center justify-between">
                  <code className="text-gray-500 text-sm font-mono">qfd_••••••••••••••••••••••••••••••••</code>
                  <div className="flex items-center space-x-3">
                    <button
                      onClick={() => copyToClipboard(key.api_key)}
                      className="text-gray-400 hover:text-white transition flex items-center space-x-1"
                    >
                      {copiedKey === key.api_key ? (
                        <>
                          <svg className="w-4 h-4 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                          </svg>
                          <span className="text-green-400 text-xs">Copied!</span>
                        </>
                      ) : (
                        <>
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                          </svg>
                          <span className="text-xs">Copy</span>
                        </>
                      )}
                    </button>
                  </div>
                </div>
              </div>

              <div className="flex items-center justify-between mb-2">
                <div className="text-sm text-gray-400">
                  Usage: {key.requests_count} / {key.requests_limit} requests
                </div>
                <button
                  onClick={() => deleteKey(key.id)}
                  className="text-red-400 hover:text-red-300 text-sm font-medium"
                >
                  Delete Key
                </button>
              </div>

              <div className="w-full bg-gray-700 rounded-full h-2">
                <div
                  className="bg-gradient-to-r from-blue-500 to-cyan-400 h-2 rounded-full transition-all"
                  style={{ width: `${(key.requests_count / key.requests_limit) * 100}%` }}
                ></div>
              </div>
            </div>
          ))}
        </div>

        {/* API Documentation */}
        <div className="mt-8 bg-blue-500/10 border border-blue-500/30 rounded-xl p-6">
          <h3 className="text-white font-semibold mb-3 flex items-center space-x-2">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span>Quick Start Guide</span>
          </h3>
          <p className="text-gray-300 text-sm mb-4">Use your API key to make fraud detection requests:</p>
          <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto">
            <code className="text-green-400 text-sm whitespace-pre">
{`curl -X POST http://localhost:8000/api/v1/fraud/predict \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "transaction_id": "txn_123",
    "amount": 1000,
    "time": "2024-01-01T10:00:00",
    "v1": 0.5, "v2": -0.3, ...
  }'`}
            </code>
          </div>
        </div>
      </div>

      {/* Create Key Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-[#1F2937] border border-gray-700 rounded-2xl p-8 max-w-md w-full mx-4">
            <h3 className="text-2xl font-bold text-white mb-4">Create New API Key</h3>
            <p className="text-gray-400 text-sm mb-6">Enter a name for your new API key</p>
            
            <input
              type="text"
              value={newKeyName}
              onChange={(e) => setNewKeyName(e.target.value)}
              placeholder="e.g., Production Key"
              className="w-full bg-gray-800 border border-gray-600 rounded-lg px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 mb-6"
            />

            <div className="flex space-x-3">
              <button
                onClick={() => {
                  setShowCreateModal(false);
                  setNewKeyName('');
                }}
                className="flex-1 bg-gray-700 text-white px-4 py-3 rounded-lg hover:bg-gray-600 transition"
              >
                Cancel
              </button>
              <button
                onClick={createNewKey}
                className="flex-1 bg-gradient-to-r from-blue-600 to-cyan-500 text-white px-4 py-3 rounded-lg hover:shadow-lg transition"
              >
                Create Key
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Show New Key Modal */}
      {showNewKeyModal && newlyCreatedKey && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-[#1F2937] border border-gray-700 rounded-2xl p-8 max-w-lg w-full mx-4">
            <div className="flex items-center space-x-3 mb-4">
              <div className="w-12 h-12 bg-green-500/20 rounded-full flex items-center justify-center">
                <svg className="w-6 h-6 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <h3 className="text-2xl font-bold text-white">API Key Created!</h3>
            </div>
            
            <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4 mb-6">
              <p className="text-yellow-400 text-sm font-medium mb-2">⚠️ Important: Save this key now!</p>
              <p className="text-gray-400 text-xs">This is the only time you'll see this key. Store it securely.</p>
            </div>

            <div className="bg-gray-800 rounded-lg p-4 mb-6">
              <div className="flex items-center justify-between">
                <code className="text-blue-400 text-sm font-mono break-all flex-1">{newlyCreatedKey}</code>
                <button
                  onClick={() => copyToClipboard(newlyCreatedKey)}
                  className="ml-4 text-gray-400 hover:text-white transition flex-shrink-0"
                >
                  {copiedKey === newlyCreatedKey ? (
                    <svg className="w-5 h-5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                  ) : (
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                    </svg>
                  )}
                </button>
              </div>
            </div>

            <button
              onClick={() => {
                setShowNewKeyModal(false);
                setNewlyCreatedKey(null);
              }}
              className="w-full bg-gradient-to-r from-blue-600 to-cyan-500 text-white px-4 py-3 rounded-lg hover:shadow-lg transition"
            >
              I've Saved My Key
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default APIKeys;
