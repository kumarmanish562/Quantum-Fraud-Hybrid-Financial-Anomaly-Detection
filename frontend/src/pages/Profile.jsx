import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

const Profile = () => {
  const navigate = useNavigate();
  const [user, setUser] = useState(null);
  const [apiKeys, setApiKeys] = useState([]);
  const [loading, setLoading] = useState(true);
  const [copiedKey, setCopiedKey] = useState('');

  useEffect(() => {
    fetchProfile();
  }, []);

  const fetchProfile = async () => {
    const token = localStorage.getItem('access_token');
    if (!token) {
      navigate('/login');
      return;
    }

    try {
      const response = await fetch(`http://localhost:8000/api/v1/user-auth/profile?token=${token}`);
      const data = await response.json();

      if (response.ok) {
        setUser(data.user);
        setApiKeys(data.api_keys);
      } else {
        navigate('/login');
      }
    } catch (err) {
      console.error('Failed to fetch profile', err);
    } finally {
      setLoading(false);
    }
  };

  const regenerateKey = async (keyId) => {
    const token = localStorage.getItem('access_token');
    
    try {
      const response = await fetch(`http://localhost:8000/api/v1/user-auth/api-keys/regenerate/${keyId}?token=${token}`, {
        method: 'POST'
      });

      if (response.ok) {
        fetchProfile();
        alert('API Key regenerated successfully!');
      }
    } catch (err) {
      alert('Failed to regenerate key');
    }
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
    setCopiedKey(text);
    setTimeout(() => setCopiedKey(''), 2000);
  };

  const logout = () => {
    localStorage.clear();
    navigate('/');
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900 flex items-center justify-center">
        <div className="text-gray-900 text-xl">Loading...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900">
      {/* Navbar */}
      <nav className="bg-gray-50/80 backdrop-blur-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-cyan-400 rounded-lg flex items-center justify-center">
                <span className="text-gray-900 font-bold text-xl">Q</span>
              </div>
              <span className="text-gray-900 text-xl font-bold">Quantum Fraud API</span>
            </div>
            <button onClick={logout} className="text-gray-600 hover:text-gray-900 transition">
              Logout
            </button>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-6 py-12">
        {/* User Info */}
        <div className="bg-gray-100 backdrop-blur-sm border border-gray-300 rounded-2xl p-8 mb-8">
          <div className="flex items-center space-x-4 mb-6">
            <div className="w-20 h-20 bg-gradient-to-br from-blue-500 to-cyan-400 rounded-full flex items-center justify-center">
              <span className="text-gray-900 text-3xl font-bold">{user?.full_name?.charAt(0)}</span>
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900">{user?.full_name}</h1>
              <p className="text-gray-500">{user?.email}</p>
              {user?.company_name && <p className="text-gray-500 text-sm">{user?.company_name}</p>}
            </div>
          </div>

          <div className="grid md:grid-cols-3 gap-6">
            <div className="bg-gray-50/50 rounded-xl p-4">
              <div className="text-gray-500 text-sm mb-1">Account Status</div>
              <div className="text-green-400 font-semibold">Active</div>
            </div>
            <div className="bg-gray-50/50 rounded-xl p-4">
              <div className="text-gray-500 text-sm mb-1">Member Since</div>
              <div className="text-gray-900 font-semibold">{new Date(user?.created_at).toLocaleDateString()}</div>
            </div>
            <div className="bg-gray-50/50 rounded-xl p-4">
              <div className="text-gray-500 text-sm mb-1">API Keys</div>
              <div className="text-gray-900 font-semibold">{apiKeys.length}</div>
            </div>
          </div>
        </div>

        {/* API Keys */}
        <div className="bg-gray-100 backdrop-blur-sm border border-gray-300 rounded-2xl p-8">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-gray-900">API Keys</h2>
            <button className="bg-gradient-to-r from-blue-600 to-cyan-500 text-white px-4 py-2 rounded-lg hover:shadow-lg transition">
              Create New Key
            </button>
          </div>

          <div className="space-y-4">
            {apiKeys.map((key) => (
              <div key={key.id} className="bg-gray-50/50 border border-gray-300 rounded-xl p-6">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h3 className="text-gray-900 font-semibold mb-1">{key.name}</h3>
                    <p className="text-gray-500 text-sm">Created: {new Date(key.created_at).toLocaleDateString()}</p>
                  </div>
                  <div className={`px-3 py-1 rounded-full text-sm ${key.is_active ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'}`}>
                    {key.is_active ? 'Active' : 'Inactive'}
                  </div>
                </div>

                <div className="bg-white rounded-lg p-4 mb-4">
                  <div className="flex items-center justify-between">
                    <code className="text-blue-400 text-sm font-mono">{key.api_key}</code>
                    <button
                      onClick={() => copyToClipboard(key.api_key)}
                      className="text-gray-500 hover:text-gray-900 transition"
                    >
                      {copiedKey === key.api_key ? (
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

                <div className="flex items-center justify-between">
                  <div className="text-sm text-gray-500">
                    Usage: {key.requests_count} / {key.requests_limit} requests
                  </div>
                  <button
                    onClick={() => regenerateKey(key.id)}
                    className="text-blue-400 hover:text-blue-300 text-sm font-medium"
                  >
                    Regenerate Key
                  </button>
                </div>

                <div className="mt-2">
                  <div className="w-full bg-gray-700 rounded-full h-2">
                    <div
                      className="bg-gradient-to-r from-blue-500 to-cyan-400 h-2 rounded-full"
                      style={{ width: `${(key.requests_count / key.requests_limit) * 100}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* API Documentation */}
          <div className="mt-8 bg-blue-500/10 border border-blue-500/30 rounded-xl p-6">
            <h3 className="text-gray-900 font-semibold mb-3">Quick Start</h3>
            <p className="text-gray-600 text-sm mb-4">Use your API key to make fraud detection requests:</p>
            <div className="bg-gray-50 rounded-lg p-4 overflow-x-auto">
              <code className="text-green-400 text-sm">
                curl -X POST https://api.quantumfraud.com/v1/detect \<br/>
                &nbsp;&nbsp;-H "Authorization: Bearer YOUR_API_KEY" \<br/>
                &nbsp;&nbsp;-H "Content-Type: application/json" \<br/>
                &nbsp;&nbsp;-d '{`{"transaction_id": "txn_123", "amount": 1000}`}'
              </code>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Profile;
