/**
 * API Service Layer
 * Connects frontend to the FastAPI backend
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const API_V1 = `${API_BASE_URL}/api/v1`;

// Helper function for API calls
async function apiCall(endpoint, options = {}) {
  const defaultOptions = {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  };

  // Add auth token if available
  const token = localStorage.getItem('auth_token');
  if (token) {
    defaultOptions.headers['Authorization'] = `Bearer ${token}`;
  }

  try {
    const response = await fetch(`${API_V1}${endpoint}`, {
      ...defaultOptions,
      ...options,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
      
      // Don't log 401 errors to console (authentication failures are expected)
      if (response.status !== 401) {
        console.error(`API call failed: ${endpoint}`, error);
      }
      
      throw new Error(error.detail || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    // Only log non-authentication errors
    if (!error.message.includes('Not authenticated') && !error.message.includes('401')) {
      console.error(`API call failed: ${endpoint}`, error);
    }
    throw error;
  }
}

// Authentication API
export const authAPI = {
  login: async (username, password) => {
    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);

    const response = await fetch(`${API_V1}/auth/login`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error('Login failed');
    }

    const data = await response.json();
    localStorage.setItem('auth_token', data.access_token);
    return data;
  },

  register: async (userData) => {
    return apiCall('/auth/register', {
      method: 'POST',
      body: JSON.stringify(userData),
    });
  },

  getCurrentUser: async () => {
    return apiCall('/auth/me');
  },

  logout: () => {
    localStorage.removeItem('auth_token');
  },
};

// Fraud Detection API
export const fraudAPI = {
  predictSingle: async (transactionData) => {
    return apiCall('/fraud/predict', {
      method: 'POST',
      body: JSON.stringify(transactionData),
    });
  },

  predictBatch: async (transactions) => {
    return apiCall('/fraud/predict/batch', {
      method: 'POST',
      body: JSON.stringify({ transactions }),
    });
  },

  predictRealtime: async (transactionData) => {
    return apiCall('/fraud/predict/realtime', {
      method: 'POST',
      body: JSON.stringify(transactionData),
    });
  },

  getModelStatus: async () => {
    return apiCall('/fraud/models/status');
  },
};

// Transactions API
export const transactionsAPI = {
  getAll: async (params = {}) => {
    const queryString = new URLSearchParams(params).toString();
    return apiCall(`/transactions/?${queryString}`);
  },

  getById: async (id) => {
    return apiCall(`/transactions/${id}`);
  },

  create: async (transactionData) => {
    return apiCall('/transactions/', {
      method: 'POST',
      body: JSON.stringify(transactionData),
    });
  },

  getStats: async () => {
    return apiCall('/transactions/stats/summary');
  },
};

// Analytics API
export const analyticsAPI = {
  getDashboard: async () => {
    return apiCall('/analytics/dashboard');
  },

  getFraudTrends: async (period = '7d') => {
    return apiCall(`/analytics/fraud-trends?period=${period}`);
  },

  getModelPerformance: async () => {
    return apiCall('/analytics/model-performance');
  },

  getRealTimeMetrics: async () => {
    return apiCall('/analytics/real-time/metrics');
  },

  getRiskDistribution: async () => {
    return apiCall('/analytics/risk-distribution');
  },

  getAlertsSummary: async () => {
    return apiCall('/analytics/alerts/summary');
  },

  getTransactionPatterns: async (patternType = 'amount', days = 7) => {
    return apiCall(`/analytics/transaction-patterns?pattern_type=${patternType}&days=${days}`);
  },
};

// WebSocket Connection for Real-time Updates
export class WebSocketService {
  constructor(clientId) {
    this.clientId = clientId || `client_${Date.now()}`;
    this.ws = null;
    this.listeners = new Map();
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
  }

  connect() {
    try {
      const wsUrl = API_BASE_URL.replace('http', 'ws');
      this.ws = new WebSocket(`${wsUrl}/ws/${this.clientId}`);

      this.ws.onopen = () => {
        this.reconnectAttempts = 0;
        this.emit('connected');
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.emit('message', data);
          
          // Emit specific event types
          if (data.type) {
            this.emit(data.type, data.data);
          }
        } catch (error) {
          // Silently ignore parse errors
        }
      };

      this.ws.onerror = () => {
        // Silently handle all WebSocket errors
        this.emit('error', null);
      };

      this.ws.onclose = () => {
        this.emit('disconnected');
        
        // Attempt to reconnect with exponential backoff
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
          this.reconnectAttempts++;
          const delay = Math.min(30000, 3000 * Math.pow(2, this.reconnectAttempts - 1));
          setTimeout(() => this.connect(), delay);
        }
      };
    } catch (error) {
      // Silently handle connection errors
      this.emit('error', null);
    }
  }

  disconnect() {
    try {
      // Clear any pending reconnection attempts first
      this.reconnectAttempts = this.maxReconnectAttempts;
      
      if (this.ws) {
        // Remove all event handlers before closing
        this.ws.onopen = null;
        this.ws.onmessage = null;
        this.ws.onerror = null;
        this.ws.onclose = null;
        
        // Only close if connection is open or connecting
        if (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING) {
          this.ws.close();
        }
        this.ws = null;
      }
    } catch (error) {
      // Silently handle disconnect errors
    }
  }

  send(data) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    } else {
      console.warn('WebSocket is not connected');
    }
  }

  on(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event).push(callback);
  }

  off(event, callback) {
    if (this.listeners.has(event)) {
      const callbacks = this.listeners.get(event);
      const index = callbacks.indexOf(callback);
      if (index > -1) {
        callbacks.splice(index, 1);
      }
    }
  }

  emit(event, data) {
    if (this.listeners.has(event)) {
      this.listeners.get(event).forEach(callback => callback(data));
    }
  }
}

// Health Check
export const healthCheck = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    return response.ok;
  } catch (error) {
    return false;
  }
};

export default {
  authAPI,
  fraudAPI,
  transactionsAPI,
  analyticsAPI,
  WebSocketService,
  healthCheck,
};
