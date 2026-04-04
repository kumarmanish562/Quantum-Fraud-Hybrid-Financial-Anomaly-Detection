import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

const Login = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({ email: '', password: '' });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [isDark, setIsDark] = useState(() => {
    const saved = localStorage.getItem('theme');
    return saved ? saved === 'dark' : true;
  });

  useEffect(() => {
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
  }, [isDark]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      const response = await fetch('http://localhost:8000/api/v1/user-auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });

      const data = await response.json();

      if (response.ok) {
        localStorage.setItem('access_token', data.access_token);
        localStorage.setItem('user', JSON.stringify(data.user));
        navigate('/dashboard');
      } else {
        setError(data.detail || 'Login failed');
      }
    } catch (err) {
      setError('Network error. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={`min-h-screen flex items-center justify-center px-6 ${isDark ? 'bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900' : 'bg-gray-50'}`}>
      {/* Theme Toggle */}
      <button
        onClick={() => setIsDark(!isDark)}
        className={`fixed top-6 right-6 p-3 rounded-xl transition ${isDark ? 'bg-white text-yellow-400 hover:bg-gray-700' : 'bg-white text-gray-700 hover:bg-gray-100 shadow-lg'}`}
      >
        {isDark ? '☀️' : '🌙'}
      </button>

      <div className="max-w-md w-full">
        <div className="text-center mb-8">
          <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-cyan-400 rounded-2xl flex items-center justify-center mx-auto mb-4">
            <span className="text-gray-900 font-bold text-2xl">Q</span>
          </div>
          <h2 className={`text-3xl font-bold mb-2 ${isDark ? 'text-gray-900' : 'text-gray-900'}`}>Welcome Back</h2>
          <p className={isDark ? 'text-gray-500' : 'text-gray-600'}>Login to access your API dashboard</p>
        </div>

        <div className={`backdrop-blur-sm rounded-2xl p-8 ${isDark ? 'bg-gray-100 border border-gray-300' : 'bg-white border border-gray-200 shadow-xl'}`}>
          {error && (
            <div className="bg-red-500/20 border border-red-500 text-red-400 px-4 py-3 rounded-lg mb-6">
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className={`block text-sm font-medium mb-2 ${isDark ? 'text-gray-600' : 'text-gray-700'}`}>Email</label>
              <input
                type="email"
                required
                value={formData.email}
                onChange={(e) => setFormData({...formData, email: e.target.value})}
                className={`w-full rounded-lg px-4 py-3 focus:outline-none focus:border-blue-500 ${isDark ? 'bg-gray-50/50 border border-gray-600 text-gray-900' : 'bg-gray-50 border border-gray-300 text-gray-900'}`}
                placeholder="john@company.com"
              />
            </div>

            <div>
              <label className={`block text-sm font-medium mb-2 ${isDark ? 'text-gray-600' : 'text-gray-700'}`}>Password</label>
              <input
                type="password"
                required
                value={formData.password}
                onChange={(e) => setFormData({...formData, password: e.target.value})}
                className={`w-full rounded-lg px-4 py-3 focus:outline-none focus:border-blue-500 ${isDark ? 'bg-gray-50/50 border border-gray-600 text-gray-900' : 'bg-gray-50 border border-gray-300 text-gray-900'}`}
                placeholder="••••••••"
              />
            </div>

            <div className="flex items-center justify-between">
              <label className="flex items-center">
                <input type="checkbox" className="mr-2" />
                <span className={`text-sm ${isDark ? 'text-gray-500' : 'text-gray-600'}`}>Remember me</span>
              </label>
              <button type="button" onClick={() => navigate('/forgot-password')} className="text-blue-400 text-sm hover:text-blue-300">
                Forgot Password?
              </button>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full bg-gradient-to-r from-blue-600 to-cyan-500 text-white py-3 rounded-lg font-semibold hover:shadow-lg transition disabled:opacity-50"
            >
              {loading ? 'Logging in...' : 'Login'}
            </button>
          </form>

          <div className="mt-6 text-center">
            <p className={`text-sm ${isDark ? 'text-gray-500' : 'text-gray-600'}`}>
              Don't have an account?{' '}
              <button onClick={() => navigate('/register')} className="text-blue-400 hover:text-blue-300">
                Sign up
              </button>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Login;
