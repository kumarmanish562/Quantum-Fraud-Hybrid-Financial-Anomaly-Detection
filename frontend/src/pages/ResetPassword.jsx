import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

const ResetPassword = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({ otp: '', new_password: '', confirm_password: '' });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [email, setEmail] = useState('');
  const [isDark, setIsDark] = useState(() => {
    const saved = localStorage.getItem('theme');
    return saved ? saved === 'dark' : true;
  });

  useEffect(() => {
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
  }, [isDark]);

  useEffect(() => {
    const resetEmail = localStorage.getItem('reset_email');
    if (!resetEmail) {
      navigate('/forgot-password');
    }
    setEmail(resetEmail);
  }, [navigate]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    if (formData.new_password !== formData.confirm_password) {
      setError('Passwords do not match');
      setLoading(false);
      return;
    }

    try {
      const response = await fetch('http://localhost:8000/api/v1/user-auth/reset-password', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email,
          otp: formData.otp,
          new_password: formData.new_password
        })
      });

      const data = await response.json();

      if (response.ok) {
        localStorage.removeItem('reset_email');
        alert('Password reset successful!');
        navigate('/login');
      } else {
        setError(data.detail || 'Failed to reset password');
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
            <svg className="w-8 h-8 text-gray-900" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
            </svg>
          </div>
          <h2 className={`text-3xl font-bold mb-2 ${isDark ? 'text-gray-900' : 'text-gray-900'}`}>Reset Password</h2>
          <p className={isDark ? 'text-gray-500' : 'text-gray-600'}>Enter OTP and new password</p>
        </div>

        <div className={`backdrop-blur-sm rounded-2xl p-8 ${isDark ? 'bg-gray-100 border border-gray-300' : 'bg-white border border-gray-200 shadow-xl'}`}>
          {error && (
            <div className="bg-red-500/20 border border-red-500 text-red-400 px-4 py-3 rounded-lg mb-6">
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className={`block text-sm font-medium mb-2 ${isDark ? 'text-gray-600' : 'text-gray-700'}`}>OTP Code</label>
              <input
                type="text"
                required
                maxLength="6"
                value={formData.otp}
                onChange={(e) => setFormData({...formData, otp: e.target.value.replace(/\D/g, '')})}
                className={`w-full rounded-lg px-4 py-3 focus:outline-none focus:border-blue-500 ${isDark ? 'bg-gray-50/50 border border-gray-600 text-gray-900' : 'bg-gray-50 border border-gray-300 text-gray-900'}`}
                placeholder="000000"
              />
            </div>

            <div>
              <label className={`block text-sm font-medium mb-2 ${isDark ? 'text-gray-600' : 'text-gray-700'}`}>New Password</label>
              <input
                type="password"
                required
                value={formData.new_password}
                onChange={(e) => setFormData({...formData, new_password: e.target.value})}
                className={`w-full rounded-lg px-4 py-3 focus:outline-none focus:border-blue-500 ${isDark ? 'bg-gray-50/50 border border-gray-600 text-gray-900' : 'bg-gray-50 border border-gray-300 text-gray-900'}`}
                placeholder="••••••••"
              />
            </div>

            <div>
              <label className={`block text-sm font-medium mb-2 ${isDark ? 'text-gray-600' : 'text-gray-700'}`}>Confirm Password</label>
              <input
                type="password"
                required
                value={formData.confirm_password}
                onChange={(e) => setFormData({...formData, confirm_password: e.target.value})}
                className={`w-full rounded-lg px-4 py-3 focus:outline-none focus:border-blue-500 ${isDark ? 'bg-gray-50/50 border border-gray-600 text-gray-900' : 'bg-gray-50 border border-gray-300 text-gray-900'}`}
                placeholder="••••••••"
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full bg-gradient-to-r from-blue-600 to-cyan-500 text-white py-3 rounded-lg font-semibold hover:shadow-lg transition disabled:opacity-50"
            >
              {loading ? 'Resetting...' : 'Reset Password'}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};

export default ResetPassword;
