import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

const ForgotPassword = () => {
  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);
  const [otpSent, setOtpSent] = useState('');
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
      const response = await fetch('http://localhost:8000/api/v1/user-auth/forgot-password', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email })
      });

      const data = await response.json();

      if (response.ok) {
        setSuccess(true);
        setOtpSent(data.otp); // For demo
        localStorage.setItem('reset_email', email);
        setTimeout(() => navigate('/reset-password'), 2000);
      } else {
        setError(data.detail || 'Failed to send OTP');
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
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z" />
            </svg>
          </div>
          <h2 className={`text-3xl font-bold mb-2 ${isDark ? 'text-gray-900' : 'text-gray-900'}`}>Forgot Password?</h2>
          <p className={isDark ? 'text-gray-500' : 'text-gray-600'}>Enter your email to receive OTP</p>
        </div>

        <div className={`backdrop-blur-sm rounded-2xl p-8 ${isDark ? 'bg-gray-100 border border-gray-300' : 'bg-white border border-gray-200 shadow-xl'}`}>
          {success && (
            <div className="bg-green-500/20 border border-green-500 text-green-400 px-4 py-3 rounded-lg mb-6">
              <div className="font-medium mb-1">OTP sent successfully!</div>
              <div className="text-sm">OTP: {otpSent}</div>
              <div className="text-sm mt-1">Redirecting to reset password...</div>
            </div>
          )}

          {error && (
            <div className="bg-red-500/20 border border-red-500 text-red-400 px-4 py-3 rounded-lg mb-6">
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className={`block text-sm font-medium mb-2 ${isDark ? 'text-gray-600' : 'text-gray-700'}`}>Email Address</label>
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className={`w-full rounded-lg px-4 py-3 focus:outline-none focus:border-blue-500 ${isDark ? 'bg-gray-50/50 border border-gray-600 text-gray-900' : 'bg-gray-50 border border-gray-300 text-gray-900'}`}
                placeholder="john@company.com"
              />
            </div>

            <button
              type="submit"
              disabled={loading || success}
              className="w-full bg-gradient-to-r from-blue-600 to-cyan-500 text-white py-3 rounded-lg font-semibold hover:shadow-lg transition disabled:opacity-50"
            >
              {loading ? 'Sending OTP...' : 'Send OTP'}
            </button>
          </form>

          <div className="mt-6 text-center">
            <button onClick={() => navigate('/login')} className="text-blue-400 hover:text-blue-300 text-sm">
              ← Back to Login
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ForgotPassword;
