import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

const VerifyOTP = () => {
  const navigate = useNavigate();
  const [otp, setOtp] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [email, setEmail] = useState('');
  const [showOTP, setShowOTP] = useState('');
  const [isDark, setIsDark] = useState(() => {
    const saved = localStorage.getItem('theme');
    return saved ? saved === 'dark' : true;
  });

  useEffect(() => {
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
  }, [isDark]);

  useEffect(() => {
    const tempEmail = localStorage.getItem('temp_email');
    const tempOTP = localStorage.getItem('temp_otp');
    if (!tempEmail) {
      navigate('/register');
    }
    setEmail(tempEmail);
    setShowOTP(tempOTP); // For demo - remove in production
  }, [navigate]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      const response = await fetch('http://localhost:8000/api/v1/user-auth/verify-otp', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, otp })
      });

      const data = await response.json();

      if (response.ok) {
        localStorage.setItem('access_token', data.access_token);
        localStorage.setItem('user', JSON.stringify(data.user));
        localStorage.setItem('api_key', data.api_key);
        localStorage.removeItem('temp_email');
        localStorage.removeItem('temp_otp');
        navigate('/dashboard');
      } else {
        setError(data.detail || 'Invalid OTP');
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
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
            </svg>
          </div>
          <h2 className={`text-3xl font-bold mb-2 ${isDark ? 'text-gray-900' : 'text-gray-900'}`}>Verify Your Email</h2>
          <p className={isDark ? 'text-gray-500' : 'text-gray-600'}>Enter the 6-digit code sent to</p>
          <p className="text-blue-400 font-medium">{email}</p>
        </div>

        <div className={`backdrop-blur-sm rounded-2xl p-8 ${isDark ? 'bg-gray-100 border border-gray-300' : 'bg-white border border-gray-200 shadow-xl'}`}>
          {showOTP && (
            <div className="bg-blue-500/20 border border-blue-500 text-blue-400 px-4 py-3 rounded-lg mb-6 text-center">
              <div className="text-sm mb-1">Demo OTP (Remove in production):</div>
              <div className="text-2xl font-bold">{showOTP}</div>
            </div>
          )}

          {error && (
            <div className="bg-red-500/20 border border-red-500 text-red-400 px-4 py-3 rounded-lg mb-6">
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label className={`block text-sm font-medium mb-2 text-center ${isDark ? 'text-gray-600' : 'text-gray-700'}`}>Enter OTP</label>
              <input
                type="text"
                required
                maxLength="6"
                value={otp}
                onChange={(e) => setOtp(e.target.value.replace(/\D/g, ''))}
                className={`w-full rounded-lg px-4 py-4 text-center text-2xl tracking-widest focus:outline-none focus:border-blue-500 ${isDark ? 'bg-gray-50/50 border border-gray-600 text-gray-900' : 'bg-gray-50 border border-gray-300 text-gray-900'}`}
                placeholder="000000"
              />
            </div>

            <button
              type="submit"
              disabled={loading || otp.length !== 6}
              className="w-full bg-gradient-to-r from-blue-600 to-cyan-500 text-white py-3 rounded-lg font-semibold hover:shadow-lg transition disabled:opacity-50"
            >
              {loading ? 'Verifying...' : 'Verify Email'}
            </button>
          </form>

          <div className="mt-6 text-center">
            <p className={`text-sm ${isDark ? 'text-gray-500' : 'text-gray-600'}`}>
              Didn't receive code?{' '}
              <button className="text-blue-400 hover:text-blue-300">
                Resend
              </button>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default VerifyOTP;
