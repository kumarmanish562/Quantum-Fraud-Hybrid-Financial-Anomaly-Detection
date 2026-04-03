import React from 'react';
import { useNavigate } from 'react-router-dom';

const Footer = ({ isDark }) => {
  const navigate = useNavigate();

  return (
    <footer className={`border-t py-12 px-6 ${isDark ? 'bg-gray-900 border-gray-800' : 'bg-gray-50 border-gray-200'}`}>
      <div className="max-w-7xl mx-auto">
        <div className="grid md:grid-cols-4 gap-8 mb-8">
          {/* Brand */}
          <div>
            <div className="flex items-center space-x-2 mb-4">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-cyan-400 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-xl">Q</span>
              </div>
              <span className={`font-bold text-lg ${isDark ? 'text-white' : 'text-gray-900'}`}>Quantum Fraud Detection</span>
            </div>
            <p className={`text-sm mb-4 ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
              Enterprise-grade fraud detection powered by quantum computing and AI.
            </p>
            <div className="flex space-x-3">
              <a href="https://github.com" target="_blank" rel="noopener noreferrer" className={`w-8 h-8 rounded-lg flex items-center justify-center transition ${isDark ? 'bg-gray-800 hover:bg-gray-700' : 'bg-white border border-gray-300 hover:bg-gray-100'}`}>
                <svg className={`w-5 h-5 ${isDark ? 'text-gray-400' : 'text-gray-600'}`} fill="currentColor" viewBox="0 0 24 24">
                  <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                </svg>
              </a>
              <a href="https://linkedin.com" target="_blank" rel="noopener noreferrer" className={`w-8 h-8 rounded-lg flex items-center justify-center transition ${isDark ? 'bg-gray-800 hover:bg-gray-700' : 'bg-white border border-gray-300 hover:bg-gray-100'}`}>
                <svg className={`w-5 h-5 ${isDark ? 'text-gray-400' : 'text-gray-600'}`} fill="currentColor" viewBox="0 0 24 24">
                  <path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/>
                </svg>
              </a>
            </div>
          </div>

          {/* Product */}
          <div>
            <h4 className={`font-semibold mb-4 ${isDark ? 'text-white' : 'text-gray-900'}`}>Product</h4>
            <ul className={`space-y-2 text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
              <li><button onClick={() => navigate('/')} className={`transition ${isDark ? 'hover:text-white' : 'hover:text-gray-900'}`}>Features</button></li>
              <li><button onClick={() => navigate('/dashboard')} className={`transition ${isDark ? 'hover:text-white' : 'hover:text-gray-900'}`}>Dashboard</button></li>
              <li><button onClick={() => navigate('/')} className={`transition ${isDark ? 'hover:text-white' : 'hover:text-gray-900'}`}>API Documentation</button></li>
              <li><button className={`transition ${isDark ? 'hover:text-white' : 'hover:text-gray-900'}`}>Pricing</button></li>
            </ul>
          </div>

          {/* Company */}
          <div>
            <h4 className={`font-semibold mb-4 ${isDark ? 'text-white' : 'text-gray-900'}`}>Company</h4>
            <ul className={`space-y-2 text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
              <li><button className={`transition ${isDark ? 'hover:text-white' : 'hover:text-gray-900'}`}>About Us</button></li>
              <li><button className={`transition ${isDark ? 'hover:text-white' : 'hover:text-gray-900'}`}>Team</button></li>
              <li><button className={`transition ${isDark ? 'hover:text-white' : 'hover:text-gray-900'}`}>Careers</button></li>
              <li><button className={`transition ${isDark ? 'hover:text-white' : 'hover:text-gray-900'}`}>Contact</button></li>
            </ul>
          </div>

          {/* Resources */}
          <div>
            <h4 className={`font-semibold mb-4 ${isDark ? 'text-white' : 'text-gray-900'}`}>Resources</h4>
            <ul className={`space-y-2 text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
              <li><button className={`transition ${isDark ? 'hover:text-white' : 'hover:text-gray-900'}`}>Documentation</button></li>
              <li><button className={`transition ${isDark ? 'hover:text-white' : 'hover:text-gray-900'}`}>Blog</button></li>
              <li><button className={`transition ${isDark ? 'hover:text-white' : 'hover:text-gray-900'}`}>Support</button></li>
              <li><button className={`transition ${isDark ? 'hover:text-white' : 'hover:text-gray-900'}`}>Status</button></li>
            </ul>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className={`border-t pt-8 flex flex-col md:flex-row items-center justify-between text-sm ${isDark ? 'border-gray-800' : 'border-gray-200'}`}>
          <div className={`mb-4 md:mb-0 ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
            © 2024 Quantum Fraud Detection System. All rights reserved.
          </div>
          <div className={`flex space-x-6 ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
            <button className={`transition ${isDark ? 'hover:text-white' : 'hover:text-gray-900'}`}>Privacy Policy</button>
            <button className={`transition ${isDark ? 'hover:text-white' : 'hover:text-gray-900'}`}>Terms of Service</button>
            <button className={`transition ${isDark ? 'hover:text-white' : 'hover:text-gray-900'}`}>Security</button>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
