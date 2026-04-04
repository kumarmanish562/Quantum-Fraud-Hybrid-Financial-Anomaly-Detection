import { useState, useEffect } from 'react';
import Sidebar from './Sidebar';
import { TopNavbar } from './ui';

const Layout = ({ children, activeMenu, onMenuClick }) => {
  const [isDark, setIsDark] = useState(() => {
    const saved = localStorage.getItem('theme');
    return saved ? saved === 'dark' : true;
  });

  useEffect(() => {
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
  }, [isDark]);

  return (
    <div className={`min-h-screen ${isDark ? 'bg-gray-50' : 'bg-gray-50'}`}>
      <Sidebar onMenuClick={onMenuClick} activeMenu={activeMenu} isDark={isDark} setIsDark={setIsDark} />
      
      {/* Main Content Area */}
      <div className="ml-64">
        {/* Top Navigation Bar */}
        <TopNavbar 
          title={activeMenu || 'Dashboard'} 
          notificationCount={3}
          userName="Admin User"
          userRole="Security Analyst"
          onMenuClick={onMenuClick}
          isDark={isDark}
        />
        
        {/* Main Content */}
        <main className="p-6">
          {children}
        </main>
      </div>
    </div>
  );
};

export default Layout;