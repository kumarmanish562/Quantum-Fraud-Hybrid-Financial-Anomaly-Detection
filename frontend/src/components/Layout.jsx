import React from 'react';
import Sidebar from './Sidebar';
import { TopNavbar } from './ui';

const Layout = ({ children, activeMenu, onMenuClick }) => {
  return (
    <div className="min-h-screen bg-[#0B0F1A]">
      <Sidebar onMenuClick={onMenuClick} activeMenu={activeMenu} />
      
      {/* Main Content Area */}
      <div className="ml-64">
        {/* Top Navigation Bar */}
        <TopNavbar 
          title={activeMenu || 'Dashboard'} 
          notificationCount={3}
          userName="Admin User"
          userRole="Security Analyst"
          onMenuClick={onMenuClick}
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