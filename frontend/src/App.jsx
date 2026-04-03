import React, { useState } from 'react'
import Layout from './components/Layout'
import Dashboard from './components/Dashboard'
import Transactions from './components/Transactions'
import Analytics from './components/Analytics'
import Alerts from './components/Alerts'
import Settings from './components/Settings'
import SecurityStatus from './components/SecurityStatus'
import MyProfile from './components/MyProfile'
import HelpSupport from './components/HelpSupport'
import UIShowcase from './components/UIShowcase'
import APIKeys from './components/APIKeys'

const App = () => {
  const [activeMenu, setActiveMenu] = useState('Dashboard');

  const handleMenuClick = (menuName) => {
    setActiveMenu(menuName);
  };

  const renderContent = () => {
    switch (activeMenu) {
      case 'Dashboard':
        return <Dashboard />;
      case 'Transactions':
        return <Transactions />;
      case 'Analytics':
        return <Analytics />;
      case 'Alerts':
        return <Alerts />;
      case 'Security Status':
        return <SecurityStatus />;
      case 'Settings':
        return <Settings />;
      case 'API Keys':
        return <APIKeys />;
      case 'My Profile':
        return <MyProfile />;
      case 'Help & Support':
        return <HelpSupport />;
      case 'UI Showcase':
        return <UIShowcase />;
      default:
        return <Dashboard />;
    }
  };

  return (
    <Layout activeMenu={activeMenu} onMenuClick={handleMenuClick}>
      {renderContent()}
    </Layout>
  )
}

export default App