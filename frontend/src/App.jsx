import React, { useState } from 'react'
import Layout from './components/Layout'
import Dashboard from './components/Dashboard'
import Transactions from './components/Transactions'
// import RealTimeDetection from './components/RealTimeDetection'
import Analytics from './components/Analytics'
import Alerts from './components/Alerts'
import Settings from './components/Settings'
import SecurityStatus from './components/SecurityStatus'
// import ReportGeneration from './components/ReportGeneration'
import MyProfile from './components/MyProfile'
import HelpSupport from './components/HelpSupport'
import UIShowcase from './components/UIShowcase'

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
      // case 'Real-Time Detection':
      //   return <RealTimeDetection />;
      case 'Analytics':
        return <Analytics />;
      case 'Alerts':
        return <Alerts />;
      case 'Security Status':
        return <SecurityStatus />;
      // case 'Report Generation':
      //   return <ReportGeneration />;
      case 'Settings':
        return <Settings />;
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