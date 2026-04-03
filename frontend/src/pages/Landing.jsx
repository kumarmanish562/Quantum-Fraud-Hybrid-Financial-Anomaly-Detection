import React, { useState } from 'react';
import Navbar from '../components/landing/Navbar';
import Hero from '../components/landing/Hero';
import Features from '../components/landing/Features';
import HowItWorks from '../components/landing/HowItWorks';
import About from '../components/landing/About';
import DashboardPreview from '../components/landing/DashboardPreview';
import Footer from '../components/landing/Footer';

const Landing = () => {
  const [isDark, setIsDark] = useState(true);

  return (
    <div className={`min-h-screen ${isDark ? 'bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900' : 'bg-white'}`}>
      <Navbar isDark={isDark} setIsDark={setIsDark} />
      <Hero isDark={isDark} />
      <Features isDark={isDark} />
      <HowItWorks isDark={isDark} />
      <About isDark={isDark} />
      <DashboardPreview isDark={isDark} />
      <Footer isDark={isDark} />
    </div>
  );
};

export default Landing;
