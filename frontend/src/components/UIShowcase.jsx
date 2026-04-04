import React from 'react';
import { Card, StatCard, AlertCard, ModelBadge, AIStatusBadge, NeuralActivityBadge } from './ui';

const UIShowcase = () => {
  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 mb-2">UI Components Showcase</h1>
        <p className="text-gray-500">Demonstration of reusable UI components for the fraud detection system</p>
      </div>

      {/* Model Badges */}
      <div className="space-y-4">
        <h2 className="text-xl font-semibold text-gray-900">AI Model Badges</h2>
        <div className="flex flex-wrap items-center gap-4">
          <ModelBadge modelType="Classical" size="sm" />
          <ModelBadge modelType="Classical" size="md" />
          <ModelBadge modelType="Classical" size="lg" />
        </div>
        <div className="flex flex-wrap items-center gap-4">
          <ModelBadge modelType="Quantum" size="sm" animated={true} />
          <ModelBadge modelType="Quantum" size="md" animated={true} />
          <ModelBadge modelType="Quantum" size="lg" animated={true} />
        </div>
      </div>

      {/* AI Status Badges */}
      <div className="space-y-4">
        <h2 className="text-xl font-semibold text-gray-900">AI Status Indicators</h2>
        <div className="flex flex-wrap items-center gap-4">
          <AIStatusBadge status="online" processingTime={847} />
          <AIStatusBadge status="processing" confidence={94} />
          <AIStatusBadge status="offline" />
          <NeuralActivityBadge activity="active" nodeCount={2048} />
        </div>
      </div>

      {/* Cards */}
      <div className="space-y-4">
        <h2 className="text-xl font-semibold text-gray-900">Card Components</h2>
        
        {/* Basic Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <Card>
            <h3 className="text-gray-900 font-semibold mb-2">Basic Card</h3>
            <p className="text-gray-500">This is a basic card with glassmorphism effect, rounded corners, and hover animation.</p>
          </Card>
          
          <Card hover={false} className="border-blue-500/30">
            <h3 className="text-gray-900 font-semibold mb-2">No Hover Card</h3>
            <p className="text-gray-500">This card has hover effects disabled and custom border color.</p>
          </Card>
          
          <Card padding="p-8" background="bg-gradient-to-br from-blue-500/10 to-purple-500/10">
            <h3 className="text-gray-900 font-semibold mb-2">Custom Card</h3>
            <p className="text-gray-500">Custom padding and gradient background.</p>
          </Card>
        </div>

        {/* Stat Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <StatCard
            title="Total Transactions"
            value="12,450"
            icon={
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            }
            change="+12.5%"
            changeType="positive"
          />
          
          <StatCard
            title="Fraud Detected"
            value="42"
            icon={
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
            }
            change="-8.2%"
            changeType="positive"
          />
          
          <StatCard
            title="Prevention Rate"
            value="99.7%"
            icon={
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
              </svg>
            }
            change="+0.3%"
            changeType="positive"
          />
          
          <StatCard
            title="Processing Time"
            value="847ms"
            icon={
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            }
            change="-15.4%"
            changeType="positive"
          />
        </div>

        {/* Alert Cards */}
        <div className="space-y-4">
          <AlertCard
            severity="High"
            title="Suspicious Pattern Detected"
            description="Multiple high-value transactions detected from unverified location"
            time="2 minutes ago"
            transactionId="#TRX-89234"
            amount="$25,890.00"
          />
          
          <AlertCard
            severity="Medium"
            title="Velocity Check Alert"
            description="Transaction frequency exceeds normal user behavior"
            time="8 minutes ago"
            transactionId="#TRX-89221"
            amount="$3,450.75"
          />
          
          <AlertCard
            severity="Low"
            title="Device Mismatch"
            description="Transaction initiated from unrecognized device"
            time="32 minutes ago"
            transactionId="#TRX-89187"
            amount="$890.25"
          />
        </div>
      </div>
    </div>
  );
};

export default UIShowcase;