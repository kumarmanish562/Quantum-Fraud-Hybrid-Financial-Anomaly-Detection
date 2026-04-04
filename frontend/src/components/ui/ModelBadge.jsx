import React from 'react';

const ModelBadge = ({ 
  modelType = 'Classical', 
  size = 'md', 
  showIcon = true,
  animated = true,
  className = '' 
}) => {
  const isQuantum = modelType === 'Quantum';
  
  const sizeClasses = {
    sm: 'px-2 py-1 text-xs',
    md: 'px-3 py-2 text-sm',
    lg: 'px-4 py-3 text-base'
  };

  const iconSizes = {
    sm: 'w-3 h-3',
    md: 'w-4 h-4',
    lg: 'w-5 h-5'
  };

  const baseClasses = `
    inline-flex 
    items-center 
    space-x-2 
    rounded-full 
    font-medium 
    uppercase 
    tracking-wider
    transition-all 
    duration-300
    ${sizeClasses[size]}
  `;

  if (isQuantum) {
    return (
      <div className={`
        ${baseClasses}
        bg-gradient-to-r from-cyan-500/20 to-purple-500/20 
        border border-cyan-500/30 
        text-cyan-400
        ${animated ? 'animate-pulse shadow-lg shadow-cyan-500/25' : ''}
        ${className}
      `}>
        {showIcon && (
          <div className="relative">
            <svg className={`${iconSizes[size]} text-cyan-400`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            {animated && (
              <div className="absolute inset-0 animate-ping">
                <svg className={`${iconSizes[size]} text-cyan-400 opacity-75`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
            )}
          </div>
        )}
        <span className="relative">
          Quantum Model Activated
          {animated && (
            <div className="absolute inset-0 bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent animate-pulse">
              Quantum Model Activated
            </div>
          )}
        </span>
      </div>
    );
  }

  return (
    <div className={`
      ${baseClasses}
      bg-blue-500/20 
      border border-blue-500/30 
      text-blue-400
      hover:bg-blue-500/30
      hover:shadow-lg 
      hover:shadow-blue-500/20
      ${className}
    `}>
      {showIcon && (
        <svg className={`${iconSizes[size]} text-blue-400`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
        </svg>
      )}
      <span>Classical Model</span>
    </div>
  );
};

// Futuristic AI Status Badge
export const AIStatusBadge = ({ 
  status = 'online', 
  processingTime = null,
  confidence = null,
  className = '' 
}) => {
  const getStatusConfig = (status) => {
    switch (status) {
      case 'online':
        return {
          bgColor: 'bg-gradient-to-r from-green-500/20 to-emerald-500/20',
          borderColor: 'border-green-500/30',
          textColor: 'text-green-400',
          dotColor: 'bg-green-400',
          label: 'AI System Online'
        };
      case 'processing':
        return {
          bgColor: 'bg-gradient-to-r from-blue-500/20 to-cyan-500/20',
          borderColor: 'border-blue-500/30',
          textColor: 'text-blue-400',
          dotColor: 'bg-blue-400',
          label: 'Processing'
        };
      case 'offline':
        return {
          bgColor: 'bg-gradient-to-r from-red-500/20 to-orange-500/20',
          borderColor: 'border-red-500/30',
          textColor: 'text-red-400',
          dotColor: 'bg-red-400',
          label: 'System Offline'
        };
      default:
        return {
          bgColor: 'bg-gradient-to-r from-gray-500/20 to-gray-600/20',
          borderColor: 'border-gray-500/30',
          textColor: 'text-gray-500',
          dotColor: 'bg-gray-400',
          label: 'Unknown Status'
        };
    }
  };

  const config = getStatusConfig(status);

  return (
    <div className={`
      inline-flex 
      items-center 
      space-x-2 
      px-3 
      py-2 
      ${config.bgColor}
      border 
      ${config.borderColor}
      rounded-full
      ${config.textColor}
      text-sm 
      font-medium
      ${className}
    `}>
      <div className={`w-2 h-2 ${config.dotColor} rounded-full animate-pulse`}></div>
      <span>{config.label}</span>
      {processingTime && (
        <span className="text-xs opacity-75">({processingTime}ms)</span>
      )}
      {confidence && (
        <span className="text-xs opacity-75">{confidence}%</span>
      )}
    </div>
  );
};

// Neural Network Activity Indicator
export const NeuralActivityBadge = ({ 
  activity = 'active',
  nodeCount = 2048,
  className = '' 
}) => {
  return (
    <div className={`
      inline-flex 
      items-center 
      space-x-3 
      px-4 
      py-2 
      bg-gradient-to-r from-purple-500/10 to-pink-500/10
      border border-purple-500/20
      rounded-full
      text-purple-400
      text-sm 
      font-medium
      ${className}
    `}>
      <div className="flex items-center space-x-1">
        {[...Array(3)].map((_, i) => (
          <div
            key={i}
            className={`w-1 h-4 bg-purple-400 rounded-full animate-pulse`}
            style={{ 
              animationDelay: `${i * 0.2}s`,
              animationDuration: '1s'
            }}
          ></div>
        ))}
      </div>
      <span>Neural Activity</span>
      <span className="text-xs opacity-75">{nodeCount.toLocaleString()} nodes</span>
    </div>
  );
};

export default ModelBadge;