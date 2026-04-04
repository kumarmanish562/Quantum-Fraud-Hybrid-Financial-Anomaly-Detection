import React from 'react';

const Card = ({ 
  children, 
  className = '', 
  hover = true, 
  padding = 'p-6',
  background = 'bg-white/60',
  border = 'border-gray-200',
  onClick,
  ...props 
}) => {
  const baseClasses = `
    ${background} 
    backdrop-blur-sm 
    border 
    ${border} 
    rounded-2xl 
    ${padding}
    shadow-lg
    transition-all 
    duration-200
    ${hover ? 'hover:bg-white/80 hover:shadow-xl hover:shadow-blue-500/10' : ''}
    ${onClick ? 'cursor-pointer' : ''}
  `;

  return (
    <div 
      className={`${baseClasses} ${className}`}
      onClick={onClick}
      {...props}
    >
      {children}
    </div>
  );
};

// Card variants for different use cases
export const StatCard = ({ title, value, icon, change, changeType = 'positive', ...props }) => {
  return (
    <Card {...props}>
      <div className="flex items-center justify-between mb-4">
        <div className="p-3 bg-blue-500/20 rounded-xl text-blue-400">
          {icon}
        </div>
        {change && (
          <span className={`text-sm font-medium ${
            changeType === 'positive' ? 'text-green-400' : 'text-red-400'
          }`}>
            {change}
          </span>
        )}
      </div>
      <h3 className="text-gray-500 text-sm font-medium mb-1 uppercase tracking-wider">{title}</h3>
      <p className="text-gray-900 text-3xl font-bold">{value}</p>
    </Card>
  );
};

export const AlertCard = ({ severity, title, description, time, transactionId, amount, ...props }) => {
  const getSeverityConfig = (severity) => {
    switch (severity) {
      case 'High':
        return {
          bgColor: 'bg-red-500/10',
          borderColor: 'border-red-500/30',
          textColor: 'text-red-400',
          badgeColor: 'bg-red-500/20 text-red-400'
        };
      case 'Medium':
        return {
          bgColor: 'bg-yellow-500/10',
          borderColor: 'border-yellow-500/30',
          textColor: 'text-yellow-400',
          badgeColor: 'bg-yellow-500/20 text-yellow-400'
        };
      case 'Low':
        return {
          bgColor: 'bg-blue-500/10',
          borderColor: 'border-blue-500/30',
          textColor: 'text-blue-400',
          badgeColor: 'bg-blue-500/20 text-blue-400'
        };
      default:
        return {
          bgColor: 'bg-gray-500/10',
          borderColor: 'border-gray-500/30',
          textColor: 'text-gray-500',
          badgeColor: 'bg-gray-500/20 text-gray-500'
        };
    }
  };

  const config = getSeverityConfig(severity);

  return (
    <Card 
      background={config.bgColor} 
      border={config.borderColor}
      {...props}
    >
      <div className="flex items-start justify-between">
        <div className="flex items-start space-x-4">
          <div className={`p-3 rounded-xl ${config.badgeColor} flex-shrink-0`}>
            <svg className={`w-6 h-6 ${config.textColor}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          </div>
          <div className="flex-1">
            <div className="flex items-center space-x-3 mb-2">
              <h3 className="text-gray-900 font-semibold">{title}</h3>
              <span className={`px-2 py-1 rounded-full text-xs font-medium ${config.badgeColor}`}>
                {severity}
              </span>
            </div>
            <p className="text-gray-600 text-sm mb-3">{description}</p>
            {transactionId && (
              <div className="flex items-center space-x-4 text-sm">
                <span className="text-blue-400 font-mono">{transactionId}</span>
                {amount && <span className="text-cyan-400 font-bold">{amount}</span>}
              </div>
            )}
          </div>
        </div>
        <div className="text-gray-500 text-sm">{time}</div>
      </div>
    </Card>
  );
};

export default Card;