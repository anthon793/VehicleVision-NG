/**
 * Stats Card Component
 * Displays a single statistic with icon
 */

import React from 'react';

const StatsCard = ({ title, value, icon: Icon, color = 'primary', trend = null }) => {
  const colorClasses = {
    primary: 'bg-primary-100 text-primary-600',
    success: 'bg-success-100 text-success-600',
    danger: 'bg-danger-100 text-danger-600',
    warning: 'bg-yellow-100 text-yellow-600',
  };

  return (
    <div className="stat-card">
      <div className={`stat-icon ${colorClasses[color]}`}>
        <Icon className="h-5 w-5" />
      </div>
      <div>
        <p className="stat-value">{value}</p>
        <p className="stat-label">{title}</p>
        {trend && (
          <p className={`text-xs mt-0.5 ${trend > 0 ? 'text-success-600' : 'text-danger-600'}`}>
            {trend > 0 ? '↑' : '↓'} {Math.abs(trend)}% from yesterday
          </p>
        )}
      </div>
    </div>
  );
};

export default StatsCard;
