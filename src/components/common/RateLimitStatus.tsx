import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { AlertTriangle, Shield, Clock, Activity } from 'lucide-react';

interface RateLimitInfo {
  limit: number;
  remaining: number;
  resetTime: number;
  retryAfter?: number;
}

interface RateLimitStatusProps {
  className?: string;
  showDetails?: boolean;
}

const RateLimitStatus: React.FC<RateLimitStatusProps> = ({ 
  className = '', 
  showDetails = false 
}) => {
  const [rateLimitInfo, setRateLimitInfo] = useState<RateLimitInfo | null>(null);
  const [isWarning, setIsWarning] = useState(false);
  const [isBlocked, setIsBlocked] = useState(false);

  useEffect(() => {
    // Check rate limit status from localStorage or API
    const checkRateLimit = () => {
      // Mock rate limit data - in real app, this would come from API headers
      const mockInfo: RateLimitInfo = {
        limit: 100,
        remaining: Math.floor(Math.random() * 100),
        resetTime: Date.now() + (60 * 1000), // Reset in 1 minute
      };

      setRateLimitInfo(mockInfo);
      
      // Calculate warning and blocked states
      const usagePercentage = ((mockInfo.limit - mockInfo.remaining) / mockInfo.limit) * 100;
      setIsWarning(usagePercentage > 80);
      setIsBlocked(mockInfo.remaining === 0);
    };

    checkRateLimit();
    const interval = setInterval(checkRateLimit, 5000); // Check every 5 seconds

    return () => clearInterval(interval);
  }, []);

  if (!rateLimitInfo) return null;

  const usagePercentage = ((rateLimitInfo.limit - rateLimitInfo.remaining) / rateLimitInfo.limit) * 100;
  const timeToReset = Math.max(0, rateLimitInfo.resetTime - Date.now());
  const minutesToReset = Math.ceil(timeToReset / (60 * 1000));

  const getStatusColor = () => {
    if (isBlocked) return 'text-red-600 dark:text-red-400';
    if (isWarning) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-green-600 dark:text-green-400';
  };

  const getProgressColor = () => {
    if (isBlocked) return 'bg-red-500';
    if (isWarning) return 'bg-yellow-500';
    return 'bg-green-500';
  };

  const getIcon = () => {
    if (isBlocked) return <AlertTriangle className="w-4 h-4" />;
    if (isWarning) return <Clock className="w-4 h-4" />;
    return <Shield className="w-4 h-4" />;
  };

  if (!showDetails) {
    // Minimal indicator
    return (
      <div className={`flex items-center space-x-2 ${className}`}>
        <div className={`${getStatusColor()}`}>
          {getIcon()}
        </div>
        <span className={`text-xs ${getStatusColor()}`}>
          {rateLimitInfo.remaining}/{rateLimitInfo.limit}
        </span>
      </div>
    );
  }

  // Detailed status component
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className={`bg-white dark:bg-neutral-800 rounded-lg border border-neutral-200 dark:border-neutral-700 p-4 ${className}`}
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-2">
          <div className={getStatusColor()}>
            {getIcon()}
          </div>
          <h4 className="text-sm font-medium text-neutral-900 dark:text-white">
            Rate Limit Status
          </h4>
        </div>
        <div className="flex items-center space-x-1 text-xs text-neutral-500 dark:text-neutral-400">
          <Activity className="w-3 h-3" />
          <span>Live</span>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="mb-3">
        <div className="flex justify-between text-xs text-neutral-600 dark:text-neutral-400 mb-1">
          <span>Usage</span>
          <span>{usagePercentage.toFixed(0)}%</span>
        </div>
        <div className="w-full bg-neutral-200 dark:bg-neutral-700 rounded-full h-2">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${usagePercentage}%` }}
            transition={{ duration: 0.5 }}
            className={`h-2 rounded-full ${getProgressColor()}`}
          />
        </div>
      </div>

      {/* Details */}
      <div className="grid grid-cols-2 gap-4 text-xs">
        <div>
          <div className="text-neutral-500 dark:text-neutral-400">Remaining</div>
          <div className="font-medium text-neutral-900 dark:text-white">
            {rateLimitInfo.remaining}
          </div>
        </div>
        <div>
          <div className="text-neutral-500 dark:text-neutral-400">Limit</div>
          <div className="font-medium text-neutral-900 dark:text-white">
            {rateLimitInfo.limit}
          </div>
        </div>
        <div>
          <div className="text-neutral-500 dark:text-neutral-400">Reset in</div>
          <div className="font-medium text-neutral-900 dark:text-white">
            {minutesToReset}m
          </div>
        </div>
        <div>
          <div className="text-neutral-500 dark:text-neutral-400">Status</div>
          <div className={`font-medium ${getStatusColor()}`}>
            {isBlocked ? 'Blocked' : isWarning ? 'Warning' : 'OK'}
          </div>
        </div>
      </div>

      {/* Warning Messages */}
      {isBlocked && (
        <div className="mt-3 p-2 bg-red-50 dark:bg-red-900/20 rounded border border-red-200 dark:border-red-800">
          <p className="text-xs text-red-700 dark:text-red-300">
            Rate limit exceeded. Please wait {minutesToReset} minute(s) before making more requests.
          </p>
        </div>
      )}

      {isWarning && !isBlocked && (
        <div className="mt-3 p-2 bg-yellow-50 dark:bg-yellow-900/20 rounded border border-yellow-200 dark:border-yellow-800">
          <p className="text-xs text-yellow-700 dark:text-yellow-300">
            Approaching rate limit. Consider reducing request frequency.
          </p>
        </div>
      )}
    </motion.div>
  );
};

export default RateLimitStatus;