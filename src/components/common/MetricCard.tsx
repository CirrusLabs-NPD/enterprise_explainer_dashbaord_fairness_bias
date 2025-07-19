import React from 'react';
import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { cn, formatNumber } from '../../utils';

interface MetricCardProps {
  title: string;
  value: number | string;
  change?: number | string;
  changeType?: 'increase' | 'decrease' | 'neutral' | 'positive' | 'negative';
  format?: 'number' | 'percentage' | 'currency';
  precision?: number;
  icon?: React.ReactNode;
  color?: 'primary' | 'secondary' | 'success' | 'warning' | 'error';
  size?: 'sm' | 'md' | 'lg';
  animated?: boolean;
  loading?: boolean;
  description?: string;
}

const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  change,
  changeType,
  format = 'number',
  precision = 2,
  icon,
  color = 'primary',
  size = 'md',
  animated = true,
  loading = false,
  description,
}) => {
  const cardVariants = {
    initial: { opacity: 0, y: 20 },
    animate: { 
      opacity: 1, 
      y: 0,
      transition: {
        duration: 0.3,
        ease: "easeOut",
      },
    },
    hover: {
      y: -2,
      transition: {
        duration: 0.2,
        ease: "easeOut",
      },
    },
  };

  const colorClasses = {
    primary: 'from-primary-50 to-primary-100 dark:from-primary-900/20 dark:to-primary-800/20 text-primary-600 dark:text-primary-400',
    secondary: 'from-secondary-50 to-secondary-100 dark:from-secondary-900/20 dark:to-secondary-800/20 text-secondary-600 dark:text-secondary-400',
    success: 'from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 text-green-600 dark:text-green-400',
    warning: 'from-amber-50 to-amber-100 dark:from-amber-900/20 dark:to-amber-800/20 text-amber-600 dark:text-amber-400',
    error: 'from-red-50 to-red-100 dark:from-red-900/20 dark:to-red-800/20 text-red-600 dark:text-red-400',
  };

  const sizeClasses = {
    sm: 'p-4',
    md: 'p-6',
    lg: 'p-8',
  };

  const formatValue = (val: number | string) => {
    if (typeof val === 'string') return val;
    
    switch (format) {
      case 'percentage':
        return formatNumber(val, { precision, percentage: true });
      case 'currency':
        return formatNumber(val, { precision, currency: true });
      default:
        return formatNumber(val, { precision });
    }
  };

  const getChangeIcon = () => {
    if (!change) return null;
    
    switch (changeType) {
      case 'increase':
      case 'positive':
        return <TrendingUp className="w-4 h-4 text-green-500" />;
      case 'decrease':
      case 'negative':
        return <TrendingDown className="w-4 h-4 text-red-500" />;
      default:
        return <Minus className="w-4 h-4 text-neutral-500" />;
    }
  };

  const getChangeColor = () => {
    switch (changeType) {
      case 'increase':
      case 'positive':
        return 'text-green-600 dark:text-green-400';
      case 'decrease':
      case 'negative':
        return 'text-red-600 dark:text-red-400';
      default:
        return 'text-neutral-600 dark:text-neutral-400';
    }
  };

  const skeletonVariants = {
    initial: { opacity: 0.6 },
    animate: { 
      opacity: [0.6, 1, 0.6],
      transition: {
        duration: 1.5,
        repeat: Infinity,
        ease: "easeInOut",
      },
    },
  };

  if (loading) {
    return (
      <motion.div
        variants={cardVariants}
        initial={animated ? "initial" : undefined}
        animate={animated ? "animate" : undefined}
        className={cn(
          'bg-white dark:bg-neutral-800 rounded-xl border border-neutral-200 dark:border-neutral-700 shadow-soft',
          sizeClasses[size]
        )}
      >
        <div className="flex items-center justify-between mb-4">
          <motion.div
            variants={skeletonVariants}
            initial="initial"
            animate="animate"
            className="h-4 bg-neutral-200 dark:bg-neutral-700 rounded w-24"
          />
          {icon && (
            <motion.div
              variants={skeletonVariants}
              initial="initial"
              animate="animate"
              className="h-8 w-8 bg-neutral-200 dark:bg-neutral-700 rounded"
            />
          )}
        </div>
        <motion.div
          variants={skeletonVariants}
          initial="initial"
          animate="animate"
          className="h-8 bg-neutral-200 dark:bg-neutral-700 rounded w-32 mb-2"
        />
        <motion.div
          variants={skeletonVariants}
          initial="initial"
          animate="animate"
          className="h-4 bg-neutral-200 dark:bg-neutral-700 rounded w-20"
        />
      </motion.div>
    );
  }

  return (
    <motion.div
      variants={cardVariants}
      initial={animated ? "initial" : undefined}
      animate={animated ? "animate" : undefined}
      whileHover={animated ? "hover" : undefined}
      className={cn(
        'bg-gradient-to-br border border-neutral-200 dark:border-neutral-700 rounded-xl shadow-soft hover:shadow-md transition-all duration-200',
        colorClasses[color],
        sizeClasses[size]
      )}
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium text-neutral-600 dark:text-neutral-400 uppercase tracking-wide">
          {title}
        </h3>
        {icon && (
          <div className="p-2 bg-white/50 dark:bg-neutral-800/50 rounded-lg">
            {icon}
          </div>
        )}
      </div>
      
      <div className="flex items-end justify-between">
        <div>
          <p className="text-2xl font-bold text-neutral-900 dark:text-neutral-100 mb-1">
            {formatValue(value)}
          </p>
          
          {change !== undefined && (
            <div className={cn(
              'flex items-center space-x-1 text-sm font-medium',
              getChangeColor()
            )}>
              {getChangeIcon()}
              <span>
                {typeof change === 'string' ? change : `${Math.abs(change)}%`}
              </span>
              {typeof change === 'number' && (
                <span className="text-neutral-500 dark:text-neutral-400">
                  vs last period
                </span>
              )}
            </div>
          )}
          
          {description && (
            <p className="text-xs text-neutral-500 dark:text-neutral-400 mt-1">
              {description}
            </p>
          )}
        </div>
      </div>
    </motion.div>
  );
};

export default MetricCard;