import React from 'react';
import { motion } from 'framer-motion';
import { cn } from '../../utils';

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg' | 'xl';
  variant?: 'primary' | 'secondary' | 'neutral';
  className?: string;
  fullScreen?: boolean;
  message?: string;
}

const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  size = 'md',
  variant = 'primary',
  className,
  fullScreen = false,
  message,
}) => {
  const sizes = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8',
    xl: 'w-12 h-12',
  };

  const colors = {
    primary: 'border-primary-500',
    secondary: 'border-secondary-500',
    neutral: 'border-neutral-400',
  };

  const spinner = (
    <div className={cn(
      'border-2 border-transparent border-t-current rounded-full animate-spin',
      sizes[size],
      colors[variant],
      className
    )} />
  );

  const pulseVariants = {
    initial: { scale: 1, opacity: 0.8 },
    animate: {
      scale: [1, 1.2, 1],
      opacity: [0.8, 0.4, 0.8],
      transition: {
        duration: 1.5,
        repeat: Infinity,
        ease: "easeInOut",
      },
    },
  };

  const containerVariants = {
    initial: { opacity: 0 },
    animate: {
      opacity: 1,
      transition: {
        duration: 0.3,
      },
    },
  };

  if (fullScreen) {
    return (
      <motion.div
        variants={containerVariants}
        initial="initial"
        animate="animate"
        className="fixed inset-0 bg-white/80 dark:bg-neutral-900/80 backdrop-blur-sm flex items-center justify-center z-50"
      >
        <div className="flex flex-col items-center space-y-4">
          <div className="relative">
            {spinner}
            <motion.div
              variants={pulseVariants}
              initial="initial"
              animate="animate"
              className={cn(
                'absolute inset-0 border-2 border-transparent rounded-full',
                colors[variant],
                sizes[size]
              )}
            />
          </div>
          {message && (
            <p className="text-neutral-600 dark:text-neutral-400 text-center max-w-xs">
              {message}
            </p>
          )}
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      variants={containerVariants}
      initial="initial"
      animate="animate"
      className="flex flex-col items-center space-y-2"
    >
      <div className="relative">
        {spinner}
        <motion.div
          variants={pulseVariants}
          initial="initial"
          animate="animate"
          className={cn(
            'absolute inset-0 border-2 border-transparent rounded-full',
            colors[variant],
            sizes[size]
          )}
        />
      </div>
      {message && (
        <p className="text-neutral-600 dark:text-neutral-400 text-sm text-center">
          {message}
        </p>
      )}
    </motion.div>
  );
};

export default LoadingSpinner;