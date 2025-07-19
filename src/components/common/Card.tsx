import React from 'react';
import { motion } from 'framer-motion';
import { cn } from '../../utils';

interface CardProps {
  children: React.ReactNode;
  className?: string;
  title?: string;
  description?: string;
  icon?: React.ReactNode;
  action?: React.ReactNode;
  hover?: boolean;
  gradient?: boolean;
  glassmorphism?: boolean;
  animated?: boolean;
}

const Card: React.FC<CardProps> = ({
  children,
  className,
  title,
  description,
  icon,
  action,
  hover = false,
  gradient = false,
  glassmorphism = false,
  animated = true,
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
      y: -4,
      transition: {
        duration: 0.2,
        ease: "easeOut",
      },
    },
  };

  return (
    <motion.div
      variants={animated ? cardVariants : undefined}
      initial={animated ? "initial" : undefined}
      animate={animated ? "animate" : undefined}
      whileHover={hover && animated ? "hover" : undefined}
      className={cn(
        'rounded-xl border shadow-soft transition-all duration-200',
        gradient && 'bg-gradient-to-br from-white to-neutral-50 dark:from-neutral-800 dark:to-neutral-900',
        glassmorphism && 'backdrop-blur-md bg-white/70 dark:bg-neutral-800/70',
        !gradient && !glassmorphism && 'bg-white dark:bg-neutral-800',
        'border-neutral-200 dark:border-neutral-700',
        hover && 'hover:shadow-lg hover:border-neutral-300 dark:hover:border-neutral-600',
        className
      )}
    >
      {(title || description || icon || action) && (
        <div className="p-6 border-b border-neutral-200 dark:border-neutral-700">
          <div className="flex items-start justify-between">
            <div className="flex items-start space-x-3">
              {icon && (
                <div className="flex-shrink-0 p-2 bg-primary-50 dark:bg-primary-900/20 rounded-lg">
                  {icon}
                </div>
              )}
              <div>
                {title && (
                  <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                    {title}
                  </h3>
                )}
                {description && (
                  <p className="text-sm text-neutral-600 dark:text-neutral-400 mt-1">
                    {description}
                  </p>
                )}
              </div>
            </div>
            {action && (
              <div className="flex-shrink-0">
                {action}
              </div>
            )}
          </div>
        </div>
      )}
      <div className="p-6">
        {children}
      </div>
    </motion.div>
  );
};

export default Card;