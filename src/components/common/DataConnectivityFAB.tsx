import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Database, Upload, Plus, X } from 'lucide-react';

interface DataConnectivityFABProps {
  onNavigateToDataConnectivity: () => void;
  isDataConnectivityActive: boolean;
}

const DataConnectivityFAB: React.FC<DataConnectivityFABProps> = ({ 
  onNavigateToDataConnectivity, 
  isDataConnectivityActive 
}) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const toggleExpanded = () => {
    setIsExpanded(!isExpanded);
  };

  const handleDataConnectivityClick = () => {
    onNavigateToDataConnectivity();
    setIsExpanded(false);
  };

  if (isDataConnectivityActive) {
    return null; // Don't show FAB when already on data connectivity page
  }

  return (
    <>
      {/* Backdrop */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setIsExpanded(false)}
            className="fixed inset-0 bg-black/20 z-40 lg:hidden"
          />
        )}
      </AnimatePresence>

      {/* FAB Container */}
      <div className="fixed bottom-6 right-6 z-50 lg:hidden">
        <AnimatePresence>
          {isExpanded && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 20 }}
              className="mb-4 space-y-3"
            >
              {/* Data Connectivity Option */}
              <motion.button
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ delay: 0.1 }}
                onClick={handleDataConnectivityClick}
                className="flex items-center space-x-3 bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100 px-4 py-3 rounded-full shadow-lg border border-neutral-200 dark:border-neutral-700 hover:shadow-xl transition-all duration-200"
              >
                <Database className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                <span className="text-sm font-medium">Data Connectivity</span>
              </motion.button>

              {/* Upload Data Option */}
              <motion.button
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ delay: 0.05 }}
                onClick={() => {
                  // This would trigger file upload directly
                  setIsExpanded(false);
                  // For now, just navigate to data connectivity
                  onNavigateToDataConnectivity();
                }}
                className="flex items-center space-x-3 bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100 px-4 py-3 rounded-full shadow-lg border border-neutral-200 dark:border-neutral-700 hover:shadow-xl transition-all duration-200"
              >
                <Upload className="w-5 h-5 text-green-600 dark:text-green-400" />
                <span className="text-sm font-medium">Upload Data</span>
              </motion.button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Main FAB */}
        <motion.button
          onClick={toggleExpanded}
          className={`w-14 h-14 rounded-full shadow-lg transition-all duration-200 flex items-center justify-center ${
            isExpanded
              ? 'bg-red-500 hover:bg-red-600 text-white'
              : 'bg-blue-500 hover:bg-blue-600 text-white'
          }`}
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
        >
          <AnimatePresence mode="wait">
            {isExpanded ? (
              <motion.div
                key="close"
                initial={{ opacity: 0, rotate: -90 }}
                animate={{ opacity: 1, rotate: 0 }}
                exit={{ opacity: 0, rotate: 90 }}
                transition={{ duration: 0.2 }}
              >
                <X className="w-6 h-6" />
              </motion.div>
            ) : (
              <motion.div
                key="open"
                initial={{ opacity: 0, rotate: 90 }}
                animate={{ opacity: 1, rotate: 0 }}
                exit={{ opacity: 0, rotate: -90 }}
                transition={{ duration: 0.2 }}
              >
                <Plus className="w-6 h-6" />
              </motion.div>
            )}
          </AnimatePresence>
        </motion.button>

        {/* Pulsing indicator */}
        {!isExpanded && (
          <div className="absolute inset-0 rounded-full bg-blue-500 animate-ping opacity-20"></div>
        )}
      </div>
    </>
  );
};

export default DataConnectivityFAB;