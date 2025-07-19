import React, { useState } from 'react';
import { motion } from 'framer-motion';
import Header from './Header';
import Sidebar from './Sidebar';
import HelpGuide from '@/components/common/HelpGuide';
import DataConnectivityFAB from '@/components/common/DataConnectivityFAB';
import { cn } from '@/utils';

interface LayoutProps {
  children: React.ReactNode;
  activeTab: string;
  onTabChange: (tabId: string) => void;
  modelType: 'classification' | 'regression';
  onModelTypeChange: (type: 'classification' | 'regression') => void;
  onShowTour?: () => void;
}

const Layout: React.FC<LayoutProps> = ({
  children,
  activeTab,
  onTabChange,
  modelType,
  onModelTypeChange,
  onShowTour,
}) => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isHelpOpen, setIsHelpOpen] = useState(false);

  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
  };

  const closeSidebar = () => {
    setIsSidebarOpen(false);
  };

  const pageVariants = {
    initial: {
      opacity: 0,
      x: -20,
    },
    animate: {
      opacity: 1,
      x: 0,
      transition: {
        duration: 0.3,
        ease: "easeOut",
      },
    },
    exit: {
      opacity: 0,
      x: 20,
      transition: {
        duration: 0.2,
        ease: "easeIn",
      },
    },
  };

  return (
    <div className="min-h-screen bg-neutral-50 dark:bg-neutral-900">
      {/* Header */}
      <Header
        onToggleSidebar={toggleSidebar}
        isSidebarOpen={isSidebarOpen}
        modelType={modelType}
        onModelTypeChange={onModelTypeChange}
        onShowHelp={() => setIsHelpOpen(true)}
        onShowTour={onShowTour}
        onTabChange={onTabChange}
      />

      <div className="flex h-[calc(100vh-4rem)]">
        {/* Sidebar */}
        <Sidebar
          isOpen={isSidebarOpen}
          onClose={closeSidebar}
          activeTab={activeTab}
          onTabChange={onTabChange}
          modelType={modelType}
        />

        {/* Main Content */}
        <main 
          className={cn(
            "flex-1 overflow-hidden",
            "lg:ml-0" // Sidebar is handled by its own positioning
          )}
        >
          <div className="h-full overflow-y-auto">
            <motion.div
              key={activeTab}
              variants={pageVariants}
              initial="initial"
              animate="animate"
              exit="exit"
              className="p-6 max-w-full"
            >
              {/* Breadcrumb */}
              <div className="mb-6">
                <nav className="flex items-center space-x-2 text-sm text-neutral-600 dark:text-neutral-400">
                  <span>Dashboard</span>
                  <span>/</span>
                  <span className="text-neutral-900 dark:text-neutral-100 font-medium">
                    {getTabLabel(activeTab)}
                  </span>
                </nav>
              </div>

              {/* Page Content */}
              <div className="space-y-6">
                {children}
              </div>
            </motion.div>
          </div>
        </main>
      </div>

      {/* Help Guide */}
      <HelpGuide
        isOpen={isHelpOpen}
        onClose={() => setIsHelpOpen(false)}
        activeTab={activeTab}
      />

      {/* Data Connectivity FAB */}
      <DataConnectivityFAB
        onNavigateToDataConnectivity={() => onTabChange('data-connectivity')}
        isDataConnectivityActive={activeTab === 'data-connectivity'}
      />
    </div>
  );
};

// Helper function to get tab label
function getTabLabel(tabId: string): string {
  const tabLabels: Record<string, string> = {
    'overview': 'Model Overview',
    'feature-importance': 'Feature Importance',
    'classification-stats': 'Classification Statistics',
    'regression-stats': 'Regression Statistics',
    'predictions': 'Individual Predictions',
    'what-if': 'What-If Analysis',
    'feature-dependence': 'Feature Dependence',
    'feature-interactions': 'Feature Interactions',
    'decision-trees': 'Decision Trees',
  };
  
  return tabLabels[tabId] || 'Dashboard';
}

export default Layout;