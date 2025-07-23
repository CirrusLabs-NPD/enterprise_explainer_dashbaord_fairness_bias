import React, { useState, useEffect } from 'react';
import { Toaster } from 'react-hot-toast';
import { ThemeProvider } from './contexts/ThemeContext';
import { useAuth } from './contexts/AuthContext';
import Layout from './components/layout/Layout';
import LoadingSpinner from './components/common/LoadingSpinner';
import OnboardingTour from './components/common/OnboardingTour';
// AuthGuard is a component that restricts access to its children based on authentication and permissions.
// It ensures that only authorized users can view or interact with protected routes or components.
import AuthGuard from './components/common/AuthGuard';

// Import page components (will be created later)
import ModelOverview from './pages/ModelOverview';
import FeatureImportance from './pages/FeatureImportance';
import ClassificationStats from './pages/ClassificationStats';
import RegressionStats from './pages/RegressionStats';
import IndividualPredictions from './pages/IndividualPredictions';
import WhatIfAnalysis from './pages/WhatIfAnalysis';
import FeatureDependence from './pages/FeatureDependence';
import FeatureInteractions from './pages/FeatureInteractions';
import DecisionTrees from './pages/DecisionTrees';
import DataConnectivity from './pages/DataConnectivity';
import DataDrift from './pages/DataDrift';
import ModelPerformance from './pages/ModelPerformance';

// Enterprise feature pages
import EnterpriseFairness from './pages/EnterpriseFairness';
import EnterpriseAlerts from './pages/EnterpriseAlerts';
import ExecutiveDashboard from './pages/ExecutiveDashboard';
// import ComplianceReporting from './pages/ComplianceReporting';
import ComplianceReportsPage from './components/ComplianceReportsPage';
import RootCauseAnalysis from './pages/RootCauseAnalysis';
import DataConnectors from './pages/DataConnectors';
import ABTestingDashboard from './pages/ABTestingDashboard';
import DataQualityDashboard from './pages/DataQualityDashboard';
import StreamDataManagement from './pages/StreamDataManagement';
import BiasMetigationDashboard from './pages/BiasMetigationDashboard';
import DataPreprocessingWorkflow from './pages/DataPreprocessingWorkflow';
import CustomDashboardBuilder from './pages/CustomDashboardBuilder';
import ModelManagement from './pages/ModelManagement';
import AdvancedDriftConfiguration from './pages/AdvancedDriftConfiguration';
import DataManagement from './pages/DataManagement';

// Import new pages
import Settings from './pages/Settings';
import UserManagement from './pages/UserManagement';
import SystemHealth from './pages/SystemHealth';
import UserProfile from './components/auth/UserProfile';

// Mock data loading
const useModelData = () => {
  const [loading, setLoading] = useState(false);
  const [error] = useState<string | null>(null);
  
  useEffect(() => {
    // Simulate data loading
    const timer = setTimeout(() => {
      setLoading(false);
    }, 500);
    
    return () => clearTimeout(timer);
  }, []);
  
  return { loading, error };
};

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [modelType, setModelType] = useState<'classification' | 'regression'>('classification');
  const [showOnboarding, setShowOnboarding] = useState(false);
  const { loading, error } = useModelData();

  // Check if user has completed onboarding
  useEffect(() => {
    const hasCompletedTour = localStorage.getItem('ml-explainer-tour-completed');
    if (!hasCompletedTour) {
      setShowOnboarding(true);
    }
  }, []);

  const handleTabChange = (tabId: string) => {
    setActiveTab(tabId);
  };

  const handleModelTypeChange = (type: 'classification' | 'regression') => {
    setModelType(type);
    // Reset to overview when model type changes
    setActiveTab('overview');
  };

  // Render appropriate page component based on active tab
  const renderPageContent = () => {
    switch (activeTab) {
      case 'overview':
        return <ModelOverview modelType={modelType} />;
      case 'feature-importance':
        return <FeatureImportance modelType={modelType} />;
      case 'classification-stats':
        return <ClassificationStats />;
      case 'regression-stats':
        return <RegressionStats />;
      case 'predictions':
        return <IndividualPredictions modelType={modelType} />;
      case 'what-if':
        return <WhatIfAnalysis modelType={modelType} />;
      case 'feature-dependence':
        return <FeatureDependence modelType={modelType} />;
      case 'feature-interactions':
        return <FeatureInteractions modelType={modelType} />;
      case 'decision-trees':
        return <DecisionTrees modelType={modelType} />;
      case 'data-connectivity':
        return <DataConnectivity />;
      case 'data-drift':
        return <DataDrift />;
      case 'model-performance':
        return <ModelPerformance />;
      case 'enterprise-fairness':
        return <EnterpriseFairness />;
      case 'enterprise-alerts':
        return <EnterpriseAlerts />;
      case 'executive-dashboard':
        return <ExecutiveDashboard />;
      case 'compliance-reporting':
        return <ComplianceReportsPage />;
      case 'root-cause-analysis':
        return <RootCauseAnalysis />;
      case 'data-connectors':
        return <DataConnectors />;
      case 'ab-testing':
        return <ABTestingDashboard />;
      case 'data-quality':
        return <DataQualityDashboard />;
      case 'stream-management':
        return <StreamDataManagement />;
      case 'bias-mitigation':
        return <BiasMetigationDashboard />;
      case 'data-preprocessing':
        return <DataPreprocessingWorkflow />;
      case 'custom-dashboard':
        return <CustomDashboardBuilder />;
      case 'model-management':
        return <ModelManagement />;
      case 'drift-configuration':
        return <AdvancedDriftConfiguration />;
      case 'data-management':
        return <DataManagement />;
      case 'settings':
        // AuthGuard with requiredPermission restricts access to users with specific permissions.
        // For example, only users with 'system:config' permission can access the Settings page.
        return (
          <AuthGuard requiredPermission="system:config">
            <Settings />
          </AuthGuard>
        );
      case 'user-management':
        // Only users with 'user:manage' permission can access the User Management page.
        return (
          <AuthGuard requiredPermission="user:manage">
            <UserManagement />
          </AuthGuard>
        );
      case 'system-health':
        // Only users with 'system:config' permission can access the System Health page.
        return (
          <AuthGuard requiredPermission="system:config">
            <SystemHealth />
          </AuthGuard>
        );
      case 'user-profile':
        return <UserProfile />;
      default:
        return <ModelOverview modelType={modelType} />;
    }
  };

  if (loading) {
    return (
      <ThemeProvider>
        <div className="min-h-screen bg-neutral-50 dark:bg-neutral-900">
          <LoadingSpinner
            fullScreen
            size="xl"
            message="Loading ML Explainer Dashboard..."
          />
        </div>
      </ThemeProvider>
    );
  }

  if (error) {
    return (
      <ThemeProvider>
        <div className="min-h-screen bg-neutral-50 dark:bg-neutral-900 flex items-center justify-center">
          <div className="text-center">
            <h1 className="text-2xl font-bold text-red-600 dark:text-red-400 mb-4">
              Error Loading Dashboard
            </h1>
            <p className="text-neutral-600 dark:text-neutral-400 mb-6">
              {error}
            </p>
            <button
              onClick={() => window.location.reload()}
              className="px-6 py-3 bg-primary-500 hover:bg-primary-600 text-white rounded-lg transition-colors"
            >
              Retry
            </button>
          </div>
        </div>
      </ThemeProvider>
    );
  }

  return (
    <ThemeProvider>
      {/* <AuthGuard> */}
        <div className="min-h-screen bg-neutral-50 dark:bg-neutral-900">
          <Layout
            activeTab={activeTab}
            onTabChange={handleTabChange}
            modelType={modelType}
            onModelTypeChange={handleModelTypeChange}
            onShowTour={() => setShowOnboarding(true)}
          >
            {renderPageContent()}
          </Layout>
          
          {/* Toast notifications */}
          <Toaster
            position="top-right"
            toastOptions={{
              className: 'bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100',
              duration: 4000,
            }}
          />
          
          {/* Onboarding Tour */}
          <OnboardingTour
            isOpen={showOnboarding}
            onClose={() => setShowOnboarding(false)}
            onTabChange={handleTabChange}
          />
        </div>
      {/* </AuthGuard> */}
    </ThemeProvider>
  );
};

export default App;