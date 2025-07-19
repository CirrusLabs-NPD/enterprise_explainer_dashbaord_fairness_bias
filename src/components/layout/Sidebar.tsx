import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  BarChart3,
  Target,
  TrendingUp,
  Users,
  User,
  Zap,
  GitBranch,
  Network,
  ChevronRight,
  ChevronDown,
  Activity,
  PieChart,
  Settings,
  FileText,
  Database,
  Brain,
  Eye,
  CheckCircle,
  Clock,
  AlertCircle,
  Shield,
  Bell,
  BarChart,
  FileCheck,
  Search,
  Plug,
  Crown,
  Layout,
  FolderOpen,
  Settings as SettingsIcon,
  TrendingDown,
} from 'lucide-react';
import { cn } from '../../utils';
import { TabItem } from '../../types';

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
  activeTab: string;
  onTabChange: (tabId: string) => void;
  modelType: 'classification' | 'regression';
}

const Sidebar: React.FC<SidebarProps> = ({
  isOpen,
  onClose,
  activeTab,
  onTabChange,
  modelType,
}) => {
  const [expandedSections, setExpandedSections] = useState<string[]>(['analysis', 'enterprise', 'data']);

  const toggleSection = (sectionId: string) => {
    setExpandedSections(prev =>
      prev.includes(sectionId)
        ? prev.filter(id => id !== sectionId)
        : [...prev, sectionId]
    );
  };

  const baseTabItems: TabItem[] = [
    {
      id: 'overview',
      label: 'Model Overview',
      icon: 'Activity',
      component: () => null,
      description: 'General model information and performance metrics',
    },
    {
      id: 'feature-importance',
      label: 'Feature Importance',
      icon: 'BarChart3',
      component: () => null,
      description: 'Global feature importance and rankings',
    },
  ];

  const classificationTabs: TabItem[] = [
    {
      id: 'classification-stats',
      label: 'Classification Stats',
      icon: 'Target',
      component: () => null,
      description: 'Confusion matrix, ROC curves, and classification metrics',
    },
    {
      id: 'predictions',
      label: 'Individual Predictions',
      icon: 'Eye',
      component: () => null,
      description: 'Analyze individual prediction explanations',
    },
    {
      id: 'what-if',
      label: 'What-If Analysis',
      icon: 'Zap',
      component: () => null,
      description: 'Interactive scenario analysis',
    },
    {
      id: 'feature-dependence',
      label: 'Feature Dependence',
      icon: 'TrendingUp',
      component: () => null,
      description: 'Feature dependence and partial dependence plots',
    },
    {
      id: 'feature-interactions',
      label: 'Feature Interactions',
      icon: 'Network',
      component: () => null,
      description: 'Feature interaction analysis',
    },
    {
      id: 'decision-trees',
      label: 'Decision Trees',
      icon: 'GitBranch',
      component: () => null,
      description: 'Decision tree visualization and analysis',
      disabled: false,
    },
  ];

  const regressionTabs: TabItem[] = [
    {
      id: 'regression-stats',
      label: 'Regression Stats',
      icon: 'PieChart',
      component: () => null,
      description: 'Residual plots, R² score, and regression metrics',
    },
    {
      id: 'predictions',
      label: 'Individual Predictions',
      icon: 'Eye',
      component: () => null,
      description: 'Analyze individual prediction explanations',
    },
    {
      id: 'what-if',
      label: 'What-If Analysis',
      icon: 'Zap',
      component: () => null,
      description: 'Interactive scenario analysis',
    },
    {
      id: 'feature-dependence',
      label: 'Feature Dependence',
      icon: 'TrendingUp',
      component: () => null,
      description: 'Feature dependence and partial dependence plots',
    },
    {
      id: 'feature-interactions',
      label: 'Feature Interactions',
      icon: 'Network',
      component: () => null,
      description: 'Feature interaction analysis',
    },
    {
      id: 'decision-trees',
      label: 'Decision Trees',
      icon: 'GitBranch',
      component: () => null,
      description: 'Decision tree visualization and analysis',
      disabled: false,
    },
  ];

  const allTabs = [
    ...baseTabItems,
    ...(modelType === 'classification' ? classificationTabs : regressionTabs),
  ];

  const getIconComponent = (iconName: string) => {
    const iconMap: Record<string, React.ComponentType<any>> = {
      Activity,
      BarChart3,
      Target,
      TrendingUp,
      Users,
      Zap,
      GitBranch,
      Network,
      PieChart,
      Eye,
      Settings,
      FileText,
      Database,
      Brain,
    };
    
    return iconMap[iconName] || Activity;
  };

  const getTabStatus = (tabId: string) => {
    // Mock status for demonstration
    const statusMap: Record<string, 'completed' | 'loading' | 'error' | 'pending'> = {
      overview: 'completed',
      'feature-importance': 'completed',
      'classification-stats': 'completed',
      'regression-stats': 'completed',
      predictions: 'loading',
      'what-if': 'pending',
      'feature-dependence': 'pending',
      'feature-interactions': 'pending',
      'decision-trees': 'pending',
    };
    
    return statusMap[tabId] || 'pending';
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-3 h-3 text-green-500" />;
      case 'loading':
        return <Clock className="w-3 h-3 text-amber-500 animate-spin" />;
      case 'error':
        return <AlertCircle className="w-3 h-3 text-red-500" />;
      default:
        return <div className="w-3 h-3 rounded-full bg-neutral-300 dark:bg-neutral-600" />;
    }
  };

  const sidebarVariants = {
    open: {
      x: 0,
      transition: {
        type: "spring",
        stiffness: 300,
        damping: 30,
      },
    },
    closed: {
      x: "-100%",
      transition: {
        type: "spring",
        stiffness: 300,
        damping: 30,
      },
    },
  };

  const overlayVariants = {
    open: {
      opacity: 1,
      pointerEvents: "auto" as const,
    },
    closed: {
      opacity: 0,
      pointerEvents: "none" as const,
    },
  };

  return (
    <>
      {/* Mobile overlay */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial="closed"
            animate="open"
            exit="closed"
            variants={overlayVariants}
            className="fixed inset-0 bg-black/50 z-40 lg:hidden"
            onClick={onClose}
          />
        )}
      </AnimatePresence>

      {/* Sidebar */}
      <motion.aside
        initial="closed"
        animate={isOpen ? "open" : "closed"}
        variants={sidebarVariants}
        className={cn(
          "fixed left-0 top-0 h-full w-80 bg-white/95 dark:bg-neutral-900/95 backdrop-blur-md border-r border-neutral-200 dark:border-neutral-800 z-50 overflow-hidden",
          "lg:relative lg:translate-x-0 lg:z-auto"
        )}
      >
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className="p-6 border-b border-neutral-200 dark:border-neutral-800">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                  Analysis Dashboard
                </h2>
                <p className="text-sm text-neutral-600 dark:text-neutral-400">
                  {modelType === 'classification' ? 'Classification' : 'Regression'} Model
                </p>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <span className="text-xs text-neutral-500 dark:text-neutral-400">
                  Live
                </span>
              </div>
            </div>
          </div>

          {/* Navigation */}
          <nav className="flex-1 overflow-y-auto p-4">
            <div className="space-y-2">
              {/* Analysis Section */}
              <div>
                <button
                  onClick={() => toggleSection('analysis')}
                  className="w-full flex items-center justify-between p-2 text-sm font-medium text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800 rounded-lg transition-colors"
                >
                  <div className="flex items-center space-x-2">
                    <Brain className="w-4 h-4" />
                    <span>Analysis</span>
                  </div>
                  {expandedSections.includes('analysis') ? (
                    <ChevronDown className="w-4 h-4" />
                  ) : (
                    <ChevronRight className="w-4 h-4" />
                  )}
                </button>
                
                <AnimatePresence>
                  {expandedSections.includes('analysis') && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.2 }}
                      className="overflow-hidden"
                    >
                      <div className="pl-4 mt-2 space-y-1">
                        {allTabs.map((tab) => {
                          const Icon = getIconComponent(tab.icon);
                          const status = getTabStatus(tab.id);
                          const isActive = activeTab === tab.id;
                          const isDisabled = tab.disabled;

                          return (
                            <button
                              key={tab.id}
                              onClick={() => {
                                if (!isDisabled) {
                                  console.log('Tab clicked:', tab.id);
                                  onTabChange(tab.id);
                                }
                              }}
                              disabled={isDisabled}
                              className={cn(
                                'w-full flex items-center justify-between p-3 rounded-lg transition-all duration-200 group',
                                isActive
                                  ? 'bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300 shadow-sm'
                                  : isDisabled
                                  ? 'text-neutral-400 dark:text-neutral-600 cursor-not-allowed'
                                  : 'text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800 hover:text-neutral-900 dark:hover:text-neutral-100',
                                'interactive-element'
                              )}
                            >
                              <div className="flex items-center space-x-3">
                                <Icon className={cn(
                                  'w-4 h-4 transition-colors',
                                  isActive && 'text-primary-600 dark:text-primary-400'
                                )} />
                                <div className="text-left">
                                  <div className="flex items-center space-x-2">
                                    <span className="font-medium text-sm">
                                      {tab.label}
                                    </span>
                                    {tab.badge && (
                                      <span className="px-2 py-0.5 text-xs bg-primary-100 dark:bg-primary-900 text-primary-700 dark:text-primary-300 rounded-full">
                                        {tab.badge}
                                      </span>
                                    )}
                                  </div>
                                  <p className="text-xs text-neutral-500 dark:text-neutral-400 mt-0.5">
                                    {tab.description}
                                  </p>
                                </div>
                              </div>
                              <div className="flex items-center space-x-2">
                                {getStatusIcon(status)}
                                {isActive && (
                                  <div className="w-2 h-2 bg-primary-500 rounded-full"></div>
                                )}
                              </div>
                            </button>
                          );
                        })}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              {/* Enterprise Section */}
              <div>
                <button
                  onClick={() => toggleSection('enterprise')}
                  className="w-full flex items-center justify-between p-2 text-sm font-medium text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800 rounded-lg transition-colors"
                >
                  <div className="flex items-center space-x-2">
                    <Crown className="w-4 h-4" />
                    <span>Enterprise</span>
                  </div>
                  {expandedSections.includes('enterprise') ? (
                    <ChevronDown className="w-4 h-4" />
                  ) : (
                    <ChevronRight className="w-4 h-4" />
                  )}
                </button>
                
                <AnimatePresence>
                  {expandedSections.includes('enterprise') && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.2 }}
                      className="overflow-hidden"
                    >
                      <div className="pl-4 mt-2 space-y-1">
                        <button 
                          onClick={() => onTabChange('executive-dashboard')}
                          className={cn(
                            "w-full flex items-center justify-between p-3 rounded-lg transition-all duration-200 group border-2",
                            activeTab === 'executive-dashboard' 
                              ? 'bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300 border-primary-200 dark:border-primary-700 shadow-sm'
                              : 'text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800 border-transparent hover:border-primary-200 dark:hover:border-primary-700'
                          )}
                        >
                          <div className="flex items-center space-x-3">
                            <div className={cn(
                              "p-2 rounded-full transition-colors",
                              activeTab === 'executive-dashboard'
                                ? 'bg-primary-100 dark:bg-primary-900/30'
                                : 'bg-purple-100 dark:bg-purple-900/30 group-hover:bg-primary-100 dark:group-hover:bg-primary-900/30'
                            )}>
                              <BarChart className={cn(
                                "w-4 h-4",
                                activeTab === 'executive-dashboard'
                                  ? 'text-primary-600 dark:text-primary-400'
                                  : 'text-purple-600 dark:text-purple-400 group-hover:text-primary-600 dark:group-hover:text-primary-400'
                              )} />
                            </div>
                            <div className="text-left">
                              <div className="font-semibold text-sm">Executive Dashboard</div>
                              <div className="text-xs text-neutral-500 dark:text-neutral-400">
                                C-suite KPIs & business metrics
                              </div>
                            </div>
                          </div>
                          <div className="flex items-center space-x-1">
                            <div className="px-2 py-1 bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 rounded-full text-xs font-medium">
                              Executive
                            </div>
                            {activeTab === 'executive-dashboard' && (
                              <div className="w-2 h-2 bg-primary-500 rounded-full"></div>
                            )}
                          </div>
                        </button>

                        <button 
                          onClick={() => onTabChange('enterprise-fairness')}
                          className={cn(
                            "w-full flex items-center justify-between p-3 rounded-lg transition-all duration-200 group border-2",
                            activeTab === 'enterprise-fairness' 
                              ? 'bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300 border-primary-200 dark:border-primary-700 shadow-sm'
                              : 'text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800 border-transparent hover:border-primary-200 dark:hover:border-primary-700'
                          )}
                        >
                          <div className="flex items-center space-x-3">
                            <div className={cn(
                              "p-2 rounded-full transition-colors",
                              activeTab === 'enterprise-fairness'
                                ? 'bg-primary-100 dark:bg-primary-900/30'
                                : 'bg-green-100 dark:bg-green-900/30 group-hover:bg-primary-100 dark:group-hover:bg-primary-900/30'
                            )}>
                              <Shield className={cn(
                                "w-4 h-4",
                                activeTab === 'enterprise-fairness'
                                  ? 'text-primary-600 dark:text-primary-400'
                                  : 'text-green-600 dark:text-green-400 group-hover:text-primary-600 dark:group-hover:text-primary-400'
                              )} />
                            </div>
                            <div className="text-left">
                              <div className="font-semibold text-sm">Enterprise Fairness</div>
                              <div className="text-xs text-neutral-500 dark:text-neutral-400">
                                80+ bias metrics & compliance
                              </div>
                            </div>
                          </div>
                          <div className="flex items-center space-x-1">
                            <div className="px-2 py-1 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 rounded-full text-xs font-medium">
                              Pro
                            </div>
                            {activeTab === 'enterprise-fairness' && (
                              <div className="w-2 h-2 bg-primary-500 rounded-full"></div>
                            )}
                          </div>
                        </button>

                        <button 
                          onClick={() => onTabChange('enterprise-alerts')}
                          className={cn(
                            "w-full flex items-center justify-between p-3 rounded-lg transition-all duration-200 group border-2",
                            activeTab === 'enterprise-alerts' 
                              ? 'bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300 border-primary-200 dark:border-primary-700 shadow-sm'
                              : 'text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800 border-transparent hover:border-primary-200 dark:hover:border-primary-700'
                          )}
                        >
                          <div className="flex items-center space-x-3">
                            <div className={cn(
                              "p-2 rounded-full transition-colors",
                              activeTab === 'enterprise-alerts'
                                ? 'bg-primary-100 dark:bg-primary-900/30'
                                : 'bg-red-100 dark:bg-red-900/30 group-hover:bg-primary-100 dark:group-hover:bg-primary-900/30'
                            )}>
                              <Bell className={cn(
                                "w-4 h-4",
                                activeTab === 'enterprise-alerts'
                                  ? 'text-primary-600 dark:text-primary-400'
                                  : 'text-red-600 dark:text-red-400 group-hover:text-primary-600 dark:group-hover:text-primary-400'
                              )} />
                            </div>
                            <div className="text-left">
                              <div className="font-semibold text-sm">Real-time Alerts</div>
                              <div className="text-xs text-neutral-500 dark:text-neutral-400">
                                Multi-channel monitoring & SLA
                              </div>
                            </div>
                          </div>
                          <div className="flex items-center space-x-1">
                            <div className="px-2 py-1 bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 rounded-full text-xs font-medium">
                              Real-time
                            </div>
                            {activeTab === 'enterprise-alerts' && (
                              <div className="w-2 h-2 bg-primary-500 rounded-full"></div>
                            )}
                          </div>
                        </button>

                        <button 
                          onClick={() => onTabChange('compliance-reporting')}
                          className={cn(
                            "w-full flex items-center justify-between p-3 rounded-lg transition-all duration-200 group border-2",
                            activeTab === 'compliance-reporting' 
                              ? 'bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300 border-primary-200 dark:border-primary-700 shadow-sm'
                              : 'text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800 border-transparent hover:border-primary-200 dark:hover:border-primary-700'
                          )}
                        >
                          <div className="flex items-center space-x-3">
                            <div className={cn(
                              "p-2 rounded-full transition-colors",
                              activeTab === 'compliance-reporting'
                                ? 'bg-primary-100 dark:bg-primary-900/30'
                                : 'bg-indigo-100 dark:bg-indigo-900/30 group-hover:bg-primary-100 dark:group-hover:bg-primary-900/30'
                            )}>
                              <FileCheck className={cn(
                                "w-4 h-4",
                                activeTab === 'compliance-reporting'
                                  ? 'text-primary-600 dark:text-primary-400'
                                  : 'text-indigo-600 dark:text-indigo-400 group-hover:text-primary-600 dark:group-hover:text-primary-400'
                              )} />
                            </div>
                            <div className="text-left">
                              <div className="font-semibold text-sm">Compliance Reports</div>
                              <div className="text-xs text-neutral-500 dark:text-neutral-400">
                                GDPR, EU AI Act, SOC2 reporting
                              </div>
                            </div>
                          </div>
                          <div className="flex items-center space-x-1">
                            <div className="px-2 py-1 bg-indigo-100 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300 rounded-full text-xs font-medium">
                              Compliance
                            </div>
                            {activeTab === 'compliance-reporting' && (
                              <div className="w-2 h-2 bg-primary-500 rounded-full"></div>
                            )}
                          </div>
                        </button>

                        <button 
                          onClick={() => onTabChange('root-cause-analysis')}
                          className={cn(
                            "w-full flex items-center justify-between p-3 rounded-lg transition-all duration-200 group border-2",
                            activeTab === 'root-cause-analysis' 
                              ? 'bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300 border-primary-200 dark:border-primary-700 shadow-sm'
                              : 'text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800 border-transparent hover:border-primary-200 dark:hover:border-primary-700'
                          )}
                        >
                          <div className="flex items-center space-x-3">
                            <div className={cn(
                              "p-2 rounded-full transition-colors",
                              activeTab === 'root-cause-analysis'
                                ? 'bg-primary-100 dark:bg-primary-900/30'
                                : 'bg-yellow-100 dark:bg-yellow-900/30 group-hover:bg-primary-100 dark:group-hover:bg-primary-900/30'
                            )}>
                              <Search className={cn(
                                "w-4 h-4",
                                activeTab === 'root-cause-analysis'
                                  ? 'text-primary-600 dark:text-primary-400'
                                  : 'text-yellow-600 dark:text-yellow-400 group-hover:text-primary-600 dark:group-hover:text-primary-400'
                              )} />
                            </div>
                            <div className="text-left">
                              <div className="font-semibold text-sm">Root Cause Analysis</div>
                              <div className="text-xs text-neutral-500 dark:text-neutral-400">
                                Automated issue diagnosis
                              </div>
                            </div>
                          </div>
                          <div className="flex items-center space-x-1">
                            <div className="px-2 py-1 bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-300 rounded-full text-xs font-medium">
                              AI-Powered
                            </div>
                            {activeTab === 'root-cause-analysis' && (
                              <div className="w-2 h-2 bg-primary-500 rounded-full"></div>
                            )}
                          </div>
                        </button>

                        <button 
                          onClick={() => onTabChange('data-connectors')}
                          className={cn(
                            "w-full flex items-center justify-between p-3 rounded-lg transition-all duration-200 group border-2",
                            activeTab === 'data-connectors' 
                              ? 'bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300 border-primary-200 dark:border-primary-700 shadow-sm'
                              : 'text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800 border-transparent hover:border-primary-200 dark:hover:border-primary-700'
                          )}
                        >
                          <div className="flex items-center space-x-3">
                            <div className={cn(
                              "p-2 rounded-full transition-colors",
                              activeTab === 'data-connectors'
                                ? 'bg-primary-100 dark:bg-primary-900/30'
                                : 'bg-teal-100 dark:bg-teal-900/30 group-hover:bg-primary-100 dark:group-hover:bg-primary-900/30'
                            )}>
                              <Plug className={cn(
                                "w-4 h-4",
                                activeTab === 'data-connectors'
                                  ? 'text-primary-600 dark:text-primary-400'
                                  : 'text-teal-600 dark:text-teal-400 group-hover:text-primary-600 dark:group-hover:text-primary-400'
                              )} />
                            </div>
                            <div className="text-left">
                              <div className="font-semibold text-sm">Data Connectors</div>
                              <div className="text-xs text-neutral-500 dark:text-neutral-400">
                                Snowflake, AWS, BigQuery, Databricks
                              </div>
                            </div>
                          </div>
                          <div className="flex items-center space-x-1">
                            <div className="px-2 py-1 bg-teal-100 dark:bg-teal-900/30 text-teal-700 dark:text-teal-300 rounded-full text-xs font-medium">
                              Enterprise
                            </div>
                            {activeTab === 'data-connectors' && (
                              <div className="w-2 h-2 bg-primary-500 rounded-full"></div>
                            )}
                          </div>
                        </button>

                        <button 
                          onClick={() => onTabChange('custom-dashboard')}
                          className={cn(
                            "w-full flex items-center justify-between p-3 rounded-lg transition-all duration-200 group border-2",
                            activeTab === 'custom-dashboard' 
                              ? 'bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300 border-primary-200 dark:border-primary-700 shadow-sm'
                              : 'text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800 border-transparent hover:border-primary-200 dark:hover:border-primary-700'
                          )}
                        >
                          <div className="flex items-center space-x-3">
                            <div className={cn(
                              "p-2 rounded-full transition-colors",
                              activeTab === 'custom-dashboard'
                                ? 'bg-primary-100 dark:bg-primary-900/30'
                                : 'bg-violet-100 dark:bg-violet-900/30 group-hover:bg-primary-100 dark:group-hover:bg-primary-900/30'
                            )}>
                              <Layout className={cn(
                                "w-4 h-4",
                                activeTab === 'custom-dashboard'
                                  ? 'text-primary-600 dark:text-primary-400'
                                  : 'text-violet-600 dark:text-violet-400 group-hover:text-primary-600 dark:group-hover:text-primary-400'
                              )} />
                            </div>
                            <div className="text-left">
                              <div className="font-semibold text-sm">Custom Dashboards</div>
                              <div className="text-xs text-neutral-500 dark:text-neutral-400">
                                Drag & drop dashboard builder
                              </div>
                            </div>
                          </div>
                          <div className="flex items-center space-x-1">
                            <div className="px-2 py-1 bg-violet-100 dark:bg-violet-900/30 text-violet-700 dark:text-violet-300 rounded-full text-xs font-medium">
                              Enterprise
                            </div>
                            {activeTab === 'custom-dashboard' && (
                              <div className="w-2 h-2 bg-primary-500 rounded-full"></div>
                            )}
                          </div>
                        </button>

                        <button 
                          onClick={() => onTabChange('model-management')}
                          className={cn(
                            "w-full flex items-center justify-between p-3 rounded-lg transition-all duration-200 group border-2",
                            activeTab === 'model-management' 
                              ? 'bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300 border-primary-200 dark:border-primary-700 shadow-sm'
                              : 'text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800 border-transparent hover:border-primary-200 dark:hover:border-primary-700'
                          )}
                        >
                          <div className="flex items-center space-x-3">
                            <div className={cn(
                              "p-2 rounded-full transition-colors",
                              activeTab === 'model-management'
                                ? 'bg-primary-100 dark:bg-primary-900/30'
                                : 'bg-emerald-100 dark:bg-emerald-900/30 group-hover:bg-primary-100 dark:group-hover:bg-primary-900/30'
                            )}>
                              <Brain className={cn(
                                "w-4 h-4",
                                activeTab === 'model-management'
                                  ? 'text-primary-600 dark:text-primary-400'
                                  : 'text-emerald-600 dark:text-emerald-400 group-hover:text-primary-600 dark:group-hover:text-primary-400'
                              )} />
                            </div>
                            <div className="text-left">
                              <div className="font-semibold text-sm">Model Management</div>
                              <div className="text-xs text-neutral-500 dark:text-neutral-400">
                                Model lifecycle & versioning
                              </div>
                            </div>
                          </div>
                          <div className="flex items-center space-x-1">
                            <div className="px-2 py-1 bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300 rounded-full text-xs font-medium">
                              Enterprise
                            </div>
                            {activeTab === 'model-management' && (
                              <div className="w-2 h-2 bg-primary-500 rounded-full"></div>
                            )}
                          </div>
                        </button>

                        <button 
                          onClick={() => onTabChange('ab-testing')}
                          className={cn(
                            "w-full flex items-center justify-between p-3 rounded-lg transition-all duration-200 group border-2",
                            activeTab === 'ab-testing' 
                              ? 'bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300 border-primary-200 dark:border-primary-700 shadow-sm'
                              : 'text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800 border-transparent hover:border-primary-200 dark:hover:border-primary-700'
                          )}
                        >
                          <div className="flex items-center space-x-3">
                            <div className={cn(
                              "p-2 rounded-full transition-colors",
                              activeTab === 'ab-testing'
                                ? 'bg-primary-100 dark:bg-primary-900/30'
                                : 'bg-sky-100 dark:bg-sky-900/30 group-hover:bg-primary-100 dark:group-hover:bg-primary-900/30'
                            )}>
                              <Target className={cn(
                                "w-4 h-4",
                                activeTab === 'ab-testing'
                                  ? 'text-primary-600 dark:text-primary-400'
                                  : 'text-sky-600 dark:text-sky-400 group-hover:text-primary-600 dark:group-hover:text-primary-400'
                              )} />
                            </div>
                            <div className="text-left">
                              <div className="font-semibold text-sm">A/B Testing</div>
                              <div className="text-xs text-neutral-500 dark:text-neutral-400">
                                Model comparison & experimentation
                              </div>
                            </div>
                          </div>
                          <div className="flex items-center space-x-1">
                            <div className="px-2 py-1 bg-sky-100 dark:bg-sky-900/30 text-sky-700 dark:text-sky-300 rounded-full text-xs font-medium">
                              Statistical
                            </div>
                            {activeTab === 'ab-testing' && (
                              <div className="w-2 h-2 bg-primary-500 rounded-full"></div>
                            )}
                          </div>
                        </button>

                        <button 
                          onClick={() => onTabChange('bias-mitigation')}
                          className={cn(
                            "w-full flex items-center justify-between p-3 rounded-lg transition-all duration-200 group border-2",
                            activeTab === 'bias-mitigation' 
                              ? 'bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300 border-primary-200 dark:border-primary-700 shadow-sm'
                              : 'text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800 border-transparent hover:border-primary-200 dark:hover:border-primary-700'
                          )}
                        >
                          <div className="flex items-center space-x-3">
                            <div className={cn(
                              "p-2 rounded-full transition-colors",
                              activeTab === 'bias-mitigation'
                                ? 'bg-primary-100 dark:bg-primary-900/30'
                                : 'bg-stone-100 dark:bg-stone-900/30 group-hover:bg-primary-100 dark:group-hover:bg-primary-900/30'
                            )}>
                              <Shield className={cn(
                                "w-4 h-4",
                                activeTab === 'bias-mitigation'
                                  ? 'text-primary-600 dark:text-primary-400'
                                  : 'text-stone-600 dark:text-stone-400 group-hover:text-primary-600 dark:group-hover:text-primary-400'
                              )} />
                            </div>
                            <div className="text-left">
                              <div className="font-semibold text-sm">Bias Mitigation</div>
                              <div className="text-xs text-neutral-500 dark:text-neutral-400">
                                Fairness interventions & workflows
                              </div>
                            </div>
                          </div>
                          <div className="flex items-center space-x-1">
                            <div className="px-2 py-1 bg-stone-100 dark:bg-stone-900/30 text-stone-700 dark:text-stone-300 rounded-full text-xs font-medium">
                              Fairness
                            </div>
                            {activeTab === 'bias-mitigation' && (
                              <div className="w-2 h-2 bg-primary-500 rounded-full"></div>
                            )}
                          </div>
                        </button>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              {/* Data Section */}
              <div>
                <button
                  onClick={() => toggleSection('data')}
                  className="w-full flex items-center justify-between p-2 text-sm font-medium text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800 rounded-lg transition-colors"
                >
                  <div className="flex items-center space-x-2">
                    <Database className="w-4 h-4" />
                    <span>Data</span>
                  </div>
                  {expandedSections.includes('data') ? (
                    <ChevronDown className="w-4 h-4" />
                  ) : (
                    <ChevronRight className="w-4 h-4" />
                  )}
                </button>
                
                <AnimatePresence>
                  {expandedSections.includes('data') && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.2 }}
                      className="overflow-hidden"
                    >
                      <div className="pl-4 mt-2 space-y-1">
                        <button 
                          onClick={() => onTabChange('data-connectivity')}
                          className={cn(
                            "w-full flex items-center justify-between p-3 rounded-lg transition-all duration-200 group border-2",
                            activeTab === 'data-connectivity' 
                              ? 'bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300 border-primary-200 dark:border-primary-700 shadow-sm'
                              : 'text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800 border-transparent hover:border-primary-200 dark:hover:border-primary-700'
                          )}
                        >
                          <div className="flex items-center space-x-3">
                            <div className={cn(
                              "p-2 rounded-full transition-colors",
                              activeTab === 'data-connectivity'
                                ? 'bg-primary-100 dark:bg-primary-900/30'
                                : 'bg-blue-100 dark:bg-blue-900/30 group-hover:bg-primary-100 dark:group-hover:bg-primary-900/30'
                            )}>
                              <Database className={cn(
                                "w-4 h-4",
                                activeTab === 'data-connectivity'
                                  ? 'text-primary-600 dark:text-primary-400'
                                  : 'text-blue-600 dark:text-blue-400 group-hover:text-primary-600 dark:group-hover:text-primary-400'
                              )} />
                            </div>
                            <div className="text-left">
                              <div className="font-semibold text-sm">Data Connectivity</div>
                              <div className="text-xs text-neutral-500 dark:text-neutral-400">
                                Upload & manage datasets
                              </div>
                            </div>
                          </div>
                          <div className="flex items-center space-x-1">
                            <div className="px-2 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded-full text-xs font-medium">
                              New
                            </div>
                            {activeTab === 'data-connectivity' && (
                              <div className="w-2 h-2 bg-primary-500 rounded-full"></div>
                            )}
                          </div>
                        </button>
                        
                        <button 
                          onClick={() => onTabChange('data-drift')}
                          className={cn(
                            "w-full flex items-center justify-between p-3 rounded-lg transition-all duration-200 group border-2",
                            activeTab === 'data-drift' 
                              ? 'bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300 border-primary-200 dark:border-primary-700 shadow-sm'
                              : 'text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800 border-transparent hover:border-primary-200 dark:hover:border-primary-700'
                          )}
                        >
                          <div className="flex items-center space-x-3">
                            <div className={cn(
                              "p-2 rounded-full transition-colors",
                              activeTab === 'data-drift'
                                ? 'bg-primary-100 dark:bg-primary-900/30'
                                : 'bg-orange-100 dark:bg-orange-900/30 group-hover:bg-primary-100 dark:group-hover:bg-primary-900/30'
                            )}>
                              <TrendingUp className={cn(
                                "w-4 h-4",
                                activeTab === 'data-drift'
                                  ? 'text-primary-600 dark:text-primary-400'
                                  : 'text-orange-600 dark:text-orange-400 group-hover:text-primary-600 dark:group-hover:text-primary-400'
                              )} />
                            </div>
                            <div className="text-left">
                              <div className="font-semibold text-sm">Data Drift</div>
                              <div className="text-xs text-neutral-500 dark:text-neutral-400">
                                Detect statistical changes in data
                              </div>
                            </div>
                          </div>
                          <div className="flex items-center space-x-1">
                            <div className="px-2 py-1 bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-300 rounded-full text-xs font-medium">
                              New
                            </div>
                            {activeTab === 'data-drift' && (
                              <div className="w-2 h-2 bg-primary-500 rounded-full"></div>
                            )}
                          </div>
                        </button>
                        
                        <button 
                          onClick={() => onTabChange('model-performance')}
                          className={cn(
                            "w-full flex items-center justify-between p-3 rounded-lg transition-all duration-200 group border-2",
                            activeTab === 'model-performance' 
                              ? 'bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300 border-primary-200 dark:border-primary-700 shadow-sm'
                              : 'text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800 border-transparent hover:border-primary-200 dark:hover:border-primary-700'
                          )}
                        >
                          <div className="flex items-center space-x-3">
                            <div className={cn(
                              "p-2 rounded-full transition-colors",
                              activeTab === 'model-performance'
                                ? 'bg-primary-100 dark:bg-primary-900/30'
                                : 'bg-purple-100 dark:bg-purple-900/30 group-hover:bg-primary-100 dark:group-hover:bg-primary-900/30'
                            )}>
                              <Activity className={cn(
                                "w-4 h-4",
                                activeTab === 'model-performance'
                                  ? 'text-primary-600 dark:text-primary-400'
                                  : 'text-purple-600 dark:text-purple-400 group-hover:text-primary-600 dark:group-hover:text-primary-400'
                              )} />
                            </div>
                            <div className="text-left">
                              <div className="font-semibold text-sm">Model Performance</div>
                              <div className="text-xs text-neutral-500 dark:text-neutral-400">
                                Enterprise monitoring & A/B testing
                              </div>
                            </div>
                          </div>
                          <div className="flex items-center space-x-1">
                            <div className="px-2 py-1 bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 rounded-full text-xs font-medium">
                              Pro
                            </div>
                            {activeTab === 'model-performance' && (
                              <div className="w-2 h-2 bg-primary-500 rounded-full"></div>
                            )}
                          </div>
                        </button>

                        <button 
                          onClick={() => onTabChange('drift-configuration')}
                          className={cn(
                            "w-full flex items-center justify-between p-3 rounded-lg transition-all duration-200 group border-2",
                            activeTab === 'drift-configuration' 
                              ? 'bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300 border-primary-200 dark:border-primary-700 shadow-sm'
                              : 'text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800 border-transparent hover:border-primary-200 dark:hover:border-primary-700'
                          )}
                        >
                          <div className="flex items-center space-x-3">
                            <div className={cn(
                              "p-2 rounded-full transition-colors",
                              activeTab === 'drift-configuration'
                                ? 'bg-primary-100 dark:bg-primary-900/30'
                                : 'bg-amber-100 dark:bg-amber-900/30 group-hover:bg-primary-100 dark:group-hover:bg-primary-900/30'
                            )}>
                              <SettingsIcon className={cn(
                                "w-4 h-4",
                                activeTab === 'drift-configuration'
                                  ? 'text-primary-600 dark:text-primary-400'
                                  : 'text-amber-600 dark:text-amber-400 group-hover:text-primary-600 dark:group-hover:text-primary-400'
                              )} />
                            </div>
                            <div className="text-left">
                              <div className="font-semibold text-sm">Drift Configuration</div>
                              <div className="text-xs text-neutral-500 dark:text-neutral-400">
                                Advanced drift detection settings
                              </div>
                            </div>
                          </div>
                          <div className="flex items-center space-x-1">
                            <div className="px-2 py-1 bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 rounded-full text-xs font-medium">
                              Advanced
                            </div>
                            {activeTab === 'drift-configuration' && (
                              <div className="w-2 h-2 bg-primary-500 rounded-full"></div>
                            )}
                          </div>
                        </button>

                        <button 
                          onClick={() => onTabChange('data-management')}
                          className={cn(
                            "w-full flex items-center justify-between p-3 rounded-lg transition-all duration-200 group border-2",
                            activeTab === 'data-management' 
                              ? 'bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300 border-primary-200 dark:border-primary-700 shadow-sm'
                              : 'text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800 border-transparent hover:border-primary-200 dark:hover:border-primary-700'
                          )}
                        >
                          <div className="flex items-center space-x-3">
                            <div className={cn(
                              "p-2 rounded-full transition-colors",
                              activeTab === 'data-management'
                                ? 'bg-primary-100 dark:bg-primary-900/30'
                                : 'bg-cyan-100 dark:bg-cyan-900/30 group-hover:bg-primary-100 dark:group-hover:bg-primary-900/30'
                            )}>
                              <FolderOpen className={cn(
                                "w-4 h-4",
                                activeTab === 'data-management'
                                  ? 'text-primary-600 dark:text-primary-400'
                                  : 'text-cyan-600 dark:text-cyan-400 group-hover:text-primary-600 dark:group-hover:text-primary-400'
                              )} />
                            </div>
                            <div className="text-left">
                              <div className="font-semibold text-sm">Data Management</div>
                              <div className="text-xs text-neutral-500 dark:text-neutral-400">
                                Complete data lifecycle portal
                              </div>
                            </div>
                          </div>
                          <div className="flex items-center space-x-1">
                            <div className="px-2 py-1 bg-cyan-100 dark:bg-cyan-900/30 text-cyan-700 dark:text-cyan-300 rounded-full text-xs font-medium">
                              Enterprise
                            </div>
                            {activeTab === 'data-management' && (
                              <div className="w-2 h-2 bg-primary-500 rounded-full"></div>
                            )}
                          </div>
                        </button>

                        <button 
                          onClick={() => onTabChange('data-quality')}
                          className={cn(
                            "w-full flex items-center justify-between p-3 rounded-lg transition-all duration-200 group border-2",
                            activeTab === 'data-quality' 
                              ? 'bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300 border-primary-200 dark:border-primary-700 shadow-sm'
                              : 'text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800 border-transparent hover:border-primary-200 dark:hover:border-primary-700'
                          )}
                        >
                          <div className="flex items-center space-x-3">
                            <div className={cn(
                              "p-2 rounded-full transition-colors",
                              activeTab === 'data-quality'
                                ? 'bg-primary-100 dark:bg-primary-900/30'
                                : 'bg-pink-100 dark:bg-pink-900/30 group-hover:bg-primary-100 dark:group-hover:bg-primary-900/30'
                            )}>
                              <CheckCircle className={cn(
                                "w-4 h-4",
                                activeTab === 'data-quality'
                                  ? 'text-primary-600 dark:text-primary-400'
                                  : 'text-pink-600 dark:text-pink-400 group-hover:text-primary-600 dark:group-hover:text-primary-400'
                              )} />
                            </div>
                            <div className="text-left">
                              <div className="font-semibold text-sm">Data Quality</div>
                              <div className="text-xs text-neutral-500 dark:text-neutral-400">
                                Quality assessment & monitoring
                              </div>
                            </div>
                          </div>
                          <div className="flex items-center space-x-1">
                            <div className="px-2 py-1 bg-pink-100 dark:bg-pink-900/30 text-pink-700 dark:text-pink-300 rounded-full text-xs font-medium">
                              New
                            </div>
                            {activeTab === 'data-quality' && (
                              <div className="w-2 h-2 bg-primary-500 rounded-full"></div>
                            )}
                          </div>
                        </button>

                        <button 
                          onClick={() => onTabChange('stream-management')}
                          className={cn(
                            "w-full flex items-center justify-between p-3 rounded-lg transition-all duration-200 group border-2",
                            activeTab === 'stream-management' 
                              ? 'bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300 border-primary-200 dark:border-primary-700 shadow-sm'
                              : 'text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800 border-transparent hover:border-primary-200 dark:hover:border-primary-700'
                          )}
                        >
                          <div className="flex items-center space-x-3">
                            <div className={cn(
                              "p-2 rounded-full transition-colors",
                              activeTab === 'stream-management'
                                ? 'bg-primary-100 dark:bg-primary-900/30'
                                : 'bg-rose-100 dark:bg-rose-900/30 group-hover:bg-primary-100 dark:group-hover:bg-primary-900/30'
                            )}>
                              <Activity className={cn(
                                "w-4 h-4",
                                activeTab === 'stream-management'
                                  ? 'text-primary-600 dark:text-primary-400'
                                  : 'text-rose-600 dark:text-rose-400 group-hover:text-primary-600 dark:group-hover:text-primary-400'
                              )} />
                            </div>
                            <div className="text-left">
                              <div className="font-semibold text-sm">Stream Management</div>
                              <div className="text-xs text-neutral-500 dark:text-neutral-400">
                                Real-time data streaming
                              </div>
                            </div>
                          </div>
                          <div className="flex items-center space-x-1">
                            <div className="px-2 py-1 bg-rose-100 dark:bg-rose-900/30 text-rose-700 dark:text-rose-300 rounded-full text-xs font-medium">
                              Real-time
                            </div>
                            {activeTab === 'stream-management' && (
                              <div className="w-2 h-2 bg-primary-500 rounded-full"></div>
                            )}
                          </div>
                        </button>

                        <button 
                          onClick={() => onTabChange('data-preprocessing')}
                          className={cn(
                            "w-full flex items-center justify-between p-3 rounded-lg transition-all duration-200 group border-2",
                            activeTab === 'data-preprocessing' 
                              ? 'bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300 border-primary-200 dark:border-primary-700 shadow-sm'
                              : 'text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800 border-transparent hover:border-primary-200 dark:hover:border-primary-700'
                          )}
                        >
                          <div className="flex items-center space-x-3">
                            <div className={cn(
                              "p-2 rounded-full transition-colors",
                              activeTab === 'data-preprocessing'
                                ? 'bg-primary-100 dark:bg-primary-900/30'
                                : 'bg-lime-100 dark:bg-lime-900/30 group-hover:bg-primary-100 dark:group-hover:bg-primary-900/30'
                            )}>
                              <Settings className={cn(
                                "w-4 h-4",
                                activeTab === 'data-preprocessing'
                                  ? 'text-primary-600 dark:text-primary-400'
                                  : 'text-lime-600 dark:text-lime-400 group-hover:text-primary-600 dark:group-hover:text-primary-400'
                              )} />
                            </div>
                            <div className="text-left">
                              <div className="font-semibold text-sm">Data Preprocessing</div>
                              <div className="text-xs text-neutral-500 dark:text-neutral-400">
                                Transform & clean data workflows
                              </div>
                            </div>
                          </div>
                          <div className="flex items-center space-x-1">
                            <div className="px-2 py-1 bg-lime-100 dark:bg-lime-900/30 text-lime-700 dark:text-lime-300 rounded-full text-xs font-medium">
                              Workflow
                            </div>
                            {activeTab === 'data-preprocessing' && (
                              <div className="w-2 h-2 bg-primary-500 rounded-full"></div>
                            )}
                          </div>
                        </button>
                        
                        <button className="w-full flex items-center space-x-3 p-3 rounded-lg text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors">
                          <FileText className="w-4 h-4" />
                          <div className="text-left">
                            <div className="font-medium text-sm">Training Data</div>
                            <div className="text-xs text-neutral-500 dark:text-neutral-400">
                              10,000 samples
                            </div>
                          </div>
                        </button>
                        <button className="w-full flex items-center space-x-3 p-3 rounded-lg text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors">
                          <FileText className="w-4 h-4" />
                          <div className="text-left">
                            <div className="font-medium text-sm">Test Data</div>
                            <div className="text-xs text-neutral-500 dark:text-neutral-400">
                              2,500 samples
                            </div>
                          </div>
                        </button>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              {/* Settings Section */}
              <div>
                <button
                  onClick={() => toggleSection('settings')}
                  className="w-full flex items-center justify-between p-2 text-sm font-medium text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800 rounded-lg transition-colors"
                >
                  <div className="flex items-center space-x-2">
                    <Settings className="w-4 h-4" />
                    <span>Settings</span>
                  </div>
                  {expandedSections.includes('settings') ? (
                    <ChevronDown className="w-4 h-4" />
                  ) : (
                    <ChevronRight className="w-4 h-4" />
                  )}
                </button>
                
                <AnimatePresence>
                  {expandedSections.includes('settings') && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.2 }}
                      className="overflow-hidden"
                    >
                      <div className="pl-4 mt-2 space-y-1">
                        <button 
                          onClick={() => onTabChange('settings')}
                          className={cn(
                            "w-full flex items-center space-x-3 p-3 rounded-lg transition-colors",
                            activeTab === 'settings'
                              ? "bg-primary-100 dark:bg-primary-900 text-primary-700 dark:text-primary-300"
                              : "text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800"
                          )}
                        >
                          <Settings className="w-4 h-4" />
                          <span className="font-medium text-sm">System Settings</span>
                        </button>
                        <button 
                          onClick={() => onTabChange('user-management')}
                          className={cn(
                            "w-full flex items-center space-x-3 p-3 rounded-lg transition-colors",
                            activeTab === 'user-management'
                              ? "bg-primary-100 dark:bg-primary-900 text-primary-700 dark:text-primary-300"
                              : "text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800"
                          )}
                        >
                          <Users className="w-4 h-4" />
                          <span className="font-medium text-sm">User Management</span>
                        </button>
                        <button 
                          onClick={() => onTabChange('system-health')}
                          className={cn(
                            "w-full flex items-center space-x-3 p-3 rounded-lg transition-colors",
                            activeTab === 'system-health'
                              ? "bg-primary-100 dark:bg-primary-900 text-primary-700 dark:text-primary-300"
                              : "text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800"
                          )}
                        >
                          <Activity className="w-4 h-4" />
                          <span className="font-medium text-sm">System Health</span>
                        </button>
                        <button 
                          onClick={() => onTabChange('user-profile')}
                          className={cn(
                            "w-full flex items-center space-x-3 p-3 rounded-lg transition-colors",
                            activeTab === 'user-profile'
                              ? "bg-primary-100 dark:bg-primary-900 text-primary-700 dark:text-primary-300"
                              : "text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800"
                          )}
                        >
                          <User className="w-4 h-4" />
                          <span className="font-medium text-sm">Profile</span>
                        </button>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </div>
          </nav>

          {/* Footer */}
          <div className="p-4 border-t border-neutral-200 dark:border-neutral-800">
            <div className="flex items-center justify-between text-xs text-neutral-500 dark:text-neutral-400">
              <span>Last updated: 2 min ago</span>
              <div className="flex items-center space-x-1">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span>Connected</span>
              </div>
            </div>
          </div>
        </div>
      </motion.aside>
    </>
  );
};

export default Sidebar;