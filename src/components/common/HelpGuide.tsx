import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Book, Play, Target, BarChart3, TrendingUp, Eye, Zap, Network, GitBranch, Activity, Lightbulb, CheckCircle } from 'lucide-react';
import Button from './Button';

interface HelpGuideProps {
  isOpen: boolean;
  onClose: () => void;
  activeTab?: string;
}

interface GuideSection {
  id: string;
  title: string;
  icon: React.ComponentType<any>;
  description: string;
  steps: string[];
  tips: string[];
  useCases: string[];
}

const HelpGuide: React.FC<HelpGuideProps> = ({ isOpen, onClose, activeTab }) => {
  const [expandedSection, setExpandedSection] = useState<string>(activeTab || 'overview');

  const guideSections: GuideSection[] = [
    {
      id: 'overview',
      title: 'Model Overview',
      icon: Activity,
      description: 'Get a high-level understanding of your model\'s performance and key metrics.',
      steps: [
        'Review the model performance metrics (accuracy, precision, recall)',
        'Check the model information (type, training date, version)',
        'Examine the feature count and data statistics',
        'Look at any alerts or warnings about model performance'
      ],
      tips: [
        'Green metrics indicate good performance',
        'Red alerts require immediate attention',
        'Compare current metrics with historical performance'
      ],
      useCases: [
        'Daily model monitoring',
        'Performance health checks',
        'Model deployment validation'
      ]
    },
    {
      id: 'feature-importance',
      title: 'Feature Importance',
      icon: BarChart3,
      description: 'Understand which features have the most impact on your model\'s predictions.',
      steps: [
        'Select the importance calculation method (SHAP, Permutation, Gain)',
        'Use the search bar to find specific features',
        'Sort features by importance or alphabetically',
        'Analyze the interactive bar chart and detailed rankings',
        'Review the correlation matrix for feature relationships'
      ],
      tips: [
        'Higher importance values indicate more influential features',
        'Use different methods to validate feature importance',
        'Focus on top 5-10 features for model interpretation'
      ],
      useCases: [
        'Feature selection for model improvement',
        'Understanding model behavior',
        'Compliance and regulatory reporting'
      ]
    },
    {
      id: 'classification-stats',
      title: 'Classification Statistics',
      icon: Target,
      description: 'Analyze detailed classification performance metrics and visualizations.',
      steps: [
        'Examine the confusion matrix for prediction accuracy',
        'Review the ROC curve and AUC score',
        'Adjust the classification threshold using the slider',
        'Monitor how threshold changes affect precision and recall',
        'Check the circular progress indicators for key metrics'
      ],
      tips: [
        'Higher AUC values (>0.8) indicate better model performance',
        'Balance precision and recall based on business needs',
        'Use threshold analysis to optimize for specific outcomes'
      ],
      useCases: [
        'Model evaluation and validation',
        'Threshold optimization',
        'Performance reporting to stakeholders'
      ]
    },
    {
      id: 'regression-stats',
      title: 'Regression Statistics',
      icon: TrendingUp,
      description: 'Evaluate regression model performance with diagnostic plots and metrics.',
      steps: [
        'Review the regression metrics (R², MSE, RMSE, MAE)',
        'Select different diagnostic plots (Residual, Q-Q, Predicted vs Actual)',
        'Analyze the residual patterns for model assumptions',
        'Check the performance summary for model quality assessment',
        'Examine error analysis for prediction accuracy'
      ],
      tips: [
        'R² values closer to 1 indicate better fit',
        'Random residual patterns suggest good model fit',
        'Look for heteroscedasticity in residual plots'
      ],
      useCases: [
        'Model diagnostics and validation',
        'Identifying model assumptions violations',
        'Comparing different regression models'
      ]
    },
    {
      id: 'predictions',
      title: 'Individual Predictions',
      icon: Eye,
      description: 'Analyze individual prediction explanations using SHAP values.',
      steps: [
        'Select an instance from the list on the left',
        'Sort instances by prediction confidence or value',
        'Examine the SHAP waterfall chart for feature contributions',
        'Review the prediction summary (actual vs predicted)',
        'Analyze feature contributions in the detailed breakdown'
      ],
      tips: [
        'Positive SHAP values increase the prediction',
        'Negative SHAP values decrease the prediction',
        'Focus on top contributing features for explanation'
      ],
      useCases: [
        'Explaining individual predictions to customers',
        'Model debugging and validation',
        'Regulatory compliance and auditing'
      ]
    },
    {
      id: 'what-if',
      title: 'What-If Analysis',
      icon: Zap,
      description: 'Perform interactive scenario analysis by modifying feature values.',
      steps: [
        'Select an instance to analyze',
        'Use the sliders to modify feature values',
        'Watch the prediction update in real-time',
        'Compare the new prediction with the original',
        'Save interesting scenarios for future reference'
      ],
      tips: [
        'Start with small changes to understand sensitivity',
        'Focus on the most important features first',
        'Use scenarios to test edge cases'
      ],
      useCases: [
        'Customer advisory and consultation',
        'Risk assessment and mitigation',
        'Product optimization and pricing'
      ]
    },
    {
      id: 'feature-dependence',
      title: 'Feature Dependence',
      icon: TrendingUp,
      description: 'Explore how individual features affect predictions across their value ranges.',
      steps: [
        'Select a feature from the searchable list',
        'Choose the plot type (Partial Dependence, SHAP, ICE)',
        'Analyze the dependence curve for the selected feature',
        'Review the impact summary and trend analysis',
        'Compare different features to understand their effects'
      ],
      tips: [
        'Steep curves indicate high feature sensitivity',
        'Flat curves suggest minimal feature impact',
        'Look for non-linear relationships and interactions'
      ],
      useCases: [
        'Understanding feature behavior',
        'Identifying optimal feature ranges',
        'Model interpretation and validation'
      ]
    },
    {
      id: 'feature-interactions',
      title: 'Feature Interactions',
      icon: Network,
      description: 'Discover how features interact with each other to influence predictions.',
      steps: [
        'Start with the interaction heatmap for an overview',
        'Switch to network view to see relationship patterns',
        'Use pairwise analysis for detailed feature comparisons',
        'Adjust the minimum interaction strength filter',
        'Review the top interactions ranking'
      ],
      tips: [
        'Strong interactions (>0.3) indicate important feature pairs',
        'Synergistic interactions amplify each other\'s effects',
        'Redundant interactions suggest correlated features'
      ],
      useCases: [
        'Feature engineering and selection',
        'Understanding complex model behavior',
        'Identifying feature redundancies'
      ]
    },
    {
      id: 'decision-trees',
      title: 'Decision Trees',
      icon: GitBranch,
      description: 'Visualize and analyze decision tree structures for tree-based models.',
      steps: [
        'Select a tree from the ensemble',
        'Adjust the maximum depth for visualization',
        'Click on nodes to highlight decision paths',
        'Switch to Rules view to see decision logic',
        'Review tree statistics and comparison metrics'
      ],
      tips: [
        'Green nodes are leaf nodes (final decisions)',
        'Blue nodes are decision points (splits)',
        'Deeper trees may indicate overfitting'
      ],
      useCases: [
        'Understanding model decision logic',
        'Debugging tree-based models',
        'Creating business rules from model insights'
      ]
    }
  ];

  const currentSection = guideSections.find(section => section.id === expandedSection);

  const containerVariants = {
    hidden: { opacity: 0, x: 300 },
    visible: { opacity: 1, x: 0 },
    exit: { opacity: 0, x: 300 }
  };

  const contentVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 },
    exit: { opacity: 0, y: -20 }
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 z-40"
            onClick={onClose}
          />

          {/* Help Panel */}
          <motion.div
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            exit="exit"
            transition={{ type: "spring", damping: 25, stiffness: 300 }}
            className="fixed right-0 top-0 h-full w-full max-w-2xl bg-white dark:bg-neutral-900 shadow-2xl z-50 overflow-hidden"
          >
            <div className="flex flex-col h-full">
              {/* Header */}
              <div className="flex items-center justify-between p-6 border-b border-neutral-200 dark:border-neutral-700">
                <div className="flex items-center space-x-3">
                  <div className="p-2 bg-primary-100 dark:bg-primary-900 rounded-lg">
                    <Book className="w-5 h-5 text-primary-600 dark:text-primary-400" />
                  </div>
                  <div>
                    <h2 className="text-xl font-bold text-neutral-900 dark:text-neutral-100">
                      User Guide
                    </h2>
                    <p className="text-sm text-neutral-600 dark:text-neutral-400">
                      Learn how to use the ML Explainer Dashboard
                    </p>
                  </div>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={onClose}
                  leftIcon={<X className="w-4 h-4" />}
                >
                  Close
                </Button>
              </div>

              <div className="flex-1 flex overflow-hidden">
                {/* Sidebar */}
                <div className="w-64 border-r border-neutral-200 dark:border-neutral-700 overflow-y-auto">
                  <div className="p-4 space-y-2">
                    {guideSections.map((section) => {
                      const Icon = section.icon;
                      return (
                        <button
                          key={section.id}
                          onClick={() => setExpandedSection(section.id)}
                          className={`w-full flex items-center space-x-3 p-3 rounded-lg text-left transition-all ${
                            expandedSection === section.id
                              ? 'bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300'
                              : 'hover:bg-neutral-50 dark:hover:bg-neutral-800 text-neutral-700 dark:text-neutral-300'
                          }`}
                        >
                          <Icon className="w-4 h-4 flex-shrink-0" />
                          <span className="font-medium text-sm">{section.title}</span>
                        </button>
                      );
                    })}
                  </div>
                </div>

                {/* Content */}
                <div className="flex-1 overflow-y-auto">
                  <AnimatePresence mode="wait">
                    {currentSection && (
                      <motion.div
                        key={currentSection.id}
                        variants={contentVariants}
                        initial="hidden"
                        animate="visible"
                        exit="exit"
                        transition={{ duration: 0.2 }}
                        className="p-6 space-y-6"
                      >
                        {/* Section Header */}
                        <div>
                          <div className="flex items-center space-x-3 mb-3">
                            <div className="p-2 bg-primary-100 dark:bg-primary-900 rounded-lg">
                              <currentSection.icon className="w-5 h-5 text-primary-600 dark:text-primary-400" />
                            </div>
                            <h3 className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
                              {currentSection.title}
                            </h3>
                          </div>
                          <p className="text-neutral-600 dark:text-neutral-400">
                            {currentSection.description}
                          </p>
                        </div>

                        {/* Steps */}
                        <div>
                          <h4 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-3 flex items-center">
                            <Play className="w-4 h-4 mr-2 text-green-600" />
                            Step-by-Step Guide
                          </h4>
                          <div className="space-y-3">
                            {currentSection.steps.map((step, index) => (
                              <div key={index} className="flex items-start space-x-3">
                                <div className="w-6 h-6 bg-primary-100 dark:bg-primary-900 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                                  <span className="text-xs font-bold text-primary-600 dark:text-primary-400">
                                    {index + 1}
                                  </span>
                                </div>
                                <p className="text-sm text-neutral-700 dark:text-neutral-300 leading-relaxed">
                                  {step}
                                </p>
                              </div>
                            ))}
                          </div>
                        </div>

                        {/* Tips */}
                        <div>
                          <h4 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-3 flex items-center">
                            <Lightbulb className="w-4 h-4 mr-2 text-amber-600" />
                            Pro Tips
                          </h4>
                          <div className="space-y-2">
                            {currentSection.tips.map((tip, index) => (
                              <div key={index} className="flex items-start space-x-3">
                                <CheckCircle className="w-4 h-4 text-green-600 flex-shrink-0 mt-0.5" />
                                <p className="text-sm text-neutral-700 dark:text-neutral-300">
                                  {tip}
                                </p>
                              </div>
                            ))}
                          </div>
                        </div>

                        {/* Use Cases */}
                        <div>
                          <h4 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-3 flex items-center">
                            <Target className="w-4 h-4 mr-2 text-blue-600" />
                            Common Use Cases
                          </h4>
                          <div className="space-y-2">
                            {currentSection.useCases.map((useCase, index) => (
                              <div key={index} className="flex items-start space-x-3">
                                <div className="w-2 h-2 bg-blue-600 rounded-full flex-shrink-0 mt-2" />
                                <p className="text-sm text-neutral-700 dark:text-neutral-300">
                                  {useCase}
                                </p>
                              </div>
                            ))}
                          </div>
                        </div>

                        {/* Quick Start */}
                        <div className="bg-gradient-to-r from-primary-50 to-blue-50 dark:from-primary-900/20 dark:to-blue-900/20 p-4 rounded-lg">
                          <h4 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-2">
                            Quick Start
                          </h4>
                          <p className="text-sm text-neutral-700 dark:text-neutral-300 mb-3">
                            New to this feature? Start with these essential actions:
                          </p>
                          <div className="flex flex-wrap gap-2">
                            <span className="px-3 py-1 bg-white dark:bg-neutral-800 rounded-full text-xs font-medium text-neutral-700 dark:text-neutral-300">
                              📊 Explore the main visualization
                            </span>
                            <span className="px-3 py-1 bg-white dark:bg-neutral-800 rounded-full text-xs font-medium text-neutral-700 dark:text-neutral-300">
                              🎛️ Try the interactive controls
                            </span>
                            <span className="px-3 py-1 bg-white dark:bg-neutral-800 rounded-full text-xs font-medium text-neutral-700 dark:text-neutral-300">
                              💡 Review the insights
                            </span>
                          </div>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};

export default HelpGuide;