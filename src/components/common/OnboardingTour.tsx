import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, ArrowRight, ArrowLeft, Play, CheckCircle } from 'lucide-react';
import Button from './Button';

interface OnboardingTourProps {
  isOpen: boolean;
  onClose: () => void;
  onTabChange: (tabId: string) => void;
}

interface TourStep {
  id: string;
  title: string;
  description: string;
  tabId: string;
  highlight?: string;
  tip: string;
}

const OnboardingTour: React.FC<OnboardingTourProps> = ({ isOpen, onClose, onTabChange }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [hasStarted, setHasStarted] = useState(false);

  const tourSteps: TourStep[] = [
    {
      id: 'welcome',
      title: 'Welcome to ML Explainer Dashboard',
      description: 'This powerful tool helps you understand and interpret your machine learning models with interactive visualizations and detailed explanations.',
      tabId: 'overview',
      tip: 'Start your journey by exploring the Model Overview to get familiar with your model\'s performance.'
    },
    {
      id: 'overview',
      title: 'Model Overview',
      description: 'Get a high-level view of your model\'s performance, key metrics, and important information at a glance.',
      tabId: 'overview',
      highlight: 'Performance metrics and model information',
      tip: 'Look for green metrics (good performance) and red alerts (needs attention).'
    },
    {
      id: 'feature-importance',
      title: 'Feature Importance',
      description: 'Discover which features have the most impact on your model\'s predictions using different calculation methods.',
      tabId: 'feature-importance',
      highlight: 'Interactive bar charts and feature rankings',
      tip: 'Try different importance methods (SHAP, Permutation, Gain) to validate your findings.'
    },
    {
      id: 'predictions',
      title: 'Individual Predictions',
      description: 'Dive deep into specific predictions with SHAP values to understand why the model made certain decisions.',
      tabId: 'predictions',
      highlight: 'SHAP waterfall charts and feature contributions',
      tip: 'Select different instances to see how feature values impact individual predictions.'
    },
    {
      id: 'what-if',
      title: 'What-If Analysis',
      description: 'Experiment with different scenarios by changing feature values and seeing how predictions change in real-time.',
      tabId: 'what-if',
      highlight: 'Interactive sliders and real-time updates',
      tip: 'Start with small changes to understand feature sensitivity and model behavior.'
    },
    {
      id: 'complete',
      title: 'You\'re Ready to Go!',
      description: 'You now have a good understanding of the main features. Explore the other tabs to discover more advanced analysis capabilities.',
      tabId: 'overview',
      tip: 'Use the help button (?) in the header anytime you need guidance on specific features.'
    }
  ];

  const currentTourStep = tourSteps[currentStep];

  const handleNext = () => {
    if (currentStep < tourSteps.length - 1) {
      const nextStep = currentStep + 1;
      setCurrentStep(nextStep);
      onTabChange(tourSteps[nextStep].tabId);
    }
  };

  const handlePrevious = () => {
    if (currentStep > 0) {
      const prevStep = currentStep - 1;
      setCurrentStep(prevStep);
      onTabChange(tourSteps[prevStep].tabId);
    }
  };

  const handleStart = () => {
    setHasStarted(true);
    onTabChange(tourSteps[0].tabId);
  };

  const handleFinish = () => {
    // Store tour completion in localStorage
    localStorage.setItem('ml-explainer-tour-completed', 'true');
    onClose();
  };

  const handleSkip = () => {
    localStorage.setItem('ml-explainer-tour-completed', 'true');
    onClose();
  };

  const containerVariants = {
    hidden: { opacity: 0, scale: 0.9 },
    visible: { opacity: 1, scale: 1 },
    exit: { opacity: 0, scale: 0.9 }
  };

  const contentVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 },
    exit: { opacity: 0, y: -20 }
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/70 z-50 flex items-center justify-center p-4"
      >
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          exit="exit"
          transition={{ type: "spring", damping: 25, stiffness: 300 }}
          className="bg-white dark:bg-neutral-900 rounded-2xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-hidden"
        >
          {!hasStarted ? (
            // Welcome Screen
            <div className="p-8 text-center">
              <div className="w-20 h-20 bg-gradient-to-br from-primary-500 to-accent-500 rounded-full flex items-center justify-center mx-auto mb-6">
                <Play className="w-10 h-10 text-white" />
              </div>
              <h2 className="text-3xl font-bold text-neutral-900 dark:text-neutral-100 mb-4">
                Welcome to ML Explainer Dashboard!
              </h2>
              <p className="text-lg text-neutral-600 dark:text-neutral-400 mb-8 leading-relaxed">
                Take a quick tour to learn how to use this powerful tool for understanding your machine learning models.
              </p>
              <div className="flex items-center justify-center space-x-4">
                <Button
                  variant="outline"
                  onClick={handleSkip}
                  className="min-w-24"
                >
                  Skip Tour
                </Button>
                <Button
                  onClick={handleStart}
                  leftIcon={<Play className="w-4 h-4" />}
                  className="min-w-32"
                >
                  Start Tour
                </Button>
              </div>
            </div>
          ) : (
            // Tour Steps
            <div className="flex flex-col h-full">
              {/* Header */}
              <div className="flex items-center justify-between p-6 border-b border-neutral-200 dark:border-neutral-700">
                <div className="flex items-center space-x-3">
                  <div className="text-sm font-medium text-neutral-500 dark:text-neutral-400">
                    Step {currentStep + 1} of {tourSteps.length}
                  </div>
                  <div className="flex space-x-1">
                    {tourSteps.map((_, index) => (
                      <div
                        key={index}
                        className={`w-2 h-2 rounded-full ${
                          index <= currentStep
                            ? 'bg-primary-500'
                            : 'bg-neutral-300 dark:bg-neutral-600'
                        }`}
                      />
                    ))}
                  </div>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleSkip}
                  leftIcon={<X className="w-4 h-4" />}
                >
                  Skip
                </Button>
              </div>

              {/* Content */}
              <div className="flex-1 p-6 space-y-6">
                <AnimatePresence mode="wait">
                  <motion.div
                    key={currentStep}
                    variants={contentVariants}
                    initial="hidden"
                    animate="visible"
                    exit="exit"
                    transition={{ duration: 0.3 }}
                    className="space-y-4"
                  >
                    <h3 className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
                      {currentTourStep.title}
                    </h3>
                    <p className="text-lg text-neutral-600 dark:text-neutral-400 leading-relaxed">
                      {currentTourStep.description}
                    </p>
                    
                    {currentTourStep.highlight && (
                      <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
                        <p className="text-sm font-medium text-blue-900 dark:text-blue-100">
                          💡 Focus on: {currentTourStep.highlight}
                        </p>
                      </div>
                    )}
                    
                    <div className="bg-amber-50 dark:bg-amber-900/20 p-4 rounded-lg">
                      <p className="text-sm text-amber-900 dark:text-amber-100">
                        <strong>Tip:</strong> {currentTourStep.tip}
                      </p>
                    </div>
                  </motion.div>
                </AnimatePresence>
              </div>

              {/* Footer */}
              <div className="flex items-center justify-between p-6 border-t border-neutral-200 dark:border-neutral-700">
                <Button
                  variant="outline"
                  onClick={handlePrevious}
                  disabled={currentStep === 0}
                  leftIcon={<ArrowLeft className="w-4 h-4" />}
                >
                  Previous
                </Button>
                
                <div className="flex items-center space-x-3">
                  {currentStep === tourSteps.length - 1 ? (
                    <Button
                      onClick={handleFinish}
                      leftIcon={<CheckCircle className="w-4 h-4" />}
                      className="min-w-32"
                    >
                      Finish Tour
                    </Button>
                  ) : (
                    <Button
                      onClick={handleNext}
                      rightIcon={<ArrowRight className="w-4 h-4" />}
                      className="min-w-24"
                    >
                      Next
                    </Button>
                  )}
                </div>
              </div>
            </div>
          )}
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

export default OnboardingTour;