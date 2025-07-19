import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Zap, Sliders, TrendingUp, RotateCcw, Save, GitCompare } from 'lucide-react';
import Card from '@/components/common/Card';
import Button from '@/components/common/Button';
import WhatIfAnalysisChart from '@/components/charts/WhatIfAnalysisChart';

interface WhatIfAnalysisProps {
  modelType: 'classification' | 'regression';
}

interface FeatureInput {
  name: string;
  type: 'numerical' | 'categorical';
  value: number | string;
  min?: number;
  max?: number;
  options?: string[];
  description: string;
}

interface Scenario {
  id: string;
  name: string;
  features: FeatureInput[];
  prediction: number;
  confidence: number;
  timestamp: Date;
}

const WhatIfAnalysis: React.FC<WhatIfAnalysisProps> = ({ modelType }) => {
  const [features, setFeatures] = useState<FeatureInput[]>([
    { name: 'Customer_Age', type: 'numerical', value: 35, min: 18, max: 80, description: 'Age of customer' },
    { name: 'Annual_Income', type: 'numerical', value: 75000, min: 20000, max: 200000, description: 'Annual income in USD' },
    { name: 'Credit_Score', type: 'numerical', value: 720, min: 300, max: 850, description: 'Credit score' },
    { name: 'Employment_Type', type: 'categorical', value: 'Full-time', options: ['Full-time', 'Part-time', 'Self-employed', 'Unemployed'], description: 'Employment status' },
    { name: 'Education_Level', type: 'categorical', value: 'Bachelor', options: ['High School', 'Bachelor', 'Master', 'PhD'], description: 'Education level' },
  ]);

  const [currentPrediction, setCurrentPrediction] = useState(0.76);
  const [confidence, setConfidence] = useState(0.89);
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Mock prediction function
  const calculatePrediction = (features: FeatureInput[]) => {
    const age = features.find(f => f.name === 'Customer_Age')?.value as number;
    const income = features.find(f => f.name === 'Annual_Income')?.value as number;
    const creditScore = features.find(f => f.name === 'Credit_Score')?.value as number;
    
    // Simple mock calculation
    const baseScore = (age * 0.01) + (income * 0.000002) + (creditScore * 0.001);
    const prediction = Math.min(Math.max(baseScore * 0.8, 0.1), 0.95);
    const confidence = Math.min(Math.max(baseScore * 0.9, 0.6), 0.98);
    
    return { prediction, confidence };
  };

  const updateFeatureValue = (name: string, value: number | string) => {
    setFeatures(prev => prev.map(f => 
      f.name === name ? { ...f, value } : f
    ));
  };

  const runAnalysis = () => {
    setIsAnalyzing(true);
    setTimeout(() => {
      const { prediction, confidence } = calculatePrediction(features);
      setCurrentPrediction(prediction);
      setConfidence(confidence);
      setIsAnalyzing(false);
    }, 1000);
  };

  const resetToDefault = () => {
    setFeatures([
      { name: 'Customer_Age', type: 'numerical', value: 35, min: 18, max: 80, description: 'Age of customer' },
      { name: 'Annual_Income', type: 'numerical', value: 75000, min: 20000, max: 200000, description: 'Annual income in USD' },
      { name: 'Credit_Score', type: 'numerical', value: 720, min: 300, max: 850, description: 'Credit score' },
      { name: 'Employment_Type', type: 'categorical', value: 'Full-time', options: ['Full-time', 'Part-time', 'Self-employed', 'Unemployed'], description: 'Employment status' },
      { name: 'Education_Level', type: 'categorical', value: 'Bachelor', options: ['High School', 'Bachelor', 'Master', 'PhD'], description: 'Education level' },
    ]);
  };

  const saveScenario = () => {
    const newScenario: Scenario = {
      id: Date.now().toString(),
      name: `Scenario ${scenarios.length + 1}`,
      features: [...features],
      prediction: currentPrediction,
      confidence,
      timestamp: new Date(),
    };
    setScenarios(prev => [...prev, newScenario]);
  };

  useEffect(() => {
    runAnalysis();
  }, [features]);

  const containerVariants = {
    initial: { opacity: 0 },
    animate: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
      },
    },
  };

  const itemVariants = {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
  };

  const getPredictionColor = (prediction: number) => {
    if (modelType === 'classification') {
      return prediction > 0.7 ? 'text-green-500' : prediction > 0.5 ? 'text-amber-500' : 'text-red-500';
    } else {
      return 'text-blue-500';
    }
  };

  return (
    <motion.div
      variants={containerVariants}
      initial="initial"
      animate="animate"
      className="space-y-6"
    >
      <motion.div variants={itemVariants} className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-neutral-900 dark:text-neutral-100">
            What-If Analysis
          </h1>
          <p className="text-neutral-600 dark:text-neutral-400 mt-1">
            Interactive scenario analysis to explore how changes in features affect predictions
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            size="sm"
            leftIcon={<RotateCcw className="w-4 h-4" />}
            onClick={resetToDefault}
          >
            Reset
          </Button>
          <Button
            variant="outline"
            size="sm"
            leftIcon={<Save className="w-4 h-4" />}
            onClick={saveScenario}
          >
            Save Scenario
          </Button>
        </div>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Feature Controls */}
        <motion.div variants={itemVariants} className="lg:col-span-2">
          <Card
            title="Feature Controls"
            icon={<Sliders className="w-5 h-5 text-primary-600 dark:text-primary-400" />}
          >
            <div className="space-y-6">
              {features.map((feature, index) => (
                <motion.div
                  key={feature.name}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="border-b border-neutral-200 dark:border-neutral-700 pb-4 last:border-b-0"
                >
                  <div className="flex items-center justify-between mb-3">
                    <div>
                      <h4 className="font-medium text-neutral-900 dark:text-neutral-100">
                        {feature.name}
                      </h4>
                      <p className="text-sm text-neutral-600 dark:text-neutral-400">
                        {feature.description}
                      </p>
                    </div>
                    <div className="text-right">
                      <div className="font-bold text-neutral-900 dark:text-neutral-100">
                        {feature.value}
                      </div>
                      <div className="text-xs text-neutral-500 dark:text-neutral-400">
                        {feature.type}
                      </div>
                    </div>
                  </div>
                  
                  {feature.type === 'numerical' ? (
                    <div className="space-y-2">
                      <input
                        type="range"
                        min={feature.min}
                        max={feature.max}
                        value={feature.value as number}
                        onChange={(e) => updateFeatureValue(feature.name, parseFloat(e.target.value))}
                        className="w-full h-2 bg-neutral-200 dark:bg-neutral-700 rounded-lg appearance-none cursor-pointer"
                      />
                      <div className="flex justify-between text-xs text-neutral-500 dark:text-neutral-400">
                        <span>{feature.min}</span>
                        <span>{feature.max}</span>
                      </div>
                    </div>
                  ) : (
                    <select
                      value={feature.value as string}
                      onChange={(e) => updateFeatureValue(feature.name, e.target.value)}
                      className="w-full p-2 border border-neutral-200 dark:border-neutral-700 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
                    >
                      {feature.options?.map(option => (
                        <option key={option} value={option}>{option}</option>
                      ))}
                    </select>
                  )}
                </motion.div>
              ))}
            </div>
          </Card>
        </motion.div>

        {/* Prediction Results */}
        <motion.div variants={itemVariants} className="space-y-6">
          <Card
            title="Current Prediction"
            icon={<Zap className="w-5 h-5 text-primary-600 dark:text-primary-400" />}
          >
            <div className="text-center space-y-4">
              <div className="relative">
                <div className="w-24 h-24 mx-auto border-4 border-neutral-200 dark:border-neutral-700 rounded-full flex items-center justify-center">
                  <div className={`text-2xl font-bold ${getPredictionColor(currentPrediction)}`}>
                    {modelType === 'classification' 
                      ? `${(currentPrediction * 100).toFixed(1)}%`
                      : currentPrediction.toFixed(3)
                    }
                  </div>
                </div>
                {isAnalyzing && (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="animate-spin rounded-full h-24 w-24 border-2 border-primary-500 border-t-transparent"></div>
                  </div>
                )}
              </div>
              <div>
                <div className="text-sm text-neutral-600 dark:text-neutral-400">
                  {modelType === 'classification' ? 'Approval Probability' : 'Predicted Value'}
                </div>
                <div className="text-xs text-neutral-500 dark:text-neutral-400 mt-1">
                  Confidence: {(confidence * 100).toFixed(1)}%
                </div>
              </div>
            </div>
          </Card>

          <Card
            title="Impact Analysis"
            icon={<TrendingUp className="w-5 h-5 text-primary-600 dark:text-primary-400" />}
          >
            <div className="space-y-3">
              {features.slice(0, 3).map((feature, index) => (
                <div key={feature.name} className="flex items-center justify-between">
                  <span className="text-sm text-neutral-600 dark:text-neutral-400">
                    {feature.name}
                  </span>
                  <div className="flex items-center space-x-2">
                    <div className="w-12 h-1 bg-neutral-200 dark:bg-neutral-700 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-primary-500 transition-all duration-300"
                        style={{ width: `${(index + 1) * 30}%` }}
                      />
                    </div>
                    <span className="text-xs text-neutral-500 dark:text-neutral-400">
                      {((index + 1) * 0.1).toFixed(1)}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </motion.div>
      </div>

      {/* Enhanced What-If Analysis Chart */}
      <motion.div variants={itemVariants}>
        <WhatIfAnalysisChart
          features={features.map(f => ({
            ...f,
            originalValue: f.value,
            impact: Math.random() > 0.5 ? 'high' : Math.random() > 0.5 ? 'medium' : 'low'
          }))}
          onFeatureChange={updateFeatureValue}
          onRunAnalysis={async (features) => {
            const { prediction, confidence } = calculatePrediction(features);
            return { prediction, confidence };
          }}
          scenarios={scenarios.map(s => ({
            id: s.id,
            name: s.name,
            features: s.features.map(f => ({
              ...f,
              originalValue: f.value,
              impact: Math.random() > 0.5 ? 'high' : Math.random() > 0.5 ? 'medium' : 'low'
            })),
            prediction: s.prediction,
            confidence: s.confidence,
            createdAt: s.timestamp.toISOString(),
            status: 'completed' as const
          }))}
          counterfactuals={[
            {
              originalPrediction: 0.3,
              targetPrediction: 0.8,
              changedFeatures: [
                {
                  feature: 'Credit_Score',
                  originalValue: 650,
                  newValue: 750,
                  changeType: 'increase' as const
                },
                {
                  feature: 'Annual_Income',
                  originalValue: 45000,
                  newValue: 85000,
                  changeType: 'increase' as const
                }
              ],
              feasibility: 0.75,
              actionability: 0.65
            }
          ]}
          originalPrediction={currentPrediction}
          modelType={modelType}
          interactive={true}
        />
      </motion.div>

      {/* Saved Scenarios */}
      {scenarios.length > 0 && (
        <motion.div variants={itemVariants}>
          <Card
            title="Saved Scenarios"
            icon={<GitCompare className="w-5 h-5 text-primary-600 dark:text-primary-400" />}
          >
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {scenarios.map((scenario) => (
                <div
                  key={scenario.id}
                  className="p-4 border border-neutral-200 dark:border-neutral-700 rounded-lg hover:bg-neutral-50 dark:hover:bg-neutral-800 transition-colors"
                >
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-medium text-neutral-900 dark:text-neutral-100">
                      {scenario.name}
                    </h4>
                    <div className={`text-sm font-bold ${getPredictionColor(scenario.prediction)}`}>
                      {modelType === 'classification' 
                        ? `${(scenario.prediction * 100).toFixed(1)}%`
                        : scenario.prediction.toFixed(3)
                      }
                    </div>
                  </div>
                  <div className="text-xs text-neutral-500 dark:text-neutral-400">
                    {scenario.timestamp.toLocaleString()}
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </motion.div>
      )}
    </motion.div>
  );
};

export default WhatIfAnalysis;