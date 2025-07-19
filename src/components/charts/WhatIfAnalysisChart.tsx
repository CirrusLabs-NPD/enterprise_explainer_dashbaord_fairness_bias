import React, { useMemo, useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  Cell,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ReferenceLine
} from 'recharts';
import {
  Play,
  RotateCcw,
  Settings,
  Sliders,
  Target,
  TrendingUp,
  ArrowRight,
  Copy,
  Download,
  Maximize2,
  AlertCircle,
  CheckCircle,
  XCircle,
  Zap
} from 'lucide-react';
import { cn } from '../../utils';

interface FeatureInput {
  name: string;
  type: 'numerical' | 'categorical' | 'boolean';
  value: number | string | boolean;
  originalValue: number | string | boolean;
  min?: number;
  max?: number;
  options?: string[];
  step?: number;
  description?: string;
  impact?: 'high' | 'medium' | 'low';
}

interface WhatIfScenario {
  id: string;
  name: string;
  features: FeatureInput[];
  prediction: number;
  confidence: number;
  createdAt: string;
  status: 'running' | 'completed' | 'failed';
}

interface CounterfactualExample {
  originalPrediction: number;
  targetPrediction: number;
  changedFeatures: {
    feature: string;
    originalValue: any;
    newValue: any;
    changeType: 'increase' | 'decrease' | 'categorical';
  }[];
  feasibility: number;
  actionability: number;
}

interface WhatIfAnalysisChartProps {
  features: FeatureInput[];
  onFeatureChange: (featureName: string, value: any) => void;
  onRunAnalysis: (features: FeatureInput[]) => Promise<{ prediction: number; confidence: number }>;
  scenarios: WhatIfScenario[];
  counterfactuals?: CounterfactualExample[];
  originalPrediction: number;
  modelType: 'classification' | 'regression';
  className?: string;
  interactive?: boolean;
}

const WhatIfAnalysisChart: React.FC<WhatIfAnalysisChartProps> = ({
  features,
  onFeatureChange,
  onRunAnalysis,
  scenarios,
  counterfactuals = [],
  originalPrediction,
  modelType,
  className,
  interactive = true
}) => {
  const [currentScenario, setCurrentScenario] = useState<WhatIfScenario | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [activeTab, setActiveTab] = useState<'builder' | 'scenarios' | 'counterfactuals'>('builder');
  const [selectedFeatures, setSelectedFeatures] = useState<Set<string>>(new Set());
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [comparisonMode, setComparisonMode] = useState(false);

  // Calculate impact scores for features
  const featureImpacts = useMemo(() => {
    return features.map(feature => {
      // Simulate running what-if with +10% change
      const impactScore = Math.random() * 0.5; // Mock impact calculation
      return {
        ...feature,
        impactScore,
        sensitivity: impactScore > 0.3 ? 'high' : impactScore > 0.15 ? 'medium' : 'low'
      };
    });
  }, [features]);

  // Prepare scenario comparison data
  const scenarioData = useMemo(() => {
    return scenarios.map((scenario, index) => ({
      name: scenario.name,
      prediction: scenario.prediction,
      confidence: scenario.confidence,
      index,
      originalPrediction,
      difference: scenario.prediction - originalPrediction
    }));
  }, [scenarios, originalPrediction]);

  const handleFeatureSliderChange = useCallback((featureName: string, value: any) => {
    onFeatureChange(featureName, value);
  }, [onFeatureChange]);

  const handleRunAnalysis = async () => {
    setIsRunning(true);
    try {
      const result = await onRunAnalysis(features);
      setCurrentScenario({
        id: `scenario_${Date.now()}`,
        name: `Scenario ${scenarios.length + 1}`,
        features: [...features],
        prediction: result.prediction,
        confidence: result.confidence,
        createdAt: new Date().toISOString(),
        status: 'completed'
      });
    } catch (error) {
      console.error('Analysis failed:', error);
    } finally {
      setIsRunning(false);
    }
  };

  const resetToOriginal = () => {
    features.forEach(feature => {
      onFeatureChange(feature.name, feature.originalValue);
    });
  };

  const getFeatureSlider = (feature: FeatureInput) => {
    switch (feature.type) {
      case 'numerical':
        return (
          <input
            type="range"
            min={feature.min || 0}
            max={feature.max || 100}
            step={feature.step || 1}
            value={feature.value as number}
            onChange={(e) => handleFeatureSliderChange(feature.name, parseFloat(e.target.value))}
            className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer"
          />
        );
      
      case 'categorical':
        return (
          <select
            value={feature.value as string}
            onChange={(e) => handleFeatureSliderChange(feature.name, e.target.value)}
            className="w-full px-3 py-2 text-sm border border-neutral-300 dark:border-neutral-600 rounded-md bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
          >
            {feature.options?.map(option => (
              <option key={option} value={option}>{option}</option>
            ))}
          </select>
        );
      
      case 'boolean':
        return (
          <div className="flex items-center space-x-2">
            <input
              type="checkbox"
              checked={feature.value as boolean}
              onChange={(e) => handleFeatureSliderChange(feature.name, e.target.checked)}
              className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500"
            />
            <span className="text-sm text-neutral-700 dark:text-neutral-300">
              {feature.value ? 'True' : 'False'}
            </span>
          </div>
        );
      
      default:
        return null;
    }
  };

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case 'high':
        return 'text-red-600 dark:text-red-400';
      case 'medium':
        return 'text-yellow-600 dark:text-yellow-400';
      case 'low':
        return 'text-green-600 dark:text-green-400';
      default:
        return 'text-neutral-600 dark:text-neutral-400';
    }
  };

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      
      return (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="bg-white dark:bg-neutral-800 p-4 rounded-lg shadow-lg border border-neutral-200 dark:border-neutral-700"
        >
          <div className="font-semibold text-neutral-900 dark:text-neutral-100 mb-2">
            {data.name}
          </div>
          
          <div className="space-y-2 text-sm">
            <div className="flex justify-between items-center">
              <span className="text-neutral-600 dark:text-neutral-400">Prediction:</span>
              <span className="font-medium text-neutral-900 dark:text-neutral-100">
                {modelType === 'classification' 
                  ? `${(data.prediction * 100).toFixed(1)}%`
                  : data.prediction.toFixed(3)
                }
              </span>
            </div>
            
            <div className="flex justify-between items-center">
              <span className="text-neutral-600 dark:text-neutral-400">Confidence:</span>
              <span className="font-medium text-blue-600 dark:text-blue-400">
                {(data.confidence * 100).toFixed(1)}%
              </span>
            </div>
            
            <div className="flex justify-between items-center">
              <span className="text-neutral-600 dark:text-neutral-400">Change:</span>
              <span className={cn(
                "font-medium",
                data.difference > 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
              )}>
                {data.difference > 0 ? '+' : ''}{data.difference.toFixed(3)}
              </span>
            </div>
          </div>
        </motion.div>
      );
    }
    return null;
  };

  const renderBuilderTab = () => (
    <div className="space-y-6">
      {/* Current Prediction Display */}
      <div className="bg-gradient-to-r from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded-lg p-6">
        <div className="flex items-center justify-between">
          <div>
            <h4 className="text-lg font-semibold text-blue-900 dark:text-blue-100">
              Current Prediction
            </h4>
            <div className="text-3xl font-bold text-blue-900 dark:text-blue-100 mt-2">
              {modelType === 'classification' 
                ? `${(originalPrediction * 100).toFixed(1)}%`
                : originalPrediction.toFixed(3)
              }
            </div>
          </div>
          <Target className="w-12 h-12 text-blue-500" />
        </div>
      </div>

      {/* Feature Controls */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="space-y-4">
          <h4 className="font-semibold text-neutral-900 dark:text-neutral-100 flex items-center gap-2">
            <Sliders className="w-5 h-5" />
            Feature Controls
          </h4>
          
          <div className="space-y-4 max-h-96 overflow-y-auto">
            {featureImpacts.map((feature) => (
              <motion.div
                key={feature.name}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="p-4 bg-white dark:bg-neutral-800 rounded-lg border border-neutral-200 dark:border-neutral-700"
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-neutral-900 dark:text-neutral-100">
                      {feature.name}
                    </span>
                    <span className={cn(
                      "px-2 py-1 rounded-full text-xs font-medium",
                      feature.sensitivity === 'high' 
                        ? 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400'
                        : feature.sensitivity === 'medium'
                        ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400'
                        : 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400'
                    )}>
                      {feature.sensitivity} impact
                    </span>
                  </div>
                  <span className="text-sm font-medium text-neutral-700 dark:text-neutral-300">
                    {typeof feature.value === 'number' ? feature.value.toFixed(2) : String(feature.value)}
                  </span>
                </div>
                
                {getFeatureSlider(feature)}
                
                {feature.description && (
                  <p className="text-xs text-neutral-500 dark:text-neutral-400 mt-2">
                    {feature.description}
                  </p>
                )}
              </motion.div>
            ))}
          </div>
        </div>

        <div className="space-y-4">
          <h4 className="font-semibold text-neutral-900 dark:text-neutral-100 flex items-center gap-2">
            <TrendingUp className="w-5 h-5" />
            Impact Analysis
          </h4>
          
          <div style={{ width: '100%', height: '300px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={featureImpacts.slice(0, 10)}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.1} />
                <XAxis 
                  dataKey="name" 
                  stroke="#6b7280" 
                  fontSize={12}
                  angle={-45}
                  textAnchor="end"
                  height={100}
                />
                <YAxis stroke="#6b7280" fontSize={12} />
                <Tooltip />
                <Bar dataKey="impactScore" fill="#3b82f6">
                  {featureImpacts.slice(0, 10).map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={
                        entry.sensitivity === 'high' ? '#ef4444' :
                        entry.sensitivity === 'medium' ? '#f59e0b' :
                        '#10b981'
                      }
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex items-center gap-4">
        <button
          onClick={handleRunAnalysis}
          disabled={isRunning}
          className={cn(
            "flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg font-medium transition-all",
            isRunning 
              ? "opacity-50 cursor-not-allowed" 
              : "hover:bg-blue-700 transform hover:scale-105"
          )}
        >
          {isRunning ? (
            <>
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
              Running Analysis...
            </>
          ) : (
            <>
              <Play className="w-4 h-4" />
              Run What-If Analysis
            </>
          )}
        </button>
        
        <button
          onClick={resetToOriginal}
          className="flex items-center gap-2 px-4 py-2 bg-neutral-200 dark:bg-neutral-700 text-neutral-700 dark:text-neutral-300 rounded-lg hover:bg-neutral-300 dark:hover:bg-neutral-600 transition-colors"
        >
          <RotateCcw className="w-4 h-4" />
          Reset to Original
        </button>
      </div>
    </div>
  );

  const renderScenariosTab = () => (
    <div className="space-y-6">
      {/* Scenario Comparison Chart */}
      <div className="bg-white dark:bg-neutral-800 rounded-lg border border-neutral-200 dark:border-neutral-700 p-6">
        <h4 className="font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
          Scenario Comparison
        </h4>
        
        <div style={{ width: '100%', height: '300px' }}>
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart data={scenarioData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.1} />
              <XAxis 
                dataKey="confidence" 
                stroke="#6b7280" 
                fontSize={12}
                label={{ value: 'Confidence', position: 'insideBottom', offset: -5 }}
              />
              <YAxis 
                dataKey="prediction" 
                stroke="#6b7280" 
                fontSize={12}
                label={{ value: 'Prediction', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip content={<CustomTooltip />} />
              
              {/* Original prediction reference line */}
              <ReferenceLine 
                y={originalPrediction} 
                stroke="#ef4444" 
                strokeDasharray="5 5"
                label={{ value: "Original", position: "topLeft" }}
              />
              
              <Scatter dataKey="prediction" fill="#3b82f6">
                {scenarioData.map((entry, index) => (
                  <Cell 
                    key={`cell-${index}`} 
                    fill={entry.difference > 0 ? '#10b981' : '#ef4444'}
                  />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Scenario List */}
      <div className="space-y-3">
        {scenarios.map((scenario, index) => (
          <motion.div
            key={scenario.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className="p-4 bg-white dark:bg-neutral-800 rounded-lg border border-neutral-200 dark:border-neutral-700"
          >
            <div className="flex items-center justify-between">
              <div>
                <h5 className="font-medium text-neutral-900 dark:text-neutral-100">
                  {scenario.name}
                </h5>
                <p className="text-sm text-neutral-600 dark:text-neutral-400">
                  Created: {new Date(scenario.createdAt).toLocaleString()}
                </p>
              </div>
              
              <div className="flex items-center gap-4">
                <div className="text-right">
                  <div className="font-medium text-neutral-900 dark:text-neutral-100">
                    {modelType === 'classification' 
                      ? `${(scenario.prediction * 100).toFixed(1)}%`
                      : scenario.prediction.toFixed(3)
                    }
                  </div>
                  <div className="text-sm text-neutral-600 dark:text-neutral-400">
                    {(scenario.confidence * 100).toFixed(1)}% confidence
                  </div>
                </div>
                
                <div className="flex items-center gap-2">
                  {scenario.status === 'completed' && (
                    <CheckCircle className="w-5 h-5 text-green-500" />
                  )}
                  {scenario.status === 'running' && (
                    <div className="w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
                  )}
                  {scenario.status === 'failed' && (
                    <XCircle className="w-5 h-5 text-red-500" />
                  )}
                </div>
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );

  const renderCounterfactualsTab = () => (
    <div className="space-y-6">
      {counterfactuals.length === 0 ? (
        <div className="text-center py-12">
          <AlertCircle className="w-12 h-12 text-neutral-400 mx-auto mb-4" />
          <p className="text-neutral-600 dark:text-neutral-400">
            No counterfactual examples available. Run what-if analyses to generate suggestions.
          </p>
        </div>
      ) : (
        counterfactuals.map((counterfactual, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className="p-6 bg-white dark:bg-neutral-800 rounded-lg border border-neutral-200 dark:border-neutral-700"
          >
            <div className="flex items-center justify-between mb-4">
              <h5 className="font-semibold text-neutral-900 dark:text-neutral-100">
                Counterfactual Example #{index + 1}
              </h5>
              <div className="flex items-center gap-4">
                <div className="text-sm">
                  <span className="text-neutral-600 dark:text-neutral-400">Feasibility: </span>
                  <span className={cn(
                    "font-medium",
                    counterfactual.feasibility > 0.7 ? 'text-green-600' :
                    counterfactual.feasibility > 0.4 ? 'text-yellow-600' :
                    'text-red-600'
                  )}>
                    {(counterfactual.feasibility * 100).toFixed(0)}%
                  </span>
                </div>
                <ArrowRight className="w-4 h-4 text-neutral-400" />
              </div>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
              <div className="text-center p-3 bg-neutral-50 dark:bg-neutral-700 rounded-lg">
                <div className="text-sm text-neutral-600 dark:text-neutral-400 mb-1">Original</div>
                <div className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                  {modelType === 'classification' 
                    ? `${(counterfactual.originalPrediction * 100).toFixed(1)}%`
                    : counterfactual.originalPrediction.toFixed(3)
                  }
                </div>
              </div>
              
              <div className="flex items-center justify-center">
                <ArrowRight className="w-6 h-6 text-blue-500" />
              </div>
              
              <div className="text-center p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                <div className="text-sm text-blue-600 dark:text-blue-400 mb-1">Target</div>
                <div className="text-lg font-semibold text-blue-900 dark:text-blue-100">
                  {modelType === 'classification' 
                    ? `${(counterfactual.targetPrediction * 100).toFixed(1)}%`
                    : counterfactual.targetPrediction.toFixed(3)
                  }
                </div>
              </div>
            </div>
            
            <div className="space-y-2">
              <h6 className="font-medium text-neutral-900 dark:text-neutral-100">Required Changes:</h6>
              {counterfactual.changedFeatures.map((change, changeIndex) => (
                <div key={changeIndex} className="flex items-center justify-between p-2 bg-neutral-50 dark:bg-neutral-700 rounded">
                  <span className="font-medium text-neutral-900 dark:text-neutral-100">
                    {change.feature}
                  </span>
                  <div className="flex items-center gap-2 text-sm">
                    <span className="text-neutral-600 dark:text-neutral-400">
                      {String(change.originalValue)}
                    </span>
                    <ArrowRight className="w-3 h-3 text-neutral-400" />
                    <span className="text-blue-600 dark:text-blue-400 font-medium">
                      {String(change.newValue)}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        ))
      )}
    </div>
  );

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn(
        'bg-white dark:bg-neutral-900 rounded-xl border border-neutral-200 dark:border-neutral-700 shadow-sm',
        isFullscreen ? 'fixed inset-4 z-50' : '',
        className
      )}
    >
      {/* Header */}
      <div className="p-6 border-b border-neutral-200 dark:border-neutral-700">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 flex items-center gap-2">
              <Zap className="w-5 h-5 text-orange-500" />
              What-If Analysis
            </h3>
            <p className="text-sm text-neutral-600 dark:text-neutral-400 mt-1">
              Explore counterfactual scenarios and understand feature impacts
            </p>
          </div>
          
          <div className="flex items-center gap-2">
            <button
              onClick={() => setComparisonMode(!comparisonMode)}
              className={cn(
                "px-3 py-1 text-sm rounded-md transition-colors",
                comparisonMode 
                  ? "bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400"
                  : "bg-neutral-100 text-neutral-600 dark:bg-neutral-800 dark:text-neutral-400"
              )}
            >
              Comparison Mode
            </button>
            <button
              onClick={() => setIsFullscreen(!isFullscreen)}
              className="p-2 text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100 transition-colors"
            >
              <Maximize2 className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex gap-1 mt-4">
          {[
            { key: 'builder', label: 'Scenario Builder', icon: Settings },
            { key: 'scenarios', label: 'Scenarios', icon: Target },
            { key: 'counterfactuals', label: 'Counterfactuals', icon: ArrowRight }
          ].map(({ key, label, icon: Icon }) => (
            <button
              key={key}
              onClick={() => setActiveTab(key as any)}
              className={cn(
                'flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all',
                activeTab === key
                  ? 'bg-blue-500 text-white shadow-lg'
                  : 'bg-neutral-100 dark:bg-neutral-800 text-neutral-700 dark:text-neutral-300 hover:bg-neutral-200 dark:hover:bg-neutral-700'
              )}
            >
              <Icon className="w-4 h-4" />
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="p-6">
        <AnimatePresence mode="wait">
          {activeTab === 'builder' && (
            <motion.div
              key="builder"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
            >
              {renderBuilderTab()}
            </motion.div>
          )}
          
          {activeTab === 'scenarios' && (
            <motion.div
              key="scenarios"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
            >
              {renderScenariosTab()}
            </motion.div>
          )}
          
          {activeTab === 'counterfactuals' && (
            <motion.div
              key="counterfactuals"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
            >
              {renderCounterfactualsTab()}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  );
};

export default WhatIfAnalysisChart;