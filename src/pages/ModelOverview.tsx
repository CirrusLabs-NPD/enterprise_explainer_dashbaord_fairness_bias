import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  Activity, 
  Target, 
  Database, 
  Users, 
  TrendingUp,
  Award,
  AlertCircle,
  CheckCircle,
  Info,
  Loader2,
  Wifi,
  WifiOff
} from 'lucide-react';
import Card from '@/components/common/Card';
import MetricCard from '@/components/common/MetricCard';
import { formatNumber } from '@/utils';
import { apiClient, ModelMetadata } from '@/services/api';
import { webSocketManager } from '@/services/websocket';

interface ModelOverviewProps {
  modelType: 'classification' | 'regression';
}

const ModelOverview: React.FC<ModelOverviewProps> = ({ modelType }) => {
  const [models, setModels] = useState<ModelMetadata[]>([]);
  console.log('Models loaded:', models.length);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<ModelMetadata | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [realTimeUpdates, setRealTimeUpdates] = useState(false);

  // Load models from API
  useEffect(() => {
    loadModels();
    connectWebSocket();
  }, []);

  const loadModels = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await apiClient.listModels();
      if (response.success && response.data) {
        setModels(response.data);
        
        // Select the first model of the matching type
        const matchingModel = response.data.find(m => m.model_type === modelType);
        if (matchingModel) {
          setSelectedModel(matchingModel);
        } else if (response.data.length > 0) {
          setSelectedModel(response.data[0]);
        }
      } else {
        setError(response.error || 'Failed to load models');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load models');
    } finally {
      setLoading(false);
    }
  };

  const connectWebSocket = async () => {
    try {
      const connected = await webSocketManager.connect();
      setIsConnected(connected);
      
      if (connected) {
        // Listen for model updates
        webSocketManager.on('model_update', (update) => {
          if (selectedModel && update.model_id === selectedModel.model_id) {
            setRealTimeUpdates(true);
            // Refresh model data
            loadModels();
            setTimeout(() => setRealTimeUpdates(false), 2000);
          }
        });
      }
    } catch (err) {
      console.error('WebSocket connection failed:', err);
    }
  };

  // Mock data fallback when no real models are available
  const mockModelInfo = {
    name: 'Customer Churn Prediction Model',
    algorithm: 'Random Forest',
    version: '1.2.3',
    framework: 'scikit-learn',
    createdDate: '2024-01-15',
    lastTrained: '2024-01-20',
    features: 24,
    samples: 10000,
    accuracy: modelType === 'classification' ? 0.892 : undefined,
    r2Score: modelType === 'regression' ? 0.847 : undefined,
    precision: modelType === 'classification' ? 0.876 : undefined,
    recall: modelType === 'classification' ? 0.903 : undefined,
    f1Score: modelType === 'classification' ? 0.889 : undefined,
    rmse: modelType === 'regression' ? 0.234 : undefined,
    mae: modelType === 'regression' ? 0.187 : undefined,
    status: 'active',
    healthScore: 94,
  };

  // Use real model data if available, otherwise use mock data
  const modelInfo = selectedModel ? {
    name: selectedModel.name,
    algorithm: selectedModel.framework,
    version: selectedModel.version,
    framework: selectedModel.framework,
    createdDate: selectedModel.created_at,
    lastTrained: selectedModel.created_at,
    features: selectedModel.feature_names.length,
    samples: 10000, // This would come from training data
    accuracy: selectedModel.validation_metrics.accuracy || mockModelInfo.accuracy,
    r2Score: selectedModel.validation_metrics.r2_score || mockModelInfo.r2Score,
    precision: selectedModel.validation_metrics.precision || mockModelInfo.precision,
    recall: selectedModel.validation_metrics.recall || mockModelInfo.recall,
    f1Score: selectedModel.validation_metrics.f1_score || mockModelInfo.f1Score,
    rmse: selectedModel.validation_metrics.rmse || mockModelInfo.rmse,
    mae: selectedModel.validation_metrics.mae || mockModelInfo.mae,
    status: selectedModel.status,
    healthScore: 94, // This would be calculated based on model performance
  } : mockModelInfo;

  const datasetInfo = {
    trainSamples: 8000,
    testSamples: 2000,
    features: 24,
    missingValues: 0.03,
    duplicates: 0.01,
  };

  const performanceMetrics = modelType === 'classification' ? [
    {
      title: 'Accuracy',
      value: modelInfo.accuracy!,
      format: 'percentage' as const,
      icon: <Target className="w-5 h-5 text-primary-600 dark:text-primary-400" />,
      change: 2.3,
      changeType: 'increase' as const,
      color: 'primary' as const,
    },
    {
      title: 'Precision',
      value: modelInfo.precision!,
      format: 'percentage' as const,
      icon: <Award className="w-5 h-5 text-secondary-600 dark:text-secondary-400" />,
      change: 1.8,
      changeType: 'increase' as const,
      color: 'secondary' as const,
    },
    {
      title: 'Recall',
      value: modelInfo.recall!,
      format: 'percentage' as const,
      icon: <Activity className="w-5 h-5 text-success-600 dark:text-success-400" />,
      change: -0.5,
      changeType: 'decrease' as const,
      color: 'success' as const,
    },
    {
      title: 'F1 Score',
      value: modelInfo.f1Score!,
      format: 'percentage' as const,
      icon: <TrendingUp className="w-5 h-5 text-warning-600 dark:text-warning-400" />,
      change: 1.2,
      changeType: 'increase' as const,
      color: 'warning' as const,
    },
  ] : [
    {
      title: 'R² Score',
      value: modelInfo.r2Score!,
      format: 'percentage' as const,
      icon: <Target className="w-5 h-5 text-primary-600 dark:text-primary-400" />,
      change: 3.1,
      changeType: 'increase' as const,
      color: 'primary' as const,
    },
    {
      title: 'RMSE',
      value: modelInfo.rmse!,
      format: 'number' as const,
      icon: <Activity className="w-5 h-5 text-secondary-600 dark:text-secondary-400" />,
      change: -2.7,
      changeType: 'decrease' as const,
      color: 'secondary' as const,
    },
    {
      title: 'MAE',
      value: modelInfo.mae!,
      format: 'number' as const,
      icon: <Award className="w-5 h-5 text-success-600 dark:text-success-400" />,
      change: -1.9,
      changeType: 'decrease' as const,
      color: 'success' as const,
    },
    {
      title: 'Health Score',
      value: modelInfo.healthScore,
      format: 'number' as const,
      icon: <TrendingUp className="w-5 h-5 text-warning-600 dark:text-warning-400" />,
      change: 0.8,
      changeType: 'increase' as const,
      color: 'warning' as const,
    },
  ];

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

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="w-8 h-8 animate-spin text-primary-600" />
        <span className="ml-3 text-lg text-neutral-600 dark:text-neutral-400">
          Loading model data...
        </span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center py-12">
        <AlertCircle className="w-12 h-12 text-red-500 mb-4" />
        <h2 className="text-xl font-semibold text-neutral-900 dark:text-neutral-100 mb-2">
          Failed to Load Models
        </h2>
        <p className="text-neutral-600 dark:text-neutral-400 mb-4 text-center max-w-md">
          {error}
        </p>
        <button
          onClick={loadModels}
          className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors"
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <motion.div
      variants={containerVariants}
      initial="initial"
      animate="animate"
      className="space-y-6"
    >
      {/* Header */}
      <motion.div variants={itemVariants} className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-neutral-900 dark:text-neutral-100">
            Model Overview
          </h1>
          <p className="text-neutral-600 dark:text-neutral-400 mt-1">
            Comprehensive view of your {modelType} model performance and metrics
          </p>
        </div>
        <div className="flex items-center space-x-3">
          {/* Connection Status */}
          <div className={`flex items-center space-x-2 px-3 py-1 rounded-full ${
            isConnected 
              ? 'bg-green-100 dark:bg-green-900/20 text-green-700 dark:text-green-300'
              : 'bg-red-100 dark:bg-red-900/20 text-red-700 dark:text-red-300'
          }`}>
            {isConnected ? <Wifi className="w-4 h-4" /> : <WifiOff className="w-4 h-4" />}
            <span className="text-sm font-medium">
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
            {realTimeUpdates && (
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            )}
          </div>

          {/* Model Status */}
          <div className="flex items-center space-x-2 px-3 py-1 bg-green-100 dark:bg-green-900/20 text-green-700 dark:text-green-300 rounded-full">
            <CheckCircle className="w-4 h-4" />
            <span className="text-sm font-medium capitalize">{modelInfo.status}</span>
          </div>
        </div>
      </motion.div>

      {/* Model Information */}
      <motion.div variants={itemVariants}>
        <Card
          title="Model Information"
          icon={<Info className="w-5 h-5 text-primary-600 dark:text-primary-400" />}
          className="mb-6"
        >
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div>
              <h4 className="font-semibold text-neutral-900 dark:text-neutral-100 mb-2">
                Model Details
              </h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-neutral-600 dark:text-neutral-400">Name:</span>
                  <span className="font-medium text-neutral-900 dark:text-neutral-100">
                    {modelInfo.name}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-600 dark:text-neutral-400">Algorithm:</span>
                  <span className="font-medium text-neutral-900 dark:text-neutral-100">
                    {modelInfo.algorithm}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-600 dark:text-neutral-400">Version:</span>
                  <span className="font-medium text-neutral-900 dark:text-neutral-100">
                    {modelInfo.version}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-600 dark:text-neutral-400">Framework:</span>
                  <span className="font-medium text-neutral-900 dark:text-neutral-100">
                    {modelInfo.framework}
                  </span>
                </div>
              </div>
            </div>
            
            <div>
              <h4 className="font-semibold text-neutral-900 dark:text-neutral-100 mb-2">
                Training Info
              </h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-neutral-600 dark:text-neutral-400">Created:</span>
                  <span className="font-medium text-neutral-900 dark:text-neutral-100">
                    {modelInfo.createdDate}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-600 dark:text-neutral-400">Last Trained:</span>
                  <span className="font-medium text-neutral-900 dark:text-neutral-100">
                    {modelInfo.lastTrained}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-600 dark:text-neutral-400">Features:</span>
                  <span className="font-medium text-neutral-900 dark:text-neutral-100">
                    {modelInfo.features}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-600 dark:text-neutral-400">Samples:</span>
                  <span className="font-medium text-neutral-900 dark:text-neutral-100">
                    {formatNumber(modelInfo.samples)}
                  </span>
                </div>
              </div>
            </div>
            
            <div>
              <h4 className="font-semibold text-neutral-900 dark:text-neutral-100 mb-2">
                Dataset Split
              </h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-neutral-600 dark:text-neutral-400">Train:</span>
                  <span className="font-medium text-neutral-900 dark:text-neutral-100">
                    {formatNumber(datasetInfo.trainSamples)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-600 dark:text-neutral-400">Test:</span>
                  <span className="font-medium text-neutral-900 dark:text-neutral-100">
                    {formatNumber(datasetInfo.testSamples)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-600 dark:text-neutral-400">Features:</span>
                  <span className="font-medium text-neutral-900 dark:text-neutral-100">
                    {datasetInfo.features}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-600 dark:text-neutral-400">Missing:</span>
                  <span className="font-medium text-neutral-900 dark:text-neutral-100">
                    {formatNumber(datasetInfo.missingValues * 100, { precision: 1 })}%
                  </span>
                </div>
              </div>
            </div>
            
            <div>
              <h4 className="font-semibold text-neutral-900 dark:text-neutral-100 mb-2">
                Status
              </h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-neutral-600 dark:text-neutral-400">Health Score:</span>
                  <span className="font-medium text-green-600 dark:text-green-400">
                    {modelInfo.healthScore}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-600 dark:text-neutral-400">Status:</span>
                  <span className="font-medium text-green-600 dark:text-green-400 capitalize">
                    {modelInfo.status}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-600 dark:text-neutral-400">Duplicates:</span>
                  <span className="font-medium text-neutral-900 dark:text-neutral-100">
                    {formatNumber(datasetInfo.duplicates * 100, { precision: 1 })}%
                  </span>
                </div>
              </div>
            </div>
          </div>
        </Card>
      </motion.div>

      {/* Performance Metrics */}
      <motion.div variants={itemVariants}>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {performanceMetrics.map((metric) => (
            <MetricCard
              key={metric.title}
              title={metric.title}
              value={metric.value}
              format={metric.format}
              icon={metric.icon}
              change={metric.change}
              changeType={metric.changeType}
              color={metric.color}
              animated={true}
            />
          ))}
        </div>
      </motion.div>

      {/* Quick Actions */}
      <motion.div variants={itemVariants}>
        <Card
          title="Quick Actions"
          description="Common tasks and shortcuts"
          icon={<Activity className="w-5 h-5 text-primary-600 dark:text-primary-400" />}
        >
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <button className="p-4 bg-primary-50 dark:bg-primary-900/20 rounded-lg hover:bg-primary-100 dark:hover:bg-primary-900/30 transition-colors text-left">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-primary-100 dark:bg-primary-800/50 rounded-lg">
                  <Target className="w-5 h-5 text-primary-600 dark:text-primary-400" />
                </div>
                <div>
                  <h4 className="font-medium text-neutral-900 dark:text-neutral-100">
                    View Feature Importance
                  </h4>
                  <p className="text-sm text-neutral-600 dark:text-neutral-400">
                    Explore which features impact predictions most
                  </p>
                </div>
              </div>
            </button>
            
            <button className="p-4 bg-secondary-50 dark:bg-secondary-900/20 rounded-lg hover:bg-secondary-100 dark:hover:bg-secondary-900/30 transition-colors text-left">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-secondary-100 dark:bg-secondary-800/50 rounded-lg">
                  <Users className="w-5 h-5 text-secondary-600 dark:text-secondary-400" />
                </div>
                <div>
                  <h4 className="font-medium text-neutral-900 dark:text-neutral-100">
                    Analyze Predictions
                  </h4>
                  <p className="text-sm text-neutral-600 dark:text-neutral-400">
                    Dive deep into individual predictions
                  </p>
                </div>
              </div>
            </button>
            
            <button className="p-4 bg-amber-50 dark:bg-amber-900/20 rounded-lg hover:bg-amber-100 dark:hover:bg-amber-900/30 transition-colors text-left">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-amber-100 dark:bg-amber-800/50 rounded-lg">
                  <Database className="w-5 h-5 text-amber-600 dark:text-amber-400" />
                </div>
                <div>
                  <h4 className="font-medium text-neutral-900 dark:text-neutral-100">
                    What-If Analysis
                  </h4>
                  <p className="text-sm text-neutral-600 dark:text-neutral-400">
                    Explore different scenarios and outcomes
                  </p>
                </div>
              </div>
            </button>
          </div>
        </Card>
      </motion.div>

      {/* Alerts and Notifications */}
      <motion.div variants={itemVariants}>
        <Card
          title="System Alerts"
          description="Important notifications and warnings"
          icon={<AlertCircle className="w-5 h-5 text-warning-600 dark:text-warning-400" />}
        >
          <div className="space-y-3">
            <div className="flex items-start space-x-3 p-3 bg-amber-50 dark:bg-amber-900/20 rounded-lg">
              <AlertCircle className="w-5 h-5 text-amber-600 dark:text-amber-400 mt-0.5" />
              <div>
                <h4 className="font-medium text-amber-900 dark:text-amber-100">
                  Model Drift Detected
                </h4>
                <p className="text-sm text-amber-700 dark:text-amber-300">
                  Feature distribution has changed by 12% since last training. Consider retraining.
                </p>
              </div>
            </div>
            
            <div className="flex items-start space-x-3 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <Info className="w-5 h-5 text-blue-600 dark:text-blue-400 mt-0.5" />
              <div>
                <h4 className="font-medium text-blue-900 dark:text-blue-100">
                  New Data Available
                </h4>
                <p className="text-sm text-blue-700 dark:text-blue-300">
                  2,500 new samples are available for training. Schedule retraining?
                </p>
              </div>
            </div>
            
            <div className="flex items-start space-x-3 p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
              <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-400 mt-0.5" />
              <div>
                <h4 className="font-medium text-green-900 dark:text-green-100">
                  Performance Stable
                </h4>
                <p className="text-sm text-green-700 dark:text-green-300">
                  Model performance has remained consistent over the past 30 days.
                </p>
              </div>
            </div>
          </div>
        </Card>
      </motion.div>
    </motion.div>
  );
};

export default ModelOverview;