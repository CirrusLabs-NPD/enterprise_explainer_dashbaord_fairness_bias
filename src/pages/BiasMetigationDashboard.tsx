import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Shield,
  AlertTriangle,
  CheckCircle,
  TrendingUp,
  TrendingDown,
  BarChart3,
  Settings,
  Play,
  Pause,
  RefreshCw,
  Download,
  Upload,
  Target,
  Users,
  Scale,
  Brain,
  Filter,
  Zap,
  Eye,
  FileText,
  Info,
  ChevronRight,
  Clock
} from 'lucide-react';
import Card from '../components/common/Card';
import MetricCard from '../components/common/MetricCard';
import { apiClient } from '../services/api';

interface BiasMetric {
  name: string;
  value: number;
  threshold: number;
  status: 'pass' | 'warning' | 'fail';
  trend: 'improving' | 'stable' | 'degrading';
  description: string;
  groups: string[];
}

interface MitigationStrategy {
  id: string;
  name: string;
  type: 'preprocessing' | 'inprocessing' | 'postprocessing';
  status: 'available' | 'running' | 'completed' | 'failed';
  effectiveness: number;
  implementation_time: number;
  description: string;
  parameters: Record<string, any>;
  results?: {
    before: Record<string, number>;
    after: Record<string, number>;
    improvement: Record<string, number>;
  };
}

interface BiasReport {
  model_id: string;
  dataset_id: string;
  sensitive_attributes: string[];
  metrics: BiasMetric[];
  overall_fairness_score: number;
  risk_level: 'low' | 'medium' | 'high' | 'critical';
  recommendations: string[];
  generated_at: string;
}

interface MitigationExperiment {
  id: string;
  name: string;
  model_id: string;
  strategies: string[];
  status: 'draft' | 'running' | 'completed' | 'failed';
  progress: number;
  start_time?: string;
  end_time?: string;
  results?: {
    baseline_metrics: Record<string, number>;
    improved_metrics: Record<string, number>;
    performance_impact: Record<string, number>;
  };
}

const BiasMetigationDashboard: React.FC = () => {
  const [selectedModel, setSelectedModel] = useState('model_v2.3');
  const [biasReport, setBiasReport] = useState<BiasReport | null>(null);
  const [strategies, setStrategies] = useState<MitigationStrategy[]>([]);
  const [experiments, setExperiments] = useState<MitigationExperiment[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'assessment' | 'strategies' | 'experiments' | 'monitoring'>('assessment');
  const [runningMitigation, setRunningMitigation] = useState(false);

  // Mock data for demonstration
  const mockBiasReport: BiasReport = {
    model_id: 'model_v2.3',
    dataset_id: 'credit_dataset',
    sensitive_attributes: ['gender', 'race', 'age_group', 'geography'],
    metrics: [
      {
        name: 'Demographic Parity',
        value: 0.15,
        threshold: 0.10,
        status: 'fail',
        trend: 'degrading',
        description: 'Difference in positive prediction rates across groups',
        groups: ['Male vs Female', 'Urban vs Rural']
      },
      {
        name: 'Equalized Odds',
        value: 0.08,
        threshold: 0.10,
        status: 'pass',
        trend: 'stable',
        description: 'Equal true positive and false positive rates',
        groups: ['All groups']
      },
      {
        name: 'Equality of Opportunity',
        value: 0.12,
        threshold: 0.10,
        status: 'warning',
        trend: 'improving',
        description: 'Equal true positive rates across groups',
        groups: ['Age groups']
      },
      {
        name: 'Calibration',
        value: 0.06,
        threshold: 0.05,
        status: 'warning',
        trend: 'stable',
        description: 'Predicted vs actual positive rates alignment',
        groups: ['Geographic regions']
      }
    ],
    overall_fairness_score: 72.5,
    risk_level: 'high',
    recommendations: [
      'Apply data resampling to balance gender representation',
      'Implement fairness constraints during model training',
      'Consider post-processing threshold optimization',
      'Monitor geographic bias more closely'
    ],
    generated_at: '2024-01-20T14:30:00Z'
  };

  const mockStrategies: MitigationStrategy[] = [
    {
      id: 'strategy_001',
      name: 'Synthetic Minority Oversampling (SMOTE)',
      type: 'preprocessing',
      status: 'available',
      effectiveness: 85,
      implementation_time: 15,
      description: 'Generate synthetic samples to balance representation across sensitive groups',
      parameters: {
        k_neighbors: 5,
        sampling_strategy: 'auto',
        random_state: 42
      }
    },
    {
      id: 'strategy_002',
      name: 'Fairness Constraints (FairLearn)',
      type: 'inprocessing',
      status: 'available',
      effectiveness: 78,
      implementation_time: 45,
      description: 'Add fairness constraints during model training to optimize for equalized odds',
      parameters: {
        constraint_type: 'equalized_odds',
        epsilon: 0.05,
        max_iterations: 100
      }
    },
    {
      id: 'strategy_003',
      name: 'Threshold Optimization',
      type: 'postprocessing',
      status: 'completed',
      effectiveness: 65,
      implementation_time: 8,
      description: 'Optimize decision thresholds separately for each sensitive group',
      parameters: {
        constraint: 'demographic_parity',
        method: 'lagrangian'
      },
      results: {
        before: {
          demographic_parity: 0.15,
          accuracy: 0.87,
          precision: 0.85
        },
        after: {
          demographic_parity: 0.08,
          accuracy: 0.85,
          precision: 0.83
        },
        improvement: {
          demographic_parity: 0.07,
          accuracy: -0.02,
          precision: -0.02
        }
      }
    },
    {
      id: 'strategy_004',
      name: 'Adversarial Debiasing',
      type: 'inprocessing',
      status: 'running',
      effectiveness: 82,
      implementation_time: 120,
      description: 'Use adversarial training to remove sensitive information from model predictions',
      parameters: {
        adversary_loss_weight: 0.1,
        learning_rate: 0.001,
        epochs: 50
      }
    }
  ];

  const mockExperiments: MitigationExperiment[] = [
    {
      id: 'exp_001',
      name: 'Multi-Strategy Bias Reduction',
      model_id: 'model_v2.3',
      strategies: ['strategy_001', 'strategy_002'],
      status: 'completed',
      progress: 100,
      start_time: '2024-01-18T09:00:00Z',
      end_time: '2024-01-18T11:30:00Z',
      results: {
        baseline_metrics: {
          demographic_parity: 0.15,
          equalized_odds: 0.12,
          accuracy: 0.87,
          precision: 0.85,
          recall: 0.89
        },
        improved_metrics: {
          demographic_parity: 0.06,
          equalized_odds: 0.04,
          accuracy: 0.85,
          precision: 0.83,
          recall: 0.87
        },
        performance_impact: {
          accuracy: -0.02,
          precision: -0.02,
          recall: -0.02,
          fairness_improvement: 0.09
        }
      }
    },
    {
      id: 'exp_002',
      name: 'Preprocessing Focus',
      model_id: 'model_v2.3',
      strategies: ['strategy_001'],
      status: 'running',
      progress: 65,
      start_time: '2024-01-20T10:00:00Z'
    }
  ];

  useEffect(() => {
    loadBiasData();
  }, [selectedModel]);

  const loadBiasData = async () => {
    try {
      setLoading(true);
      
      // Try to fetch from API, fallback to mock data
      const response = await apiClient.getBiasMetrics();
      
      if (response.success && response.data) {
        setBiasReport(response.data);
      } else {
        // Use mock data as fallback
        setTimeout(() => {
          setBiasReport(mockBiasReport);
          setStrategies(mockStrategies);
          setExperiments(mockExperiments);
          setLoading(false);
        }, 1000);
        return;
      }
    } catch (err) {
      console.warn('API call failed, using mock data:', err);
      // Use mock data as fallback
      setTimeout(() => {
        setBiasReport(mockBiasReport);
        setStrategies(mockStrategies);
        setExperiments(mockExperiments);
        setLoading(false);
      }, 1000);
      return;
    }
    
    setLoading(false);
  };

  const runMitigationStrategy = async (strategyId: string) => {
    try {
      setRunningMitigation(true);
      
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Update strategy status
      setStrategies(prev => 
        prev.map(s => 
          s.id === strategyId 
            ? { ...s, status: 'running' }
            : s
        )
      );
      
      // Simulate completion after some time
      setTimeout(() => {
        setStrategies(prev => 
          prev.map(s => 
            s.id === strategyId 
              ? { ...s, status: 'completed' }
              : s
          )
        );
        setRunningMitigation(false);
      }, 5000);
      
    } catch (err) {
      console.error('Failed to run mitigation strategy:', err);
      setRunningMitigation(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'pass': return 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300';
      case 'warning': return 'bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300';
      case 'fail': return 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300';
      case 'available': return 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300';
      case 'running': return 'bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300';
      case 'completed': return 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300';
      case 'failed': return 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300';
      default: return 'bg-neutral-100 text-neutral-800 dark:bg-neutral-800 dark:text-neutral-300';
    }
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'improving': return <TrendingUp className="w-4 h-4 text-green-500" />;
      case 'degrading': return <TrendingDown className="w-4 h-4 text-red-500" />;
      case 'stable': return <Target className="w-4 h-4 text-blue-500" />;
      default: return <Target className="w-4 h-4 text-neutral-500" />;
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'preprocessing': return <Filter className="w-5 h-5" />;
      case 'inprocessing': return <Brain className="w-5 h-5" />;
      case 'postprocessing': return <Settings className="w-5 h-5" />;
      default: return <Settings className="w-5 h-5" />;
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <RefreshCw className="w-8 h-8 animate-spin text-primary-600" />
        <span className="ml-3 text-lg text-neutral-600 dark:text-neutral-400">
          Loading bias mitigation data...
        </span>
      </div>
    );
  }

  if (!biasReport) {
    return (
      <div className="flex flex-col items-center justify-center py-12">
        <AlertTriangle className="w-12 h-12 text-red-500 mb-4" />
        <h2 className="text-xl font-semibold text-neutral-900 dark:text-neutral-100 mb-2">
          Failed to Load Bias Data
        </h2>
        <button
          onClick={loadBiasData}
          className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors"
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-neutral-900 dark:text-neutral-100">
            Advanced Bias Mitigation
          </h1>
          <p className="text-neutral-600 dark:text-neutral-400 mt-1">
            Detect, analyze, and mitigate algorithmic bias with advanced ML fairness techniques
          </p>
        </div>
        
        <div className="flex items-center space-x-4">
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="px-4 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
          >
            <option value="model_v2.3">Credit Risk Model v2.3</option>
            <option value="fraud_model">Fraud Detection Model</option>
            <option value="recommendation_model">Recommendation Engine</option>
          </select>
          
          <button
            onClick={loadBiasData}
            className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
            <span>Refresh</span>
          </button>
          
          <button className="flex items-center space-x-2 px-4 py-2 bg-neutral-600 hover:bg-neutral-700 text-white rounded-lg transition-colors">
            <Download className="w-4 h-4" />
            <span>Export Report</span>
          </button>
        </div>
      </div>

      {/* Overview Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Fairness Score"
          value={biasReport.overall_fairness_score / 100}
          format="percentage"
          icon={<Scale className="w-5 h-5" />}
          change={biasReport.overall_fairness_score > 75 ? "+2.3%" : "-1.8%"}
          changeType={biasReport.overall_fairness_score > 75 ? "positive" : "negative"}
        />
        
        <MetricCard
          title="Risk Level"
          value={biasReport.risk_level.charAt(0).toUpperCase() + biasReport.risk_level.slice(1)}
          format="text"
          icon={<AlertTriangle className="w-5 h-5" />}
          change={biasReport.risk_level === 'low' ? 'Improving' : 'Needs Attention'}
          changeType={biasReport.risk_level === 'low' ? "positive" : "negative"}
        />
        
        <MetricCard
          title="Metrics Passing"
          value={`${biasReport.metrics.filter(m => m.status === 'pass').length}/${biasReport.metrics.length}`}
          format="text"
          icon={<CheckCircle className="w-5 h-5" />}
          change={`${Math.round(biasReport.metrics.filter(m => m.status === 'pass').length / biasReport.metrics.length * 100)}%`}
          changeType="neutral"
        />
        
        <MetricCard
          title="Sensitive Attributes"
          value={biasReport.sensitive_attributes.length}
          format="number"
          icon={<Users className="w-5 h-5" />}
          change="Protected"
          changeType="neutral"
        />
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-neutral-200 dark:border-neutral-700">
        <nav className="flex space-x-8">
          {[
            { id: 'assessment', label: 'Bias Assessment', icon: BarChart3 },
            { id: 'strategies', label: 'Mitigation Strategies', icon: Shield },
            { id: 'experiments', label: 'Experiments', icon: Brain },
            { id: 'monitoring', label: 'Monitoring', icon: Eye }
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                activeTab === tab.id
                  ? 'border-primary-500 text-primary-600 dark:text-primary-400'
                  : 'border-transparent text-neutral-500 hover:text-neutral-700 dark:hover:text-neutral-300'
              }`}
            >
              <tab.icon className="w-4 h-4" />
              <span>{tab.label}</span>
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      {activeTab === 'assessment' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Bias Metrics */}
          <Card className="lg:col-span-2">
            <div className="p-6">
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-6">
                Fairness Metrics Analysis
              </h3>
              
              <div className="space-y-4">
                {biasReport.metrics.map((metric, index) => (
                  <motion.div
                    key={metric.name}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className={`p-4 rounded-lg border-l-4 ${
                      metric.status === 'fail'
                        ? 'border-red-500 bg-red-50 dark:bg-red-900/20'
                        : metric.status === 'warning'
                        ? 'border-amber-500 bg-amber-50 dark:bg-amber-900/20'
                        : 'border-green-500 bg-green-50 dark:bg-green-900/20'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center space-x-3">
                        <span className="font-medium text-neutral-900 dark:text-neutral-100">
                          {metric.name}
                        </span>
                        <span className={`px-2 py-1 rounded text-xs font-medium ${getStatusColor(metric.status)}`}>
                          {metric.status.charAt(0).toUpperCase() + metric.status.slice(1)}
                        </span>
                        {getTrendIcon(metric.trend)}
                      </div>
                      
                      <div className="text-right">
                        <span className={`font-bold text-lg ${
                          metric.status === 'fail' ? 'text-red-600 dark:text-red-400' :
                          metric.status === 'warning' ? 'text-amber-600 dark:text-amber-400' :
                          'text-green-600 dark:text-green-400'
                        }`}>
                          {metric.value.toFixed(3)}
                        </span>
                        <div className="text-xs text-neutral-500">
                          Threshold: {metric.threshold.toFixed(2)}
                        </div>
                      </div>
                    </div>
                    
                    <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-2">
                      {metric.description}
                    </p>
                    
                    <div className="flex items-center justify-between">
                      <div className="text-xs text-neutral-500">
                        Affected: {metric.groups.join(', ')}
                      </div>
                      
                      <div className="w-32 bg-neutral-200 dark:bg-neutral-700 rounded-full h-2">
                        <div
                          className={`h-2 rounded-full ${
                            metric.value <= metric.threshold ? 'bg-green-500' :
                            metric.value <= metric.threshold * 1.5 ? 'bg-amber-500' : 'bg-red-500'
                          }`}
                          style={{ width: `${Math.min((metric.value / (metric.threshold * 2)) * 100, 100)}%` }}
                        />
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>
          </Card>

          {/* Recommendations */}
          <Card>
            <div className="p-6">
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
                Recommendations
              </h3>
              
              <div className="space-y-3">
                {biasReport.recommendations.map((rec, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800"
                  >
                    <div className="flex items-start space-x-3">
                      <Info className="w-4 h-4 text-blue-600 dark:text-blue-400 mt-0.5" />
                      <p className="text-sm text-blue-900 dark:text-blue-100">
                        {rec}
                      </p>
                    </div>
                  </motion.div>
                ))}
              </div>
              
              <button className="w-full mt-4 px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors">
                Generate Action Plan
              </button>
            </div>
          </Card>
        </div>
      )}

      {activeTab === 'strategies' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {strategies.map((strategy) => {
            const TypeIcon = getTypeIcon(strategy.type);
            
            return (
              <Card key={strategy.id}>
                <div className="p-6">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center space-x-3">
                      <div className={`p-2 rounded-lg ${
                        strategy.type === 'preprocessing' ? 'bg-blue-100 dark:bg-blue-900/20' :
                        strategy.type === 'inprocessing' ? 'bg-green-100 dark:bg-green-900/20' :
                        'bg-purple-100 dark:bg-purple-900/20'
                      }`}>
                        <TypeIcon className={`${
                          strategy.type === 'preprocessing' ? 'text-blue-600 dark:text-blue-400' :
                          strategy.type === 'inprocessing' ? 'text-green-600 dark:text-green-400' :
                          'text-purple-600 dark:text-purple-400'
                        }`} />
                      </div>
                      <div>
                        <h4 className="font-medium text-neutral-900 dark:text-neutral-100">
                          {strategy.name}
                        </h4>
                        <p className="text-sm text-neutral-600 dark:text-neutral-400 capitalize">
                          {strategy.type.replace('processing', '-processing')}
                        </p>
                      </div>
                    </div>
                    
                    <span className={`px-2 py-1 rounded text-xs font-medium ${getStatusColor(strategy.status)}`}>
                      {strategy.status.charAt(0).toUpperCase() + strategy.status.slice(1)}
                    </span>
                  </div>
                  
                  <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-4">
                    {strategy.description}
                  </p>
                  
                  <div className="grid grid-cols-2 gap-4 mb-4">
                    <div className="text-center">
                      <div className="text-lg font-bold text-neutral-900 dark:text-neutral-100">
                        {strategy.effectiveness}%
                      </div>
                      <div className="text-xs text-neutral-500">Effectiveness</div>
                    </div>
                    <div className="text-center">
                      <div className="text-lg font-bold text-neutral-900 dark:text-neutral-100">
                        {strategy.implementation_time}m
                      </div>
                      <div className="text-xs text-neutral-500">Est. Time</div>
                    </div>
                  </div>
                  
                  {strategy.results && (
                    <div className="mb-4 p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                      <h5 className="text-sm font-medium text-green-900 dark:text-green-100 mb-2">
                        Previous Results
                      </h5>
                      <div className="space-y-1 text-xs">
                        {Object.entries(strategy.results.improvement).map(([metric, value]) => (
                          <div key={metric} className="flex justify-between">
                            <span className="text-green-700 dark:text-green-300 capitalize">
                              {metric.replace('_', ' ')}:
                            </span>
                            <span className={`font-medium ${
                              value > 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
                            }`}>
                              {value > 0 ? '+' : ''}{(value * 100).toFixed(1)}%
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  <div className="flex items-center space-x-2">
                    {strategy.status === 'available' && (
                      <button
                        onClick={() => runMitigationStrategy(strategy.id)}
                        disabled={runningMitigation}
                        className="flex items-center space-x-2 px-3 py-2 bg-primary-600 hover:bg-primary-700 disabled:bg-primary-400 text-white rounded-lg transition-colors text-sm"
                      >
                        <Play className="w-4 h-4" />
                        <span>Run Strategy</span>
                      </button>
                    )}
                    
                    {strategy.status === 'running' && (
                      <div className="flex items-center space-x-2 px-3 py-2 bg-amber-100 text-amber-800 rounded-lg text-sm">
                        <RefreshCw className="w-4 h-4 animate-spin" />
                        <span>Running...</span>
                      </div>
                    )}
                    
                    <button className="flex items-center space-x-2 px-3 py-2 bg-neutral-100 dark:bg-neutral-800 text-neutral-700 dark:text-neutral-300 rounded-lg transition-colors text-sm">
                      <Settings className="w-4 h-4" />
                      <span>Configure</span>
                    </button>
                  </div>
                </div>
              </Card>
            );
          })}
        </div>
      )}

      {activeTab === 'experiments' && (
        <div className="space-y-6">
          {experiments.map((experiment) => (
            <Card key={experiment.id}>
              <div className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h4 className="font-medium text-neutral-900 dark:text-neutral-100">
                      {experiment.name}
                    </h4>
                    <p className="text-sm text-neutral-600 dark:text-neutral-400">
                      {experiment.strategies.length} strategies • Model: {experiment.model_id}
                    </p>
                  </div>
                  
                  <div className="flex items-center space-x-3">
                    <span className={`px-2 py-1 rounded text-xs font-medium ${getStatusColor(experiment.status)}`}>
                      {experiment.status.charAt(0).toUpperCase() + experiment.status.slice(1)}
                    </span>
                    
                    {experiment.status === 'running' && (
                      <div className="flex items-center space-x-2">
                        <div className="w-16 bg-neutral-200 dark:bg-neutral-700 rounded-full h-2">
                          <div
                            className="h-2 bg-primary-500 rounded-full transition-all duration-300"
                            style={{ width: `${experiment.progress}%` }}
                          />
                        </div>
                        <span className="text-xs text-neutral-500">{experiment.progress}%</span>
                      </div>
                    )}
                  </div>
                </div>
                
                {experiment.start_time && (
                  <div className="flex items-center space-x-4 mb-4 text-sm text-neutral-600 dark:text-neutral-400">
                    <div className="flex items-center space-x-1">
                      <Clock className="w-4 h-4" />
                      <span>Started: {new Date(experiment.start_time).toLocaleString()}</span>
                    </div>
                    {experiment.end_time && (
                      <div>
                        Completed: {new Date(experiment.end_time).toLocaleString()}
                      </div>
                    )}
                  </div>
                )}
                
                {experiment.results && (
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="p-4 bg-neutral-50 dark:bg-neutral-800 rounded-lg">
                      <h5 className="font-medium text-neutral-900 dark:text-neutral-100 mb-2">
                        Baseline Metrics
                      </h5>
                      <div className="space-y-1 text-sm">
                        {Object.entries(experiment.results.baseline_metrics).map(([metric, value]) => (
                          <div key={metric} className="flex justify-between">
                            <span className="text-neutral-600 dark:text-neutral-400 capitalize">
                              {metric.replace('_', ' ')}:
                            </span>
                            <span className="font-medium">
                              {(value * 100).toFixed(1)}%
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                    
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                      <h5 className="font-medium text-green-900 dark:text-green-100 mb-2">
                        Improved Metrics
                      </h5>
                      <div className="space-y-1 text-sm">
                        {Object.entries(experiment.results.improved_metrics).map(([metric, value]) => (
                          <div key={metric} className="flex justify-between">
                            <span className="text-green-700 dark:text-green-300 capitalize">
                              {metric.replace('_', ' ')}:
                            </span>
                            <span className="font-medium text-green-600 dark:text-green-400">
                              {(value * 100).toFixed(1)}%
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                    
                    <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                      <h5 className="font-medium text-blue-900 dark:text-blue-100 mb-2">
                        Impact Analysis
                      </h5>
                      <div className="space-y-1 text-sm">
                        {Object.entries(experiment.results.performance_impact).map(([metric, value]) => (
                          <div key={metric} className="flex justify-between">
                            <span className="text-blue-700 dark:text-blue-300 capitalize">
                              {metric.replace('_', ' ')}:
                            </span>
                            <span className={`font-medium ${
                              value > 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
                            }`}>
                              {value > 0 ? '+' : ''}{(value * 100).toFixed(1)}%
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </Card>
          ))}
          
          <Card>
            <div className="p-6 text-center">
              <h4 className="font-medium text-neutral-900 dark:text-neutral-100 mb-2">
                Start New Experiment
              </h4>
              <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-4">
                Combine multiple mitigation strategies to optimize fairness
              </p>
              <button className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors">
                Create Experiment
              </button>
            </div>
          </Card>
        </div>
      )}

      {activeTab === 'monitoring' && (
        <Card>
          <div className="p-6">
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
              Continuous Fairness Monitoring
            </h3>
            <div className="text-center py-12 text-neutral-500 dark:text-neutral-400">
              <Eye className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p>Real-time fairness monitoring dashboard coming soon</p>
              <p className="text-sm">Track bias metrics and alerts in production</p>
            </div>
          </div>
        </Card>
      )}
    </div>
  );
};

export default BiasMetigationDashboard;