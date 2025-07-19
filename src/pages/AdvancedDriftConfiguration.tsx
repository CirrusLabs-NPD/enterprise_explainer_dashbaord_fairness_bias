import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Settings,
  Save,
  RotateCcw,
  AlertTriangle,
  TrendingUp,
  Database,
  Clock,
  Target,
  Zap,
  Bell,
  Mail,
  MessageSquare,
  Phone,
  Plus,
  Trash2,
  Edit,
  Play,
  Pause,
  CheckCircle,
  Info,
  Filter,
  BarChart3,
} from 'lucide-react';
import Card from '../components/common/Card';
import MetricCard from '../components/common/MetricCard';

interface DriftDetectionMethod {
  id: string;
  name: string;
  description: string;
  suitable_for: string[];
  complexity: 'low' | 'medium' | 'high';
  computational_cost: 'low' | 'medium' | 'high';
}

interface MonitoringConfig {
  model_id: string;
  model_name: string;
  drift_detection_enabled: boolean;
  performance_monitoring_enabled: boolean;
  data_quality_monitoring_enabled: boolean;
  drift_threshold: number;
  performance_threshold: number;
  quality_threshold: number;
  detection_methods: string[];
  monitoring_frequency: 'real_time' | 'hourly' | 'daily' | 'weekly';
  alert_channels: string[];
  reference_data_size: number;
  last_updated: string;
  status: 'active' | 'paused' | 'error';
}

interface AlertRule {
  id: string;
  name: string;
  trigger_type: 'threshold' | 'trend' | 'anomaly';
  metric: string;
  condition: 'greater_than' | 'less_than' | 'equals' | 'changes_by';
  threshold: number;
  severity: 'low' | 'medium' | 'high' | 'critical';
  notification_channels: string[];
  cooldown_period: number;
  is_active: boolean;
}

const AdvancedDriftConfiguration: React.FC = () => {
  const [selectedModel, setSelectedModel] = useState<string>('all');
  const [activeTab, setActiveTab] = useState<'models' | 'methods' | 'alerts' | 'global'>('models');
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [editingConfig, setEditingConfig] = useState<MonitoringConfig | null>(null);

  // Mock data - in production, this would come from API
  const driftMethods: DriftDetectionMethod[] = [
    {
      id: 'kolmogorov_smirnov',
      name: 'Kolmogorov-Smirnov',
      description: 'Two-sample test comparing distributions of continuous features',
      suitable_for: ['continuous', 'numerical'],
      complexity: 'low',
      computational_cost: 'low',
    },
    {
      id: 'jensen_shannon',
      name: 'Jensen-Shannon Divergence',
      description: 'Measures similarity between probability distributions',
      suitable_for: ['continuous', 'categorical', 'mixed'],
      complexity: 'medium',
      computational_cost: 'medium',
    },
    {
      id: 'wasserstein',
      name: 'Wasserstein Distance',
      description: 'Earth Mover\'s Distance for comparing distributions',
      suitable_for: ['continuous', 'ordinal'],
      complexity: 'medium',
      computational_cost: 'medium',
    },
    {
      id: 'population_stability_index',
      name: 'Population Stability Index (PSI)',
      description: 'Measures population shifts between reference and current data',
      suitable_for: ['categorical', 'binned_continuous'],
      complexity: 'low',
      computational_cost: 'low',
    },
    {
      id: 'adversarial_accuracy',
      name: 'Adversarial Accuracy',
      description: 'Uses adversarial model to detect distribution changes',
      suitable_for: ['high_dimensional', 'complex_features'],
      complexity: 'high',
      computational_cost: 'high',
    },
    {
      id: 'chi_square',
      name: 'Chi-Square Test',
      description: 'Tests independence between categorical variables',
      suitable_for: ['categorical', 'discrete'],
      complexity: 'low',
      computational_cost: 'low',
    },
  ];

  const monitoringConfigs: MonitoringConfig[] = [
    {
      model_id: 'credit_risk_v2.3',
      model_name: 'Credit Risk Model v2.3',
      drift_detection_enabled: true,
      performance_monitoring_enabled: true,
      data_quality_monitoring_enabled: true,
      drift_threshold: 0.15,
      performance_threshold: 0.10,
      quality_threshold: 0.80,
      detection_methods: ['kolmogorov_smirnov', 'jensen_shannon'],
      monitoring_frequency: 'hourly',
      alert_channels: ['email', 'slack'],
      reference_data_size: 50000,
      last_updated: '2024-01-19T10:30:00Z',
      status: 'active',
    },
    {
      model_id: 'fraud_detection_v1.8',
      model_name: 'Fraud Detection v1.8',
      drift_detection_enabled: true,
      performance_monitoring_enabled: true,
      data_quality_monitoring_enabled: false,
      drift_threshold: 0.20,
      performance_threshold: 0.05,
      quality_threshold: 0.75,
      detection_methods: ['adversarial_accuracy', 'wasserstein'],
      monitoring_frequency: 'real_time',
      alert_channels: ['email', 'slack', 'sms'],
      reference_data_size: 100000,
      last_updated: '2024-01-19T09:45:00Z',
      status: 'active',
    },
    {
      model_id: 'recommendation_v3.1',
      model_name: 'Recommendation Engine v3.1',
      drift_detection_enabled: false,
      performance_monitoring_enabled: true,
      data_quality_monitoring_enabled: true,
      drift_threshold: 0.25,
      performance_threshold: 0.15,
      quality_threshold: 0.70,
      detection_methods: ['population_stability_index'],
      monitoring_frequency: 'daily',
      alert_channels: ['email'],
      reference_data_size: 25000,
      last_updated: '2024-01-18T16:20:00Z',
      status: 'paused',
    },
  ];

  const alertRules: AlertRule[] = [
    {
      id: 'rule_001',
      name: 'High Drift Alert',
      trigger_type: 'threshold',
      metric: 'drift_score',
      condition: 'greater_than',
      threshold: 0.3,
      severity: 'high',
      notification_channels: ['email', 'slack'],
      cooldown_period: 3600,
      is_active: true,
    },
    {
      id: 'rule_002',
      name: 'Performance Degradation',
      trigger_type: 'threshold',
      metric: 'accuracy',
      condition: 'less_than',
      threshold: 0.85,
      severity: 'medium',
      notification_channels: ['email'],
      cooldown_period: 7200,
      is_active: true,
    },
    {
      id: 'rule_003',
      name: 'Data Quality Drop',
      trigger_type: 'threshold',
      metric: 'quality_score',
      condition: 'less_than',
      threshold: 0.75,
      severity: 'medium',
      notification_channels: ['email', 'slack'],
      cooldown_period: 1800,
      is_active: true,
    },
  ];

  const globalSettings = {
    default_drift_threshold: 0.20,
    default_performance_threshold: 0.10,
    default_quality_threshold: 0.80,
    monitoring_interval: 300,
    max_reference_data_age_days: 30,
    auto_reference_update: true,
    enable_trend_analysis: true,
    retain_monitoring_history_days: 90,
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'text-green-600 bg-green-100 dark:bg-green-900/30 dark:text-green-300';
      case 'paused':
        return 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900/30 dark:text-yellow-300';
      case 'error':
        return 'text-red-600 bg-red-100 dark:bg-red-900/30 dark:text-red-300';
      default:
        return 'text-gray-600 bg-gray-100 dark:bg-gray-900/30 dark:text-gray-300';
    }
  };

  const getComplexityColor = (complexity: string) => {
    switch (complexity) {
      case 'low':
        return 'text-green-600 bg-green-100 dark:bg-green-900/30 dark:text-green-300';
      case 'medium':
        return 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900/30 dark:text-yellow-300';
      case 'high':
        return 'text-red-600 bg-red-100 dark:bg-red-900/30 dark:text-red-300';
      default:
        return 'text-gray-600 bg-gray-100 dark:bg-gray-900/30 dark:text-gray-300';
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'text-red-600 bg-red-100 dark:bg-red-900/30 dark:text-red-300';
      case 'high':
        return 'text-orange-600 bg-orange-100 dark:bg-orange-900/30 dark:text-orange-300';
      case 'medium':
        return 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900/30 dark:text-yellow-300';
      case 'low':
        return 'text-blue-600 bg-blue-100 dark:bg-blue-900/30 dark:text-blue-300';
      default:
        return 'text-gray-600 bg-gray-100 dark:bg-gray-900/30 dark:text-gray-300';
    }
  };

  const handleSaveConfig = (config: MonitoringConfig) => {
    console.log('Saving configuration:', config);
    setEditingConfig(null);
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-neutral-900 dark:text-neutral-100">
            Advanced Drift Configuration
          </h1>
          <p className="text-neutral-600 dark:text-neutral-400 mt-1">
            Configure drift detection, monitoring thresholds, and alerting rules
          </p>
        </div>
        
        <div className="flex items-center space-x-4">
          <button className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors">
            <Plus className="w-4 h-4" />
            <span>New Configuration</span>
          </button>
          
          <button className="flex items-center space-x-2 px-4 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg hover:bg-neutral-50 dark:hover:bg-neutral-800 transition-colors">
            <Settings className="w-4 h-4" />
            <span>Global Settings</span>
          </button>
        </div>
      </div>

      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Active Monitors"
          value={monitoringConfigs.filter(c => c.status === 'active').length.toString()}
          change="+1"
          changeType="positive"
          icon={Play}
          description="Models with active monitoring"
        />
        
        <MetricCard
          title="Detection Methods"
          value={driftMethods.length.toString()}
          change="0"
          changeType="neutral"
          icon={BarChart3}
          description="Available drift detection algorithms"
        />
        
        <MetricCard
          title="Alert Rules"
          value={alertRules.filter(r => r.is_active).length.toString()}
          change="+2"
          changeType="positive"
          icon={Bell}
          description="Active alert configurations"
        />
        
        <MetricCard
          title="Avg Threshold"
          value="0.18"
          change="-0.02"
          changeType="positive"
          icon={Target}
          description="Average drift threshold across models"
        />
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-neutral-200 dark:border-neutral-700">
        <nav className="flex space-x-8">
          {[
            { id: 'models', label: 'Model Configurations', icon: Database },
            { id: 'methods', label: 'Detection Methods', icon: BarChart3 },
            { id: 'alerts', label: 'Alert Rules', icon: Bell },
            { id: 'global', label: 'Global Settings', icon: Settings },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                activeTab === tab.id
                  ? 'border-blue-500 text-blue-600 dark:text-blue-400'
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
      {activeTab === 'models' && (
        <div className="space-y-6">
          {/* Model Selection and Filters */}
          <Card>
            <div className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                  Model Monitoring Configurations
                </h3>
                <div className="flex items-center space-x-4">
                  <select
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                    className="px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
                  >
                    <option value="all">All Models</option>
                    {monitoringConfigs.map((config) => (
                      <option key={config.model_id} value={config.model_id}>
                        {config.model_name}
                      </option>
                    ))}
                  </select>
                  
                  <button className="flex items-center space-x-2 px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg hover:bg-neutral-50 dark:hover:bg-neutral-800 transition-colors">
                    <Filter className="w-4 h-4" />
                    <span>Filter</span>
                  </button>
                </div>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-neutral-200 dark:border-neutral-700">
                      <th className="text-left py-3 px-4 font-medium text-neutral-900 dark:text-neutral-100">Model</th>
                      <th className="text-left py-3 px-4 font-medium text-neutral-900 dark:text-neutral-100">Status</th>
                      <th className="text-left py-3 px-4 font-medium text-neutral-900 dark:text-neutral-100">Drift Threshold</th>
                      <th className="text-left py-3 px-4 font-medium text-neutral-900 dark:text-neutral-100">Methods</th>
                      <th className="text-left py-3 px-4 font-medium text-neutral-900 dark:text-neutral-100">Frequency</th>
                      <th className="text-left py-3 px-4 font-medium text-neutral-900 dark:text-neutral-100">Alerts</th>
                      <th className="text-left py-3 px-4 font-medium text-neutral-900 dark:text-neutral-100">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {monitoringConfigs
                      .filter(config => selectedModel === 'all' || config.model_id === selectedModel)
                      .map((config, index) => (
                        <motion.tr
                          key={config.model_id}
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          transition={{ delay: index * 0.1 }}
                          className="border-b border-neutral-100 dark:border-neutral-800 hover:bg-neutral-50 dark:hover:bg-neutral-900/50"
                        >
                          <td className="py-3 px-4">
                            <div>
                              <span className="font-medium text-neutral-900 dark:text-neutral-100">
                                {config.model_name}
                              </span>
                              <div className="text-xs text-neutral-500 mt-1">
                                ID: {config.model_id}
                              </div>
                            </div>
                          </td>
                          <td className="py-3 px-4">
                            <span className={`px-2 py-1 rounded text-xs font-medium ${getStatusColor(config.status)}`}>
                              {config.status}
                            </span>
                          </td>
                          <td className="py-3 px-4 text-neutral-600 dark:text-neutral-400">
                            {config.drift_threshold.toFixed(3)}
                          </td>
                          <td className="py-3 px-4">
                            <div className="flex flex-wrap gap-1">
                              {config.detection_methods.slice(0, 2).map((method) => (
                                <span
                                  key={method}
                                  className="px-2 py-1 bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300 rounded text-xs"
                                >
                                  {method.replace('_', ' ')}
                                </span>
                              ))}
                              {config.detection_methods.length > 2 && (
                                <span className="px-2 py-1 bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-300 rounded text-xs">
                                  +{config.detection_methods.length - 2}
                                </span>
                              )}
                            </div>
                          </td>
                          <td className="py-3 px-4 text-neutral-600 dark:text-neutral-400">
                            {config.monitoring_frequency.replace('_', ' ')}
                          </td>
                          <td className="py-3 px-4">
                            <div className="flex space-x-1">
                              {config.alert_channels.includes('email') && (
                                <Mail className="w-4 h-4 text-blue-500" />
                              )}
                              {config.alert_channels.includes('slack') && (
                                <MessageSquare className="w-4 h-4 text-green-500" />
                              )}
                              {config.alert_channels.includes('sms') && (
                                <Phone className="w-4 h-4 text-purple-500" />
                              )}
                            </div>
                          </td>
                          <td className="py-3 px-4">
                            <div className="flex items-center space-x-2">
                              <button
                                onClick={() => setEditingConfig(config)}
                                className="text-blue-600 hover:text-blue-800"
                              >
                                <Edit className="w-4 h-4" />
                              </button>
                              <button className="text-neutral-600 hover:text-neutral-800">
                                {config.status === 'active' ? (
                                  <Pause className="w-4 h-4" />
                                ) : (
                                  <Play className="w-4 h-4" />
                                )}
                              </button>
                            </div>
                          </td>
                        </motion.tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </div>
          </Card>
        </div>
      )}

      {activeTab === 'methods' && (
        <div className="space-y-6">
          <Card>
            <div className="p-6">
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-6">
                Available Drift Detection Methods
              </h3>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {driftMethods.map((method, index) => (
                  <motion.div
                    key={method.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className="p-6 border border-neutral-200 dark:border-neutral-700 rounded-lg"
                  >
                    <div className="flex items-center justify-between mb-4">
                      <h4 className="font-semibold text-neutral-900 dark:text-neutral-100">
                        {method.name}
                      </h4>
                      <div className="flex space-x-2">
                        <span className={`px-2 py-1 rounded text-xs font-medium ${getComplexityColor(method.complexity)}`}>
                          {method.complexity} complexity
                        </span>
                      </div>
                    </div>
                    
                    <p className="text-neutral-600 dark:text-neutral-400 text-sm mb-4">
                      {method.description}
                    </p>
                    
                    <div className="space-y-3">
                      <div>
                        <span className="text-xs font-medium text-neutral-700 dark:text-neutral-300">
                          Suitable for:
                        </span>
                        <div className="flex flex-wrap gap-1 mt-1">
                          {method.suitable_for.map((type) => (
                            <span
                              key={type}
                              className="px-2 py-1 bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300 rounded text-xs"
                            >
                              {type.replace('_', ' ')}
                            </span>
                          ))}
                        </div>
                      </div>
                      
                      <div className="flex items-center justify-between text-xs">
                        <div className="flex items-center space-x-4">
                          <span className="text-neutral-500">
                            Cost: <span className={`font-medium ${
                              method.computational_cost === 'low' ? 'text-green-600' :
                              method.computational_cost === 'medium' ? 'text-yellow-600' : 'text-red-600'
                            }`}>
                              {method.computational_cost}
                            </span>
                          </span>
                        </div>
                        
                        <button className="text-blue-600 hover:text-blue-800 font-medium">
                          Configure →
                        </button>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>
          </Card>
        </div>
      )}

      {activeTab === 'alerts' && (
        <div className="space-y-6">
          <Card>
            <div className="p-6">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                  Alert Rules Configuration
                </h3>
                <button className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors">
                  <Plus className="w-4 h-4" />
                  <span>New Alert Rule</span>
                </button>
              </div>

              <div className="space-y-4">
                {alertRules.map((rule, index) => (
                  <motion.div
                    key={rule.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className="p-4 border border-neutral-200 dark:border-neutral-700 rounded-lg"
                  >
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center space-x-3">
                        <h4 className="font-medium text-neutral-900 dark:text-neutral-100">
                          {rule.name}
                        </h4>
                        <span className={`px-2 py-1 rounded text-xs font-medium ${getSeverityColor(rule.severity)}`}>
                          {rule.severity}
                        </span>
                        <span className={`px-2 py-1 rounded text-xs font-medium ${
                          rule.is_active ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300' :
                          'bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-300'
                        }`}>
                          {rule.is_active ? 'Active' : 'Inactive'}
                        </span>
                      </div>
                      
                      <div className="flex items-center space-x-2">
                        <button className="text-blue-600 hover:text-blue-800">
                          <Edit className="w-4 h-4" />
                        </button>
                        <button className="text-neutral-600 hover:text-neutral-800">
                          {rule.is_active ? (
                            <Pause className="w-4 h-4" />
                          ) : (
                            <Play className="w-4 h-4" />
                          )}
                        </button>
                        <button className="text-red-600 hover:text-red-800">
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-sm">
                      <div>
                        <span className="text-neutral-500">Trigger:</span>
                        <div className="font-medium text-neutral-900 dark:text-neutral-100">
                          {rule.trigger_type.replace('_', ' ')}
                        </div>
                      </div>
                      
                      <div>
                        <span className="text-neutral-500">Condition:</span>
                        <div className="font-medium text-neutral-900 dark:text-neutral-100">
                          {rule.metric} {rule.condition.replace('_', ' ')} {rule.threshold}
                        </div>
                      </div>
                      
                      <div>
                        <span className="text-neutral-500">Cooldown:</span>
                        <div className="font-medium text-neutral-900 dark:text-neutral-100">
                          {Math.floor(rule.cooldown_period / 60)} minutes
                        </div>
                      </div>
                      
                      <div>
                        <span className="text-neutral-500">Channels:</span>
                        <div className="flex space-x-1 mt-1">
                          {rule.notification_channels.includes('email') && (
                            <Mail className="w-4 h-4 text-blue-500" />
                          )}
                          {rule.notification_channels.includes('slack') && (
                            <MessageSquare className="w-4 h-4 text-green-500" />
                          )}
                          {rule.notification_channels.includes('sms') && (
                            <Phone className="w-4 h-4 text-purple-500" />
                          )}
                        </div>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>
          </Card>
        </div>
      )}

      {activeTab === 'global' && (
        <div className="space-y-6">
          <Card>
            <div className="p-6">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                    Global Configuration Settings
                  </h3>
                  <p className="text-sm text-neutral-600 dark:text-neutral-400">
                    Default settings applied to all new model configurations
                  </p>
                </div>
                <div className="flex space-x-2">
                  <button className="flex items-center space-x-2 px-4 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg hover:bg-neutral-50 dark:hover:bg-neutral-800 transition-colors">
                    <RotateCcw className="w-4 h-4" />
                    <span>Reset</span>
                  </button>
                  <button className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors">
                    <Save className="w-4 h-4" />
                    <span>Save Changes</span>
                  </button>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div className="space-y-6">
                  <div>
                    <h4 className="font-medium text-neutral-900 dark:text-neutral-100 mb-4">
                      Default Thresholds
                    </h4>
                    
                    <div className="space-y-4">
                      <div>
                        <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                          Drift Threshold
                        </label>
                        <input
                          type="number"
                          step="0.01"
                          min="0"
                          max="1"
                          defaultValue={globalSettings.default_drift_threshold}
                          className="w-full px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
                        />
                        <p className="text-xs text-neutral-500 mt-1">
                          Values above this threshold will trigger drift alerts
                        </p>
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                          Performance Threshold
                        </label>
                        <input
                          type="number"
                          step="0.01"
                          min="0"
                          max="1"
                          defaultValue={globalSettings.default_performance_threshold}
                          className="w-full px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
                        />
                        <p className="text-xs text-neutral-500 mt-1">
                          Performance degradation threshold for alerts
                        </p>
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                          Data Quality Threshold
                        </label>
                        <input
                          type="number"
                          step="0.01"
                          min="0"
                          max="1"
                          defaultValue={globalSettings.default_quality_threshold}
                          className="w-full px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
                        />
                        <p className="text-xs text-neutral-500 mt-1">
                          Minimum acceptable data quality score
                        </p>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="space-y-6">
                  <div>
                    <h4 className="font-medium text-neutral-900 dark:text-neutral-100 mb-4">
                      Monitoring Settings
                    </h4>
                    
                    <div className="space-y-4">
                      <div>
                        <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                          Monitoring Interval (seconds)
                        </label>
                        <input
                          type="number"
                          min="60"
                          max="3600"
                          defaultValue={globalSettings.monitoring_interval}
                          className="w-full px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
                        />
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                          Reference Data Max Age (days)
                        </label>
                        <input
                          type="number"
                          min="1"
                          max="365"
                          defaultValue={globalSettings.max_reference_data_age_days}
                          className="w-full px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
                        />
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                          History Retention (days)
                        </label>
                        <input
                          type="number"
                          min="7"
                          max="365"
                          defaultValue={globalSettings.retain_monitoring_history_days}
                          className="w-full px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
                        />
                      </div>
                    </div>
                  </div>

                  <div>
                    <h4 className="font-medium text-neutral-900 dark:text-neutral-100 mb-4">
                      Advanced Options
                    </h4>
                    
                    <div className="space-y-3">
                      <label className="flex items-center space-x-3">
                        <input
                          type="checkbox"
                          defaultChecked={globalSettings.auto_reference_update}
                          className="rounded border-neutral-300 dark:border-neutral-600"
                        />
                        <span className="text-sm text-neutral-700 dark:text-neutral-300">
                          Auto-update reference data
                        </span>
                      </label>
                      
                      <label className="flex items-center space-x-3">
                        <input
                          type="checkbox"
                          defaultChecked={globalSettings.enable_trend_analysis}
                          className="rounded border-neutral-300 dark:border-neutral-600"
                        />
                        <span className="text-sm text-neutral-700 dark:text-neutral-300">
                          Enable trend analysis
                        </span>
                      </label>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
};

export default AdvancedDriftConfiguration;