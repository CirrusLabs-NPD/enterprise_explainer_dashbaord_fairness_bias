import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Filter,
  Database,
  Play,
  Pause,
  Square,
  Settings,
  CheckCircle,
  AlertTriangle,
  Clock,
  BarChart3,
  ArrowRight,
  Upload,
  Download,
  RefreshCw,
  Eye,
  FileText,
  Zap,
  Target,
  Users,
  TrendingUp,
  Edit,
  Trash2,
  Copy,
  Save,
  Plus,
  ChevronDown,
  ChevronRight
} from 'lucide-react';
import Card from '../components/common/Card';
import MetricCard from '../components/common/MetricCard';
import { apiClient } from '../services/api';

interface PreprocessingStep {
  id: string;
  name: string;
  type: 'cleaning' | 'transformation' | 'feature_engineering' | 'validation' | 'encoding';
  description: string;
  parameters: Record<string, any>;
  enabled: boolean;
  order: number;
  execution_time?: number;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress?: number;
  output_preview?: {
    rows_affected: number;
    columns_affected: number;
    summary: string;
  };
}

interface Workflow {
  id: string;
  name: string;
  description: string;
  dataset_id: string;
  dataset_name: string;
  steps: PreprocessingStep[];
  status: 'draft' | 'running' | 'completed' | 'failed' | 'paused';
  created_at: string;
  last_modified: string;
  execution_stats?: {
    total_runtime: number;
    rows_processed: number;
    columns_processed: number;
    success_rate: number;
  };
}

interface DatasetInfo {
  id: string;
  name: string;
  rows: number;
  columns: number;
  size: string;
  format: string;
  quality_score: number;
  missing_values: number;
  duplicates: number;
  outliers: number;
}

const DataPreprocessingWorkflow: React.FC = () => {
  const [workflows, setWorkflows] = useState<Workflow[]>([]);
  const [selectedWorkflow, setSelectedWorkflow] = useState<Workflow | null>(null);
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'workflows' | 'builder' | 'monitoring' | 'templates'>('workflows');
  const [expandedSteps, setExpandedSteps] = useState<Set<string>>(new Set());

  // Mock data for demonstration
  const mockDatasets: DatasetInfo[] = [
    {
      id: 'dataset_001',
      name: 'Customer Demographics',
      rows: 125000,
      columns: 28,
      size: '45.2 MB',
      format: 'CSV',
      quality_score: 72,
      missing_values: 15.6,
      duplicates: 2.3,
      outliers: 4.1
    },
    {
      id: 'dataset_002',
      name: 'Transaction History',
      rows: 2500000,
      columns: 15,
      size: '380 MB',
      format: 'Parquet',
      quality_score: 85,
      missing_values: 8.2,
      duplicates: 0.5,
      outliers: 2.8
    }
  ];

  const mockWorkflows: Workflow[] = [
    {
      id: 'workflow_001',
      name: 'Credit Model Data Pipeline',
      description: 'Complete preprocessing pipeline for credit risk modeling',
      dataset_id: 'dataset_001',
      dataset_name: 'Customer Demographics',
      status: 'completed',
      created_at: '2024-01-15T10:00:00Z',
      last_modified: '2024-01-20T14:30:00Z',
      execution_stats: {
        total_runtime: 245,
        rows_processed: 125000,
        columns_processed: 28,
        success_rate: 98.5
      },
      steps: [
        {
          id: 'step_001',
          name: 'Missing Value Imputation',
          type: 'cleaning',
          description: 'Handle missing values using median/mode imputation',
          parameters: {
            strategy: 'median',
            numeric_strategy: 'median',
            categorical_strategy: 'most_frequent'
          },
          enabled: true,
          order: 1,
          execution_time: 15,
          status: 'completed',
          output_preview: {
            rows_affected: 19500,
            columns_affected: 8,
            summary: 'Imputed 19,500 missing values across 8 columns'
          }
        },
        {
          id: 'step_002',
          name: 'Duplicate Removal',
          type: 'cleaning',
          description: 'Remove duplicate records based on key fields',
          parameters: {
            subset: ['customer_id', 'email'],
            keep: 'first'
          },
          enabled: true,
          order: 2,
          execution_time: 8,
          status: 'completed',
          output_preview: {
            rows_affected: 2875,
            columns_affected: 28,
            summary: 'Removed 2,875 duplicate records'
          }
        },
        {
          id: 'step_003',
          name: 'Outlier Detection',
          type: 'cleaning',
          description: 'Detect and handle outliers using IQR method',
          parameters: {
            method: 'iqr',
            threshold: 1.5,
            action: 'cap'
          },
          enabled: true,
          order: 3,
          execution_time: 22,
          status: 'completed',
          output_preview: {
            rows_affected: 5125,
            columns_affected: 12,
            summary: 'Capped 5,125 outlier values in numeric columns'
          }
        },
        {
          id: 'step_004',
          name: 'Feature Scaling',
          type: 'transformation',
          description: 'Standardize numeric features using StandardScaler',
          parameters: {
            method: 'standard',
            with_mean: true,
            with_std: true
          },
          enabled: true,
          order: 4,
          execution_time: 12,
          status: 'completed',
          output_preview: {
            rows_affected: 122125,
            columns_affected: 15,
            summary: 'Scaled 15 numeric features to standard distribution'
          }
        },
        {
          id: 'step_005',
          name: 'Categorical Encoding',
          type: 'encoding',
          description: 'One-hot encode categorical variables',
          parameters: {
            method: 'onehot',
            drop_first: true,
            handle_unknown: 'ignore'
          },
          enabled: true,
          order: 5,
          execution_time: 18,
          status: 'completed',
          output_preview: {
            rows_affected: 122125,
            columns_affected: 13,
            summary: 'Created 47 new binary features from 13 categorical columns'
          }
        },
        {
          id: 'step_006',
          name: 'Feature Engineering',
          type: 'feature_engineering',
          description: 'Create new features from existing ones',
          parameters: {
            interactions: ['income_age', 'credit_score_debt_ratio'],
            polynomials: ['income^2', 'age^2'],
            ratios: ['debt_to_income', 'utilization_ratio']
          },
          enabled: true,
          order: 6,
          execution_time: 35,
          status: 'completed',
          output_preview: {
            rows_affected: 122125,
            columns_affected: 7,
            summary: 'Generated 7 new engineered features'
          }
        }
      ]
    },
    {
      id: 'workflow_002',
      name: 'Transaction Data Processing',
      description: 'Real-time transaction preprocessing for fraud detection',
      dataset_id: 'dataset_002',
      dataset_name: 'Transaction History',
      status: 'running',
      created_at: '2024-01-20T09:00:00Z',
      last_modified: '2024-01-20T14:35:00Z',
      steps: [
        {
          id: 'step_007',
          name: 'Data Validation',
          type: 'validation',
          description: 'Validate transaction data integrity',
          parameters: {
            checks: ['schema', 'ranges', 'business_rules'],
            fail_on_error: false
          },
          enabled: true,
          order: 1,
          execution_time: 45,
          status: 'completed',
          progress: 100
        },
        {
          id: 'step_008',
          name: 'Time Feature Extraction',
          type: 'feature_engineering',
          description: 'Extract temporal features from timestamps',
          parameters: {
            features: ['hour', 'day_of_week', 'month', 'is_weekend'],
            timezone: 'UTC'
          },
          enabled: true,
          order: 2,
          execution_time: 28,
          status: 'running',
          progress: 65
        },
        {
          id: 'step_009',
          name: 'Amount Normalization',
          type: 'transformation',
          description: 'Normalize transaction amounts by merchant category',
          parameters: {
            method: 'robust',
            by_group: 'merchant_category'
          },
          enabled: true,
          order: 3,
          status: 'pending'
        }
      ]
    }
  ];

  const stepTemplates = [
    {
      name: 'Missing Value Imputation',
      type: 'cleaning',
      description: 'Handle missing values using various imputation strategies',
      parameters: {
        strategy: 'median',
        numeric_strategy: 'median',
        categorical_strategy: 'most_frequent'
      }
    },
    {
      name: 'Outlier Detection',
      type: 'cleaning',
      description: 'Detect and handle outliers using statistical methods',
      parameters: {
        method: 'iqr',
        threshold: 1.5,
        action: 'cap'
      }
    },
    {
      name: 'Feature Scaling',
      type: 'transformation',
      description: 'Scale features using various normalization methods',
      parameters: {
        method: 'standard',
        with_mean: true,
        with_std: true
      }
    },
    {
      name: 'One-Hot Encoding',
      type: 'encoding',
      description: 'Convert categorical variables to binary features',
      parameters: {
        method: 'onehot',
        drop_first: true,
        handle_unknown: 'ignore'
      }
    }
  ];

  useEffect(() => {
    loadWorkflows();
  }, []);

  const loadWorkflows = async () => {
    try {
      setLoading(true);
      
      // Try to fetch from API, fallback to mock data
      const response = await apiClient.preprocessDataset('dataset_001', []);
      
      if (response.success) {
        // API would return workflows, but for now use mock data
        setTimeout(() => {
          setWorkflows(mockWorkflows);
          setDatasets(mockDatasets);
          if (mockWorkflows.length > 0) {
            setSelectedWorkflow(mockWorkflows[0]);
          }
          setLoading(false);
        }, 1000);
      } else {
        // Use mock data as fallback
        setTimeout(() => {
          setWorkflows(mockWorkflows);
          setDatasets(mockDatasets);
          if (mockWorkflows.length > 0) {
            setSelectedWorkflow(mockWorkflows[0]);
          }
          setLoading(false);
        }, 1000);
      }
    } catch (err) {
      console.warn('API call failed, using mock data:', err);
      // Use mock data as fallback
      setTimeout(() => {
        setWorkflows(mockWorkflows);
        setDatasets(mockDatasets);
        if (mockWorkflows.length > 0) {
          setSelectedWorkflow(mockWorkflows[0]);
        }
        setLoading(false);
      }, 1000);
    }
  };

  const executeWorkflow = async (workflowId: string) => {
    try {
      const workflow = workflows.find(w => w.id === workflowId);
      if (!workflow) return;

      // Update workflow status
      setWorkflows(prev => 
        prev.map(w => 
          w.id === workflowId 
            ? { ...w, status: 'running' }
            : w
        )
      );

      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Update status to completed
      setWorkflows(prev => 
        prev.map(w => 
          w.id === workflowId 
            ? { ...w, status: 'completed' }
            : w
        )
      );
    } catch (err) {
      console.error('Failed to execute workflow:', err);
    }
  };

  const toggleStepExpansion = (stepId: string) => {
    setExpandedSteps(prev => {
      const newSet = new Set(prev);
      if (newSet.has(stepId)) {
        newSet.delete(stepId);
      } else {
        newSet.add(stepId);
      }
      return newSet;
    });
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300';
      case 'running': return 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300';
      case 'failed': return 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300';
      case 'paused': return 'bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300';
      case 'pending': return 'bg-neutral-100 text-neutral-800 dark:bg-neutral-800 dark:text-neutral-300';
      default: return 'bg-neutral-100 text-neutral-800 dark:bg-neutral-800 dark:text-neutral-300';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'cleaning': return Filter;
      case 'transformation': return Zap;
      case 'feature_engineering': return Target;
      case 'validation': return CheckCircle;
      case 'encoding': return Settings;
      default: return Settings;
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'cleaning': return 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300';
      case 'transformation': return 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300';
      case 'feature_engineering': return 'bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300';
      case 'validation': return 'bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300';
      case 'encoding': return 'bg-pink-100 text-pink-800 dark:bg-pink-900/30 dark:text-pink-300';
      default: return 'bg-neutral-100 text-neutral-800 dark:bg-neutral-800 dark:text-neutral-300';
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <RefreshCw className="w-8 h-8 animate-spin text-primary-600" />
        <span className="ml-3 text-lg text-neutral-600 dark:text-neutral-400">
          Loading preprocessing workflows...
        </span>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-neutral-900 dark:text-neutral-100">
            Data Preprocessing Workflows
          </h1>
          <p className="text-neutral-600 dark:text-neutral-400 mt-1">
            Design, execute, and monitor automated data preprocessing pipelines
          </p>
        </div>
        
        <div className="flex items-center space-x-4">
          <button className="flex items-center space-x-2 px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors">
            <Plus className="w-4 h-4" />
            <span>New Workflow</span>
          </button>
          
          <button
            onClick={loadWorkflows}
            className="flex items-center space-x-2 px-4 py-2 bg-neutral-600 hover:bg-neutral-700 text-white rounded-lg transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Overview Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Total Workflows"
          value={workflows.length}
          format="number"
          icon={<BarChart3 className="w-5 h-5" />}
          change="+2"
          changeType="positive"
        />
        
        <MetricCard
          title="Active Workflows"
          value={workflows.filter(w => w.status === 'running').length}
          format="number"
          icon={<Play className="w-5 h-5" />}
          change={workflows.filter(w => w.status === 'running').length > 0 ? "Running" : "Idle"}
          changeType={workflows.filter(w => w.status === 'running').length > 0 ? "positive" : "neutral"}
        />
        
        <MetricCard
          title="Success Rate"
          value={workflows.length > 0 ? workflows.filter(w => w.status === 'completed').length / workflows.length : 0}
          format="percentage"
          icon={<CheckCircle className="w-5 h-5" />}
          change="+5.2%"
          changeType="positive"
        />
        
        <MetricCard
          title="Avg Runtime"
          value={`${Math.round(workflows.reduce((sum, w) => sum + (w.execution_stats?.total_runtime || 0), 0) / workflows.length || 0)}m`}
          format="text"
          icon={<Clock className="w-5 h-5" />}
          change="-12m"
          changeType="positive"
        />
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-neutral-200 dark:border-neutral-700">
        <nav className="flex space-x-8">
          {[
            { id: 'workflows', label: 'Workflows', icon: BarChart3 },
            { id: 'builder', label: 'Workflow Builder', icon: Settings },
            { id: 'monitoring', label: 'Monitoring', icon: Eye },
            { id: 'templates', label: 'Templates', icon: FileText }
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
      {activeTab === 'workflows' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Workflow List */}
          <Card className="lg:col-span-2">
            <div className="p-6">
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
                All Workflows
              </h3>
              
              <div className="space-y-4">
                {workflows.map((workflow) => (
                  <div
                    key={workflow.id}
                    className={`p-4 border rounded-lg transition-colors cursor-pointer ${
                      selectedWorkflow?.id === workflow.id
                        ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                        : 'border-neutral-200 dark:border-neutral-700 hover:border-primary-300 dark:hover:border-primary-600'
                    }`}
                    onClick={() => setSelectedWorkflow(workflow)}
                  >
                    <div className="flex items-center justify-between mb-3">
                      <div>
                        <h4 className="font-medium text-neutral-900 dark:text-neutral-100">
                          {workflow.name}
                        </h4>
                        <p className="text-sm text-neutral-600 dark:text-neutral-400">
                          {workflow.dataset_name} • {workflow.steps.length} steps
                        </p>
                      </div>
                      
                      <div className="flex items-center space-x-3">
                        <span className={`px-2 py-1 rounded text-xs font-medium ${getStatusColor(workflow.status)}`}>
                          {workflow.status.charAt(0).toUpperCase() + workflow.status.slice(1)}
                        </span>
                        
                        {workflow.status === 'draft' && (
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              executeWorkflow(workflow.id);
                            }}
                            className="p-1 text-green-600 hover:text-green-700 transition-colors"
                          >
                            <Play className="w-4 h-4" />
                          </button>
                        )}
                      </div>
                    </div>
                    
                    {workflow.execution_stats && (
                      <div className="grid grid-cols-3 gap-4 text-sm">
                        <div>
                          <span className="text-neutral-500">Runtime:</span>
                          <div className="font-medium text-neutral-900 dark:text-neutral-100">
                            {workflow.execution_stats.total_runtime}m
                          </div>
                        </div>
                        <div>
                          <span className="text-neutral-500">Rows:</span>
                          <div className="font-medium text-neutral-900 dark:text-neutral-100">
                            {workflow.execution_stats.rows_processed.toLocaleString()}
                          </div>
                        </div>
                        <div>
                          <span className="text-neutral-500">Success:</span>
                          <div className="font-medium text-green-600 dark:text-green-400">
                            {workflow.execution_stats.success_rate}%
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </Card>

          {/* Workflow Details */}
          {selectedWorkflow && (
            <Card>
              <div className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                    Workflow Steps
                  </h3>
                  <span className={`px-2 py-1 rounded text-xs font-medium ${getStatusColor(selectedWorkflow.status)}`}>
                    {selectedWorkflow.status}
                  </span>
                </div>
                
                <div className="space-y-3">
                  {selectedWorkflow.steps
                    .sort((a, b) => a.order - b.order)
                    .map((step, index) => {
                      const TypeIcon = getTypeIcon(step.type);
                      const isExpanded = expandedSteps.has(step.id);
                      
                      return (
                        <div key={step.id} className="border border-neutral-200 dark:border-neutral-700 rounded-lg">
                          <div
                            className="p-3 cursor-pointer hover:bg-neutral-50 dark:hover:bg-neutral-800 transition-colors"
                            onClick={() => toggleStepExpansion(step.id)}
                          >
                            <div className="flex items-center justify-between">
                              <div className="flex items-center space-x-3">
                                <div className="flex items-center space-x-2">
                                  {isExpanded ? (
                                    <ChevronDown className="w-4 h-4 text-neutral-500" />
                                  ) : (
                                    <ChevronRight className="w-4 h-4 text-neutral-500" />
                                  )}
                                  <span className="w-6 h-6 bg-neutral-100 dark:bg-neutral-800 rounded-full flex items-center justify-center text-xs font-medium">
                                    {step.order}
                                  </span>
                                </div>
                                
                                <div className={`p-1 rounded ${
                                  step.type === 'cleaning' ? 'bg-blue-100 dark:bg-blue-900/20' :
                                  step.type === 'transformation' ? 'bg-green-100 dark:bg-green-900/20' :
                                  step.type === 'feature_engineering' ? 'bg-purple-100 dark:bg-purple-900/20' :
                                  'bg-neutral-100 dark:bg-neutral-800'
                                }`}>
                                  <TypeIcon className={`w-4 h-4 ${
                                    step.type === 'cleaning' ? 'text-blue-600 dark:text-blue-400' :
                                    step.type === 'transformation' ? 'text-green-600 dark:text-green-400' :
                                    step.type === 'feature_engineering' ? 'text-purple-600 dark:text-purple-400' :
                                    'text-neutral-500'
                                  }`} />
                                </div>
                                
                                <div>
                                  <h5 className="font-medium text-sm text-neutral-900 dark:text-neutral-100">
                                    {step.name}
                                  </h5>
                                  <p className="text-xs text-neutral-600 dark:text-neutral-400">
                                    {step.type.replace('_', ' ')}
                                  </p>
                                </div>
                              </div>
                              
                              <div className="flex items-center space-x-2">
                                {step.status === 'running' && step.progress && (
                                  <div className="flex items-center space-x-2">
                                    <div className="w-16 bg-neutral-200 dark:bg-neutral-700 rounded-full h-1.5">
                                      <div
                                        className="h-1.5 bg-blue-500 rounded-full transition-all duration-300"
                                        style={{ width: `${step.progress}%` }}
                                      />
                                    </div>
                                    <span className="text-xs text-neutral-500">{step.progress}%</span>
                                  </div>
                                )}
                                
                                <span className={`px-2 py-1 rounded text-xs font-medium ${getStatusColor(step.status)}`}>
                                  {step.status}
                                </span>
                              </div>
                            </div>
                          </div>
                          
                          {isExpanded && (
                            <div className="px-3 pb-3 border-t border-neutral-200 dark:border-neutral-700">
                              <div className="pt-3 space-y-3">
                                <p className="text-sm text-neutral-600 dark:text-neutral-400">
                                  {step.description}
                                </p>
                                
                                {step.output_preview && (
                                  <div className="bg-neutral-50 dark:bg-neutral-800 p-3 rounded-lg">
                                    <h6 className="text-xs font-medium text-neutral-900 dark:text-neutral-100 mb-2">
                                      Output Preview
                                    </h6>
                                    <div className="text-xs text-neutral-600 dark:text-neutral-400 space-y-1">
                                      <div>Rows affected: {step.output_preview.rows_affected.toLocaleString()}</div>
                                      <div>Columns affected: {step.output_preview.columns_affected}</div>
                                      <div className="text-neutral-900 dark:text-neutral-100 font-medium">
                                        {step.output_preview.summary}
                                      </div>
                                    </div>
                                  </div>
                                )}
                                
                                {step.execution_time && (
                                  <div className="flex items-center space-x-2 text-xs text-neutral-500">
                                    <Clock className="w-3 h-3" />
                                    <span>Execution time: {step.execution_time}s</span>
                                  </div>
                                )}
                              </div>
                            </div>
                          )}
                        </div>
                      );
                    })}
                </div>
              </div>
            </Card>
          )}
        </div>
      )}

      {activeTab === 'builder' && (
        <Card>
          <div className="p-6">
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
              Workflow Builder
            </h3>
            <div className="text-center py-12 text-neutral-500 dark:text-neutral-400">
              <Settings className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p>Visual workflow builder coming soon</p>
              <p className="text-sm">Drag and drop interface for creating preprocessing pipelines</p>
            </div>
          </div>
        </Card>
      )}

      {activeTab === 'monitoring' && (
        <Card>
          <div className="p-6">
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
              Real-time Monitoring
            </h3>
            <div className="text-center py-12 text-neutral-500 dark:text-neutral-400">
              <Eye className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p>Real-time workflow monitoring coming soon</p>
              <p className="text-sm">Track execution progress and performance metrics</p>
            </div>
          </div>
        </Card>
      )}

      {activeTab === 'templates' && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {stepTemplates.map((template, index) => {
            const TypeIcon = getTypeIcon(template.type);
            
            return (
              <Card key={index}>
                <div className="p-6">
                  <div className="flex items-center space-x-3 mb-4">
                    <div className={`p-2 rounded-lg ${
                      template.type === 'cleaning' ? 'bg-blue-100 dark:bg-blue-900/20' :
                      template.type === 'transformation' ? 'bg-green-100 dark:bg-green-900/20' :
                      template.type === 'feature_engineering' ? 'bg-purple-100 dark:bg-purple-900/20' :
                      'bg-neutral-100 dark:bg-neutral-800'
                    }`}>
                      <TypeIcon className={`w-5 h-5 ${
                        template.type === 'cleaning' ? 'text-blue-600 dark:text-blue-400' :
                        template.type === 'transformation' ? 'text-green-600 dark:text-green-400' :
                        template.type === 'feature_engineering' ? 'text-purple-600 dark:text-purple-400' :
                        'text-neutral-500'
                      }`} />
                    </div>
                    <div>
                      <h4 className="font-medium text-neutral-900 dark:text-neutral-100">
                        {template.name}
                      </h4>
                      <span className={`px-2 py-1 rounded text-xs font-medium ${getTypeColor(template.type)}`}>
                        {template.type.replace('_', ' ')}
                      </span>
                    </div>
                  </div>
                  
                  <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-4">
                    {template.description}
                  </p>
                  
                  <div className="flex items-center space-x-2">
                    <button className="flex items-center space-x-2 px-3 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors text-sm">
                      <Plus className="w-4 h-4" />
                      <span>Add to Workflow</span>
                    </button>
                    
                    <button className="flex items-center space-x-2 px-3 py-2 bg-neutral-100 dark:bg-neutral-800 text-neutral-700 dark:text-neutral-300 rounded-lg transition-colors text-sm">
                      <Eye className="w-4 h-4" />
                      <span>Preview</span>
                    </button>
                  </div>
                </div>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default DataPreprocessingWorkflow;