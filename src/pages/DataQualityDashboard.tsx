import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Database,
  AlertTriangle,
  CheckCircle,
  TrendingUp,
  TrendingDown,
  FileText,
  BarChart3,
  PieChart,
  Activity,
  Settings,
  RefreshCw,
  Download,
  Filter,
  Search,
  Calendar,
  Target,
  Zap,
  Eye,
  Info
} from 'lucide-react';
import Card from '../components/common/Card';
import MetricCard from '../components/common/MetricCard';
import { apiClient } from '../services/api';

interface DataQualityMetrics {
  completeness: number;
  accuracy: number;
  consistency: number;
  validity: number;
  uniqueness: number;
  timeliness: number;
  overall_score: number;
}

interface QualityIssue {
  id: string;
  type: 'missing_values' | 'duplicates' | 'outliers' | 'format_errors' | 'invalid_values';
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  affected_rows: number;
  affected_columns: string[];
  suggested_action: string;
  detected_at: string;
}

interface DatasetQuality {
  dataset_id: string;
  dataset_name: string;
  total_rows: number;
  total_columns: number;
  metrics: DataQualityMetrics;
  issues: QualityIssue[];
  last_assessed: string;
  trend: 'improving' | 'stable' | 'declining';
}

interface QualityRules {
  id: string;
  name: string;
  type: 'completeness' | 'range' | 'format' | 'uniqueness' | 'consistency';
  description: string;
  threshold: number;
  enabled: boolean;
  violations: number;
}

const DataQualityDashboard: React.FC = () => {
  const [selectedDataset, setSelectedDataset] = useState<string>('customer_data');
  const [qualityData, setQualityData] = useState<DatasetQuality | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'issues' | 'rules' | 'trends'>('overview');
  const [assessmentRunning, setAssessmentRunning] = useState(false);

  // Mock data for demonstration
  const mockQualityData: DatasetQuality = {
    dataset_id: 'customer_data',
    dataset_name: 'Customer Demographics',
    total_rows: 125000,
    total_columns: 28,
    metrics: {
      completeness: 87.3,
      accuracy: 91.2,
      consistency: 85.7,
      validity: 93.8,
      uniqueness: 96.1,
      timeliness: 78.9,
      overall_score: 88.8
    },
    issues: [
      {
        id: 'iss_001',
        type: 'missing_values',
        severity: 'high',
        description: 'High rate of missing values in income field',
        affected_rows: 15600,
        affected_columns: ['annual_income'],
        suggested_action: 'Implement income estimation model or require field completion',
        detected_at: '2024-01-20T10:30:00Z'
      },
      {
        id: 'iss_002',
        type: 'duplicates',
        severity: 'medium',
        description: 'Duplicate customer records based on email',
        affected_rows: 3200,
        affected_columns: ['email', 'customer_id'],
        suggested_action: 'Merge duplicate records and establish unique constraints',
        detected_at: '2024-01-20T10:45:00Z'
      },
      {
        id: 'iss_003',
        type: 'outliers',
        severity: 'low',
        description: 'Age values outside expected range',
        affected_rows: 145,
        affected_columns: ['age'],
        suggested_action: 'Review and validate outlier records manually',
        detected_at: '2024-01-20T11:15:00Z'
      },
      {
        id: 'iss_004',
        type: 'format_errors',
        severity: 'medium',
        description: 'Inconsistent phone number formats',
        affected_rows: 8900,
        affected_columns: ['phone_number'],
        suggested_action: 'Standardize phone number format using validation rules',
        detected_at: '2024-01-20T11:30:00Z'
      }
    ],
    last_assessed: '2024-01-20T12:00:00Z',
    trend: 'improving'
  };

  const mockQualityRules: QualityRules[] = [
    {
      id: 'rule_001',
      name: 'Email Format Validation',
      type: 'format',
      description: 'Ensures all email addresses follow valid format',
      threshold: 95.0,
      enabled: true,
      violations: 245
    },
    {
      id: 'rule_002',
      name: 'Age Range Check',
      type: 'range',
      description: 'Age should be between 18 and 120',
      threshold: 99.0,
      enabled: true,
      violations: 12
    },
    {
      id: 'rule_003',
      name: 'Customer ID Uniqueness',
      type: 'uniqueness',
      description: 'Customer IDs must be unique across dataset',
      threshold: 100.0,
      enabled: true,
      violations: 0
    },
    {
      id: 'rule_004',
      name: 'Required Fields Completeness',
      type: 'completeness',
      description: 'Critical fields must not be null',
      threshold: 98.0,
      enabled: true,
      violations: 1876
    }
  ];

  useEffect(() => {
    loadQualityData();
  }, [selectedDataset]);

  const loadQualityData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Try to fetch from API, fallback to mock data
      const response = await apiClient.getDataQuality(selectedDataset);
      
      if (response.success && response.data) {
        setQualityData(response.data);
      } else {
        // Use mock data as fallback
        setTimeout(() => {
          setQualityData(mockQualityData);
          setLoading(false);
        }, 1000);
        return;
      }
    } catch (err) {
      console.warn('API call failed, using mock data:', err);
      // Use mock data as fallback
      setTimeout(() => {
        setQualityData(mockQualityData);
        setLoading(false);
      }, 1000);
      return;
    }
    
    setLoading(false);
  };

  const runQualityAssessment = async () => {
    try {
      setAssessmentRunning(true);
      
      const response = await apiClient.runQualityAssessment(selectedDataset);
      
      if (response.success) {
        // Refresh data after assessment
        await loadQualityData();
      }
    } catch (err) {
      console.error('Failed to run quality assessment:', err);
      // Simulate assessment completion with mock data
      setTimeout(() => {
        loadQualityData();
      }, 3000);
    } finally {
      setAssessmentRunning(false);
    }
  };

  const qualityMetrics = qualityData ? [
    {
      title: 'Overall Score',
      value: qualityData.metrics.overall_score / 100,
      format: 'percentage' as const,
      icon: Target,
      change: 2.3,
      changeType: 'positive' as const,
      color: 'primary' as const
    },
    {
      title: 'Completeness',
      value: qualityData.metrics.completeness / 100,
      format: 'percentage' as const,
      icon: CheckCircle,
      change: -1.2,
      changeType: 'negative' as const,
      color: 'success' as const
    },
    {
      title: 'Accuracy',
      value: qualityData.metrics.accuracy / 100,
      format: 'percentage' as const,
      icon: Zap,
      change: 0.8,
      changeType: 'positive' as const,
      color: 'secondary' as const
    },
    {
      title: 'Uniqueness',
      value: qualityData.metrics.uniqueness / 100,
      format: 'percentage' as const,
      icon: Database,
      change: 1.5,
      changeType: 'positive' as const,
      color: 'warning' as const
    }
  ] : [];

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'bg-red-500 text-white';
      case 'high': return 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300';
      case 'medium': return 'bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300';
      case 'low': return 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300';
      default: return 'bg-neutral-100 text-neutral-800 dark:bg-neutral-800 dark:text-neutral-300';
    }
  };

  const getIssueIcon = (type: string) => {
    switch (type) {
      case 'missing_values': return AlertTriangle;
      case 'duplicates': return FileText;
      case 'outliers': return TrendingUp;
      case 'format_errors': return Settings;
      case 'invalid_values': return AlertTriangle;
      default: return Info;
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <RefreshCw className="w-8 h-8 animate-spin text-primary-600" />
        <span className="ml-3 text-lg text-neutral-600 dark:text-neutral-400">
          Loading data quality assessment...
        </span>
      </div>
    );
  }

  if (error || !qualityData) {
    return (
      <div className="flex flex-col items-center justify-center py-12">
        <AlertTriangle className="w-12 h-12 text-red-500 mb-4" />
        <h2 className="text-xl font-semibold text-neutral-900 dark:text-neutral-100 mb-2">
          Failed to Load Quality Data
        </h2>
        <p className="text-neutral-600 dark:text-neutral-400 mb-4 text-center max-w-md">
          {error || 'Unable to load data quality information'}
        </p>
        <button
          onClick={loadQualityData}
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
            Data Quality Assessment
          </h1>
          <p className="text-neutral-600 dark:text-neutral-400 mt-1">
            Monitor and improve data quality across your datasets
          </p>
        </div>
        
        <div className="flex items-center space-x-4">
          <select
            value={selectedDataset}
            onChange={(e) => setSelectedDataset(e.target.value)}
            className="px-4 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
          >
            <option value="customer_data">Customer Demographics</option>
            <option value="transaction_data">Transaction History</option>
            <option value="product_data">Product Catalog</option>
            <option value="behavioral_data">User Behavior</option>
          </select>
          
          <button
            onClick={runQualityAssessment}
            disabled={assessmentRunning}
            className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white rounded-lg transition-colors"
          >
            {assessmentRunning ? (
              <RefreshCw className="w-4 h-4 animate-spin" />
            ) : (
              <Activity className="w-4 h-4" />
            )}
            <span>{assessmentRunning ? 'Running...' : 'Run Assessment'}</span>
          </button>
          
          <button className="flex items-center space-x-2 px-4 py-2 bg-neutral-600 hover:bg-neutral-700 text-white rounded-lg transition-colors">
            <Download className="w-4 h-4" />
            <span>Export Report</span>
          </button>
        </div>
      </div>

      {/* Quality Metrics Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {qualityMetrics.map((metric) => (
          <MetricCard
            key={metric.title}
            title={metric.title}
            value={metric.value}
            format={metric.format}
            icon={<metric.icon className="w-5 h-5" />}
            change={`${metric.change > 0 ? '+' : ''}${metric.change}%`}
            changeType={metric.changeType}
          />
        ))}
      </div>

      {/* Dataset Summary */}
      <Card>
        <div className="p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
              Dataset Summary
            </h3>
            <div className="flex items-center space-x-2">
              <Calendar className="w-4 h-4 text-neutral-500" />
              <span className="text-sm text-neutral-500">
                Last assessed: {new Date(qualityData.last_assessed).toLocaleDateString()}
              </span>
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <div className="text-center">
              <div className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
                {qualityData.total_rows.toLocaleString()}
              </div>
              <div className="text-sm text-neutral-600 dark:text-neutral-400">Total Rows</div>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
                {qualityData.total_columns}
              </div>
              <div className="text-sm text-neutral-600 dark:text-neutral-400">Columns</div>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
                {qualityData.issues.length}
              </div>
              <div className="text-sm text-neutral-600 dark:text-neutral-400">Issues Found</div>
            </div>
            
            <div className="text-center">
              <div className={`text-2xl font-bold ${
                qualityData.trend === 'improving' ? 'text-green-600' :
                qualityData.trend === 'declining' ? 'text-red-600' : 'text-neutral-600'
              }`}>
                {qualityData.trend === 'improving' ? <TrendingUp className="w-8 h-8 mx-auto" /> :
                 qualityData.trend === 'declining' ? <TrendingDown className="w-8 h-8 mx-auto" /> :
                 <Activity className="w-8 h-8 mx-auto" />}
              </div>
              <div className="text-sm text-neutral-600 dark:text-neutral-400 capitalize">
                {qualityData.trend}
              </div>
            </div>
          </div>
        </div>
      </Card>

      {/* Tab Navigation */}
      <div className="border-b border-neutral-200 dark:border-neutral-700">
        <nav className="flex space-x-8">
          {[
            { id: 'overview', label: 'Quality Overview', icon: BarChart3 },
            { id: 'issues', label: 'Issues', icon: AlertTriangle },
            { id: 'rules', label: 'Quality Rules', icon: Settings },
            { id: 'trends', label: 'Trends', icon: TrendingUp }
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
      {activeTab === 'overview' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Quality Dimensions */}
          <Card>
            <div className="p-6">
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
                Quality Dimensions
              </h3>
              
              <div className="space-y-4">
                {Object.entries(qualityData.metrics).filter(([key]) => key !== 'overall_score').map(([dimension, score]) => (
                  <div key={dimension} className="flex items-center justify-between">
                    <span className="text-sm font-medium text-neutral-900 dark:text-neutral-100 capitalize">
                      {dimension.replace('_', ' ')}
                    </span>
                    <div className="flex items-center space-x-3">
                      <div className="w-32 bg-neutral-200 dark:bg-neutral-700 rounded-full h-2">
                        <div
                          className={`h-2 rounded-full transition-all duration-500 ${
                            score >= 90 ? 'bg-green-500' :
                            score >= 70 ? 'bg-amber-500' : 'bg-red-500'
                          }`}
                          style={{ width: `${score}%` }}
                        />
                      </div>
                      <span className="text-sm font-medium text-neutral-900 dark:text-neutral-100 w-12">
                        {score.toFixed(1)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </Card>

          {/* Issue Distribution */}
          <Card>
            <div className="p-6">
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
                Issue Distribution
              </h3>
              
              <div className="space-y-3">
                {['critical', 'high', 'medium', 'low'].map((severity) => {
                  const count = qualityData.issues.filter(issue => issue.severity === severity).length;
                  const percentage = (count / qualityData.issues.length) * 100;
                  
                  return (
                    <div key={severity} className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <span className={`px-2 py-1 rounded text-xs font-medium ${getSeverityColor(severity)}`}>
                          {severity.charAt(0).toUpperCase() + severity.slice(1)}
                        </span>
                        <span className="text-sm text-neutral-600 dark:text-neutral-400">
                          {count} issues
                        </span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <div className="w-16 bg-neutral-200 dark:bg-neutral-700 rounded-full h-2">
                          <div
                            className="h-2 bg-primary-500 rounded-full"
                            style={{ width: `${percentage}%` }}
                          />
                        </div>
                        <span className="text-xs text-neutral-500 w-8">
                          {percentage.toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </Card>
        </div>
      )}

      {activeTab === 'issues' && (
        <Card>
          <div className="p-6">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                Data Quality Issues
              </h3>
              <div className="flex items-center space-x-4">
                <div className="relative">
                  <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-neutral-400" />
                  <input
                    type="text"
                    placeholder="Search issues..."
                    className="pl-10 pr-4 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
                  />
                </div>
                <select className="px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100">
                  <option value="">All Severities</option>
                  <option value="critical">Critical</option>
                  <option value="high">High</option>
                  <option value="medium">Medium</option>
                  <option value="low">Low</option>
                </select>
              </div>
            </div>
            
            <div className="space-y-4">
              {qualityData.issues.map((issue) => {
                const IconComponent = getIssueIcon(issue.type);
                
                return (
                  <motion.div
                    key={issue.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="p-4 border border-neutral-200 dark:border-neutral-700 rounded-lg hover:border-primary-300 dark:hover:border-primary-600 transition-colors"
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex items-center space-x-3">
                        <IconComponent className="w-5 h-5 text-neutral-500" />
                        <div>
                          <h4 className="font-medium text-neutral-900 dark:text-neutral-100">
                            {issue.description}
                          </h4>
                          <p className="text-sm text-neutral-600 dark:text-neutral-400">
                            Affects {issue.affected_rows.toLocaleString()} rows in {issue.affected_columns.join(', ')}
                          </p>
                        </div>
                      </div>
                      <span className={`px-2 py-1 rounded text-xs font-medium ${getSeverityColor(issue.severity)}`}>
                        {issue.severity.charAt(0).toUpperCase() + issue.severity.slice(1)}
                      </span>
                    </div>
                    
                    <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded-lg">
                      <div className="flex items-start space-x-2">
                        <Info className="w-4 h-4 text-blue-600 dark:text-blue-400 mt-0.5" />
                        <div>
                          <h5 className="text-sm font-medium text-blue-900 dark:text-blue-100">
                            Suggested Action
                          </h5>
                          <p className="text-sm text-blue-700 dark:text-blue-300">
                            {issue.suggested_action}
                          </p>
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex items-center justify-between mt-3 pt-3 border-t border-neutral-200 dark:border-neutral-700">
                      <span className="text-xs text-neutral-500">
                        Detected: {new Date(issue.detected_at).toLocaleDateString()}
                      </span>
                      <div className="flex items-center space-x-2">
                        <button className="text-xs text-primary-600 hover:text-primary-700 dark:text-primary-400 dark:hover:text-primary-300">
                          View Details
                        </button>
                        <button className="text-xs text-green-600 hover:text-green-700 dark:text-green-400 dark:hover:text-green-300">
                          Mark Resolved
                        </button>
                      </div>
                    </div>
                  </motion.div>
                );
              })}
            </div>
          </div>
        </Card>
      )}

      {activeTab === 'rules' && (
        <Card>
          <div className="p-6">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                Quality Rules
              </h3>
              <button className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors">
                Add Rule
              </button>
            </div>
            
            <div className="space-y-4">
              {mockQualityRules.map((rule) => (
                <div
                  key={rule.id}
                  className="p-4 border border-neutral-200 dark:border-neutral-700 rounded-lg"
                >
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center space-x-3">
                      <div className={`w-3 h-3 rounded-full ${rule.enabled ? 'bg-green-500' : 'bg-red-500'}`} />
                      <div>
                        <h4 className="font-medium text-neutral-900 dark:text-neutral-100">
                          {rule.name}
                        </h4>
                        <p className="text-sm text-neutral-600 dark:text-neutral-400">
                          {rule.description}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-4">
                      <span className="text-sm text-neutral-600 dark:text-neutral-400">
                        Threshold: {rule.threshold}%
                      </span>
                      <span className={`px-2 py-1 rounded text-xs font-medium ${
                        rule.violations === 0 
                          ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300'
                          : 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300'
                      }`}>
                        {rule.violations} violations
                      </span>
                    </div>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      rule.type === 'completeness' ? 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300' :
                      rule.type === 'format' ? 'bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300' :
                      rule.type === 'range' ? 'bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300' :
                      rule.type === 'uniqueness' ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300' :
                      'bg-neutral-100 text-neutral-800 dark:bg-neutral-800 dark:text-neutral-300'
                    }`}>
                      {rule.type}
                    </span>
                    
                    <div className="flex items-center space-x-2">
                      <button className="text-sm text-primary-600 hover:text-primary-700 dark:text-primary-400">
                        Edit
                      </button>
                      <button className={`text-sm ${
                        rule.enabled 
                          ? 'text-red-600 hover:text-red-700 dark:text-red-400'
                          : 'text-green-600 hover:text-green-700 dark:text-green-400'
                      }`}>
                        {rule.enabled ? 'Disable' : 'Enable'}
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </Card>
      )}

      {activeTab === 'trends' && (
        <Card>
          <div className="p-6">
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
              Quality Trends
            </h3>
            <div className="text-center py-12 text-neutral-500 dark:text-neutral-400">
              <TrendingUp className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p>Historical trend visualization coming soon</p>
              <p className="text-sm">Track quality improvements over time</p>
            </div>
          </div>
        </Card>
      )}
    </div>
  );
};

export default DataQualityDashboard;