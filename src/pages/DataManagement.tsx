import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Database,
  Upload,
  Download,
  RefreshCw,
  Search,
  Filter,
  Eye,
  Edit,
  Trash2,
  Plus,
  FileText,
  BarChart3,
  Calendar,
  Users,
  Lock,
  Unlock,
  CheckCircle,
  AlertTriangle,
  Clock,
  HardDrive,
  Activity,
  TrendingUp,
  Settings,
  Share2,
  Copy,
  ExternalLink,
  FolderOpen,
  Tag,
  Info,
} from 'lucide-react';
import Card from '../components/common/Card';
import MetricCard from '../components/common/MetricCard';

interface Dataset {
  id: string;
  name: string;
  description: string;
  type: 'training' | 'validation' | 'test' | 'production' | 'reference';
  format: 'csv' | 'json' | 'parquet' | 'avro' | 'orc';
  size_mb: number;
  rows: number;
  columns: number;
  created_at: string;
  updated_at: string;
  created_by: string;
  status: 'processing' | 'ready' | 'error' | 'archived';
  quality_score: number;
  schema_version: string;
  tags: string[];
  access_level: 'public' | 'restricted' | 'private';
  location: string;
  lineage?: {
    source_datasets: string[];
    derived_datasets: string[];
  };
}

interface DataQualityMetrics {
  dataset_id: string;
  completeness: number;
  uniqueness: number;
  validity: number;
  consistency: number;
  accuracy: number;
  missing_values: number;
  duplicate_rows: number;
  outliers: number;
  schema_violations: number;
  last_checked: string;
}

interface DataPipeline {
  id: string;
  name: string;
  source_dataset: string;
  target_dataset: string;
  transformation_steps: string[];
  schedule: 'manual' | 'hourly' | 'daily' | 'weekly';
  status: 'active' | 'paused' | 'error' | 'completed';
  last_run: string;
  next_run?: string;
  success_rate: number;
}

const DataManagement: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'datasets' | 'quality' | 'pipelines' | 'lineage'>('datasets');
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState<string>('all');
  const [showUploadModal, setShowUploadModal] = useState(false);

  // Mock data - in production, this would come from API calls
  const datasets: Dataset[] = [
    {
      id: 'ds_001',
      name: 'Customer Transaction Data',
      description: 'Historical customer transaction records for fraud detection model training',
      type: 'training',
      format: 'parquet',
      size_mb: 2450,
      rows: 1250000,
      columns: 47,
      created_at: '2024-01-15T08:30:00Z',
      updated_at: '2024-01-19T10:15:00Z',
      created_by: 'data.team@company.com',
      status: 'ready',
      quality_score: 0.94,
      schema_version: 'v2.1',
      tags: ['fraud', 'transactions', 'customers'],
      access_level: 'restricted',
      location: 's3://ml-data-lake/fraud/training/',
      lineage: {
        source_datasets: ['raw_transactions', 'customer_profiles'],
        derived_datasets: ['fraud_features_v1', 'fraud_features_v2'],
      },
    },
    {
      id: 'ds_002',
      name: 'Credit Risk Features',
      description: 'Engineered features for credit risk assessment model',
      type: 'production',
      format: 'csv',
      size_mb: 890,
      rows: 500000,
      columns: 23,
      created_at: '2024-01-18T14:20:00Z',
      updated_at: '2024-01-19T09:30:00Z',
      created_by: 'ml.platform@company.com',
      status: 'ready',
      quality_score: 0.89,
      schema_version: 'v1.8',
      tags: ['credit', 'risk', 'features'],
      access_level: 'private',
      location: 's3://ml-data-lake/credit/features/',
      lineage: {
        source_datasets: ['loan_applications', 'credit_history'],
        derived_datasets: [],
      },
    },
    {
      id: 'ds_003',
      name: 'Reference Data Snapshot',
      description: 'Reference dataset for drift detection baseline',
      type: 'reference',
      format: 'json',
      size_mb: 156,
      rows: 75000,
      columns: 15,
      created_at: '2024-01-10T11:45:00Z',
      updated_at: '2024-01-10T11:45:00Z',
      created_by: 'monitoring.service@company.com',
      status: 'ready',
      quality_score: 0.97,
      schema_version: 'v1.0',
      tags: ['reference', 'baseline', 'monitoring'],
      access_level: 'public',
      location: 's3://ml-data-lake/reference/',
    },
    {
      id: 'ds_004',
      name: 'Real-time Stream Data',
      description: 'Live streaming data for real-time model inference',
      type: 'production',
      format: 'avro',
      size_mb: 45,
      rows: 12000,
      columns: 32,
      created_at: '2024-01-19T10:00:00Z',
      updated_at: '2024-01-19T10:45:00Z',
      created_by: 'stream.processor@company.com',
      status: 'processing',
      quality_score: 0.91,
      schema_version: 'v3.2',
      tags: ['streaming', 'real-time', 'inference'],
      access_level: 'restricted',
      location: 'kafka://ml-streams/inference-data/',
    },
  ];

  const qualityMetrics: DataQualityMetrics[] = [
    {
      dataset_id: 'ds_001',
      completeness: 0.96,
      uniqueness: 0.94,
      validity: 0.93,
      consistency: 0.95,
      accuracy: 0.92,
      missing_values: 4250,
      duplicate_rows: 145,
      outliers: 1850,
      schema_violations: 12,
      last_checked: '2024-01-19T10:15:00Z',
    },
    {
      dataset_id: 'ds_002',
      completeness: 0.91,
      uniqueness: 0.89,
      validity: 0.88,
      consistency: 0.90,
      accuracy: 0.87,
      missing_values: 2890,
      duplicate_rows: 78,
      outliers: 920,
      schema_violations: 5,
      last_checked: '2024-01-19T09:30:00Z',
    },
    {
      dataset_id: 'ds_003',
      completeness: 0.98,
      uniqueness: 0.97,
      validity: 0.96,
      consistency: 0.98,
      accuracy: 0.96,
      missing_values: 150,
      duplicate_rows: 2,
      outliers: 45,
      schema_violations: 0,
      last_checked: '2024-01-19T08:00:00Z',
    },
  ];

  const pipelines: DataPipeline[] = [
    {
      id: 'pipe_001',
      name: 'Transaction Data Preprocessing',
      source_dataset: 'raw_transactions',
      target_dataset: 'ds_001',
      transformation_steps: [
        'Remove PII fields',
        'Apply feature engineering',
        'Data validation',
        'Quality checks',
      ],
      schedule: 'daily',
      status: 'active',
      last_run: '2024-01-19T02:00:00Z',
      next_run: '2024-01-20T02:00:00Z',
      success_rate: 0.98,
    },
    {
      id: 'pipe_002',
      name: 'Credit Features Pipeline',
      source_dataset: 'loan_applications',
      target_dataset: 'ds_002',
      transformation_steps: [
        'Feature extraction',
        'Normalization',
        'Encoding categorical variables',
        'Validation',
      ],
      schedule: 'hourly',
      status: 'active',
      last_run: '2024-01-19T10:00:00Z',
      next_run: '2024-01-19T11:00:00Z',
      success_rate: 0.95,
    },
    {
      id: 'pipe_003',
      name: 'Reference Data Update',
      source_dataset: 'ds_001',
      target_dataset: 'ds_003',
      transformation_steps: [
        'Sample representative data',
        'Statistical validation',
        'Version control',
      ],
      schedule: 'weekly',
      status: 'completed',
      last_run: '2024-01-15T06:00:00Z',
      next_run: '2024-01-22T06:00:00Z',
      success_rate: 1.0,
    },
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'ready':
        return 'text-green-600 bg-green-100 dark:bg-green-900/30 dark:text-green-300';
      case 'processing':
        return 'text-blue-600 bg-blue-100 dark:bg-blue-900/30 dark:text-blue-300';
      case 'error':
        return 'text-red-600 bg-red-100 dark:bg-red-900/30 dark:text-red-300';
      case 'archived':
        return 'text-gray-600 bg-gray-100 dark:bg-gray-900/30 dark:text-gray-300';
      case 'active':
        return 'text-green-600 bg-green-100 dark:bg-green-900/30 dark:text-green-300';
      case 'paused':
        return 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900/30 dark:text-yellow-300';
      case 'completed':
        return 'text-blue-600 bg-blue-100 dark:bg-blue-900/30 dark:text-blue-300';
      default:
        return 'text-gray-600 bg-gray-100 dark:bg-gray-900/30 dark:text-gray-300';
    }
  };

  const getAccessLevelColor = (level: string) => {
    switch (level) {
      case 'public':
        return 'text-green-600 bg-green-100 dark:bg-green-900/30 dark:text-green-300';
      case 'restricted':
        return 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900/30 dark:text-yellow-300';
      case 'private':
        return 'text-red-600 bg-red-100 dark:bg-red-900/30 dark:text-red-300';
      default:
        return 'text-gray-600 bg-gray-100 dark:bg-gray-900/30 dark:text-gray-300';
    }
  };

  const getQualityScoreColor = (score: number) => {
    if (score >= 0.9) return 'text-green-600';
    if (score >= 0.8) return 'text-yellow-600';
    return 'text-red-600';
  };

  const formatFileSize = (sizeMb: number) => {
    if (sizeMb >= 1024) {
      return `${(sizeMb / 1024).toFixed(1)} GB`;
    }
    return `${sizeMb} MB`;
  };

  const filteredDatasets = datasets.filter(dataset => {
    const matchesSearch = dataset.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         dataset.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         dataset.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()));
    const matchesFilter = filterType === 'all' || dataset.type === filterType;
    return matchesSearch && matchesFilter;
  });

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-neutral-900 dark:text-neutral-100">
            Data Management
          </h1>
          <p className="text-neutral-600 dark:text-neutral-400 mt-1">
            Comprehensive data lifecycle management and quality monitoring
          </p>
        </div>
        
        <div className="flex items-center space-x-4">
          <button
            onClick={() => setShowUploadModal(true)}
            className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
          >
            <Upload className="w-4 h-4" />
            <span>Upload Dataset</span>
          </button>
          
          <button className="flex items-center space-x-2 px-4 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg hover:bg-neutral-50 dark:hover:bg-neutral-800 transition-colors">
            <Settings className="w-4 h-4" />
            <span>Settings</span>
          </button>
        </div>
      </div>

      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
        <MetricCard
          title="Total Datasets"
          value={datasets.length.toString()}
          change="+3"
          changeType="positive"
          icon={Database}
          description="Managed datasets across all environments"
        />
        
        <MetricCard
          title="Storage Used"
          value={`${(datasets.reduce((sum, ds) => sum + ds.size_mb, 0) / 1024).toFixed(1)} GB`}
          change="+12%"
          changeType="neutral"
          icon={HardDrive}
          description="Total storage consumption"
        />
        
        <MetricCard
          title="Avg Quality Score"
          value={`${(datasets.reduce((sum, ds) => sum + ds.quality_score, 0) / datasets.length * 100).toFixed(1)}%`}
          change="+2.1%"
          changeType="positive"
          icon={CheckCircle}
          description="Average data quality across datasets"
        />
        
        <MetricCard
          title="Active Pipelines"
          value={pipelines.filter(p => p.status === 'active').length.toString()}
          change="+1"
          changeType="positive"
          icon={Activity}
          description="Currently running data pipelines"
        />
        
        <MetricCard
          title="Data Freshness"
          value="4.2h"
          change="-0.8h"
          changeType="positive"
          icon={Clock}
          description="Average data age across production datasets"
        />
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-neutral-200 dark:border-neutral-700">
        <nav className="flex space-x-8">
          {[
            { id: 'datasets', label: 'Datasets', icon: Database },
            { id: 'quality', label: 'Data Quality', icon: CheckCircle },
            { id: 'pipelines', label: 'Pipelines', icon: Activity },
            { id: 'lineage', label: 'Data Lineage', icon: Share2 },
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
      {activeTab === 'datasets' && (
        <div className="space-y-6">
          {/* Search and Filters */}
          <Card>
            <div className="p-6">
              <div className="flex items-center justify-between space-x-4">
                <div className="flex-1 flex items-center space-x-4">
                  <div className="relative flex-1 max-w-md">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-neutral-400 w-4 h-4" />
                    <input
                      type="text"
                      placeholder="Search datasets..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      className="w-full pl-10 pr-4 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
                    />
                  </div>
                  
                  <select
                    value={filterType}
                    onChange={(e) => setFilterType(e.target.value)}
                    className="px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
                  >
                    <option value="all">All Types</option>
                    <option value="training">Training</option>
                    <option value="validation">Validation</option>
                    <option value="test">Test</option>
                    <option value="production">Production</option>
                    <option value="reference">Reference</option>
                  </select>
                </div>
                
                <div className="flex items-center space-x-2">
                  <button className="flex items-center space-x-2 px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg hover:bg-neutral-50 dark:hover:bg-neutral-800 transition-colors">
                    <Filter className="w-4 h-4" />
                    <span>Advanced Filters</span>
                  </button>
                  
                  <button className="flex items-center space-x-2 px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg hover:bg-neutral-50 dark:hover:bg-neutral-800 transition-colors">
                    <RefreshCw className="w-4 h-4" />
                    <span>Refresh</span>
                  </button>
                </div>
              </div>
            </div>
          </Card>

          {/* Datasets Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredDatasets.map((dataset, index) => (
              <motion.div
                key={dataset.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                <Card className="h-full">
                  <div className="p-6">
                    <div className="flex items-start justify-between mb-4">
                      <div className="flex-1">
                        <h3 className="font-semibold text-neutral-900 dark:text-neutral-100 mb-2">
                          {dataset.name}
                        </h3>
                        <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-3">
                          {dataset.description}
                        </p>
                      </div>
                      
                      <div className="flex space-x-1">
                        <button className="text-neutral-600 hover:text-neutral-800">
                          <Eye className="w-4 h-4" />
                        </button>
                        <button className="text-neutral-600 hover:text-neutral-800">
                          <Edit className="w-4 h-4" />
                        </button>
                        <button className="text-red-600 hover:text-red-800">
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                    
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-neutral-500">Type</span>
                        <span className={`px-2 py-1 rounded text-xs font-medium ${getStatusColor(dataset.type)}`}>
                          {dataset.type}
                        </span>
                      </div>
                      
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-neutral-500">Status</span>
                        <span className={`px-2 py-1 rounded text-xs font-medium ${getStatusColor(dataset.status)}`}>
                          {dataset.status}
                        </span>
                      </div>
                      
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-neutral-500">Size</span>
                        <span className="text-sm font-medium text-neutral-900 dark:text-neutral-100">
                          {formatFileSize(dataset.size_mb)}
                        </span>
                      </div>
                      
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-neutral-500">Rows</span>
                        <span className="text-sm font-medium text-neutral-900 dark:text-neutral-100">
                          {dataset.rows.toLocaleString()}
                        </span>
                      </div>
                      
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-neutral-500">Quality</span>
                        <span className={`text-sm font-medium ${getQualityScoreColor(dataset.quality_score)}`}>
                          {(dataset.quality_score * 100).toFixed(1)}%
                        </span>
                      </div>
                      
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-neutral-500">Access</span>
                        <span className={`px-2 py-1 rounded text-xs font-medium ${getAccessLevelColor(dataset.access_level)}`}>
                          {dataset.access_level}
                        </span>
                      </div>
                    </div>
                    
                    {dataset.tags.length > 0 && (
                      <div className="mt-4">
                        <div className="flex flex-wrap gap-1">
                          {dataset.tags.slice(0, 3).map((tag) => (
                            <span
                              key={tag}
                              className="px-2 py-1 bg-neutral-100 text-neutral-700 dark:bg-neutral-800 dark:text-neutral-300 rounded text-xs"
                            >
                              {tag}
                            </span>
                          ))}
                          {dataset.tags.length > 3 && (
                            <span className="px-2 py-1 bg-neutral-100 text-neutral-700 dark:bg-neutral-800 dark:text-neutral-300 rounded text-xs">
                              +{dataset.tags.length - 3}
                            </span>
                          )}
                        </div>
                      </div>
                    )}
                    
                    <div className="mt-4 flex items-center justify-between">
                      <span className="text-xs text-neutral-500">
                        Updated {new Date(dataset.updated_at).toLocaleDateString()}
                      </span>
                      
                      <div className="flex space-x-2">
                        <button className="text-blue-600 hover:text-blue-800">
                          <Download className="w-4 h-4" />
                        </button>
                        <button className="text-blue-600 hover:text-blue-800">
                          <Share2 className="w-4 h-4" />
                        </button>
                        <button className="text-blue-600 hover:text-blue-800">
                          <ExternalLink className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  </div>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      )}

      {activeTab === 'quality' && (
        <div className="space-y-6">
          <Card>
            <div className="p-6">
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-6">
                Data Quality Dashboard
              </h3>
              
              <div className="space-y-6">
                {qualityMetrics.map((metrics, index) => {
                  const dataset = datasets.find(d => d.id === metrics.dataset_id);
                  if (!dataset) return null;
                  
                  return (
                    <motion.div
                      key={metrics.dataset_id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.1 }}
                      className="p-6 border border-neutral-200 dark:border-neutral-700 rounded-lg"
                    >
                      <div className="flex items-center justify-between mb-4">
                        <div>
                          <h4 className="font-semibold text-neutral-900 dark:text-neutral-100">
                            {dataset.name}
                          </h4>
                          <p className="text-sm text-neutral-600 dark:text-neutral-400">
                            Last checked: {new Date(metrics.last_checked).toLocaleString()}
                          </p>
                        </div>
                        
                        <div className="flex items-center space-x-2">
                          <span className={`text-lg font-bold ${getQualityScoreColor(dataset.quality_score)}`}>
                            {(dataset.quality_score * 100).toFixed(1)}%
                          </span>
                          <button className="text-blue-600 hover:text-blue-800">
                            <RefreshCw className="w-4 h-4" />
                          </button>
                        </div>
                      </div>
                      
                      <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-4">
                        {[
                          { label: 'Completeness', value: metrics.completeness },
                          { label: 'Uniqueness', value: metrics.uniqueness },
                          { label: 'Validity', value: metrics.validity },
                          { label: 'Consistency', value: metrics.consistency },
                          { label: 'Accuracy', value: metrics.accuracy },
                        ].map((metric) => (
                          <div key={metric.label} className="text-center">
                            <div className="text-sm text-neutral-500 mb-1">{metric.label}</div>
                            <div className={`text-lg font-semibold ${getQualityScoreColor(metric.value)}`}>
                              {(metric.value * 100).toFixed(1)}%
                            </div>
                            <div className="w-full bg-neutral-200 dark:bg-neutral-700 rounded-full h-2 mt-1">
                              <div
                                className={`h-2 rounded-full transition-all duration-500 ${
                                  metric.value >= 0.9 ? 'bg-green-500' :
                                  metric.value >= 0.8 ? 'bg-yellow-500' : 'bg-red-500'
                                }`}
                                style={{ width: `${metric.value * 100}%` }}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                      
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                        <div className="flex items-center justify-between">
                          <span className="text-neutral-500">Missing Values:</span>
                          <span className="font-medium text-neutral-900 dark:text-neutral-100">
                            {metrics.missing_values.toLocaleString()}
                          </span>
                        </div>
                        
                        <div className="flex items-center justify-between">
                          <span className="text-neutral-500">Duplicates:</span>
                          <span className="font-medium text-neutral-900 dark:text-neutral-100">
                            {metrics.duplicate_rows.toLocaleString()}
                          </span>
                        </div>
                        
                        <div className="flex items-center justify-between">
                          <span className="text-neutral-500">Outliers:</span>
                          <span className="font-medium text-neutral-900 dark:text-neutral-100">
                            {metrics.outliers.toLocaleString()}
                          </span>
                        </div>
                        
                        <div className="flex items-center justify-between">
                          <span className="text-neutral-500">Schema Violations:</span>
                          <span className={`font-medium ${metrics.schema_violations > 0 ? 'text-red-600' : 'text-green-600'}`}>
                            {metrics.schema_violations}
                          </span>
                        </div>
                      </div>
                    </motion.div>
                  );
                })}
              </div>
            </div>
          </Card>
        </div>
      )}

      {activeTab === 'pipelines' && (
        <div className="space-y-6">
          <Card>
            <div className="p-6">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                  Data Processing Pipelines
                </h3>
                <button className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors">
                  <Plus className="w-4 h-4" />
                  <span>Create Pipeline</span>
                </button>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-neutral-200 dark:border-neutral-700">
                      <th className="text-left py-3 px-4 font-medium text-neutral-900 dark:text-neutral-100">Pipeline</th>
                      <th className="text-left py-3 px-4 font-medium text-neutral-900 dark:text-neutral-100">Source</th>
                      <th className="text-left py-3 px-4 font-medium text-neutral-900 dark:text-neutral-100">Target</th>
                      <th className="text-left py-3 px-4 font-medium text-neutral-900 dark:text-neutral-100">Schedule</th>
                      <th className="text-left py-3 px-4 font-medium text-neutral-900 dark:text-neutral-100">Status</th>
                      <th className="text-left py-3 px-4 font-medium text-neutral-900 dark:text-neutral-100">Success Rate</th>
                      <th className="text-left py-3 px-4 font-medium text-neutral-900 dark:text-neutral-100">Last Run</th>
                      <th className="text-left py-3 px-4 font-medium text-neutral-900 dark:text-neutral-100">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {pipelines.map((pipeline, index) => (
                      <motion.tr
                        key={pipeline.id}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: index * 0.1 }}
                        className="border-b border-neutral-100 dark:border-neutral-800 hover:bg-neutral-50 dark:hover:bg-neutral-900/50"
                      >
                        <td className="py-3 px-4">
                          <div>
                            <span className="font-medium text-neutral-900 dark:text-neutral-100">
                              {pipeline.name}
                            </span>
                            <div className="text-xs text-neutral-500 mt-1">
                              {pipeline.transformation_steps.length} steps
                            </div>
                          </div>
                        </td>
                        <td className="py-3 px-4 text-neutral-600 dark:text-neutral-400">
                          {pipeline.source_dataset}
                        </td>
                        <td className="py-3 px-4 text-neutral-600 dark:text-neutral-400">
                          {pipeline.target_dataset}
                        </td>
                        <td className="py-3 px-4 text-neutral-600 dark:text-neutral-400">
                          {pipeline.schedule}
                        </td>
                        <td className="py-3 px-4">
                          <span className={`px-2 py-1 rounded text-xs font-medium ${getStatusColor(pipeline.status)}`}>
                            {pipeline.status}
                          </span>
                        </td>
                        <td className="py-3 px-4">
                          <span className={`font-medium ${getQualityScoreColor(pipeline.success_rate)}`}>
                            {(pipeline.success_rate * 100).toFixed(1)}%
                          </span>
                        </td>
                        <td className="py-3 px-4 text-neutral-600 dark:text-neutral-400">
                          {new Date(pipeline.last_run).toLocaleString()}
                        </td>
                        <td className="py-3 px-4">
                          <div className="flex items-center space-x-2">
                            <button className="text-blue-600 hover:text-blue-800">
                              <Edit className="w-4 h-4" />
                            </button>
                            <button className="text-green-600 hover:text-green-800">
                              <Activity className="w-4 h-4" />
                            </button>
                            <button className="text-red-600 hover:text-red-800">
                              <Trash2 className="w-4 h-4" />
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

      {activeTab === 'lineage' && (
        <div className="space-y-6">
          <Card>
            <div className="p-6">
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-6">
                Data Lineage Visualization
              </h3>
              
              {/* Mock Lineage Visualization */}
              <div className="h-96 bg-gradient-to-br from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg flex items-center justify-center border-2 border-dashed border-blue-200 dark:border-blue-700">
                <div className="text-center">
                  <Share2 className="w-16 h-16 text-blue-400 mx-auto mb-4" />
                  <h4 className="text-xl font-semibold text-blue-600 dark:text-blue-400 mb-2">
                    Interactive Data Lineage Graph
                  </h4>
                  <p className="text-blue-500 dark:text-blue-300 mb-4">
                    Visualize data flow and dependencies across your ML pipeline
                  </p>
                  <div className="flex items-center justify-center space-x-4 text-sm text-blue-500">
                    <span>• Source tracking</span>
                    <span>• Impact analysis</span>
                    <span>• Dependency mapping</span>
                  </div>
                </div>
              </div>
              
              {/* Lineage Summary */}
              <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
                {datasets.filter(d => d.lineage).map((dataset) => (
                  <div
                    key={dataset.id}
                    className="p-4 border border-neutral-200 dark:border-neutral-700 rounded-lg"
                  >
                    <h4 className="font-medium text-neutral-900 dark:text-neutral-100 mb-2">
                      {dataset.name}
                    </h4>
                    
                    {dataset.lineage?.source_datasets && dataset.lineage.source_datasets.length > 0 && (
                      <div className="mb-3">
                        <span className="text-xs text-neutral-500 font-medium">Sources:</span>
                        <div className="flex flex-wrap gap-1 mt-1">
                          {dataset.lineage.source_datasets.map((source) => (
                            <span
                              key={source}
                              className="px-2 py-1 bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300 rounded text-xs"
                            >
                              {source}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                    
                    {dataset.lineage?.derived_datasets && dataset.lineage.derived_datasets.length > 0 && (
                      <div>
                        <span className="text-xs text-neutral-500 font-medium">Derived:</span>
                        <div className="flex flex-wrap gap-1 mt-1">
                          {dataset.lineage.derived_datasets.map((derived) => (
                            <span
                              key={derived}
                              className="px-2 py-1 bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300 rounded text-xs"
                            >
                              {derived}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
};

export default DataManagement;