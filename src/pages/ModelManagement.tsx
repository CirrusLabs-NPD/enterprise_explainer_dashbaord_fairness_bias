import React, { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import {
  Upload,
  Download,
  Archive,
  Package,
  GitBranch,
  Settings,
  PlayCircle,
  PauseCircle,
  StopCircle,
  RefreshCw,
  Search,
  Filter,
  MoreVertical,
  Edit3,
  Trash2,
  Copy,
  Eye,
  FileText,
  Database,
  Cloud,
  Server,
  Code,
  Zap,
  Target,
  AlertTriangle,
  CheckCircle,
  Clock,
  Users,
  Tag,
  FolderOpen,
  Link,
  Star,
  TrendingUp,
  Activity,
  BarChart3,
  Monitor,
  Shield,
  Globe,
  Lock,
  Unlock
} from 'lucide-react';
import Card from '../components/common/Card';
import MetricCard from '../components/common/MetricCard';
import { apiClient } from '../services/api';

interface ModelVersion {
  version: string;
  description: string;
  createdAt: string;
  createdBy: string;
  status: 'active' | 'deprecated' | 'testing';
  metrics: Record<string, number>;
  size: string;
  format: string;
  checksum: string;
}

interface ModelMetadata {
  id: string;
  name: string;
  description: string;
  type: 'classification' | 'regression' | 'clustering' | 'nlp' | 'computer_vision';
  framework: 'sklearn' | 'tensorflow' | 'pytorch' | 'xgboost' | 'onnx' | 'custom';
  currentVersion: string;
  versions: ModelVersion[];
  status: 'active' | 'inactive' | 'training' | 'deploying' | 'deprecated';
  deployment: {
    status: 'deployed' | 'not_deployed' | 'deploying' | 'failed';
    environment: 'development' | 'staging' | 'production';
    endpoint?: string;
    instances: number;
    lastDeployment: string;
  };
  performance: {
    accuracy?: number;
    precision?: number;
    recall?: number;
    f1Score?: number;
    rmse?: number;
    mae?: number;
    rSquared?: number;
  };
  usage: {
    totalPredictions: number;
    dailyPredictions: number;
    avgLatency: number;
    errorRate: number;
  };
  features: {
    inputFeatures: string[];
    outputShape: string;
    featureImportance?: Record<string, number>;
  };
  tags: string[];
  owner: string;
  team: string;
  createdAt: string;
  updatedAt: string;
  lastTrainedAt: string;
  modelCard?: {
    intendedUse: string;
    limitations: string;
    tradeOffs: string;
    ethicalConsiderations: string;
  };
  artifacts: {
    modelFile: string;
    configFile?: string;
    requirementsFile?: string;
    documentationFile?: string;
  };
}

interface ModelTemplate {
  id: string;
  name: string;
  description: string;
  type: ModelMetadata['type'];
  framework: ModelMetadata['framework'];
  icon: React.ReactNode;
  template: Partial<ModelMetadata>;
}

const ModelManagement: React.FC = () => {
  const [models, setModels] = useState<ModelMetadata[]>([]);
  const [selectedModel, setSelectedModel] = useState<ModelMetadata | null>(null);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState<string>('all');
  const [filterStatus, setFilterStatus] = useState<string>('all');
  const [sortBy, setSortBy] = useState<'name' | 'created' | 'updated' | 'performance'>('updated');
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [activeTab, setActiveTab] = useState<'overview' | 'versions' | 'deployment' | 'performance' | 'settings'>('overview');
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [showVersionModal, setShowVersionModal] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const modelTemplates: ModelTemplate[] = [
    {
      id: 'sklearn-classifier',
      name: 'Scikit-learn Classifier',
      description: 'Standard classification model using scikit-learn',
      type: 'classification',
      framework: 'sklearn',
      icon: <Target className="w-6 h-6" />,
      template: {
        type: 'classification',
        framework: 'sklearn',
        features: { inputFeatures: [], outputShape: 'binary' }
      }
    },
    {
      id: 'tensorflow-neural-net',
      name: 'TensorFlow Neural Network',
      description: 'Deep learning model using TensorFlow/Keras',
      type: 'classification',
      framework: 'tensorflow',
      icon: <Zap className="w-6 h-6" />,
      template: {
        type: 'classification',
        framework: 'tensorflow',
        features: { inputFeatures: [], outputShape: 'multi-class' }
      }
    },
    {
      id: 'onnx-model',
      name: 'ONNX Model',
      description: 'Cross-platform ONNX format model',
      type: 'classification',
      framework: 'onnx',
      icon: <Package className="w-6 h-6" />,
      template: {
        type: 'classification',
        framework: 'onnx',
        features: { inputFeatures: [], outputShape: 'tensor' }
      }
    }
  ];

  // Mock data
  const mockModels: ModelMetadata[] = [
    {
      id: 'model_001',
      name: 'Credit Risk Classifier v2.3',
      description: 'Advanced credit risk assessment model with improved accuracy',
      type: 'classification',
      framework: 'sklearn',
      currentVersion: 'v2.3.1',
      versions: [
        {
          version: 'v2.3.1',
          description: 'Bug fixes and performance improvements',
          createdAt: '2024-01-20T14:30:00Z',
          createdBy: 'john.doe',
          status: 'active',
          metrics: { accuracy: 0.892, precision: 0.876, recall: 0.903 },
          size: '45.2 MB',
          format: 'pkl',
          checksum: 'sha256:abc123...'
        },
        {
          version: 'v2.3.0',
          description: 'Feature engineering improvements',
          createdAt: '2024-01-18T10:00:00Z',
          createdBy: 'jane.smith',
          status: 'deprecated',
          metrics: { accuracy: 0.885, precision: 0.870, recall: 0.898 },
          size: '43.8 MB',
          format: 'pkl',
          checksum: 'sha256:def456...'
        }
      ],
      status: 'active',
      deployment: {
        status: 'deployed',
        environment: 'production',
        endpoint: 'https://api.ml.company.com/v1/credit-risk',
        instances: 3,
        lastDeployment: '2024-01-20T15:00:00Z'
      },
      performance: {
        accuracy: 0.892,
        precision: 0.876,
        recall: 0.903,
        f1Score: 0.889
      },
      usage: {
        totalPredictions: 1234567,
        dailyPredictions: 8945,
        avgLatency: 45,
        errorRate: 0.02
      },
      features: {
        inputFeatures: ['income', 'age', 'credit_score', 'employment_length', 'debt_ratio'],
        outputShape: 'binary',
        featureImportance: {
          'credit_score': 0.35,
          'income': 0.28,
          'debt_ratio': 0.22,
          'age': 0.10,
          'employment_length': 0.05
        }
      },
      tags: ['credit', 'risk', 'production', 'high-performance'],
      owner: 'John Doe',
      team: 'Risk Analytics',
      createdAt: '2024-01-15T10:00:00Z',
      updatedAt: '2024-01-20T14:30:00Z',
      lastTrainedAt: '2024-01-18T08:00:00Z',
      modelCard: {
        intendedUse: 'Credit risk assessment for loan applications',
        limitations: 'Not suitable for high-value commercial loans above $1M',
        tradeOffs: 'Optimized for recall over precision to minimize false negatives',
        ethicalConsiderations: 'Regular bias monitoring for protected characteristics'
      },
      artifacts: {
        modelFile: 'credit_risk_v2.3.1.pkl',
        configFile: 'config.json',
        requirementsFile: 'requirements.txt',
        documentationFile: 'model_documentation.md'
      }
    },
    {
      id: 'model_002',
      name: 'Fraud Detection Neural Network',
      description: 'Real-time fraud detection using deep learning',
      type: 'classification',
      framework: 'tensorflow',
      currentVersion: 'v1.2.0',
      versions: [
        {
          version: 'v1.2.0',
          description: 'Added real-time feature engineering pipeline',
          createdAt: '2024-01-19T16:20:00Z',
          createdBy: 'alex.chen',
          status: 'testing',
          metrics: { accuracy: 0.943, precision: 0.921, recall: 0.965 },
          size: '128.5 MB',
          format: 'h5',
          checksum: 'sha256:ghi789...'
        }
      ],
      status: 'training',
      deployment: {
        status: 'not_deployed',
        environment: 'staging',
        instances: 0,
        lastDeployment: '2024-01-15T12:00:00Z'
      },
      performance: {
        accuracy: 0.943,
        precision: 0.921,
        recall: 0.965,
        f1Score: 0.942
      },
      usage: {
        totalPredictions: 0,
        dailyPredictions: 0,
        avgLatency: 0,
        errorRate: 0
      },
      features: {
        inputFeatures: ['transaction_amount', 'merchant_category', 'time_features', 'user_behavior'],
        outputShape: 'binary'
      },
      tags: ['fraud', 'neural-network', 'real-time', 'experimental'],
      owner: 'Alex Chen',
      team: 'Security AI',
      createdAt: '2024-01-10T14:30:00Z',
      updatedAt: '2024-01-19T16:20:00Z',
      lastTrainedAt: '2024-01-19T12:00:00Z',
      artifacts: {
        modelFile: 'fraud_detection_v1.2.0.h5',
        configFile: 'model_config.json',
        requirementsFile: 'requirements.txt'
      }
    }
  ];

  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    try {
      setLoading(true);
      
      // Try to fetch from API, fallback to mock data
      const response = await apiClient.listModels();
      
      if (response.success && response.data) {
        // Transform API data to match our interface
        setModels(response.data.map(transformApiModel));
      } else {
        // Use mock data as fallback
        setTimeout(() => {
          setModels(mockModels);
          setSelectedModel(mockModels[0]);
          setLoading(false);
        }, 1000);
        return;
      }
    } catch (err) {
      console.warn('API call failed, using mock data:', err);
      // Use mock data as fallback
      setTimeout(() => {
        setModels(mockModels);
        setSelectedModel(mockModels[0]);
        setLoading(false);
      }, 1000);
      return;
    }
    
    setLoading(false);
  };

  const transformApiModel = (apiModel: any): ModelMetadata => {
    return {
      id: apiModel.id,
      name: apiModel.name,
      description: apiModel.description || '',
      type: apiModel.model_type,
      framework: apiModel.framework,
      currentVersion: apiModel.version,
      versions: [{
        version: apiModel.version,
        description: 'Current version',
        createdAt: apiModel.created_at,
        createdBy: apiModel.created_by || 'unknown',
        status: 'active',
        metrics: apiModel.validation_metrics || {},
        size: 'Unknown',
        format: 'pkl',
        checksum: 'Unknown'
      }],
      status: apiModel.status,
      deployment: {
        status: 'not_deployed',
        environment: 'development',
        instances: 0,
        lastDeployment: apiModel.created_at
      },
      performance: apiModel.validation_metrics || {},
      usage: {
        totalPredictions: 0,
        dailyPredictions: 0,
        avgLatency: 0,
        errorRate: 0
      },
      features: {
        inputFeatures: apiModel.feature_names || [],
        outputShape: apiModel.target_names?.length > 2 ? 'multi-class' : 'binary'
      },
      tags: [],
      owner: apiModel.created_by || 'Unknown',
      team: 'Unknown',
      createdAt: apiModel.created_at,
      updatedAt: apiModel.updated_at || apiModel.created_at,
      lastTrainedAt: apiModel.created_at,
      artifacts: {
        modelFile: 'model.pkl'
      }
    };
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      setUploading(true);
      
      const formData = new FormData();
      formData.append('model_file', file);
      formData.append('model_name', 'Uploaded Model');
      formData.append('model_type', 'classification');
      
      const response = await apiClient.uploadModel(file, {
        model_name: 'Uploaded Model',
        model_type: 'classification'
      });
      
      if (response.success) {
        await loadModels();
        setShowUploadModal(false);
      }
    } catch (err) {
      console.error('Upload failed:', err);
    } finally {
      setUploading(false);
    }
  };

  const deployModel = async (modelId: string, environment: string) => {
    try {
      // API call would go here
      console.log(`Deploying model ${modelId} to ${environment}`);
      
      // Update local state
      setModels(prev => prev.map(model => 
        model.id === modelId 
          ? {
              ...model,
              deployment: {
                ...model.deployment,
                status: 'deploying'
              }
            }
          : model
      ));
      
      // Simulate deployment completion
      setTimeout(() => {
        setModels(prev => prev.map(model => 
          model.id === modelId 
            ? {
                ...model,
                deployment: {
                  ...model.deployment,
                  status: 'deployed',
                  environment: environment as any,
                  lastDeployment: new Date().toISOString()
                }
              }
            : model
        ));
      }, 3000);
    } catch (err) {
      console.error('Deployment failed:', err);
    }
  };

  const filteredModels = models.filter(model => {
    const matchesSearch = model.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         model.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         model.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()));
    
    const matchesType = filterType === 'all' || model.type === filterType;
    const matchesStatus = filterStatus === 'all' || model.status === filterStatus;
    
    return matchesSearch && matchesType && matchesStatus;
  });

  const sortedModels = [...filteredModels].sort((a, b) => {
    switch (sortBy) {
      case 'name':
        return a.name.localeCompare(b.name);
      case 'created':
        return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();
      case 'updated':
        return new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime();
      case 'performance':
        const perfA = a.performance.accuracy || a.performance.f1Score || 0;
        const perfB = b.performance.accuracy || b.performance.f1Score || 0;
        return perfB - perfA;
      default:
        return 0;
    }
  });

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300';
      case 'inactive': return 'bg-neutral-100 text-neutral-800 dark:bg-neutral-800 dark:text-neutral-300';
      case 'training': return 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300';
      case 'deploying': return 'bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300';
      case 'deprecated': return 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300';
      default: return 'bg-neutral-100 text-neutral-800 dark:bg-neutral-800 dark:text-neutral-300';
    }
  };

  const getDeploymentStatusColor = (status: string) => {
    switch (status) {
      case 'deployed': return 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300';
      case 'deploying': return 'bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300';
      case 'failed': return 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300';
      case 'not_deployed': return 'bg-neutral-100 text-neutral-800 dark:bg-neutral-800 dark:text-neutral-300';
      default: return 'bg-neutral-100 text-neutral-800 dark:bg-neutral-800 dark:text-neutral-300';
    }
  };

  const getFrameworkIcon = (framework: string) => {
    switch (framework) {
      case 'sklearn': return <Target className="w-5 h-5" />;
      case 'tensorflow': return <Zap className="w-5 h-5" />;
      case 'pytorch': return <Code className="w-5 h-5" />;
      case 'xgboost': return <TrendingUp className="w-5 h-5" />;
      case 'onnx': return <Package className="w-5 h-5" />;
      default: return <Database className="w-5 h-5" />;
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <RefreshCw className="w-8 h-8 animate-spin text-primary-600" />
        <span className="ml-3 text-lg text-neutral-600 dark:text-neutral-400">
          Loading models...
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
            Model Management
          </h1>
          <p className="text-neutral-600 dark:text-neutral-400 mt-1">
            Upload, version, deploy, and manage ML models across environments
          </p>
        </div>
        
        <div className="flex items-center space-x-4">
          <button
            onClick={() => setShowUploadModal(true)}
            className="flex items-center space-x-2 px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors"
          >
            <Upload className="w-4 h-4" />
            <span>Upload Model</span>
          </button>
          
          <button
            onClick={loadModels}
            className="flex items-center space-x-2 px-4 py-2 bg-neutral-600 hover:bg-neutral-700 text-white rounded-lg transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Total Models"
          value={models.length}
          format="number"
          icon={<Database className="w-5 h-5" />}
          change="+3"
          changeType="positive"
        />
        
        <MetricCard
          title="Active Models"
          value={models.filter(m => m.status === 'active').length}
          format="number"
          icon={<PlayCircle className="w-5 h-5" />}
          change="+2"
          changeType="positive"
        />
        
        <MetricCard
          title="Deployed Models"
          value={models.filter(m => m.deployment.status === 'deployed').length}
          format="number"
          icon={<Cloud className="w-5 h-5" />}
          change="+1"
          changeType="positive"
        />
        
        <MetricCard
          title="Avg Performance"
          value={models.reduce((sum, m) => sum + (m.performance.accuracy || m.performance.f1Score || 0), 0) / models.length || 0}
          format="percentage"
          icon={<TrendingUp className="w-5 h-5" />}
          change="+2.3%"
          changeType="positive"
        />
      </div>

      {/* Filters and Search */}
      <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between">
        <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center">
          <div className="relative">
            <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-neutral-400" />
            <input
              type="text"
              placeholder="Search models..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10 pr-4 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
            />
          </div>
          
          <select
            value={filterType}
            onChange={(e) => setFilterType(e.target.value)}
            className="px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
          >
            <option value="all">All Types</option>
            <option value="classification">Classification</option>
            <option value="regression">Regression</option>
            <option value="clustering">Clustering</option>
            <option value="nlp">NLP</option>
            <option value="computer_vision">Computer Vision</option>
          </select>
          
          <select
            value={filterStatus}
            onChange={(e) => setFilterStatus(e.target.value)}
            className="px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
          >
            <option value="all">All Statuses</option>
            <option value="active">Active</option>
            <option value="inactive">Inactive</option>
            <option value="training">Training</option>
            <option value="deprecated">Deprecated</option>
          </select>
        </div>
        
        <div className="flex items-center space-x-2">
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as any)}
            className="px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
          >
            <option value="updated">Recently Updated</option>
            <option value="created">Recently Created</option>
            <option value="name">Name</option>
            <option value="performance">Performance</option>
          </select>
          
          <div className="flex border border-neutral-300 dark:border-neutral-600 rounded-lg">
            <button
              onClick={() => setViewMode('grid')}
              className={`p-2 ${viewMode === 'grid' ? 'bg-primary-100 text-primary-600' : 'text-neutral-500'}`}
            >
              <BarChart3 className="w-4 h-4" />
            </button>
            <button
              onClick={() => setViewMode('list')}
              className={`p-2 ${viewMode === 'list' ? 'bg-primary-100 text-primary-600' : 'text-neutral-500'}`}
            >
              <Archive className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Models Grid/List */}
      {viewMode === 'grid' ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
          {sortedModels.map((model) => (
            <Card
              key={model.id}
              className="cursor-pointer hover:shadow-lg transition-shadow"
              onClick={() => setSelectedModel(model)}
            >
              <div className="p-6">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    <div className="p-2 bg-neutral-100 dark:bg-neutral-800 rounded-lg">
                      {getFrameworkIcon(model.framework)}
                    </div>
                    <div>
                      <h3 className="font-semibold text-neutral-900 dark:text-neutral-100 truncate">
                        {model.name}
                      </h3>
                      <p className="text-sm text-neutral-600 dark:text-neutral-400">
                        {model.framework} • {model.type}
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    <span className={`px-2 py-1 rounded text-xs font-medium ${getStatusColor(model.status)}`}>
                      {model.status}
                    </span>
                  </div>
                </div>
                
                <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-4 line-clamp-2">
                  {model.description}
                </p>
                
                <div className="space-y-3">
                  <div className="flex justify-between text-sm">
                    <span className="text-neutral-500">Version:</span>
                    <span className="font-medium">{model.currentVersion}</span>
                  </div>
                  
                  <div className="flex justify-between text-sm">
                    <span className="text-neutral-500">Performance:</span>
                    <span className="font-medium">
                      {((model.performance.accuracy || model.performance.f1Score || 0) * 100).toFixed(1)}%
                    </span>
                  </div>
                  
                  <div className="flex justify-between text-sm">
                    <span className="text-neutral-500">Deployment:</span>
                    <span className={`px-2 py-1 rounded text-xs font-medium ${getDeploymentStatusColor(model.deployment.status)}`}>
                      {model.deployment.status.replace('_', ' ')}
                    </span>
                  </div>
                  
                  <div className="flex justify-between text-sm">
                    <span className="text-neutral-500">Predictions:</span>
                    <span className="font-medium">{model.usage.totalPredictions.toLocaleString()}</span>
                  </div>
                </div>
                
                {model.tags.length > 0 && (
                  <div className="flex flex-wrap gap-1 mt-4">
                    {model.tags.slice(0, 3).map((tag) => (
                      <span
                        key={tag}
                        className="px-2 py-1 bg-neutral-100 dark:bg-neutral-800 text-neutral-600 dark:text-neutral-400 rounded text-xs"
                      >
                        {tag}
                      </span>
                    ))}
                    {model.tags.length > 3 && (
                      <span className="px-2 py-1 bg-neutral-100 dark:bg-neutral-800 text-neutral-600 dark:text-neutral-400 rounded text-xs">
                        +{model.tags.length - 3}
                      </span>
                    )}
                  </div>
                )}
                
                <div className="flex items-center justify-between mt-4 pt-4 border-t border-neutral-200 dark:border-neutral-700">
                  <span className="text-xs text-neutral-500">
                    Updated {new Date(model.updatedAt).toLocaleDateString()}
                  </span>
                  
                  <div className="flex items-center space-x-1">
                    {model.deployment.status !== 'deployed' && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          deployModel(model.id, 'staging');
                        }}
                        className="p-1 text-neutral-500 hover:text-green-600 transition-colors"
                        title="Deploy"
                      >
                        <PlayCircle className="w-4 h-4" />
                      </button>
                    )}
                    
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setSelectedModel(model);
                        setShowVersionModal(true);
                      }}
                      className="p-1 text-neutral-500 hover:text-blue-600 transition-colors"
                      title="Versions"
                    >
                      <GitBranch className="w-4 h-4" />
                    </button>
                    
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        // Handle settings
                      }}
                      className="p-1 text-neutral-500 hover:text-neutral-700 transition-colors"
                      title="Settings"
                    >
                      <Settings className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </div>
            </Card>
          ))}
        </div>
      ) : (
        <Card>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-neutral-50 dark:bg-neutral-800">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">
                    Model
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">
                    Type
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">
                    Version
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">
                    Performance
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">
                    Status
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">
                    Deployment
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white dark:bg-neutral-900 divide-y divide-neutral-200 dark:divide-neutral-700">
                {sortedModels.map((model) => (
                  <tr
                    key={model.id}
                    className="hover:bg-neutral-50 dark:hover:bg-neutral-800 cursor-pointer"
                    onClick={() => setSelectedModel(model)}
                  >
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center space-x-3">
                        <div className="p-2 bg-neutral-100 dark:bg-neutral-800 rounded-lg">
                          {getFrameworkIcon(model.framework)}
                        </div>
                        <div>
                          <div className="text-sm font-medium text-neutral-900 dark:text-neutral-100">
                            {model.name}
                          </div>
                          <div className="text-sm text-neutral-500">
                            {model.framework}
                          </div>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className="text-sm text-neutral-900 dark:text-neutral-100 capitalize">
                        {model.type}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className="text-sm text-neutral-900 dark:text-neutral-100">
                        {model.currentVersion}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className="text-sm text-neutral-900 dark:text-neutral-100">
                        {((model.performance.accuracy || model.performance.f1Score || 0) * 100).toFixed(1)}%
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`px-2 py-1 rounded text-xs font-medium ${getStatusColor(model.status)}`}>
                        {model.status}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`px-2 py-1 rounded text-xs font-medium ${getDeploymentStatusColor(model.deployment.status)}`}>
                        {model.deployment.status.replace('_', ' ')}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                      <div className="flex items-center space-x-2">
                        {model.deployment.status !== 'deployed' && (
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              deployModel(model.id, 'staging');
                            }}
                            className="text-green-600 hover:text-green-900"
                          >
                            <PlayCircle className="w-4 h-4" />
                          </button>
                        )}
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setSelectedModel(model);
                            setShowVersionModal(true);
                          }}
                          className="text-blue-600 hover:text-blue-900"
                        >
                          <GitBranch className="w-4 h-4" />
                        </button>
                        <button className="text-neutral-400 hover:text-neutral-600">
                          <MoreVertical className="w-4 h-4" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* Model Details Modal */}
      {selectedModel && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-neutral-800 rounded-lg shadow-xl max-w-6xl w-full mx-4 max-h-[90vh] overflow-y-auto">
            <div className="p-6 border-b border-neutral-200 dark:border-neutral-700">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-xl font-semibold text-neutral-900 dark:text-neutral-100">
                    {selectedModel.name}
                  </h3>
                  <p className="text-neutral-600 dark:text-neutral-400">
                    {selectedModel.description}
                  </p>
                </div>
                <button
                  onClick={() => setSelectedModel(null)}
                  className="text-neutral-500 hover:text-neutral-700 dark:hover:text-neutral-300"
                >
                  ×
                </button>
              </div>
              
              {/* Tab Navigation */}
              <div className="mt-6 border-b border-neutral-200 dark:border-neutral-700">
                <nav className="flex space-x-8">
                  {[
                    { id: 'overview', label: 'Overview', icon: Eye },
                    { id: 'versions', label: 'Versions', icon: GitBranch },
                    { id: 'deployment', label: 'Deployment', icon: Cloud },
                    { id: 'performance', label: 'Performance', icon: BarChart3 },
                    { id: 'settings', label: 'Settings', icon: Settings }
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
            </div>
            
            <div className="p-6">
              {/* Tab content would go here */}
              <div className="text-center py-8 text-neutral-500 dark:text-neutral-400">
                <Monitor className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p>Model details panel coming soon</p>
                <p className="text-sm">View versions, deployment status, and performance metrics</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Upload Modal */}
      {showUploadModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-neutral-800 rounded-lg shadow-xl max-w-2xl w-full mx-4">
            <div className="p-6 border-b border-neutral-200 dark:border-neutral-700">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                  Upload New Model
                </h3>
                <button
                  onClick={() => setShowUploadModal(false)}
                  className="text-neutral-500 hover:text-neutral-700 dark:hover:text-neutral-300"
                >
                  ×
                </button>
              </div>
            </div>
            
            <div className="p-6">
              <div className="space-y-6">
                {/* Model Templates */}
                <div>
                  <h4 className="font-medium text-neutral-900 dark:text-neutral-100 mb-4">
                    Choose Model Template
                  </h4>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {modelTemplates.map((template) => (
                      <div
                        key={template.id}
                        className="p-4 border border-neutral-200 dark:border-neutral-700 rounded-lg hover:border-primary-300 dark:hover:border-primary-600 cursor-pointer transition-colors"
                      >
                        <div className="flex items-center space-x-3 mb-2">
                          <div className="p-2 bg-primary-100 dark:bg-primary-900/20 rounded-lg">
                            {template.icon}
                          </div>
                          <div>
                            <h5 className="font-medium text-neutral-900 dark:text-neutral-100">
                              {template.name}
                            </h5>
                            <p className="text-xs text-neutral-600 dark:text-neutral-400">
                              {template.framework}
                            </p>
                          </div>
                        </div>
                        <p className="text-sm text-neutral-600 dark:text-neutral-400">
                          {template.description}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
                
                {/* File Upload */}
                <div>
                  <h4 className="font-medium text-neutral-900 dark:text-neutral-100 mb-4">
                    Upload Model File
                  </h4>
                  <div
                    className="border-2 border-dashed border-neutral-300 dark:border-neutral-600 rounded-lg p-8 text-center cursor-pointer hover:border-primary-400"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <Upload className="w-8 h-8 text-neutral-400 mx-auto mb-4" />
                    <p className="text-neutral-600 dark:text-neutral-400 mb-2">
                      Click to upload or drag and drop
                    </p>
                    <p className="text-sm text-neutral-500">
                      Supports .pkl, .joblib, .h5, .onnx, .pt files
                    </p>
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept=".pkl,.joblib,.h5,.onnx,.pt"
                      onChange={handleFileUpload}
                      className="hidden"
                    />
                  </div>
                </div>
                
                <div className="flex justify-end space-x-3">
                  <button
                    onClick={() => setShowUploadModal(false)}
                    className="px-4 py-2 text-neutral-600 hover:text-neutral-800 dark:text-neutral-400 dark:hover:text-neutral-200"
                  >
                    Cancel
                  </button>
                  <button
                    disabled={uploading}
                    className="px-4 py-2 bg-primary-600 hover:bg-primary-700 disabled:bg-primary-400 text-white rounded-lg transition-colors"
                  >
                    {uploading ? 'Uploading...' : 'Upload Model'}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelManagement;