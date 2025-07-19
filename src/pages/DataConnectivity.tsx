import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Upload, 
  Database, 
  FileText, 
  CheckCircle2, 
  AlertCircle, 
  Loader2,
  Eye,
  Trash2,
  Search,
  Activity,
  AlertTriangle,
  Info,
  Link,
  Server,
  Wifi
} from 'lucide-react';
import Card from '../components/common/Card';
import Button from '../components/common/Button';
import { apiClient, DatasetMetadata, DataQualityReport } from '../services/api';
import { webSocketManager } from '../services/websocket';

interface DataSource {
  id: string;
  name: string;
  type: 'file' | 'database' | 'stream' | 'api';
  status: 'connected' | 'disconnected' | 'error' | 'connecting';
  lastSync: string;
  recordCount: number;
  icon: React.ReactNode;
  description: string;
}

interface StreamConnection {
  id: string;
  name: string;
  type: 'kafka' | 'kinesis' | 'pubsub' | 'webhook';
  status: 'active' | 'inactive' | 'error';
  messagesProcessed: number;
  lastMessage: string;
  config: Record<string, any>;
}

const DataConnectivity: React.FC = () => {
  const [datasets, setDatasets] = useState<DatasetMetadata[]>([]);
  const [dataSources, setDataSources] = useState<DataSource[]>([]);
  const [streamConnections, setStreamConnections] = useState<StreamConnection[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploadProgress, setUploadProgress] = useState<Record<string, number>>({});
  // Log for debugging
  console.log('Upload progress state:', uploadProgress, setUploadProgress);
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null);
  const [qualityReport, setQualityReport] = useState<DataQualityReport | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState<'all' | 'csv' | 'json' | 'parquet'>('all');
  const [isUploading, setIsUploading] = useState(false);
  const [connectionStats, setConnectionStats] = useState({
    totalConnections: 0,
    activeConnections: 0,
    totalRecords: 0,
    realTimeStreams: 0
  });

  // Load initial data
  useEffect(() => {
    loadData();
    setupWebSocket();
    
    // Setup periodic refresh
    const interval = setInterval(loadData, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const setupWebSocket = useCallback(() => {
    // Connect to WebSocket for real-time updates
    webSocketManager.connect();
    
    // Listen for data updates
    webSocketManager.on('system_notification', (notification) => {
      if (notification.message.includes('dataset') || notification.message.includes('data')) {
        loadData();
      }
    });
    
    // Listen for connection stats
    webSocketManager.on('stats', (stats) => {
      setConnectionStats(prev => ({
        ...prev,
        totalConnections: stats.active_connections || 0,
        activeConnections: stats.active_connections || 0
      }));
    });
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      
      // Load datasets
      const datasetsResponse = await apiClient.listDatasets();
      if (datasetsResponse.success && datasetsResponse.data) {
        setDatasets(datasetsResponse.data);
        
        // Update connection stats
        const totalRecords = datasetsResponse.data.reduce((sum, dataset) => sum + dataset.num_rows, 0);
        setConnectionStats(prev => ({
          ...prev,
          totalRecords
        }));
      }
      
      // Load mock data sources (replace with real API calls)
      setDataSources([
        {
          id: 'postgres-1',
          name: 'Production Database',
          type: 'database',
          status: 'connected',
          lastSync: '2024-01-15T10:30:00Z',
          recordCount: 1250000,
          icon: <Database className="w-5 h-5" />,
          description: 'Main production PostgreSQL database'
        },
        {
          id: 'api-1',
          name: 'Customer API',
          type: 'api',
          status: 'connected',
          lastSync: '2024-01-15T10:25:00Z',
          recordCount: 45000,
          icon: <Link className="w-5 h-5" />,
          description: 'REST API for customer data'
        },
        {
          id: 'stream-1',
          name: 'Event Stream',
          type: 'stream',
          status: 'connected',
          lastSync: '2024-01-15T10:35:00Z',
          recordCount: 980000,
          icon: <Activity className="w-5 h-5" />,
          description: 'Real-time event streaming pipeline'
        }
      ]);
      
      // Load mock stream connections
      setStreamConnections([
        {
          id: 'kafka-1',
          name: 'User Events',
          type: 'kafka',
          status: 'active',
          messagesProcessed: 15420,
          lastMessage: '2024-01-15T10:34:30Z',
          config: { topic: 'user-events', partition: 3 }
        },
        {
          id: 'kinesis-1',
          name: 'Transaction Stream',
          type: 'kinesis',
          status: 'active',
          messagesProcessed: 8930,
          lastMessage: '2024-01-15T10:34:45Z',
          config: { stream: 'transactions', shard: 'shard-001' }
        }
      ]);
      
    } catch (error) {
      console.error('Error loading data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    setIsUploading(true);
    
    for (const file of files) {
      try {
        const response = await apiClient.uploadDataset(file, {
          dataset_name: file.name.replace(/\.[^/.]+$/, ""),
          description: `Uploaded dataset from ${file.name}`
        });
        
        if (response.success) {
          console.log('Dataset uploaded successfully:', response.data);
          await loadData(); // Refresh the list
        } else {
          console.error('Upload failed:', response.error);
        }
      } catch (error) {
        console.error('Upload error:', error);
      }
    }
    
    setIsUploading(false);
  };

  const handleDeleteDataset = async (datasetId: string) => {
    try {
      const response = await apiClient.deleteDataset(datasetId);
      if (response.success) {
        setDatasets(datasets.filter(d => d.dataset_id !== datasetId));
        if (selectedDataset === datasetId) {
          setSelectedDataset(null);
          setQualityReport(null);
        }
      }
    } catch (error) {
      console.error('Delete error:', error);
    }
  };

  const handleViewDataset = async (datasetId: string) => {
    try {
      setSelectedDataset(datasetId);
      
      // Load quality report
      const qualityResponse = await apiClient.assessDataQuality(datasetId);
      if (qualityResponse.success && qualityResponse.data) {
        setQualityReport(qualityResponse.data);
      }
    } catch (error) {
      console.error('Error loading dataset details:', error);
    }
  };

  const filteredDatasets = datasets.filter(dataset => {
    const matchesSearch = dataset.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         dataset.description.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesType = filterType === 'all' || dataset.format === filterType;
    return matchesSearch && matchesType;
  });

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'connected':
      case 'active':
        return <CheckCircle2 className="w-4 h-4 text-green-500" />;
      case 'disconnected':
      case 'inactive':
        return <AlertCircle className="w-4 h-4 text-red-500" />;
      case 'error':
        return <AlertTriangle className="w-4 h-4 text-red-500" />;
      case 'connecting':
        return <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />;
      default:
        return <Info className="w-4 h-4 text-gray-500" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'connected':
      case 'active':
        return 'bg-green-100 text-green-800 border-green-200';
      case 'disconnected':
      case 'inactive':
        return 'bg-red-100 text-red-800 border-red-200';
      case 'error':
        return 'bg-red-100 text-red-800 border-red-200';
      case 'connecting':
        return 'bg-blue-100 text-blue-800 border-blue-200';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatNumber = (num: number) => {
    return new Intl.NumberFormat().format(num);
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Data Connectivity</h1>
          <p className="text-gray-600 mt-1">Manage datasets, connections, and streaming data sources</p>
        </div>
        <div className="flex items-center space-x-3">
          <Button
            onClick={() => webSocketManager.getStats()}
            className="flex items-center space-x-2"
          >
            <Activity className="w-4 h-4" />
            <span>Refresh Stats</span>
          </Button>
          <label className="relative">
            <input
              type="file"
              multiple
              accept=".csv,.json,.parquet"
              onChange={handleFileUpload}
              className="hidden"
              disabled={isUploading}
            />
            <Button
              disabled={isUploading}
              className="flex items-center space-x-2"
            >
              {isUploading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Upload className="w-4 h-4" />
              )}
              <span>{isUploading ? 'Uploading...' : 'Upload Dataset'}</span>
            </Button>
          </label>
        </div>
      </div>

      {/* Connection Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Total Connections</p>
              <p className="text-2xl font-bold text-gray-900">{connectionStats.totalConnections}</p>
            </div>
            <div className="p-3 bg-blue-100 rounded-full">
              <Server className="w-6 h-6 text-blue-600" />
            </div>
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Active Connections</p>
              <p className="text-2xl font-bold text-green-600">{connectionStats.activeConnections}</p>
            </div>
            <div className="p-3 bg-green-100 rounded-full">
              <Wifi className="w-6 h-6 text-green-600" />
            </div>
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Total Records</p>
              <p className="text-2xl font-bold text-gray-900">{formatNumber(connectionStats.totalRecords)}</p>
            </div>
            <div className="p-3 bg-purple-100 rounded-full">
              <Database className="w-6 h-6 text-purple-600" />
            </div>
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Real-time Streams</p>
              <p className="text-2xl font-bold text-orange-600">{streamConnections.length}</p>
            </div>
            <div className="p-3 bg-orange-100 rounded-full">
              <Activity className="w-6 h-6 text-orange-600" />
            </div>
          </div>
        </Card>
      </div>

      {/* Data Sources */}
      <Card className="p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Data Sources</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {dataSources.map((source) => (
            <motion.div
              key={source.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="p-4 border border-gray-200 rounded-lg hover:border-blue-300 transition-colors"
            >
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center space-x-3">
                  {source.icon}
                  <h3 className="font-medium text-gray-900">{source.name}</h3>
                </div>
                <div className={`px-2 py-1 rounded-full text-xs border ${getStatusColor(source.status)}`}>
                  <div className="flex items-center space-x-1">
                    {getStatusIcon(source.status)}
                    <span className="capitalize">{source.status}</span>
                  </div>
                </div>
              </div>
              <p className="text-sm text-gray-600 mb-3">{source.description}</p>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-500">Records:</span>
                  <span className="font-medium">{formatNumber(source.recordCount)}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-500">Last Sync:</span>
                  <span className="font-medium">{formatDate(source.lastSync)}</span>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </Card>

      {/* Streaming Connections */}
      <Card className="p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Real-time Streaming</h2>
        <div className="space-y-4">
          {streamConnections.map((connection) => (
            <motion.div
              key={connection.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="flex items-center justify-between p-4 border border-gray-200 rounded-lg hover:border-blue-300 transition-colors"
            >
              <div className="flex items-center space-x-4">
                <div className="p-2 bg-blue-100 rounded-full">
                  <Activity className="w-5 h-5 text-blue-600" />
                </div>
                <div>
                  <h3 className="font-medium text-gray-900">{connection.name}</h3>
                  <p className="text-sm text-gray-600">{connection.type.toUpperCase()} Stream</p>
                </div>
              </div>
              <div className="flex items-center space-x-6">
                <div className="text-center">
                  <p className="text-sm text-gray-500">Messages Processed</p>
                  <p className="font-semibold text-gray-900">{formatNumber(connection.messagesProcessed)}</p>
                </div>
                <div className="text-center">
                  <p className="text-sm text-gray-500">Last Message</p>
                  <p className="font-semibold text-gray-900">{formatDate(connection.lastMessage)}</p>
                </div>
                <div className={`px-3 py-1 rounded-full text-sm border ${getStatusColor(connection.status)}`}>
                  <div className="flex items-center space-x-1">
                    {getStatusIcon(connection.status)}
                    <span className="capitalize">{connection.status}</span>
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </Card>

      {/* Datasets */}
      <Card className="p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-900">Datasets</h2>
          <div className="flex items-center space-x-3">
            <div className="relative">
              <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
              <input
                type="text"
                placeholder="Search datasets..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
            <select
              value={filterType}
              onChange={(e) => setFilterType(e.target.value as any)}
              className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="all">All Types</option>
              <option value="csv">CSV</option>
              <option value="json">JSON</option>
              <option value="parquet">Parquet</option>
            </select>
          </div>
        </div>

        {loading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="w-8 h-8 animate-spin text-blue-600" />
            <span className="ml-2 text-gray-600">Loading datasets...</span>
          </div>
        ) : filteredDatasets.length === 0 ? (
          <div className="text-center py-8">
            <Database className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-600">No datasets found. Upload your first dataset to get started.</p>
          </div>
        ) : (
          <div className="space-y-4">
            {filteredDatasets.map((dataset) => (
              <motion.div
                key={dataset.dataset_id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex items-center justify-between p-4 border border-gray-200 rounded-lg hover:border-blue-300 transition-colors"
              >
                <div className="flex items-center space-x-4">
                  <div className="p-2 bg-blue-100 rounded-full">
                    <FileText className="w-5 h-5 text-blue-600" />
                  </div>
                  <div>
                    <h3 className="font-medium text-gray-900">{dataset.name}</h3>
                    <p className="text-sm text-gray-600">{dataset.description || 'No description'}</p>
                    <div className="flex items-center space-x-4 mt-1">
                      <span className="text-xs text-gray-500">
                        {formatNumber(dataset.num_rows)} rows × {dataset.num_columns} columns
                      </span>
                      <span className="text-xs text-gray-500">
                        {formatFileSize(dataset.size_bytes)}
                      </span>
                      <span className="text-xs text-gray-500">
                        {dataset.format.toUpperCase()}
                      </span>
                    </div>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <Button
                    onClick={() => handleViewDataset(dataset.dataset_id)}
                    size="sm"
                    variant="outline"
                    className="flex items-center space-x-1"
                  >
                    <Eye className="w-4 h-4" />
                    <span>View</span>
                  </Button>
                  <Button
                    onClick={() => handleDeleteDataset(dataset.dataset_id)}
                    size="sm"
                    variant="outline"
                    className="flex items-center space-x-1 text-red-600 hover:text-red-700"
                  >
                    <Trash2 className="w-4 h-4" />
                    <span>Delete</span>
                  </Button>
                </div>
              </motion.div>
            ))}
          </div>
        )}
      </Card>

      {/* Dataset Details Modal */}
      <AnimatePresence>
        {selectedDataset && qualityReport && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="bg-white rounded-lg max-w-4xl w-full max-h-[90vh] overflow-y-auto"
            >
              <div className="p-6">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-xl font-bold text-gray-900">Dataset Quality Report</h2>
                  <Button
                    onClick={() => {
                      setSelectedDataset(null);
                      setQualityReport(null);
                    }}
                    variant="outline"
                    size="sm"
                  >
                    Close
                  </Button>
                </div>

                {/* Quality Score */}
                <div className="mb-6">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-gray-700">Quality Score</span>
                    <span className="text-sm font-bold text-gray-900">
                      {Math.round(qualityReport.quality_score * 100)}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full transition-all duration-500 ${
                        qualityReport.quality_score > 0.8
                          ? 'bg-green-500'
                          : qualityReport.quality_score > 0.6
                          ? 'bg-yellow-500'
                          : 'bg-red-500'
                      }`}
                      style={{ width: `${qualityReport.quality_score * 100}%` }}
                    />
                  </div>
                </div>

                {/* Issues */}
                {qualityReport.issues.length > 0 && (
                  <div className="mb-6">
                    <h3 className="text-sm font-medium text-gray-700 mb-3">Data Quality Issues</h3>
                    <div className="space-y-2">
                      {qualityReport.issues.map((issue, index) => (
                        <div
                          key={index}
                          className="flex items-center space-x-2 p-2 bg-red-50 rounded-lg"
                        >
                          <AlertTriangle className="w-4 h-4 text-red-500" />
                          <span className="text-sm text-red-700">{issue}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Statistics Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <Card className="p-4">
                    <h3 className="font-medium text-gray-900 mb-3">Dataset Overview</h3>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-sm text-gray-600">Total Rows:</span>
                        <span className="text-sm font-medium">{formatNumber(qualityReport.total_rows)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-gray-600">Total Features:</span>
                        <span className="text-sm font-medium">{qualityReport.total_features}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-gray-600">Duplicate Rows:</span>
                        <span className="text-sm font-medium">{formatNumber(qualityReport.duplicate_rows)}</span>
                      </div>
                    </div>
                  </Card>

                  <Card className="p-4">
                    <h3 className="font-medium text-gray-900 mb-3">Missing Values</h3>
                    <div className="space-y-2 max-h-32 overflow-y-auto">
                      {Object.entries(qualityReport.missing_values).map(([column, count]) => (
                        <div key={column} className="flex justify-between">
                          <span className="text-sm text-gray-600 truncate">{column}:</span>
                          <span className="text-sm font-medium">{count}</span>
                        </div>
                      ))}
                    </div>
                  </Card>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default DataConnectivity;