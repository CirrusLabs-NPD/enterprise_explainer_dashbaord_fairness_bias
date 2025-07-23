import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  Database,
  Cloud,
  Plug,
  CheckCircle,
  AlertTriangle,
  XCircle,
  Plus,
  Settings,
  RefreshCw,
  Download,
  Upload,
  Lock,
  Unlock,
  Activity,
  BarChart3,
  Clock,
  Shield,
  Key,
  Globe,
} from 'lucide-react';
import Card from '../components/common/Card';
import MetricCard from '../components/common/MetricCard';

const DataConnectors: React.FC = () => {
  const [selectedConnector, setSelectedConnector] = useState<string | null>(null);
  const [showCreateModal, setShowCreateModal] = useState(false);

  // Mock data connector information
  const connectorsOverview = {
    totalConnectors: 8,
    activeConnections: 6,
    dataTransferred: '2.4TB',
    lastSync: '5 minutes ago',
    successRate: 98.5,
    errorCount: 2,
  };

  const connectorTypes = [
    {
      id: 'snowflake',
      name: 'Snowflake',
      description: 'Enterprise data warehouse connectivity',
      icon: '❄️',
      category: 'Data Warehouse',
      isEnabled: true,
      connectionCount: 2,
      features: ['Bulk data loading', 'Real-time streaming', 'SQL queries', 'Schema detection'],
    },
    {
      id: 'aws_s3',
      name: 'AWS S3',
      description: 'Amazon S3 object storage integration',
      icon: '☁️',
      category: 'Cloud Storage',
      isEnabled: true,
      connectionCount: 3,
      features: ['File upload/download', 'Batch processing', 'Event triggers', 'Versioning'],
    },
    {
      id: 'databricks',
      name: 'Databricks',
      description: 'Unified analytics platform connector',
      icon: '🧱',
      category: 'Analytics Platform',
      isEnabled: true,
      connectionCount: 1,
      features: ['MLflow integration', 'Spark clusters', 'Delta Lake', 'Notebooks'],
    },
    {
      id: 'bigquery',
      name: 'BigQuery',
      description: 'Google Cloud data warehouse',
      icon: '🔍',
      category: 'Data Warehouse',
      isEnabled: true,
      connectionCount: 1,
      features: ['SQL analytics', 'ML models', 'Real-time data', 'Federated queries'],
    },
    {
      id: 'kafka',
      name: 'Apache Kafka',
      description: 'Real-time data streaming platform',
      icon: '🌊',
      category: 'Streaming',
      isEnabled: false,
      connectionCount: 0,
      features: ['Event streaming', 'Real-time processing', 'High throughput', 'Fault tolerance'],
    },
    {
      id: 'postgresql',
      name: 'PostgreSQL',
      description: 'Relational database connectivity',
      icon: '🐘',
      category: 'Database',
      isEnabled: true,
      connectionCount: 1,
      features: ['ACID compliance', 'JSON support', 'Full-text search', 'Replication'],
    },
  ];

  const activeConnections = [
    {
      id: 'conn_001',
      name: 'Production Data Warehouse',
      type: 'Snowflake',
      status: 'connected',
      lastSync: '2024-01-19T14:30:00Z',
      dataTransferred: '458.2 GB',
      errorCount: 0,
      uptime: '99.8%',
      environment: 'Production',
      credentials: 'prod-snowflake-key',
      syncFrequency: 'Real-time',
      tables: 45,
    },
    {
      id: 'conn_002',
      name: 'ML Training Data',
      type: 'AWS S3',
      status: 'connected',
      lastSync: '2024-01-19T14:25:00Z',
      dataTransferred: '1.2 TB',
      errorCount: 0,
      uptime: '99.9%',
      environment: 'Production',
      credentials: 'aws-s3-role',
      syncFrequency: 'Hourly',
      buckets: 8,
    },
    {
      id: 'conn_003',
      name: 'Analytics Workspace',
      type: 'Databricks',
      status: 'connected',
      lastSync: '2024-01-19T14:20:00Z',
      dataTransferred: '234.5 GB',
      errorCount: 1,
      uptime: '98.2%',
      environment: 'Development',
      credentials: 'databricks-token',
      syncFrequency: 'Daily',
      notebooks: 12,
    },
    {
      id: 'conn_004',
      name: 'Compliance Reporting DB',
      type: 'BigQuery',
      status: 'connected',
      lastSync: '2024-01-19T14:15:00Z',
      dataTransferred: '89.3 GB',
      errorCount: 0,
      uptime: '99.5%',
      environment: 'Production',
      credentials: 'gcp-service-account',
      syncFrequency: 'Daily',
      datasets: 6,
    },
    {
      id: 'conn_005',
      name: 'Customer Database',
      type: 'PostgreSQL',
      status: 'error',
      lastSync: '2024-01-19T12:45:00Z',
      dataTransferred: '45.7 GB',
      errorCount: 3,
      uptime: '95.1%',
      environment: 'Staging',
      credentials: 'postgres-staging',
      syncFrequency: '6 hours',
      tables: 18,
    },
    {
      id: 'conn_006',
      name: 'Feature Store',
      type: 'AWS S3',
      status: 'syncing',
      lastSync: '2024-01-19T14:35:00Z',
      dataTransferred: '567.8 GB',
      errorCount: 0,
      uptime: '99.7%',
      environment: 'Production',
      credentials: 'aws-feature-store',
      syncFrequency: 'Real-time',
      buckets: 3,
    },
  ];

  const syncHistory = [
    {
      timestamp: '2024-01-19T14:30:00Z',
      connection: 'Production Data Warehouse',
      type: 'Scheduled Sync',
      status: 'success',
      duration: '2.3 minutes',
      recordsProcessed: 12500,
      dataSize: '45.2 MB',
    },
    {
      timestamp: '2024-01-19T14:25:00Z',
      connection: 'ML Training Data',
      type: 'Incremental Sync',
      status: 'success',
      duration: '5.7 minutes',
      recordsProcessed: 8900,
      dataSize: '123.4 MB',
    },
    {
      timestamp: '2024-01-19T14:20:00Z',
      connection: 'Analytics Workspace',
      type: 'Full Sync',
      status: 'warning',
      duration: '15.2 minutes',
      recordsProcessed: 45000,
      dataSize: '567.8 MB',
    },
    {
      timestamp: '2024-01-19T14:15:00Z',
      connection: 'Compliance Reporting DB',
      type: 'Scheduled Sync',
      status: 'success',
      duration: '3.1 minutes',
      recordsProcessed: 6700,
      dataSize: '78.9 MB',
    },
    {
      timestamp: '2024-01-19T12:45:00Z',
      connection: 'Customer Database',
      type: 'Scheduled Sync',
      status: 'error',
      duration: '1.2 minutes',
      recordsProcessed: 0,
      dataSize: '0 MB',
    },
  ];

  const dataQualityMetrics = {
    completeness: 94.2,
    accuracy: 96.8,
    consistency: 91.5,
    timeliness: 98.1,
    validity: 93.7,
    uniqueness: 99.2,
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'connected':
        return 'text-green-600 bg-green-100 dark:bg-green-900/30 dark:text-green-300';
      case 'syncing':
        return 'text-blue-600 bg-blue-100 dark:bg-blue-900/30 dark:text-blue-300';
      case 'error':
        return 'text-red-600 bg-red-100 dark:bg-red-900/30 dark:text-red-300';
      case 'disconnected':
        return 'text-gray-600 bg-gray-100 dark:bg-gray-900/30 dark:text-gray-300';
      default:
        return 'text-gray-600 bg-gray-100 dark:bg-gray-900/30 dark:text-gray-300';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'connected':
        return <CheckCircle className="w-4 h-4" />;
      case 'syncing':
        return <RefreshCw className="w-4 h-4 animate-spin" />;
      case 'error':
        return <XCircle className="w-4 h-4" />;
      case 'disconnected':
        return <AlertTriangle className="w-4 h-4" />;
      default:
        return <AlertTriangle className="w-4 h-4" />;
    }
  };

  const getSyncStatusColor = (status: string) => {
    switch (status) {
      case 'success':
        return 'text-green-600';
      case 'warning':
        return 'text-yellow-600';
      case 'error':
        return 'text-red-600';
      default:
        return 'text-gray-600';
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-neutral-900 dark:text-neutral-100">
            Data Connectors
          </h1>
          <p className="text-neutral-600 dark:text-neutral-400 mt-1">
            Enterprise data source integrations and real-time synchronization
          </p>
        </div>
        
        <div className="flex items-center space-x-4">
          <button className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors">
            <Plus className="w-4 h-4" />
            <span>Add Connector</span>
          </button>
          
          <button className="flex items-center space-x-2 px-4 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg hover:bg-neutral-50 dark:hover:bg-neutral-800 transition-colors">
            <Settings className="w-4 h-4" />
            <span>Settings</span>
          </button>
        </div>
      </div>

      {/* Connectors Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-6">
        <MetricCard
          title="Total Connectors"
          value={connectorsOverview.totalConnectors.toString()}
          change="+1"
          changeType="positive"
          icon={<Plug/>}
          description="Available connector types"
        />
        
        <MetricCard
          title="Active Connections"
          value={connectorsOverview.activeConnections.toString()}
          change="0"
          changeType="neutral"
          icon={<Activity/>}
          description="Currently connected sources"
        />
        
        <MetricCard
          title="Data Transferred"
          value={connectorsOverview.dataTransferred}
          change="+15.2%"
          changeType="positive"
          icon={<Database/>}
          description="Total data processed today"
        />
        
        <MetricCard
          title="Last Sync"
          value={connectorsOverview.lastSync}
          change="Real-time"
          changeType="positive"
          icon={<Clock/>}
          description="Most recent synchronization"
        />
        
        <MetricCard
          title="Success Rate"
          value={`${connectorsOverview.successRate}%`}
          change="+0.3%"
          changeType="positive"
          icon={<CheckCircle/>}
          description="Sync success percentage"
        />
        
        <MetricCard
          title="Error Count"
          value={connectorsOverview.errorCount.toString()}
          change="-1"
          changeType="positive"
          icon={<AlertTriangle/>}
          description="Sync errors in last 24h"
        />
      </div>

      {/* Available Connectors */}
      <Card>
        <div className="p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
              Available Connectors
            </h3>
            <div className="text-sm text-neutral-500">
              {connectorTypes.filter(c => c.isEnabled).length} of {connectorTypes.length} enabled
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {connectorTypes.map((connector, index) => (
              <motion.div
                key={connector.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                className={`p-6 border rounded-lg transition-all cursor-pointer ${
                  connector.isEnabled
                    ? 'border-blue-200 bg-blue-50 dark:bg-blue-900/20 dark:border-blue-800'
                    : 'border-neutral-200 dark:border-neutral-700 hover:border-blue-300 dark:hover:border-blue-600'
                }`}
                onClick={() => setSelectedConnector(connector.id)}
              >
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    <span className="text-2xl">{connector.icon}</span>
                    <div>
                      <h4 className="font-semibold text-neutral-900 dark:text-neutral-100">
                        {connector.name}
                      </h4>
                      <p className="text-sm text-neutral-500">{connector.category}</p>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    {connector.isEnabled ? (
                      <CheckCircle className="w-5 h-5 text-green-500" />
                    ) : (
                      <Plus className="w-5 h-5 text-neutral-400" />
                    )}
                  </div>
                </div>
                
                <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-4">
                  {connector.description}
                </p>
                
                <div className="mb-4">
                  <div className="text-xs text-neutral-500 mb-2">Key Features:</div>
                  <div className="flex flex-wrap gap-1">
                    {connector.features.slice(0, 2).map((feature) => (
                      <span
                        key={feature}
                        className="px-2 py-1 bg-neutral-100 dark:bg-neutral-800 text-neutral-700 dark:text-neutral-300 rounded text-xs"
                      >
                        {feature}
                      </span>
                    ))}
                    {connector.features.length > 2 && (
                      <span className="px-2 py-1 bg-neutral-100 dark:bg-neutral-800 text-neutral-500 rounded text-xs">
                        +{connector.features.length - 2} more
                      </span>
                    )}
                  </div>
                </div>
                
                <div className="flex items-center justify-between">
                  <span className="text-sm text-neutral-600 dark:text-neutral-400">
                    {connector.connectionCount} connection{connector.connectionCount !== 1 ? 's' : ''}
                  </span>
                  
                  <button className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                    connector.isEnabled
                      ? 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300 hover:bg-blue-200'
                      : 'bg-neutral-100 text-neutral-800 dark:bg-neutral-800 dark:text-neutral-300 hover:bg-neutral-200'
                  }`}>
                    {connector.isEnabled ? 'Configure' : 'Enable'}
                  </button>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </Card>

      {/* Active Connections and Data Quality */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Active Connections */}
        <Card className="lg:col-span-2">
          <div className="p-6">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                Active Connections
              </h3>
              <div className="flex items-center space-x-2">
                <RefreshCw className="w-4 h-4 text-neutral-500" />
                <span className="text-sm text-neutral-500">Auto-refresh enabled</span>
              </div>
            </div>
            
            <div className="space-y-4">
              {activeConnections.map((connection, index) => (
                <motion.div
                  key={connection.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="p-4 border border-neutral-200 dark:border-neutral-700 rounded-lg"
                >
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center space-x-3">
                      <div className="flex items-center space-x-2">
                        {getStatusIcon(connection.status)}
                        <span className="font-medium text-neutral-900 dark:text-neutral-100">
                          {connection.name}
                        </span>
                      </div>
                      <span className="text-sm text-neutral-500">({connection.type})</span>
                    </div>
                    
                    <div className="flex items-center space-x-3">
                      <span className={`px-2 py-1 rounded text-xs font-medium ${getStatusColor(connection.status)}`}>
                        {connection.status}
                      </span>
                      <span className="text-xs text-neutral-500">{connection.environment}</span>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div>
                      <span className="text-neutral-500">Last Sync:</span>
                      <div className="font-medium">{new Date(connection.lastSync).toLocaleString()}</div>
                    </div>
                    <div>
                      <span className="text-neutral-500">Data Transferred:</span>
                      <div className="font-medium">{connection.dataTransferred}</div>
                    </div>
                    <div>
                      <span className="text-neutral-500">Uptime:</span>
                      <div className="font-medium">{connection.uptime}</div>
                    </div>
                    <div>
                      <span className="text-neutral-500">Sync Frequency:</span>
                      <div className="font-medium">{connection.syncFrequency}</div>
                    </div>
                  </div>
                  
                  {connection.errorCount > 0 && (
                    <div className="mt-3 p-2 bg-red-50 dark:bg-red-900/20 rounded">
                      <div className="flex items-center space-x-2">
                        <AlertTriangle className="w-4 h-4 text-red-600" />
                        <span className="text-sm text-red-700 dark:text-red-300">
                          {connection.errorCount} error{connection.errorCount !== 1 ? 's' : ''} in last 24 hours
                        </span>
                      </div>
                    </div>
                  )}
                  
                  <div className="mt-3 flex items-center justify-between">
                    <div className="flex items-center space-x-4 text-xs text-neutral-500">
                      <span>
                        {connection.tables && `${connection.tables} tables`}
                        {connection.buckets && `${connection.buckets} buckets`}
                        {connection.notebooks && `${connection.notebooks} notebooks`}
                        {connection.datasets && `${connection.datasets} datasets`}
                      </span>
                      <span>Credential: {connection.credentials}</span>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <button className="text-blue-600 hover:text-blue-800 transition-colors">
                        <Settings className="w-4 h-4" />
                      </button>
                      <button className="text-green-600 hover:text-green-800 transition-colors">
                        <RefreshCw className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </Card>

        {/* Data Quality Metrics */}
        <Card>
          <div className="p-6">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                Data Quality
              </h3>
              <Shield className="w-5 h-5 text-green-500" />
            </div>
            
            <div className="space-y-4">
              {Object.entries(dataQualityMetrics).map(([metric, value], index) => (
                <motion.div
                  key={metric}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="space-y-2"
                >
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-neutral-900 dark:text-neutral-100 capitalize">
                      {metric}
                    </span>
                    <span className="text-sm font-semibold">{value.toFixed(1)}%</span>
                  </div>
                  
                  <div className="w-full bg-neutral-200 dark:bg-neutral-700 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full transition-all duration-500 ${
                        value >= 95
                          ? 'bg-green-500'
                          : value >= 85
                          ? 'bg-yellow-500'
                          : 'bg-red-500'
                      }`}
                      style={{ width: `${value}%` }}
                    />
                  </div>
                </motion.div>
              ))}
            </div>
            
            <div className="mt-6 p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
              <div className="flex items-center space-x-2">
                <CheckCircle className="w-5 h-5 text-green-600" />
                <div>
                  <p className="text-sm font-medium text-green-800 dark:text-green-300">
                    Overall Quality Score: 95.6%
                  </p>
                  <p className="text-xs text-green-600 dark:text-green-400">
                    Excellent data quality across all sources
                  </p>
                </div>
              </div>
            </div>
          </div>
        </Card>
      </div>

      {/* Sync History */}
      <Card>
        <div className="p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
              Recent Sync History
            </h3>
            <div className="flex items-center space-x-4">
              <select className="px-3 py-1 border border-neutral-300 dark:border-neutral-600 rounded text-sm">
                <option>All Connections</option>
                <option>Success Only</option>
                <option>Errors Only</option>
              </select>
              <Download className="w-4 h-4 text-neutral-500 cursor-pointer" />
            </div>
          </div>
          
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-neutral-200 dark:border-neutral-700">
                  <th className="text-left py-3 px-4 font-medium text-neutral-900 dark:text-neutral-100">Timestamp</th>
                  <th className="text-left py-3 px-4 font-medium text-neutral-900 dark:text-neutral-100">Connection</th>
                  <th className="text-left py-3 px-4 font-medium text-neutral-900 dark:text-neutral-100">Type</th>
                  <th className="text-left py-3 px-4 font-medium text-neutral-900 dark:text-neutral-100">Status</th>
                  <th className="text-left py-3 px-4 font-medium text-neutral-900 dark:text-neutral-100">Duration</th>
                  <th className="text-left py-3 px-4 font-medium text-neutral-900 dark:text-neutral-100">Records</th>
                  <th className="text-left py-3 px-4 font-medium text-neutral-900 dark:text-neutral-100">Data Size</th>
                </tr>
              </thead>
              <tbody>
                {syncHistory.map((sync, index) => (
                  <motion.tr
                    key={index}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: index * 0.1 }}
                    className="border-b border-neutral-100 dark:border-neutral-800 hover:bg-neutral-50 dark:hover:bg-neutral-900/50"
                  >
                    <td className="py-3 px-4 text-sm text-neutral-600 dark:text-neutral-400">
                      {new Date(sync.timestamp).toLocaleString()}
                    </td>
                    <td className="py-3 px-4">
                      <span className="font-medium text-neutral-900 dark:text-neutral-100">
                        {sync.connection}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-sm text-neutral-600 dark:text-neutral-400">
                      {sync.type}
                    </td>
                    <td className="py-3 px-4">
                      <span className={`px-2 py-1 rounded text-xs font-medium ${
                        sync.status === 'success'
                          ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300'
                          : sync.status === 'warning'
                          ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300'
                          : 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300'
                      }`}>
                        {sync.status}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-sm text-neutral-600 dark:text-neutral-400">
                      {sync.duration}
                    </td>
                    <td className="py-3 px-4 text-sm text-neutral-600 dark:text-neutral-400">
                      {sync.recordsProcessed.toLocaleString()}
                    </td>
                    <td className="py-3 px-4 text-sm text-neutral-600 dark:text-neutral-400">
                      {sync.dataSize}
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default DataConnectors;