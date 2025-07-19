import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Activity,
  Database,
  Play,
  Pause,
  StopCircle,
  Settings,
  AlertTriangle,
  CheckCircle,
  TrendingUp,
  TrendingDown,
  Clock,
  BarChart3,
  Filter,
  Zap,
  RefreshCw,
  Eye,
  Download,
  Upload,
  Server,
  Wifi,
  WifiOff,
  FileText,
  Users,
  Target
} from 'lucide-react';
import Card from '../components/common/Card';
import MetricCard from '../components/common/MetricCard';
import { apiClient } from '../services/api';

interface DataStream {
  id: string;
  name: string;
  type: 'kafka' | 'kinesis' | 'pubsub' | 'rabbitmq' | 'webhook';
  status: 'connected' | 'disconnected' | 'error' | 'connecting';
  source: {
    url: string;
    topic?: string;
    partition?: number;
    format: 'json' | 'avro' | 'csv' | 'parquet';
  };
  metrics: {
    recordsPerSecond: number;
    totalRecords: number;
    lastReceived: string;
    errorRate: number;
    latency: number;
    throughput: number;
  };
  schema: {
    fields: Array<{
      name: string;
      type: string;
      required: boolean;
    }>;
    version: string;
  };
  processing: {
    enabled: boolean;
    transformations: string[];
    validations: string[];
    enrichments: string[];
  };
  targets: Array<{
    id: string;
    type: 'database' | 'warehouse' | 'lake' | 'model';
    name: string;
    enabled: boolean;
  }>;
  createdAt: string;
  lastActive: string;
}

interface StreamAlert {
  id: string;
  streamId: string;
  type: 'error' | 'warning' | 'info';
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  timestamp: string;
  resolved: boolean;
}

const StreamDataManagement: React.FC = () => {
  const [streams, setStreams] = useState<DataStream[]>([]);
  const [alerts, setAlerts] = useState<StreamAlert[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedStream, setSelectedStream] = useState<DataStream | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'streams' | 'monitoring' | 'alerts'>('overview');
  const [showNewStreamModal, setShowNewStreamModal] = useState(false);

  // Mock data for demonstration
  const mockStreams: DataStream[] = [
    {
      id: 'stream_001',
      name: 'Customer Transaction Feed',
      type: 'kafka',
      status: 'connected',
      source: {
        url: 'kafka://localhost:9092',
        topic: 'transactions',
        partition: 0,
        format: 'json'
      },
      metrics: {
        recordsPerSecond: 1250,
        totalRecords: 4567892,
        lastReceived: '2024-01-20T14:35:22Z',
        errorRate: 0.02,
        latency: 45,
        throughput: 2.8
      },
      schema: {
        fields: [
          { name: 'transaction_id', type: 'string', required: true },
          { name: 'customer_id', type: 'string', required: true },
          { name: 'amount', type: 'decimal', required: true },
          { name: 'timestamp', type: 'datetime', required: true },
          { name: 'merchant_id', type: 'string', required: true },
          { name: 'category', type: 'string', required: false }
        ],
        version: 'v2.1'
      },
      processing: {
        enabled: true,
        transformations: ['normalize_amounts', 'extract_features'],
        validations: ['schema_validation', 'business_rules'],
        enrichments: ['merchant_data', 'customer_profile']
      },
      targets: [
        { id: 'db_01', type: 'database', name: 'PostgreSQL Analytics', enabled: true },
        { id: 'model_01', type: 'model', name: 'Fraud Detection Model', enabled: true },
        { id: 'lake_01', type: 'lake', name: 'Data Lake Storage', enabled: false }
      ],
      createdAt: '2024-01-10T09:00:00Z',
      lastActive: '2024-01-20T14:35:22Z'
    },
    {
      id: 'stream_002',
      name: 'User Behavior Events',
      type: 'kinesis',
      status: 'error',
      source: {
        url: 'kinesis://us-east-1/user-events',
        format: 'json'
      },
      metrics: {
        recordsPerSecond: 0,
        totalRecords: 2345678,
        lastReceived: '2024-01-20T13:45:15Z',
        errorRate: 0.85,
        latency: 0,
        throughput: 0
      },
      schema: {
        fields: [
          { name: 'user_id', type: 'string', required: true },
          { name: 'event_type', type: 'string', required: true },
          { name: 'timestamp', type: 'datetime', required: true },
          { name: 'properties', type: 'json', required: false }
        ],
        version: 'v1.3'
      },
      processing: {
        enabled: false,
        transformations: ['event_parsing'],
        validations: ['user_validation'],
        enrichments: ['session_data']
      },
      targets: [
        { id: 'warehouse_01', type: 'warehouse', name: 'BigQuery DW', enabled: true }
      ],
      createdAt: '2024-01-08T14:30:00Z',
      lastActive: '2024-01-20T13:45:15Z'
    },
    {
      id: 'stream_003',
      name: 'Model Predictions',
      type: 'webhook',
      status: 'connected',
      source: {
        url: 'https://api.ml-service.com/predictions',
        format: 'json'
      },
      metrics: {
        recordsPerSecond: 450,
        totalRecords: 987654,
        lastReceived: '2024-01-20T14:34:58Z',
        errorRate: 0.01,
        latency: 23,
        throughput: 1.2
      },
      schema: {
        fields: [
          { name: 'prediction_id', type: 'string', required: true },
          { name: 'model_id', type: 'string', required: true },
          { name: 'input_data', type: 'json', required: true },
          { name: 'prediction', type: 'decimal', required: true },
          { name: 'confidence', type: 'decimal', required: true }
        ],
        version: 'v1.0'
      },
      processing: {
        enabled: true,
        transformations: ['prediction_parsing'],
        validations: ['confidence_check'],
        enrichments: ['model_metadata']
      },
      targets: [
        { id: 'db_02', type: 'database', name: 'Predictions DB', enabled: true },
        { id: 'warehouse_02', type: 'warehouse', name: 'Analytics DW', enabled: true }
      ],
      createdAt: '2024-01-15T11:20:00Z',
      lastActive: '2024-01-20T14:34:58Z'
    }
  ];

  const mockAlerts: StreamAlert[] = [
    {
      id: 'alert_001',
      streamId: 'stream_002',
      type: 'error',
      severity: 'critical',
      message: 'Stream connection lost - Authentication failed',
      timestamp: '2024-01-20T13:45:30Z',
      resolved: false
    },
    {
      id: 'alert_002',
      streamId: 'stream_001',
      type: 'warning',
      severity: 'medium',
      message: 'High error rate detected (0.02%)',
      timestamp: '2024-01-20T14:20:15Z',
      resolved: false
    },
    {
      id: 'alert_003',
      streamId: 'stream_003',
      type: 'info',
      severity: 'low',
      message: 'Schema version updated to v1.0',
      timestamp: '2024-01-20T12:30:00Z',
      resolved: true
    }
  ];

  useEffect(() => {
    loadStreams();
  }, []);

  const loadStreams = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Try to fetch from API, fallback to mock data
      const response = await apiClient.getStreamStatus('all');
      
      if (response.success && response.data) {
        setStreams(response.data);
      } else {
        // Use mock data as fallback
        setTimeout(() => {
          setStreams(mockStreams);
          setAlerts(mockAlerts);
          if (mockStreams.length > 0) {
            setSelectedStream(mockStreams[0]);
          }
          setLoading(false);
        }, 1000);
        return;
      }
    } catch (err) {
      console.warn('API call failed, using mock data:', err);
      // Use mock data as fallback
      setTimeout(() => {
        setStreams(mockStreams);
        setAlerts(mockAlerts);
        if (mockStreams.length > 0) {
          setSelectedStream(mockStreams[0]);
        }
        setLoading(false);
      }, 1000);
      return;
    }
    
    setLoading(false);
  };

  const handleStreamAction = async (streamId: string, action: 'start' | 'stop' | 'restart') => {
    try {
      const stream = streams.find(s => s.id === streamId);
      if (!stream) return;

      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Update local state
      setStreams(prevStreams => 
        prevStreams.map(s => 
          s.id === streamId 
            ? { 
                ...s, 
                status: action === 'stop' ? 'disconnected' : 
                       action === 'start' ? 'connected' : 'connecting'
              }
            : s
        )
      );
    } catch (err) {
      console.error('Stream action failed:', err);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'connected': return 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300';
      case 'disconnected': return 'bg-neutral-100 text-neutral-800 dark:bg-neutral-800 dark:text-neutral-300';
      case 'error': return 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300';
      case 'connecting': return 'bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300';
      default: return 'bg-neutral-100 text-neutral-800 dark:bg-neutral-800 dark:text-neutral-300';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'kafka': return Database;
      case 'kinesis': return Activity;
      case 'pubsub': return Server;
      case 'rabbitmq': return FileText;
      case 'webhook': return Zap;
      default: return Database;
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'bg-red-500 text-white';
      case 'high': return 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300';
      case 'medium': return 'bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300';
      case 'low': return 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300';
      default: return 'bg-neutral-100 text-neutral-800 dark:bg-neutral-800 dark:text-neutral-300';
    }
  };

  const overallMetrics = {
    totalStreams: streams.length,
    activeStreams: streams.filter(s => s.status === 'connected').length,
    totalThroughput: streams.reduce((sum, s) => sum + s.metrics.recordsPerSecond, 0),
    avgLatency: streams.length > 0 ? streams.reduce((sum, s) => sum + s.metrics.latency, 0) / streams.length : 0,
    errorRate: streams.length > 0 ? streams.reduce((sum, s) => sum + s.metrics.errorRate, 0) / streams.length : 0
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <RefreshCw className="w-8 h-8 animate-spin text-primary-600" />
        <span className="ml-3 text-lg text-neutral-600 dark:text-neutral-400">
          Loading stream data...
        </span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center py-12">
        <AlertTriangle className="w-12 h-12 text-red-500 mb-4" />
        <h2 className="text-xl font-semibold text-neutral-900 dark:text-neutral-100 mb-2">
          Failed to Load Stream Data
        </h2>
        <p className="text-neutral-600 dark:text-neutral-400 mb-4 text-center max-w-md">
          {error}
        </p>
        <button
          onClick={loadStreams}
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
            Stream Data Management
          </h1>
          <p className="text-neutral-600 dark:text-neutral-400 mt-1">
            Monitor and manage real-time data streams for ML pipelines
          </p>
        </div>
        
        <div className="flex items-center space-x-4">
          <button
            onClick={() => setShowNewStreamModal(true)}
            className="flex items-center space-x-2 px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors"
          >
            <Upload className="w-4 h-4" />
            <span>Add Stream</span>
          </button>
          
          <button
            onClick={loadStreams}
            className="flex items-center space-x-2 px-4 py-2 bg-neutral-600 hover:bg-neutral-700 text-white rounded-lg transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Overall Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
        <MetricCard
          title="Total Streams"
          value={overallMetrics.totalStreams}
          format="number"
          icon={<Database className="w-5 h-5" />}
          change={overallMetrics.totalStreams > 0 ? "+1" : "0"}
          changeType="positive"
        />
        
        <MetricCard
          title="Active Streams"
          value={overallMetrics.activeStreams}
          format="number"
          icon={<Activity className="w-5 h-5" />}
          change={`${overallMetrics.activeStreams}/${overallMetrics.totalStreams}`}
          changeType={overallMetrics.activeStreams === overallMetrics.totalStreams ? "positive" : "negative"}
        />
        
        <MetricCard
          title="Total Throughput"
          value={`${overallMetrics.totalThroughput.toLocaleString()}/s`}
          format="text"
          icon={<TrendingUp className="w-5 h-5" />}
          change="+5.2%"
          changeType="positive"
        />
        
        <MetricCard
          title="Avg Latency"
          value={`${overallMetrics.avgLatency.toFixed(0)}ms`}
          format="text"
          icon={<Clock className="w-5 h-5" />}
          change="-2.3ms"
          changeType="positive"
        />
        
        <MetricCard
          title="Error Rate"
          value={`${(overallMetrics.errorRate * 100).toFixed(2)}%`}
          format="text"
          icon={<AlertTriangle className="w-5 h-5" />}
          change={overallMetrics.errorRate < 0.01 ? "Low" : "High"}
          changeType={overallMetrics.errorRate < 0.01 ? "positive" : "negative"}
        />
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-neutral-200 dark:border-neutral-700">
        <nav className="flex space-x-8">
          {[
            { id: 'overview', label: 'Overview', icon: BarChart3 },
            { id: 'streams', label: 'Streams', icon: Activity },
            { id: 'monitoring', label: 'Monitoring', icon: Eye },
            { id: 'alerts', label: 'Alerts', icon: AlertTriangle }
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
              {tab.id === 'alerts' && alerts.filter(a => !a.resolved).length > 0 && (
                <span className="ml-1 px-2 py-0.5 bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300 rounded-full text-xs">
                  {alerts.filter(a => !a.resolved).length}
                </span>
              )}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      {activeTab === 'overview' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Stream Status Overview */}
          <Card className="lg:col-span-2">
            <div className="p-6">
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
                Stream Status Overview
              </h3>
              
              <div className="space-y-4">
                {streams.map((stream) => {
                  const TypeIcon = getTypeIcon(stream.type);
                  
                  return (
                    <div
                      key={stream.id}
                      className="p-4 border border-neutral-200 dark:border-neutral-700 rounded-lg hover:border-primary-300 dark:hover:border-primary-600 transition-colors cursor-pointer"
                      onClick={() => setSelectedStream(stream)}
                    >
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center space-x-3">
                          <div className={`p-2 rounded-lg ${
                            stream.status === 'connected' ? 'bg-green-100 dark:bg-green-900/20' :
                            stream.status === 'error' ? 'bg-red-100 dark:bg-red-900/20' :
                            'bg-neutral-100 dark:bg-neutral-800'
                          }`}>
                            <TypeIcon className={`w-5 h-5 ${
                              stream.status === 'connected' ? 'text-green-600 dark:text-green-400' :
                              stream.status === 'error' ? 'text-red-600 dark:text-red-400' :
                              'text-neutral-500'
                            }`} />
                          </div>
                          <div>
                            <h4 className="font-medium text-neutral-900 dark:text-neutral-100">
                              {stream.name}
                            </h4>
                            <p className="text-sm text-neutral-600 dark:text-neutral-400">
                              {stream.type.toUpperCase()} • {stream.source.format.toUpperCase()}
                            </p>
                          </div>
                        </div>
                        
                        <div className="flex items-center space-x-3">
                          <span className={`px-2 py-1 rounded text-xs font-medium ${getStatusColor(stream.status)}`}>
                            {stream.status.charAt(0).toUpperCase() + stream.status.slice(1)}
                          </span>
                          {stream.status === 'connected' ? (
                            <Wifi className="w-4 h-4 text-green-500" />
                          ) : (
                            <WifiOff className="w-4 h-4 text-red-500" />
                          )}
                        </div>
                      </div>
                      
                      <div className="grid grid-cols-3 gap-4 text-sm">
                        <div>
                          <span className="text-neutral-500">Records/sec:</span>
                          <div className="font-medium text-neutral-900 dark:text-neutral-100">
                            {stream.metrics.recordsPerSecond.toLocaleString()}
                          </div>
                        </div>
                        <div>
                          <span className="text-neutral-500">Latency:</span>
                          <div className="font-medium text-neutral-900 dark:text-neutral-100">
                            {stream.metrics.latency}ms
                          </div>
                        </div>
                        <div>
                          <span className="text-neutral-500">Error Rate:</span>
                          <div className={`font-medium ${
                            stream.metrics.errorRate < 0.01 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
                          }`}>
                            {(stream.metrics.errorRate * 100).toFixed(2)}%
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </Card>

          {/* Recent Alerts */}
          <Card>
            <div className="p-6">
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
                Recent Alerts
              </h3>
              
              <div className="space-y-3">
                {alerts.slice(0, 5).map((alert) => (
                  <div
                    key={alert.id}
                    className={`p-3 rounded-lg border-l-4 ${
                      alert.type === 'error' ? 'border-red-500 bg-red-50 dark:bg-red-900/20' :
                      alert.type === 'warning' ? 'border-amber-500 bg-amber-50 dark:bg-amber-900/20' :
                      'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className={`px-2 py-1 rounded text-xs font-medium ${getSeverityColor(alert.severity)}`}>
                        {alert.severity.charAt(0).toUpperCase() + alert.severity.slice(1)}
                      </span>
                      <span className="text-xs text-neutral-500">
                        {new Date(alert.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                    <p className="text-sm text-neutral-700 dark:text-neutral-300">
                      {alert.message}
                    </p>
                  </div>
                ))}
              </div>
              
              {alerts.length === 0 && (
                <div className="text-center py-8 text-neutral-500 dark:text-neutral-400">
                  <CheckCircle className="w-8 h-8 mx-auto mb-2 opacity-50" />
                  <p>No recent alerts</p>
                </div>
              )}
            </div>
          </Card>
        </div>
      )}

      {activeTab === 'streams' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Stream List */}
          <Card className="lg:col-span-2">
            <div className="p-6">
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
                All Streams
              </h3>
              
              <div className="space-y-4">
                {streams.map((stream) => {
                  const TypeIcon = getTypeIcon(stream.type);
                  
                  return (
                    <div
                      key={stream.id}
                      className={`p-4 border rounded-lg transition-colors cursor-pointer ${
                        selectedStream?.id === stream.id
                          ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                          : 'border-neutral-200 dark:border-neutral-700 hover:border-primary-300 dark:hover:border-primary-600'
                      }`}
                      onClick={() => setSelectedStream(stream)}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                          <TypeIcon className="w-5 h-5 text-neutral-500" />
                          <div>
                            <h4 className="font-medium text-neutral-900 dark:text-neutral-100">
                              {stream.name}
                            </h4>
                            <p className="text-sm text-neutral-600 dark:text-neutral-400">
                              {stream.source.url}
                            </p>
                          </div>
                        </div>
                        
                        <div className="flex items-center space-x-2">
                          <span className={`px-2 py-1 rounded text-xs font-medium ${getStatusColor(stream.status)}`}>
                            {stream.status}
                          </span>
                          
                          <div className="flex items-center space-x-1">
                            {stream.status === 'connected' && (
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleStreamAction(stream.id, 'stop');
                                }}
                                className="p-1 text-neutral-500 hover:text-red-600 transition-colors"
                              >
                                <Pause className="w-4 h-4" />
                              </button>
                            )}
                            
                            {stream.status === 'disconnected' && (
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleStreamAction(stream.id, 'start');
                                }}
                                className="p-1 text-neutral-500 hover:text-green-600 transition-colors"
                              >
                                <Play className="w-4 h-4" />
                              </button>
                            )}
                            
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                handleStreamAction(stream.id, 'restart');
                              }}
                              className="p-1 text-neutral-500 hover:text-blue-600 transition-colors"
                            >
                              <RefreshCw className="w-4 h-4" />
                            </button>
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </Card>

          {/* Stream Details */}
          {selectedStream && (
            <Card>
              <div className="p-6">
                <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
                  Stream Details
                </h3>
                
                <div className="space-y-4">
                  <div>
                    <h4 className="font-medium text-neutral-900 dark:text-neutral-100 mb-2">
                      Configuration
                    </h4>
                    <div className="text-sm space-y-1">
                      <div className="flex justify-between">
                        <span className="text-neutral-500">Type:</span>
                        <span className="font-medium">{selectedStream.type.toUpperCase()}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-neutral-500">Format:</span>
                        <span className="font-medium">{selectedStream.source.format.toUpperCase()}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-neutral-500">Schema:</span>
                        <span className="font-medium">{selectedStream.schema.version}</span>
                      </div>
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="font-medium text-neutral-900 dark:text-neutral-100 mb-2">
                      Performance
                    </h4>
                    <div className="text-sm space-y-2">
                      <div>
                        <div className="flex justify-between mb-1">
                          <span className="text-neutral-500">Records/sec:</span>
                          <span className="font-medium">{selectedStream.metrics.recordsPerSecond.toLocaleString()}</span>
                        </div>
                        <div className="w-full bg-neutral-200 dark:bg-neutral-700 rounded-full h-2">
                          <div 
                            className="h-2 bg-primary-500 rounded-full"
                            style={{ width: `${Math.min(selectedStream.metrics.recordsPerSecond / 2000 * 100, 100)}%` }}
                          />
                        </div>
                      </div>
                      
                      <div className="flex justify-between">
                        <span className="text-neutral-500">Total Records:</span>
                        <span className="font-medium">{selectedStream.metrics.totalRecords.toLocaleString()}</span>
                      </div>
                      
                      <div className="flex justify-between">
                        <span className="text-neutral-500">Latency:</span>
                        <span className="font-medium">{selectedStream.metrics.latency}ms</span>
                      </div>
                      
                      <div className="flex justify-between">
                        <span className="text-neutral-500">Error Rate:</span>
                        <span className={`font-medium ${
                          selectedStream.metrics.errorRate < 0.01 ? 'text-green-600' : 'text-red-600'
                        }`}>
                          {(selectedStream.metrics.errorRate * 100).toFixed(2)}%
                        </span>
                      </div>
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="font-medium text-neutral-900 dark:text-neutral-100 mb-2">
                      Targets ({selectedStream.targets.filter(t => t.enabled).length} active)
                    </h4>
                    <div className="space-y-2">
                      {selectedStream.targets.map((target) => (
                        <div key={target.id} className="flex items-center justify-between text-sm">
                          <div className="flex items-center space-x-2">
                            <div className={`w-2 h-2 rounded-full ${target.enabled ? 'bg-green-500' : 'bg-neutral-400'}`} />
                            <span className="text-neutral-700 dark:text-neutral-300">{target.name}</span>
                          </div>
                          <span className="text-xs text-neutral-500 capitalize">{target.type}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </Card>
          )}
        </div>
      )}

      {activeTab === 'monitoring' && (
        <Card>
          <div className="p-6">
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
              Real-time Monitoring
            </h3>
            <div className="text-center py-12 text-neutral-500 dark:text-neutral-400">
              <BarChart3 className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p>Real-time monitoring charts coming soon</p>
              <p className="text-sm">Track throughput, latency, and error rates over time</p>
            </div>
          </div>
        </Card>
      )}

      {activeTab === 'alerts' && (
        <Card>
          <div className="p-6">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                Stream Alerts
              </h3>
              <div className="flex items-center space-x-4">
                <select className="px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100">
                  <option value="">All Severities</option>
                  <option value="critical">Critical</option>
                  <option value="high">High</option>
                  <option value="medium">Medium</option>
                  <option value="low">Low</option>
                </select>
                <select className="px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100">
                  <option value="">All Streams</option>
                  {streams.map(stream => (
                    <option key={stream.id} value={stream.id}>{stream.name}</option>
                  ))}
                </select>
              </div>
            </div>
            
            <div className="space-y-4">
              {alerts.map((alert) => (
                <div
                  key={alert.id}
                  className={`p-4 border rounded-lg transition-colors ${
                    alert.resolved ? 'border-neutral-200 dark:border-neutral-700 opacity-60' : 'border-neutral-300 dark:border-neutral-600'
                  }`}
                >
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center space-x-3">
                      <span className={`px-2 py-1 rounded text-xs font-medium ${getSeverityColor(alert.severity)}`}>
                        {alert.severity.toUpperCase()}
                      </span>
                      <span className="text-sm text-neutral-600 dark:text-neutral-400">
                        {streams.find(s => s.id === alert.streamId)?.name}
                      </span>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <span className="text-xs text-neutral-500">
                        {new Date(alert.timestamp).toLocaleString()}
                      </span>
                      {alert.resolved ? (
                        <CheckCircle className="w-4 h-4 text-green-500" />
                      ) : (
                        <AlertTriangle className="w-4 h-4 text-amber-500" />
                      )}
                    </div>
                  </div>
                  
                  <p className="text-neutral-900 dark:text-neutral-100 mb-3">
                    {alert.message}
                  </p>
                  
                  {!alert.resolved && (
                    <div className="flex items-center space-x-2">
                      <button className="text-sm text-blue-600 hover:text-blue-700 dark:text-blue-400">
                        Acknowledge
                      </button>
                      <button className="text-sm text-green-600 hover:text-green-700 dark:text-green-400">
                        Resolve
                      </button>
                      <button className="text-sm text-neutral-600 hover:text-neutral-700 dark:text-neutral-400">
                        View Details
                      </button>
                    </div>
                  )}
                </div>
              ))}
            </div>
            
            {alerts.length === 0 && (
              <div className="text-center py-12 text-neutral-500 dark:text-neutral-400">
                <CheckCircle className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p>No alerts found</p>
              </div>
            )}
          </div>
        </Card>
      )}
    </div>
  );
};

export default StreamDataManagement;