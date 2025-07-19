import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  Activity, Server, Database, Cpu, HardDrive, Wifi, 
  AlertTriangle, CheckCircle, XCircle, RefreshCw,
  BarChart3, Zap, Clock, Users, Globe
} from 'lucide-react';
import Card from '../components/common/Card';
import Button from '../components/common/Button';
import LoadingSpinner from '../components/common/LoadingSpinner';
import { toast } from 'react-hot-toast';

interface SystemMetrics {
  cpu: {
    usage: number;
    cores: number;
    temperature?: number;
  };
  memory: {
    used: number;
    total: number;
    percentage: number;
  };
  disk: {
    used: number;
    total: number;
    percentage: number;
  };
  network: {
    bytesIn: number;
    bytesOut: number;
    connections: number;
  };
}

interface ServiceStatus {
  name: string;
  status: 'healthy' | 'warning' | 'critical' | 'unknown';
  uptime: number;
  lastCheck: string;
  message?: string;
  responseTime?: number;
}

interface ApplicationMetrics {
  activeUsers: number;
  requestsPerMinute: number;
  averageResponseTime: number;
  errorRate: number;
  totalRequests: number;
  uptime: number;
}

const SystemHealth: React.FC = () => {
  const [isLoading, setIsLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  const [autoRefresh, setAutoRefresh] = useState(true);
  
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics>({
    cpu: { usage: 0, cores: 0 },
    memory: { used: 0, total: 0, percentage: 0 },
    disk: { used: 0, total: 0, percentage: 0 },
    network: { bytesIn: 0, bytesOut: 0, connections: 0 }
  });

  const [services, setServices] = useState<ServiceStatus[]>([
    {
      name: 'ML Explainer API',
      status: 'healthy',
      uptime: 99.9,
      lastCheck: new Date().toISOString(),
      responseTime: 45
    },
    {
      name: 'PostgreSQL Database',
      status: 'healthy',
      uptime: 99.8,
      lastCheck: new Date().toISOString(),
      responseTime: 12
    },
    {
      name: 'Redis Cache',
      status: 'healthy',
      uptime: 99.5,
      lastCheck: new Date().toISOString(),
      responseTime: 3
    },
    {
      name: 'Authentication Service',
      status: 'warning',
      uptime: 98.2,
      lastCheck: new Date().toISOString(),
      responseTime: 150,
      message: 'High response time detected'
    },
    {
      name: 'File Storage',
      status: 'healthy',
      uptime: 99.7,
      lastCheck: new Date().toISOString(),
      responseTime: 28
    }
  ]);

  const [appMetrics, setAppMetrics] = useState<ApplicationMetrics>({
    activeUsers: 0,
    requestsPerMinute: 0,
    averageResponseTime: 0,
    errorRate: 0,
    totalRequests: 0,
    uptime: 0
  });

  useEffect(() => {
    fetchHealthData();
    
    if (autoRefresh) {
      const interval = setInterval(fetchHealthData, 30000); // Refresh every 30 seconds
      return () => clearInterval(interval);
    }
  }, [autoRefresh]);

  const fetchHealthData = async () => {
    try {
      // Mock API calls - replace with actual health endpoints
      await Promise.all([
        fetchSystemMetrics(),
        fetchServiceStatus(),
        fetchApplicationMetrics()
      ]);
      
      setLastUpdate(new Date());
    } catch (error) {
      toast.error('Failed to fetch health data');
    } finally {
      setIsLoading(false);
    }
  };

  const fetchSystemMetrics = async () => {
    // Mock system metrics - replace with actual API call
    setSystemMetrics({
      cpu: {
        usage: Math.random() * 80 + 10,
        cores: 8,
        temperature: Math.random() * 20 + 40
      },
      memory: {
        used: Math.random() * 6 + 2,
        total: 16,
        percentage: Math.random() * 60 + 20
      },
      disk: {
        used: Math.random() * 200 + 50,
        total: 500,
        percentage: Math.random() * 40 + 30
      },
      network: {
        bytesIn: Math.random() * 1000000 + 500000,
        bytesOut: Math.random() * 800000 + 300000,
        connections: Math.floor(Math.random() * 100) + 20
      }
    });
  };

  const fetchServiceStatus = async () => {
    // Mock service status updates
    setServices(prev => prev.map(service => ({
      ...service,
      lastCheck: new Date().toISOString(),
      responseTime: Math.random() * 100 + 10,
      uptime: Math.random() * 2 + 98
    })));
  };

  const fetchApplicationMetrics = async () => {
    // Mock application metrics
    setAppMetrics({
      activeUsers: Math.floor(Math.random() * 50) + 10,
      requestsPerMinute: Math.floor(Math.random() * 200) + 50,
      averageResponseTime: Math.random() * 100 + 50,
      errorRate: Math.random() * 2,
      totalRequests: Math.floor(Math.random() * 10000) + 50000,
      uptime: Math.random() * 2 + 98
    });
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'text-green-600 dark:text-green-400';
      case 'warning':
        return 'text-yellow-600 dark:text-yellow-400';
      case 'critical':
        return 'text-red-600 dark:text-red-400';
      default:
        return 'text-neutral-600 dark:text-neutral-400';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
        return <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-400" />;
      case 'warning':
        return <AlertTriangle className="w-5 h-5 text-yellow-600 dark:text-yellow-400" />;
      case 'critical':
        return <XCircle className="w-5 h-5 text-red-600 dark:text-red-400" />;
      default:
        return <AlertTriangle className="w-5 h-5 text-neutral-600 dark:text-neutral-400" />;
    }
  };

  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  const formatUptime = (uptime: number): string => {
    if (uptime >= 99) return `${uptime.toFixed(1)}%`;
    if (uptime >= 95) return `${uptime.toFixed(1)}%`;
    return `${uptime.toFixed(1)}%`;
  };

  if (isLoading) {
    return <LoadingSpinner fullScreen size="xl" message="Loading system health..." />;
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-neutral-900 dark:text-white">
            System Health
          </h1>
          <p className="text-neutral-600 dark:text-neutral-400 mt-1">
            Monitor system performance and service status
          </p>
        </div>
        <div className="flex items-center space-x-3">
          <div className="text-sm text-neutral-600 dark:text-neutral-400">
            Last updated: {lastUpdate.toLocaleTimeString()}
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setAutoRefresh(!autoRefresh)}
          >
            <Activity className={`w-4 h-4 mr-2 ${autoRefresh ? 'text-green-500' : 'text-neutral-400'}`} />
            Auto Refresh
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={fetchHealthData}
          >
            <RefreshCw className="w-4 h-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Overall Status */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="p-4">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-green-100 dark:bg-green-900/20 rounded-lg">
              <CheckCircle className="w-6 h-6 text-green-600 dark:text-green-400" />
            </div>
            <div>
              <p className="text-sm text-neutral-600 dark:text-neutral-400">System Status</p>
              <p className="text-lg font-semibold text-green-600 dark:text-green-400">Healthy</p>
            </div>
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-blue-100 dark:bg-blue-900/20 rounded-lg">
              <Users className="w-6 h-6 text-blue-600 dark:text-blue-400" />
            </div>
            <div>
              <p className="text-sm text-neutral-600 dark:text-neutral-400">Active Users</p>
              <p className="text-lg font-semibold text-neutral-900 dark:text-white">
                {appMetrics.activeUsers}
              </p>
            </div>
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-purple-100 dark:bg-purple-900/20 rounded-lg">
              <BarChart3 className="w-6 h-6 text-purple-600 dark:text-purple-400" />
            </div>
            <div>
              <p className="text-sm text-neutral-600 dark:text-neutral-400">Requests/min</p>
              <p className="text-lg font-semibold text-neutral-900 dark:text-white">
                {appMetrics.requestsPerMinute}
              </p>
            </div>
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-orange-100 dark:bg-orange-900/20 rounded-lg">
              <Clock className="w-6 h-6 text-orange-600 dark:text-orange-400" />
            </div>
            <div>
              <p className="text-sm text-neutral-600 dark:text-neutral-400">Avg Response</p>
              <p className="text-lg font-semibold text-neutral-900 dark:text-white">
                {appMetrics.averageResponseTime.toFixed(0)}ms
              </p>
            </div>
          </div>
        </Card>
      </div>

      {/* System Metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="p-6">
          <h3 className="text-lg font-semibold text-neutral-900 dark:text-white mb-4">
            System Resources
          </h3>
          <div className="space-y-6">
            {/* CPU */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center space-x-2">
                  <Cpu className="w-4 h-4 text-blue-500" />
                  <span className="text-sm font-medium text-neutral-700 dark:text-neutral-300">
                    CPU Usage
                  </span>
                </div>
                <span className="text-sm text-neutral-600 dark:text-neutral-400">
                  {systemMetrics.cpu.usage.toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-neutral-200 dark:bg-neutral-700 rounded-full h-2">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${systemMetrics.cpu.usage}%` }}
                  transition={{ duration: 0.5 }}
                  className={`h-2 rounded-full ${
                    systemMetrics.cpu.usage > 80 
                      ? 'bg-red-500' 
                      : systemMetrics.cpu.usage > 60 
                        ? 'bg-yellow-500' 
                        : 'bg-green-500'
                  }`}
                />
              </div>
              <div className="flex justify-between text-xs text-neutral-500 dark:text-neutral-400 mt-1">
                <span>{systemMetrics.cpu.cores} cores</span>
                {systemMetrics.cpu.temperature && (
                  <span>{systemMetrics.cpu.temperature.toFixed(1)}°C</span>
                )}
              </div>
            </div>

            {/* Memory */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center space-x-2">
                  <Server className="w-4 h-4 text-green-500" />
                  <span className="text-sm font-medium text-neutral-700 dark:text-neutral-300">
                    Memory Usage
                  </span>
                </div>
                <span className="text-sm text-neutral-600 dark:text-neutral-400">
                  {systemMetrics.memory.percentage.toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-neutral-200 dark:bg-neutral-700 rounded-full h-2">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${systemMetrics.memory.percentage}%` }}
                  transition={{ duration: 0.5 }}
                  className={`h-2 rounded-full ${
                    systemMetrics.memory.percentage > 90 
                      ? 'bg-red-500' 
                      : systemMetrics.memory.percentage > 70 
                        ? 'bg-yellow-500' 
                        : 'bg-green-500'
                  }`}
                />
              </div>
              <div className="flex justify-between text-xs text-neutral-500 dark:text-neutral-400 mt-1">
                <span>{systemMetrics.memory.used.toFixed(1)} GB used</span>
                <span>{systemMetrics.memory.total} GB total</span>
              </div>
            </div>

            {/* Disk */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center space-x-2">
                  <HardDrive className="w-4 h-4 text-purple-500" />
                  <span className="text-sm font-medium text-neutral-700 dark:text-neutral-300">
                    Disk Usage
                  </span>
                </div>
                <span className="text-sm text-neutral-600 dark:text-neutral-400">
                  {systemMetrics.disk.percentage.toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-neutral-200 dark:bg-neutral-700 rounded-full h-2">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${systemMetrics.disk.percentage}%` }}
                  transition={{ duration: 0.5 }}
                  className={`h-2 rounded-full ${
                    systemMetrics.disk.percentage > 90 
                      ? 'bg-red-500' 
                      : systemMetrics.disk.percentage > 70 
                        ? 'bg-yellow-500' 
                        : 'bg-green-500'
                  }`}
                />
              </div>
              <div className="flex justify-between text-xs text-neutral-500 dark:text-neutral-400 mt-1">
                <span>{systemMetrics.disk.used} GB used</span>
                <span>{systemMetrics.disk.total} GB total</span>
              </div>
            </div>

            {/* Network */}
            <div>
              <div className="flex items-center space-x-2 mb-2">
                <Wifi className="w-4 h-4 text-orange-500" />
                <span className="text-sm font-medium text-neutral-700 dark:text-neutral-300">
                  Network Activity
                </span>
              </div>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <p className="text-neutral-600 dark:text-neutral-400">Incoming</p>
                  <p className="font-medium text-neutral-900 dark:text-white">
                    {formatBytes(systemMetrics.network.bytesIn)}
                  </p>
                </div>
                <div>
                  <p className="text-neutral-600 dark:text-neutral-400">Outgoing</p>
                  <p className="font-medium text-neutral-900 dark:text-white">
                    {formatBytes(systemMetrics.network.bytesOut)}
                  </p>
                </div>
              </div>
              <div className="mt-2">
                <p className="text-xs text-neutral-500 dark:text-neutral-400">
                  {systemMetrics.network.connections} active connections
                </p>
              </div>
            </div>
          </div>
        </Card>

        <Card className="p-6">
          <h3 className="text-lg font-semibold text-neutral-900 dark:text-white mb-4">
            Application Metrics
          </h3>
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="text-center p-4 bg-neutral-50 dark:bg-neutral-800 rounded-lg">
                <div className="text-2xl font-bold text-neutral-900 dark:text-white">
                  {formatUptime(appMetrics.uptime)}
                </div>
                <div className="text-sm text-neutral-600 dark:text-neutral-400">Uptime</div>
              </div>
              <div className="text-center p-4 bg-neutral-50 dark:bg-neutral-800 rounded-lg">
                <div className="text-2xl font-bold text-neutral-900 dark:text-white">
                  {appMetrics.errorRate.toFixed(2)}%
                </div>
                <div className="text-sm text-neutral-600 dark:text-neutral-400">Error Rate</div>
              </div>
            </div>
            
            <div className="p-4 bg-neutral-50 dark:bg-neutral-800 rounded-lg">
              <div className="flex items-center justify-between">
                <span className="text-sm text-neutral-600 dark:text-neutral-400">Total Requests</span>
                <span className="font-medium text-neutral-900 dark:text-white">
                  {appMetrics.totalRequests.toLocaleString()}
                </span>
              </div>
            </div>
          </div>
        </Card>
      </div>

      {/* Service Status */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold text-neutral-900 dark:text-white mb-4">
          Service Status
        </h3>
        <div className="space-y-4">
          {services.map((service, index) => (
            <motion.div
              key={service.name}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="flex items-center justify-between p-4 bg-neutral-50 dark:bg-neutral-800 rounded-lg"
            >
              <div className="flex items-center space-x-3">
                {getStatusIcon(service.status)}
                <div>
                  <h4 className="font-medium text-neutral-900 dark:text-white">
                    {service.name}
                  </h4>
                  {service.message && (
                    <p className="text-sm text-neutral-600 dark:text-neutral-400">
                      {service.message}
                    </p>
                  )}
                </div>
              </div>
              
              <div className="flex items-center space-x-6 text-sm">
                <div className="text-center">
                  <p className="text-neutral-600 dark:text-neutral-400">Uptime</p>
                  <p className={`font-medium ${getStatusColor(service.status)}`}>
                    {formatUptime(service.uptime)}
                  </p>
                </div>
                <div className="text-center">
                  <p className="text-neutral-600 dark:text-neutral-400">Response</p>
                  <p className="font-medium text-neutral-900 dark:text-white">
                    {service.responseTime}ms
                  </p>
                </div>
                <div className="text-center">
                  <p className="text-neutral-600 dark:text-neutral-400">Last Check</p>
                  <p className="font-medium text-neutral-900 dark:text-white">
                    {new Date(service.lastCheck).toLocaleTimeString()}
                  </p>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </Card>
    </div>
  );
};

export default SystemHealth;