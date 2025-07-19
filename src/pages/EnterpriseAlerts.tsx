import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  Bell,
  AlertTriangle,
  CheckCircle,
  Clock,
  Mail,
  MessageSquare,
  Phone,
  Zap,
  Settings,
  Plus,
  Play,
  Pause,
  Edit,
  Trash2,
  BarChart3,
  TrendingUp,
  Shield,
} from 'lucide-react';
import Card from '../components/common/Card';
import MetricCard from '../components/common/MetricCard';

const EnterpriseAlerts: React.FC = () => {
  const [selectedAlertRule, setSelectedAlertRule] = useState<string | null>(null);
  const [showCreateModal, setShowCreateModal] = useState(false);

  // Mock alert data
  const alertOverview = {
    activeAlerts: 7,
    resolvedToday: 23,
    criticalAlerts: 2,
    averageResponseTime: '4.2 min',
    alertRules: 15,
    notificationChannels: 4,
  };

  const activeAlerts = [
    {
      id: 'alert_001',
      title: 'High Bias Detected in Credit Model',
      severity: 'critical',
      model: 'credit_risk_v2.3',
      metric: 'Demographic Parity',
      value: 0.15,
      threshold: 0.10,
      duration: '15 minutes',
      description: 'Gender bias exceeding acceptable threshold',
      channels: ['email', 'slack', 'sms'],
      assignee: 'ML Team',
      created: '2024-01-19T10:30:00Z',
    },
    {
      id: 'alert_002',
      title: 'Data Drift Detected',
      severity: 'high',
      model: 'fraud_detection_v1.8',
      metric: 'Jensen-Shannon Divergence',
      value: 0.28,
      threshold: 0.20,
      duration: '32 minutes',
      description: 'Significant drift in user behavior patterns',
      channels: ['email', 'slack'],
      assignee: 'Data Science Team',
      created: '2024-01-19T10:15:00Z',
    },
    {
      id: 'alert_003',
      title: 'Model Performance Degradation',
      severity: 'medium',
      model: 'recommendation_engine_v3.1',
      metric: 'Accuracy',
      value: 0.78,
      threshold: 0.80,
      duration: '1 hour 12 minutes',
      description: 'Accuracy below expected performance threshold',
      channels: ['email'],
      assignee: 'Platform Team',
      created: '2024-01-19T09:18:00Z',
    },
  ];

  const alertRules = [
    {
      id: 'rule_001',
      name: 'Bias Threshold Monitor',
      metric: 'Demographic Parity',
      condition: 'greater_than',
      threshold: 0.10,
      severity: 'critical',
      models: ['credit_risk_v2.3', 'loan_approval_v1.5'],
      channels: ['email', 'slack', 'sms'],
      isActive: true,
      lastTriggered: '2024-01-19T10:30:00Z',
    },
    {
      id: 'rule_002',
      name: 'Data Drift Alert',
      metric: 'Jensen-Shannon Divergence',
      condition: 'greater_than',
      threshold: 0.20,
      severity: 'high',
      models: ['all'],
      channels: ['email', 'slack'],
      isActive: true,
      lastTriggered: '2024-01-19T10:15:00Z',
    },
    {
      id: 'rule_003',
      name: 'Performance Degradation',
      metric: 'Model Accuracy',
      condition: 'less_than',
      threshold: 0.80,
      severity: 'medium',
      models: ['all'],
      channels: ['email'],
      isActive: true,
      lastTriggered: '2024-01-19T09:18:00Z',
    },
    {
      id: 'rule_004',
      name: 'Compliance Violation',
      metric: 'GDPR Compliance Score',
      condition: 'less_than',
      threshold: 0.95,
      severity: 'critical',
      models: ['all'],
      channels: ['email', 'slack', 'pagerduty'],
      isActive: true,
      lastTriggered: null,
    },
  ];

  const notificationChannels = [
    {
      type: 'email',
      name: 'Email Notifications',
      endpoint: 'ml-alerts@company.com',
      status: 'active',
      lastUsed: '2024-01-19T10:30:00Z',
      deliveryRate: '99.2%',
    },
    {
      type: 'slack',
      name: 'Slack #ml-alerts',
      endpoint: '#ml-alerts',
      status: 'active',
      lastUsed: '2024-01-19T10:30:00Z',
      deliveryRate: '98.7%',
    },
    {
      type: 'sms',
      name: 'SMS Alerts',
      endpoint: '+1-555-****',
      status: 'active',
      lastUsed: '2024-01-19T10:30:00Z',
      deliveryRate: '100%',
    },
    {
      type: 'pagerduty',
      name: 'PagerDuty',
      endpoint: 'ml-team-service',
      status: 'active',
      lastUsed: '2024-01-18T22:15:00Z',
      deliveryRate: '100%',
    },
  ];

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

  const getChannelIcon = (type: string) => {
    switch (type) {
      case 'email':
        return <Mail className="w-4 h-4" />;
      case 'slack':
        return <MessageSquare className="w-4 h-4" />;
      case 'sms':
        return <Phone className="w-4 h-4" />;
      case 'pagerduty':
        return <Zap className="w-4 h-4" />;
      default:
        return <Bell className="w-4 h-4" />;
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-neutral-900 dark:text-neutral-100">
            Enterprise Alerts
          </h1>
          <p className="text-neutral-600 dark:text-neutral-400 mt-1">
            Real-time monitoring and alerting across all ML models and metrics
          </p>
        </div>
        
        <div className="flex items-center space-x-4">
          <button className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors">
            <Plus className="w-4 h-4" />
            <span>Create Alert Rule</span>
          </button>
          
          <button className="flex items-center space-x-2 px-4 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg hover:bg-neutral-50 dark:hover:bg-neutral-800 transition-colors">
            <Settings className="w-4 h-4" />
            <span>Settings</span>
          </button>
        </div>
      </div>

      {/* Alert Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-6">
        <MetricCard
          title="Active Alerts"
          value={alertOverview.activeAlerts.toString()}
          change="+2"
          changeType="negative"
          icon={AlertTriangle}
          description="Currently active alerts"
        />
        
        <MetricCard
          title="Resolved Today"
          value={alertOverview.resolvedToday.toString()}
          change="+5"
          changeType="positive"
          icon={CheckCircle}
          description="Alerts resolved in last 24h"
        />
        
        <MetricCard
          title="Critical Alerts"
          value={alertOverview.criticalAlerts.toString()}
          change="0"
          changeType="neutral"
          icon={Shield}
          description="High priority alerts"
        />
        
        <MetricCard
          title="Avg Response Time"
          value={alertOverview.averageResponseTime}
          change="-0.8 min"
          changeType="positive"
          icon={Clock}
          description="Average time to response"
        />
        
        <MetricCard
          title="Alert Rules"
          value={alertOverview.alertRules.toString()}
          change="+1"
          changeType="positive"
          icon={BarChart3}
          description="Configured alert rules"
        />
        
        <MetricCard
          title="Notification Channels"
          value={alertOverview.notificationChannels.toString()}
          change="0"
          changeType="neutral"
          icon={Bell}
          description="Active notification channels"
        />
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Active Alerts */}
        <Card className="lg:col-span-2">
          <div className="p-6">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                Active Alerts
              </h3>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
                <span className="text-sm text-neutral-500">Live monitoring</span>
              </div>
            </div>
            
            <div className="space-y-4">
              {activeAlerts.map((alert, index) => (
                <motion.div
                  key={alert.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className={`p-4 border-l-4 rounded-lg ${
                    alert.severity === 'critical'
                      ? 'border-red-500 bg-red-50 dark:bg-red-900/20'
                      : alert.severity === 'high'
                      ? 'border-orange-500 bg-orange-50 dark:bg-orange-900/20'
                      : 'border-yellow-500 bg-yellow-50 dark:bg-yellow-900/20'
                  }`}
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex-1">
                      <div className="flex items-center space-x-3 mb-2">
                        <h4 className="font-medium text-neutral-900 dark:text-neutral-100">
                          {alert.title}
                        </h4>
                        <span className={`px-2 py-1 rounded text-xs font-medium ${getSeverityColor(alert.severity)}`}>
                          {alert.severity.toUpperCase()}
                        </span>
                      </div>
                      
                      <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-2">
                        {alert.description}
                      </p>
                      
                      <div className="flex items-center space-x-4 text-xs text-neutral-500">
                        <span>Model: {alert.model}</span>
                        <span>Metric: {alert.metric}</span>
                        <span>Value: {alert.value}</span>
                        <span>Threshold: {alert.threshold}</span>
                      </div>
                    </div>
                    
                    <div className="flex items-center space-x-2 ml-4">
                      {alert.channels.map((channel) => (
                        <div key={channel} className="text-neutral-400">
                          {getChannelIcon(channel)}
                        </div>
                      ))}
                    </div>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-4 text-xs text-neutral-500">
                      <span>Duration: {alert.duration}</span>
                      <span>Assignee: {alert.assignee}</span>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <button className="px-3 py-1 bg-blue-100 hover:bg-blue-200 text-blue-800 rounded text-xs font-medium transition-colors">
                        Acknowledge
                      </button>
                      <button className="px-3 py-1 bg-green-100 hover:bg-green-200 text-green-800 rounded text-xs font-medium transition-colors">
                        Resolve
                      </button>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </Card>

        {/* Notification Channels */}
        <Card>
          <div className="p-6">
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
              Notification Channels
            </h3>
            
            <div className="space-y-4">
              {notificationChannels.map((channel, index) => (
                <motion.div
                  key={channel.type}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="p-4 border border-neutral-200 dark:border-neutral-700 rounded-lg"
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-3">
                      {getChannelIcon(channel.type)}
                      <span className="font-medium text-neutral-900 dark:text-neutral-100">
                        {channel.name}
                      </span>
                    </div>
                    <span className="px-2 py-1 bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300 rounded text-xs font-medium">
                      {channel.status}
                    </span>
                  </div>
                  
                  <div className="text-sm text-neutral-600 dark:text-neutral-400 mb-2">
                    {channel.endpoint}
                  </div>
                  
                  <div className="flex items-center justify-between text-xs text-neutral-500">
                    <span>Delivery: {channel.deliveryRate}</span>
                    <span>Last used: {new Date(channel.lastUsed).toLocaleDateString()}</span>
                  </div>
                </motion.div>
              ))}
            </div>
            
            <button className="w-full mt-4 px-4 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg hover:bg-neutral-50 dark:hover:bg-neutral-800 transition-colors">
              + Add Channel
            </button>
          </div>
        </Card>
      </div>

      {/* Alert Rules */}
      <Card>
        <div className="p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
              Alert Rules
            </h3>
            <button className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors">
              <Plus className="w-4 h-4" />
              <span>New Rule</span>
            </button>
          </div>
          
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-neutral-200 dark:border-neutral-700">
                  <th className="text-left py-3 px-4 font-medium text-neutral-900 dark:text-neutral-100">Rule Name</th>
                  <th className="text-left py-3 px-4 font-medium text-neutral-900 dark:text-neutral-100">Metric</th>
                  <th className="text-left py-3 px-4 font-medium text-neutral-900 dark:text-neutral-100">Condition</th>
                  <th className="text-left py-3 px-4 font-medium text-neutral-900 dark:text-neutral-100">Severity</th>
                  <th className="text-left py-3 px-4 font-medium text-neutral-900 dark:text-neutral-100">Models</th>
                  <th className="text-left py-3 px-4 font-medium text-neutral-900 dark:text-neutral-100">Channels</th>
                  <th className="text-left py-3 px-4 font-medium text-neutral-900 dark:text-neutral-100">Status</th>
                  <th className="text-left py-3 px-4 font-medium text-neutral-900 dark:text-neutral-100">Actions</th>
                </tr>
              </thead>
              <tbody>
                {alertRules.map((rule, index) => (
                  <motion.tr
                    key={rule.id}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: index * 0.1 }}
                    className="border-b border-neutral-100 dark:border-neutral-800 hover:bg-neutral-50 dark:hover:bg-neutral-900/50"
                  >
                    <td className="py-3 px-4">
                      <span className="font-medium text-neutral-900 dark:text-neutral-100">
                        {rule.name}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-neutral-600 dark:text-neutral-400">
                      {rule.metric}
                    </td>
                    <td className="py-3 px-4 text-neutral-600 dark:text-neutral-400">
                      {rule.condition.replace('_', ' ')} {rule.threshold}
                    </td>
                    <td className="py-3 px-4">
                      <span className={`px-2 py-1 rounded text-xs font-medium ${getSeverityColor(rule.severity)}`}>
                        {rule.severity}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-neutral-600 dark:text-neutral-400">
                      {Array.isArray(rule.models) && rule.models.includes('all') ? 'All Models' : `${rule.models.length} models`}
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex items-center space-x-1">
                        {rule.channels.map((channel) => (
                          <div key={channel} className="text-neutral-400">
                            {getChannelIcon(channel)}
                          </div>
                        ))}
                      </div>
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex items-center space-x-2">
                        {rule.isActive ? (
                          <Play className="w-4 h-4 text-green-500" />
                        ) : (
                          <Pause className="w-4 h-4 text-gray-500" />
                        )}
                        <span className={`text-sm ${rule.isActive ? 'text-green-600' : 'text-gray-500'}`}>
                          {rule.isActive ? 'Active' : 'Paused'}
                        </span>
                      </div>
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex items-center space-x-2">
                        <button className="text-blue-600 hover:text-blue-800">
                          <Edit className="w-4 h-4" />
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

      {/* Alert Trends */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <div className="p-6">
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
              Alert Volume Trends
            </h3>
            
            {/* Mock Chart Placeholder */}
            <div className="h-64 bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-lg flex items-center justify-center border-2 border-dashed border-blue-200 dark:border-blue-700">
              <div className="text-center">
                <TrendingUp className="w-12 h-12 text-blue-400 mx-auto mb-2" />
                <p className="text-blue-600 dark:text-blue-400 font-medium">
                  Alert Volume Chart
                </p>
                <p className="text-sm text-blue-500 dark:text-blue-300 mt-1">
                  7-day trend analysis
                </p>
              </div>
            </div>
          </div>
        </Card>
        
        <Card>
          <div className="p-6">
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
              Response Time Metrics
            </h3>
            
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-neutral-600 dark:text-neutral-400">Average Response Time</span>
                <span className="font-semibold text-neutral-900 dark:text-neutral-100">4.2 min</span>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-neutral-600 dark:text-neutral-400">Fastest Response</span>
                <span className="font-semibold text-neutral-900 dark:text-neutral-100">45 sec</span>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-neutral-600 dark:text-neutral-400">SLA Compliance</span>
                <span className="font-semibold text-green-600 dark:text-green-400">98.5%</span>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-neutral-600 dark:text-neutral-400">Escalated Alerts</span>
                <span className="font-semibold text-neutral-900 dark:text-neutral-100">3</span>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-neutral-600 dark:text-neutral-400">Auto-Resolved</span>
                <span className="font-semibold text-neutral-900 dark:text-neutral-100">12</span>
              </div>
            </div>
            
            <div className="mt-6 p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
              <div className="flex items-center space-x-2">
                <CheckCircle className="w-5 h-5 text-green-600" />
                <span className="text-green-700 dark:text-green-300 font-medium">
                  All critical alerts resolved within SLA
                </span>
              </div>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
};

export default EnterpriseAlerts;