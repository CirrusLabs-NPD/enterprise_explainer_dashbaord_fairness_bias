import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  BarChart3,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  Shield,
  CheckCircle,
  Clock,
  DollarSign,
  Users,
  Activity,
  Target,
  Bell,
} from 'lucide-react';
import Card from '../components/common/Card';
import MetricCard from '../components/common/MetricCard';

const ExecutiveDashboard: React.FC = () => {
  const [selectedPeriod, setSelectedPeriod] = useState('monthly');
  const [realTimeMetrics, setRealTimeMetrics] = useState({
    modelsInProduction: 12,
    totalPredictions: 2.4,
    averageAccuracy: 87.3,
    biasScore: 0.08,
    complianceStatus: 'compliant',
    costPerPrediction: 0.002,
    revenueImpact: 1.2,
    alertsCount: 3,
  });

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setRealTimeMetrics(prev => ({
        ...prev,
        totalPredictions: prev.totalPredictions + Math.random() * 0.1,
        averageAccuracy: Math.max(75, Math.min(95, prev.averageAccuracy + (Math.random() - 0.5) * 2)),
      }));
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const kpiData = [
    {
      title: 'Models in Production',
      value: realTimeMetrics.modelsInProduction.toString(),
      change: '+2',
      changeType: 'positive' as const,
      icon: Activity,
      description: 'Active ML models serving predictions',
    },
    {
      title: 'Daily Predictions',
      value: `${realTimeMetrics.totalPredictions.toFixed(1)}M`,
      change: '+15.3%',
      changeType: 'positive' as const,
      icon: TrendingUp,
      description: 'Total predictions served today',
    },
    {
      title: 'Average Accuracy',
      value: `${realTimeMetrics.averageAccuracy.toFixed(1)}%`,
      change: realTimeMetrics.averageAccuracy > 85 ? '+1.2%' : '-0.8%',
      changeType: realTimeMetrics.averageAccuracy > 85 ? 'positive' as const : 'negative' as const,
      icon: Target,
      description: 'Weighted average across all models',
    },
    {
      title: 'Bias Score',
      value: realTimeMetrics.biasScore.toFixed(3),
      change: '-12%',
      changeType: 'positive' as const,
      icon: Shield,
      description: 'Lower is better - EU AI Act compliant',
    },
    {
      title: 'Revenue Impact',
      value: `$${realTimeMetrics.revenueImpact.toFixed(1)}M`,
      change: '+18.7%',
      changeType: 'positive' as const,
      icon: DollarSign,
      description: 'Monthly revenue attributed to ML',
    },
    {
      title: 'Cost per Prediction',
      value: `$${realTimeMetrics.costPerPrediction.toFixed(4)}`,
      change: '-5.2%',
      changeType: 'positive' as const,
      icon: TrendingDown,
      description: 'Infrastructure cost optimization',
    },
  ];

  const riskIndicators = [
    {
      type: 'Bias Alert',
      severity: 'Medium',
      count: 1,
      description: 'Gender bias detected in model_v2.3',
      color: 'amber',
    },
    {
      type: 'Drift Alert',
      severity: 'High', 
      count: 2,
      description: 'Significant data drift in user behavior',
      color: 'red',
    },
    {
      type: 'Performance',
      severity: 'Low',
      count: 0,
      description: 'All models performing within SLA',
      color: 'green',
    },
  ];

  const complianceStatus = [
    { framework: 'GDPR', status: 'Compliant', score: 98, lastAudit: '2024-01-15' },
    { framework: 'EU AI Act', status: 'Compliant', score: 95, lastAudit: '2024-01-10' },
    { framework: 'SOC2 Type II', status: 'Compliant', score: 99, lastAudit: '2024-01-20' },
    { framework: 'ISO 27001', status: 'In Progress', score: 87, lastAudit: '2024-01-05' },
  ];

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-neutral-900 dark:text-neutral-100">
            Executive Dashboard
          </h1>
          <p className="text-neutral-600 dark:text-neutral-400 mt-1">
            Real-time ML operations overview and business metrics
          </p>
        </div>
        
        <div className="flex items-center space-x-4">
          <select
            value={selectedPeriod}
            onChange={(e) => setSelectedPeriod(e.target.value)}
            className="px-4 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
          >
            <option value="daily">Daily</option>
            <option value="weekly">Weekly</option>
            <option value="monthly">Monthly</option>
            <option value="quarterly">Quarterly</option>
          </select>
          
          <div className="flex items-center space-x-2 px-3 py-2 bg-green-50 dark:bg-green-900/20 rounded-lg">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            <span className="text-sm text-green-700 dark:text-green-300 font-medium">
              Live Data
            </span>
          </div>
        </div>
      </div>

      {/* KPI Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {kpiData.map((kpi, index) => (
          <motion.div
            key={kpi.title}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
          >
            <MetricCard
              title={kpi.title}
              value={kpi.value}
              change={kpi.change}
              changeType={kpi.changeType}
              icon={React.createElement(kpi.icon, { className: "w-5 h-5" })}
              description={kpi.description}
            />
          </motion.div>
        ))}
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Risk & Alerts Panel */}
        <Card className="lg:col-span-1">
          <div className="p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                Risk Indicators
              </h3>
              <Bell className="w-5 h-5 text-neutral-500" />
            </div>
            
            <div className="space-y-4">
              {riskIndicators.map((risk, index) => (
                <div
                  key={index}
                  className={`p-4 rounded-lg border-l-4 ${
                    risk.color === 'red' 
                      ? 'border-red-500 bg-red-50 dark:bg-red-900/20'
                      : risk.color === 'amber'
                      ? 'border-amber-500 bg-amber-50 dark:bg-amber-900/20'
                      : 'border-green-500 bg-green-50 dark:bg-green-900/20'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium text-neutral-900 dark:text-neutral-100">
                      {risk.type}
                    </span>
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      risk.severity === 'High'
                        ? 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300'
                        : risk.severity === 'Medium'
                        ? 'bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300'
                        : 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300'
                    }`}>
                      {risk.severity}
                    </span>
                  </div>
                  <p className="text-sm text-neutral-600 dark:text-neutral-400">
                    {risk.description}
                  </p>
                  {risk.count > 0 && (
                    <div className="mt-2 text-xs text-neutral-500 dark:text-neutral-400">
                      {risk.count} active alert{risk.count !== 1 ? 's' : ''}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </Card>

        {/* Performance Trends */}
        <Card className="lg:col-span-2">
          <div className="p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                Model Performance Trends
              </h3>
              <BarChart3 className="w-5 h-5 text-neutral-500" />
            </div>
            
            {/* Mock Chart Placeholder */}
            <div className="h-64 bg-gradient-to-br from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg flex items-center justify-center border-2 border-dashed border-blue-200 dark:border-blue-700">
              <div className="text-center">
                <BarChart3 className="w-12 h-12 text-blue-400 mx-auto mb-2" />
                <p className="text-blue-600 dark:text-blue-400 font-medium">
                  Interactive Performance Chart
                </p>
                <p className="text-sm text-blue-500 dark:text-blue-300 mt-1">
                  Real-time accuracy, latency, and throughput metrics
                </p>
              </div>
            </div>
          </div>
        </Card>
      </div>

      {/* Compliance Status */}
      <Card>
        <div className="p-6">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                Compliance Status
              </h3>
              <p className="text-sm text-neutral-600 dark:text-neutral-400">
                Regulatory compliance across all frameworks
              </p>
            </div>
            <div className="flex items-center space-x-2 px-3 py-2 bg-green-50 dark:bg-green-900/20 rounded-lg">
              <CheckCircle className="w-4 h-4 text-green-600" />
              <span className="text-sm text-green-700 dark:text-green-300 font-medium">
                Overall Compliant
              </span>
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {complianceStatus.map((compliance, index) => (
              <motion.div
                key={compliance.framework}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                className="p-4 border border-neutral-200 dark:border-neutral-700 rounded-lg"
              >
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-medium text-neutral-900 dark:text-neutral-100">
                    {compliance.framework}
                  </h4>
                  <span className={`px-2 py-1 rounded text-xs font-medium ${
                    compliance.status === 'Compliant'
                      ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300'
                      : 'bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300'
                  }`}>
                    {compliance.status}
                  </span>
                </div>
                
                <div className="mb-3">
                  <div className="flex items-center justify-between text-sm mb-1">
                    <span className="text-neutral-600 dark:text-neutral-400">Score</span>
                    <span className="font-medium text-neutral-900 dark:text-neutral-100">
                      {compliance.score}%
                    </span>
                  </div>
                  <div className="w-full bg-neutral-200 dark:bg-neutral-700 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full transition-all duration-500 ${
                        compliance.score >= 95
                          ? 'bg-green-500'
                          : compliance.score >= 85
                          ? 'bg-amber-500'
                          : 'bg-red-500'
                      }`}
                      style={{ width: `${compliance.score}%` }}
                    />
                  </div>
                </div>
                
                <div className="flex items-center text-xs text-neutral-500 dark:text-neutral-400">
                  <Clock className="w-3 h-3 mr-1" />
                  Last audit: {compliance.lastAudit}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </Card>

      {/* Business Impact Summary */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <div className="p-6">
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
              Business Impact Summary
            </h3>
            
            <div className="space-y-4">
              <div className="flex items-center justify-between p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                <div>
                  <p className="font-medium text-green-900 dark:text-green-100">Revenue Attribution</p>
                  <p className="text-sm text-green-700 dark:text-green-300">Monthly ML-driven revenue</p>
                </div>
                <div className="text-right">
                  <p className="text-2xl font-bold text-green-600 dark:text-green-400">$1.2M</p>
                  <p className="text-sm text-green-500">+18.7% MoM</p>
                </div>
              </div>
              
              <div className="flex items-center justify-between p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                <div>
                  <p className="font-medium text-blue-900 dark:text-blue-100">Cost Savings</p>
                  <p className="text-sm text-blue-700 dark:text-blue-300">Automation efficiency gains</p>
                </div>
                <div className="text-right">
                  <p className="text-2xl font-bold text-blue-600 dark:text-blue-400">$450K</p>
                  <p className="text-sm text-blue-500">+12.3% MoM</p>
                </div>
              </div>
              
              <div className="flex items-center justify-between p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                <div>
                  <p className="font-medium text-purple-900 dark:text-purple-100">Risk Mitigation</p>
                  <p className="text-sm text-purple-700 dark:text-purple-300">Prevented compliance violations</p>
                </div>
                <div className="text-right">
                  <p className="text-2xl font-bold text-purple-600 dark:text-purple-400">$2.1M</p>
                  <p className="text-sm text-purple-500">Potential savings</p>
                </div>
              </div>
            </div>
          </div>
        </Card>
        
        <Card>
          <div className="p-6">
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
              Operational Excellence
            </h3>
            
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-neutral-600 dark:text-neutral-400">Model Uptime</span>
                <div className="flex items-center space-x-2">
                  <div className="text-right">
                    <span className="font-semibold text-neutral-900 dark:text-neutral-100">99.97%</span>
                  </div>
                  <CheckCircle className="w-4 h-4 text-green-500" />
                </div>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-neutral-600 dark:text-neutral-400">Average Latency</span>
                <div className="flex items-center space-x-2">
                  <span className="font-semibold text-neutral-900 dark:text-neutral-100">45ms</span>
                  <TrendingDown className="w-4 h-4 text-green-500" />
                </div>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-neutral-600 dark:text-neutral-400">Error Rate</span>
                <div className="flex items-center space-x-2">
                  <span className="font-semibold text-neutral-900 dark:text-neutral-100">0.03%</span>
                  <TrendingDown className="w-4 h-4 text-green-500" />
                </div>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-neutral-600 dark:text-neutral-400">Throughput</span>
                <div className="flex items-center space-x-2">
                  <span className="font-semibold text-neutral-900 dark:text-neutral-100">1.2K RPS</span>
                  <TrendingUp className="w-4 h-4 text-blue-500" />
                </div>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-neutral-600 dark:text-neutral-400">Resource Utilization</span>
                <div className="flex items-center space-x-2">
                  <span className="font-semibold text-neutral-900 dark:text-neutral-100">68%</span>
                  <Activity className="w-4 h-4 text-blue-500" />
                </div>
              </div>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
};

export default ExecutiveDashboard;