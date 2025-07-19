import React, { useMemo, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
  Cell,
  ScatterChart,
  Scatter,
  ComposedChart,
  ReferenceLine
} from 'recharts';
import {
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Activity,
  Bell,
  Eye,
  EyeOff,
  Calendar,
  Filter,
  Download,
  Maximize2,
  RotateCcw,
  Zap,
  Shield
} from 'lucide-react';
import { cn } from '../../utils';

interface DriftPoint {
  timestamp: string;
  value: number;
  threshold: number;
  severity: 'low' | 'medium' | 'high' | 'critical';
  featureName?: string;
  driftType: 'feature' | 'prediction' | 'concept';
}

interface FeatureDrift {
  feature: string;
  currentValue: number;
  baselineValue: number;
  driftScore: number;
  pValue: number;
  alertLevel: 'none' | 'warning' | 'critical';
  trend: 'increasing' | 'decreasing' | 'stable';
  lastUpdated: string;
  distribution?: {
    current: Array<{ value: number; frequency: number }>;
    baseline: Array<{ value: number; frequency: number }>;
  };
}

interface DriftAlert {
  id: string;
  feature: string;
  alertType: 'feature_drift' | 'prediction_drift' | 'concept_drift';
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  timestamp: string;
  acknowledged: boolean;
  suggestedAction?: string;
}

interface DataDriftChartsProps {
  driftTimeSeries: DriftPoint[];
  featureDrifts: FeatureDrift[];
  alerts: DriftAlert[];
  className?: string;
  interactive?: boolean;
  onFeatureSelect?: (feature: string) => void;
  onAlertAcknowledge?: (alertId: string) => void;
  refreshInterval?: number;
}

const DataDriftCharts: React.FC<DataDriftChartsProps> = ({
  driftTimeSeries,
  featureDrifts,
  alerts,
  className,
  interactive = true,
  onFeatureSelect,
  onAlertAcknowledge,
  refreshInterval = 30000
}) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'features' | 'alerts'>('overview');
  const [selectedTimeRange, setSelectedTimeRange] = useState<'1h' | '24h' | '7d' | '30d'>('24h');
  const [selectedFeature, setSelectedFeature] = useState<string | null>(null);
  const [hiddenFeatures, setHiddenFeatures] = useState<Set<string>>(new Set());
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);

  // Filter time series data based on selected range
  const filteredTimeSeries = useMemo(() => {
    const now = new Date();
    const timeRanges = {
      '1h': 60 * 60 * 1000,
      '24h': 24 * 60 * 60 * 1000,
      '7d': 7 * 24 * 60 * 60 * 1000,
      '30d': 30 * 24 * 60 * 60 * 1000
    };
    
    const cutoff = new Date(now.getTime() - timeRanges[selectedTimeRange]);
    
    return driftTimeSeries.filter(point => 
      new Date(point.timestamp) >= cutoff
    );
  }, [driftTimeSeries, selectedTimeRange]);

  // Calculate drift statistics
  const driftStats = useMemo(() => {
    const criticalAlerts = alerts.filter(a => a.severity === 'critical' && !a.acknowledged).length;
    const highAlerts = alerts.filter(a => a.severity === 'high' && !a.acknowledged).length;
    const driftingFeatures = featureDrifts.filter(f => f.alertLevel !== 'none').length;
    const avgDriftScore = featureDrifts.reduce((sum, f) => sum + f.driftScore, 0) / featureDrifts.length;
    
    return {
      criticalAlerts,
      highAlerts,
      driftingFeatures,
      avgDriftScore: avgDriftScore || 0,
      totalFeatures: featureDrifts.length
    };
  }, [alerts, featureDrifts]);

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return '#dc2626';
      case 'high':
        return '#ea580c';
      case 'medium':
        return '#d97706';
      case 'low':
        return '#65a30d';
      default:
        return '#6b7280';
    }
  };

  const getDriftTrendIcon = (trend: string) => {
    switch (trend) {
      case 'increasing':
        return <TrendingUp className="w-4 h-4 text-red-500" />;
      case 'decreasing':
        return <TrendingDown className="w-4 h-4 text-blue-500" />;
      default:
        return <Activity className="w-4 h-4 text-gray-500" />;
    }
  };

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      
      return (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="bg-white dark:bg-neutral-800 p-4 rounded-lg shadow-lg border border-neutral-200 dark:border-neutral-700"
        >
          <div className="font-semibold text-neutral-900 dark:text-neutral-100 mb-2">
            {new Date(label).toLocaleString()}
          </div>
          
          <div className="space-y-2 text-sm">
            <div className="flex justify-between items-center">
              <span className="text-neutral-600 dark:text-neutral-400">Drift Score:</span>
              <span className={cn(
                "font-medium",
                data.severity === 'critical' ? 'text-red-600' :
                data.severity === 'high' ? 'text-orange-600' :
                data.severity === 'medium' ? 'text-yellow-600' :
                'text-green-600'
              )}>
                {data.value.toFixed(4)}
              </span>
            </div>
            
            <div className="flex justify-between items-center">
              <span className="text-neutral-600 dark:text-neutral-400">Threshold:</span>
              <span className="font-medium text-neutral-900 dark:text-neutral-100">
                {data.threshold.toFixed(4)}
              </span>
            </div>
            
            <div className="flex justify-between items-center">
              <span className="text-neutral-600 dark:text-neutral-400">Severity:</span>
              <span className={cn(
                "font-medium capitalize",
                data.severity === 'critical' ? 'text-red-600' :
                data.severity === 'high' ? 'text-orange-600' :
                data.severity === 'medium' ? 'text-yellow-600' :
                'text-green-600'
              )}>
                {data.severity}
              </span>
            </div>
            
            {data.featureName && (
              <div className="flex justify-between items-center">
                <span className="text-neutral-600 dark:text-neutral-400">Feature:</span>
                <span className="font-medium text-blue-600 dark:text-blue-400">
                  {data.featureName}
                </span>
              </div>
            )}
          </div>
        </motion.div>
      );
    }
    return null;
  };

  const handleFeatureClick = (feature: string) => {
    if (interactive) {
      setSelectedFeature(feature);
      onFeatureSelect?.(feature);
    }
  };

  const toggleFeatureVisibility = (feature: string) => {
    const newHidden = new Set(hiddenFeatures);
    if (newHidden.has(feature)) {
      newHidden.delete(feature);
    } else {
      newHidden.add(feature);
    }
    setHiddenFeatures(newHidden);
  };

  const acknowledgeAlert = (alertId: string) => {
    onAlertAcknowledge?.(alertId);
  };

  const renderOverviewTab = () => (
    <div className="space-y-6">
      {/* Summary Statistics */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-gradient-to-r from-red-50 to-red-100 dark:from-red-900/20 dark:to-red-800/20 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs font-medium text-red-600 dark:text-red-400 uppercase tracking-wider">
                Critical Alerts
              </div>
              <div className="text-2xl font-bold text-red-900 dark:text-red-100">
                {driftStats.criticalAlerts}
              </div>
            </div>
            <AlertTriangle className="w-8 h-8 text-red-500" />
          </div>
        </div>
        
        <div className="bg-gradient-to-r from-orange-50 to-orange-100 dark:from-orange-900/20 dark:to-orange-800/20 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs font-medium text-orange-600 dark:text-orange-400 uppercase tracking-wider">
                Drifting Features
              </div>
              <div className="text-2xl font-bold text-orange-900 dark:text-orange-100">
                {driftStats.driftingFeatures}
              </div>
            </div>
            <TrendingUp className="w-8 h-8 text-orange-500" />
          </div>
        </div>
        
        <div className="bg-gradient-to-r from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs font-medium text-blue-600 dark:text-blue-400 uppercase tracking-wider">
                Avg Drift Score
              </div>
              <div className="text-2xl font-bold text-blue-900 dark:text-blue-100">
                {driftStats.avgDriftScore.toFixed(3)}
              </div>
            </div>
            <Activity className="w-8 h-8 text-blue-500" />
          </div>
        </div>
        
        <div className="bg-gradient-to-r from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs font-medium text-green-600 dark:text-green-400 uppercase tracking-wider">
                Monitored Features
              </div>
              <div className="text-2xl font-bold text-green-900 dark:text-green-100">
                {driftStats.totalFeatures}
              </div>
            </div>
            <Shield className="w-8 h-8 text-green-500" />
          </div>
        </div>
      </div>

      {/* Drift Time Series Chart */}
      <div className="bg-white dark:bg-neutral-800 rounded-lg border border-neutral-200 dark:border-neutral-700 p-6">
        <div className="flex items-center justify-between mb-4">
          <h4 className="font-semibold text-neutral-900 dark:text-neutral-100">
            Drift Score Timeline
          </h4>
          <div className="flex items-center gap-2">
            <select
              value={selectedTimeRange}
              onChange={(e) => setSelectedTimeRange(e.target.value as any)}
              className="px-3 py-1 text-sm border border-neutral-300 dark:border-neutral-600 rounded-md bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
            >
              <option value="1h">Last Hour</option>
              <option value="24h">Last 24 Hours</option>
              <option value="7d">Last 7 Days</option>
              <option value="30d">Last 30 Days</option>
            </select>
          </div>
        </div>
        
        <div style={{ width: '100%', height: '300px' }}>
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={filteredTimeSeries}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.1} />
              <XAxis 
                dataKey="timestamp"
                type="category"
                scale="time"
                stroke="#6b7280"
                fontSize={12}
                tickFormatter={(value) => new Date(value).toLocaleTimeString()}
              />
              <YAxis stroke="#6b7280" fontSize={12} />
              <Tooltip content={<CustomTooltip />} />
              
              {/* Threshold line */}
              <ReferenceLine 
                y={0.05} 
                stroke="#ef4444" 
                strokeDasharray="5 5"
                label={{ value: "Alert Threshold", position: "topLeft" }}
              />
              
              {/* Drift score area */}
              <Area
                type="monotone"
                dataKey="value"
                stroke="#3b82f6"
                fill="url(#driftGradient)"
                strokeWidth={2}
              />
              
              <defs>
                <linearGradient id="driftGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.05}/>
                </linearGradient>
              </defs>
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );

  const renderFeaturesTab = () => (
    <div className="space-y-6">
      {/* Feature Drift Heatmap */}
      <div className="bg-white dark:bg-neutral-800 rounded-lg border border-neutral-200 dark:border-neutral-700 p-6">
        <div className="flex items-center justify-between mb-4">
          <h4 className="font-semibold text-neutral-900 dark:text-neutral-100">
            Feature Drift Heatmap
          </h4>
          <div className="text-xs text-neutral-500 dark:text-neutral-400">
            Click features to explore details
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {featureDrifts.map((feature, index) => (
            <motion.div
              key={feature.feature}
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: index * 0.05 }}
              onClick={() => handleFeatureClick(feature.feature)}
              className={cn(
                "p-4 rounded-lg border cursor-pointer transition-all hover:shadow-md",
                feature.alertLevel === 'critical' ? 'bg-red-50 border-red-200 dark:bg-red-900/20 dark:border-red-700' :
                feature.alertLevel === 'warning' ? 'bg-yellow-50 border-yellow-200 dark:bg-yellow-900/20 dark:border-yellow-700' :
                'bg-neutral-50 border-neutral-200 dark:bg-neutral-800 dark:border-neutral-600',
                selectedFeature === feature.feature ? 'ring-2 ring-blue-500' : ''
              )}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="font-medium text-neutral-900 dark:text-neutral-100 text-sm truncate">
                  {feature.feature}
                </span>
                {getDriftTrendIcon(feature.trend)}
              </div>
              
              <div className="space-y-1 text-xs">
                <div className="flex justify-between">
                  <span className="text-neutral-600 dark:text-neutral-400">Drift Score:</span>
                  <span className={cn(
                    "font-medium",
                    feature.driftScore > 0.1 ? 'text-red-600' :
                    feature.driftScore > 0.05 ? 'text-yellow-600' :
                    'text-green-600'
                  )}>
                    {feature.driftScore.toFixed(4)}
                  </span>
                </div>
                
                <div className="flex justify-between">
                  <span className="text-neutral-600 dark:text-neutral-400">p-value:</span>
                  <span className="font-medium text-neutral-900 dark:text-neutral-100">
                    {feature.pValue.toFixed(4)}
                  </span>
                </div>
                
                <div className="flex justify-between">
                  <span className="text-neutral-600 dark:text-neutral-400">Status:</span>
                  <span className={cn(
                    "font-medium capitalize",
                    feature.alertLevel === 'critical' ? 'text-red-600' :
                    feature.alertLevel === 'warning' ? 'text-yellow-600' :
                    'text-green-600'
                  )}>
                    {feature.alertLevel === 'none' ? 'Normal' : feature.alertLevel}
                  </span>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Feature Distribution Comparison */}
      {selectedFeature && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          className="bg-white dark:bg-neutral-800 rounded-lg border border-neutral-200 dark:border-neutral-700 p-6"
        >
          <div className="flex items-center justify-between mb-4">
            <h4 className="font-semibold text-neutral-900 dark:text-neutral-100">
              Distribution Comparison: {selectedFeature}
            </h4>
            <button
              onClick={() => setSelectedFeature(null)}
              className="text-neutral-500 hover:text-neutral-700 dark:hover:text-neutral-300"
            >
              ×
            </button>
          </div>
          
          <div style={{ width: '100%', height: '250px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart
                data={featureDrifts.find(f => f.feature === selectedFeature)?.distribution?.current || []}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.1} />
                <XAxis dataKey="value" stroke="#6b7280" fontSize={12} />
                <YAxis stroke="#6b7280" fontSize={12} />
                <Tooltip />
                
                <Bar dataKey="frequency" fill="#3b82f6" fillOpacity={0.7} />
                
                {/* Baseline overlay would go here */}
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </motion.div>
      )}
    </div>
  );

  const renderAlertsTab = () => (
    <div className="space-y-4">
      {alerts.length === 0 ? (
        <div className="text-center py-12">
          <Shield className="w-12 h-12 text-green-500 mx-auto mb-4" />
          <p className="text-neutral-600 dark:text-neutral-400">
            No active drift alerts. System is operating normally.
          </p>
        </div>
      ) : (
        alerts.map((alert) => (
          <motion.div
            key={alert.id}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className={cn(
              "p-4 rounded-lg border-l-4",
              alert.severity === 'critical' ? 'bg-red-50 border-red-500 dark:bg-red-900/20' :
              alert.severity === 'high' ? 'bg-orange-50 border-orange-500 dark:bg-orange-900/20' :
              alert.severity === 'medium' ? 'bg-yellow-50 border-yellow-500 dark:bg-yellow-900/20' :
              'bg-blue-50 border-blue-500 dark:bg-blue-900/20',
              alert.acknowledged ? 'opacity-60' : ''
            )}
          >
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-1">
                  <span className={cn(
                    "px-2 py-1 rounded text-xs font-medium uppercase",
                    alert.severity === 'critical' ? 'bg-red-100 text-red-800 dark:bg-red-800 dark:text-red-100' :
                    alert.severity === 'high' ? 'bg-orange-100 text-orange-800 dark:bg-orange-800 dark:text-orange-100' :
                    alert.severity === 'medium' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-800 dark:text-yellow-100' :
                    'bg-blue-100 text-blue-800 dark:bg-blue-800 dark:text-blue-100'
                  )}>
                    {alert.severity}
                  </span>
                  <span className="text-sm font-medium text-neutral-900 dark:text-neutral-100">
                    {alert.feature}
                  </span>
                  <span className="text-xs text-neutral-500 dark:text-neutral-400">
                    {new Date(alert.timestamp).toLocaleString()}
                  </span>
                </div>
                
                <p className="text-sm text-neutral-700 dark:text-neutral-300 mb-2">
                  {alert.message}
                </p>
                
                {alert.suggestedAction && (
                  <div className="text-xs text-neutral-600 dark:text-neutral-400 bg-neutral-100 dark:bg-neutral-700 rounded p-2">
                    <strong>Suggested Action:</strong> {alert.suggestedAction}
                  </div>
                )}
              </div>
              
              <div className="flex items-center gap-2">
                {!alert.acknowledged && (
                  <button
                    onClick={() => acknowledgeAlert(alert.id)}
                    className="px-3 py-1 text-xs bg-neutral-200 dark:bg-neutral-700 text-neutral-700 dark:text-neutral-300 rounded hover:bg-neutral-300 dark:hover:bg-neutral-600"
                  >
                    Acknowledge
                  </button>
                )}
                <Bell className={cn(
                  "w-4 h-4",
                  alert.acknowledged ? 'text-neutral-400' : 'text-orange-500'
                )} />
              </div>
            </div>
          </motion.div>
        ))
      )}
    </div>
  );

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn(
        'bg-white dark:bg-neutral-900 rounded-xl border border-neutral-200 dark:border-neutral-700 shadow-sm',
        isFullscreen ? 'fixed inset-4 z-50' : '',
        className
      )}
    >
      {/* Header */}
      <div className="p-6 border-b border-neutral-200 dark:border-neutral-700">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 flex items-center gap-2">
              <Zap className="w-5 h-5 text-yellow-500" />
              Real-time Data Drift Monitoring
            </h3>
            <p className="text-sm text-neutral-600 dark:text-neutral-400 mt-1">
              Monitor feature distributions, detect concept drift, and prevent model degradation
            </p>
          </div>
          
          <div className="flex items-center gap-2">
            <button
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={cn(
                "p-2 rounded-md transition-colors",
                autoRefresh 
                  ? "bg-green-100 text-green-600 dark:bg-green-900/20 dark:text-green-400" 
                  : "text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100"
              )}
              title={autoRefresh ? "Disable auto-refresh" : "Enable auto-refresh"}
            >
              <RotateCcw className="w-4 h-4" />
            </button>
            <button
              onClick={() => setIsFullscreen(!isFullscreen)}
              className="p-2 text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100 transition-colors"
            >
              <Maximize2 className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex gap-1 mt-4">
          {[
            { key: 'overview', label: 'Overview', icon: Activity },
            { key: 'features', label: 'Features', icon: Filter },
            { key: 'alerts', label: 'Alerts', icon: Bell }
          ].map(({ key, label, icon: Icon }) => (
            <button
              key={key}
              onClick={() => setActiveTab(key as any)}
              className={cn(
                'flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all',
                activeTab === key
                  ? 'bg-blue-500 text-white shadow-lg'
                  : 'bg-neutral-100 dark:bg-neutral-800 text-neutral-700 dark:text-neutral-300 hover:bg-neutral-200 dark:hover:bg-neutral-700'
              )}
            >
              <Icon className="w-4 h-4" />
              {label}
              {key === 'alerts' && driftStats.criticalAlerts + driftStats.highAlerts > 0 && (
                <span className="bg-red-500 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center">
                  {driftStats.criticalAlerts + driftStats.highAlerts}
                </span>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="p-6">
        <AnimatePresence mode="wait">
          {activeTab === 'overview' && (
            <motion.div
              key="overview"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
            >
              {renderOverviewTab()}
            </motion.div>
          )}
          
          {activeTab === 'features' && (
            <motion.div
              key="features"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
            >
              {renderFeaturesTab()}
            </motion.div>
          )}
          
          {activeTab === 'alerts' && (
            <motion.div
              key="alerts"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
            >
              {renderAlertsTab()}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  );
};

export default DataDriftCharts;