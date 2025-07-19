import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Upload, 
  Database, 
  AlertCircle, 
  CheckCircle, 
  TrendingUp, 
  TrendingDown,
  Activity,
  BarChart3,
  Settings,
  Download,
  Info,
  Play,
  Users,
  DollarSign,
  Zap,
  Shield,
  Eye,
  Bell,
  GitBranch,
  RefreshCw,
  Target,
  Gauge,
  LineChart
} from 'lucide-react';
import Button from '@/components/common/Button';
import Card from '@/components/common/Card';
import LoadingSpinner from '@/components/common/LoadingSpinner';
import { cn } from '@/utils';

interface ModelInfo {
  model_id: string;
  model_type: string;
  status: string;
  registered_at: string;
  version: string;
  has_tracking_data: boolean;
  business_metrics_count: number;
}

interface PerformanceMetrics {
  accuracy?: number;
  f1_score?: number;
  precision?: number;
  recall?: number;
  roc_auc?: number;
  mse?: number;
  mae?: number;
  r2_score?: number;
}

interface BusinessMetrics {
  total_revenue: number;
  average_conversion_rate: number;
  total_predictions: number;
  revenue_per_prediction: number;
}

interface Alert {
  alert_id: string;
  model_id: string;
  metric: string;
  value: number;
  threshold: number;
  severity: string;
  timestamp: string;
  message: string;
}

interface ABTest {
  test_id: string;
  champion_model_id: string;
  challenger_model_id: string;
  traffic_split: number;
  status: string;
  start_date: string;
  end_date: string;
}

const ModelPerformance: React.FC = () => {
  const [models, setModels] = useState<Record<string, ModelInfo>>({});
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [performanceData, setPerformanceData] = useState<any>(null);
  const [businessMetrics, setBusinessMetrics] = useState<BusinessMetrics | null>(null);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [abTests, setAbTests] = useState<Record<string, ABTest>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'performance' | 'business' | 'explainability' | 'alerts' | 'ab-tests'>('overview');

  // Model registration state
  const [showRegistration, setShowRegistration] = useState(false);
  const [registrationData, setRegistrationData] = useState({
    model_id: '',
    model_type: 'classification',
    business_metrics: ['conversion_rate', 'revenue_per_prediction'],
    performance_thresholds: {
      accuracy: 0.8,
      f1_score: 0.75
    }
  });

  // Load models and data
  useEffect(() => {
    loadModels();
  }, []);

  useEffect(() => {
    if (selectedModel) {
      loadModelData(selectedModel);
    }
  }, [selectedModel]);

  const loadModels = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/v1/model-monitoring/models');
      if (response.ok) {
        const data = await response.json();
        setModels(data.models);
        
        // Select first model if none selected
        if (!selectedModel && Object.keys(data.models).length > 0) {
          setSelectedModel(Object.keys(data.models)[0]);
        }
      }
    } catch (err) {
      setError('Failed to load models');
      console.error('Error loading models:', err);
    } finally {
      setLoading(false);
    }
  };

  const loadModelData = async (modelId: string) => {
    try {
      setLoading(true);
      
      // Load performance dashboard
      const dashboardResponse = await fetch(`/api/v1/model-monitoring/model-health/${modelId}`);
      if (dashboardResponse.ok) {
        const dashboardData = await dashboardResponse.json();
        setPerformanceData(dashboardData);
      }
      
      // Load business metrics
      const businessResponse = await fetch(`/api/v1/model-monitoring/business-metrics/${modelId}`);
      if (businessResponse.ok) {
        const businessData = await businessResponse.json();
        setBusinessMetrics(businessData.business_impact);
      }
      
      // Load alerts
      const alertsResponse = await fetch(`/api/v1/model-monitoring/alerts/${modelId}`);
      if (alertsResponse.ok) {
        const alertsData = await alertsResponse.json();
        setAlerts(alertsData.alerts);
      }
      
      // Load A/B tests
      const abTestsResponse = await fetch('/api/v1/model-monitoring/ab-tests');
      if (abTestsResponse.ok) {
        const abTestsData = await abTestsResponse.json();
        setAbTests(abTestsData.ab_tests);
      }
      
    } catch (err) {
      setError('Failed to load model data');
      console.error('Error loading model data:', err);
    } finally {
      setLoading(false);
    }
  };

  const registerModel = async () => {
    try {
      const formData = new FormData();
      
      // Create a mock model file for demo
      const mockModelFile = new Blob(['mock model content'], { type: 'application/octet-stream' });
      formData.append('model_file', mockModelFile, 'model.pkl');
      formData.append('request_data', JSON.stringify(registrationData));
      
      const response = await fetch('/api/v1/model-monitoring/register-model', {
        method: 'POST',
        body: formData
      });
      
      if (response.ok) {
        await loadModels();
        setShowRegistration(false);
        setSelectedModel(registrationData.model_id);
      } else {
        const errorData = await response.json();
        setError(errorData.detail || 'Failed to register model');
      }
    } catch (err) {
      setError('Failed to register model');
      console.error('Error registering model:', err);
    }
  };

  const MetricCard = ({ 
    title, 
    value, 
    change, 
    icon: Icon, 
    color = 'blue' 
  }: { 
    title: string; 
    value: string | number; 
    change?: number; 
    icon: any; 
    color?: string;
  }) => (
    <Card className="p-4">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-gray-600 dark:text-gray-400">{title}</p>
          <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">{value}</p>
          {change !== undefined && (
            <div className={cn(
              "flex items-center space-x-1 text-sm",
              change > 0 ? "text-green-600" : change < 0 ? "text-red-600" : "text-gray-600"
            )}>
              {change > 0 ? <TrendingUp className="h-3 w-3" /> : change < 0 ? <TrendingDown className="h-3 w-3" /> : null}
              <span>{change > 0 ? '+' : ''}{change?.toFixed(1)}%</span>
            </div>
          )}
        </div>
        <div className={cn(
          "p-3 rounded-full",
          color === 'blue' && "bg-blue-100 dark:bg-blue-900/20",
          color === 'green' && "bg-green-100 dark:bg-green-900/20",
          color === 'orange' && "bg-orange-100 dark:bg-orange-900/20",
          color === 'purple' && "bg-purple-100 dark:bg-purple-900/20"
        )}>
          <Icon className={cn(
            "w-6 h-6",
            color === 'blue' && "text-blue-600 dark:text-blue-400",
            color === 'green' && "text-green-600 dark:text-green-400",
            color === 'orange' && "text-orange-600 dark:text-orange-400",
            color === 'purple' && "text-purple-600 dark:text-purple-400"
          )} />
        </div>
      </div>
    </Card>
  );

  const PerformanceChart = ({ data }: { data: any }) => {
    if (!data || !data.charts?.performance_trends) {
      return (
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">Performance Trends</h3>
          <div className="flex items-center justify-center h-64 text-gray-500">
            No performance trend data available
          </div>
        </Card>
      );
    }

    return (
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">Performance Trends</h3>
        <div className="h-64 flex items-center justify-center">
          <div dangerouslySetInnerHTML={{ __html: data.charts.performance_trends }} />
        </div>
      </Card>
    );
  };

  const AlertsList = ({ alerts }: { alerts: Alert[] }) => (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">Recent Alerts</h3>
        <Button variant="outline" size="sm">
          <Bell className="h-4 w-4 mr-2" />
          Configure
        </Button>
      </div>
      
      {alerts.length === 0 ? (
        <div className="text-center py-8 text-gray-500">
          No alerts found
        </div>
      ) : (
        <div className="space-y-3">
          {alerts.slice(0, 5).map((alert) => (
            <div
              key={alert.alert_id}
              className={cn(
                "p-3 border rounded-lg",
                alert.severity === 'high' && "border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-900/20",
                alert.severity === 'medium' && "border-yellow-200 bg-yellow-50 dark:border-yellow-800 dark:bg-yellow-900/20"
              )}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <AlertCircle className={cn(
                    "h-4 w-4",
                    alert.severity === 'high' ? "text-red-600" : "text-yellow-600"
                  )} />
                  <span className="font-medium">{alert.metric}</span>
                </div>
                <span className="text-sm text-gray-500">
                  {new Date(alert.timestamp).toLocaleDateString()}
                </span>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                {alert.message}
              </p>
            </div>
          ))}
        </div>
      )}
    </Card>
  );

  const ABTestsCard = ({ tests }: { tests: Record<string, ABTest> }) => (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">A/B Tests</h3>
        <Button variant="outline" size="sm">
          <GitBranch className="h-4 w-4 mr-2" />
          Create Test
        </Button>
      </div>
      
      {Object.keys(tests).length === 0 ? (
        <div className="text-center py-8 text-gray-500">
          No A/B tests configured
        </div>
      ) : (
        <div className="space-y-3">
          {Object.entries(tests).slice(0, 3).map(([testId, test]) => (
            <div key={testId} className="p-3 border rounded-lg">
              <div className="flex items-center justify-between">
                <span className="font-medium">{testId}</span>
                <span className={cn(
                  "px-2 py-1 rounded-full text-xs",
                  test.status === 'active' ? "bg-green-100 text-green-800" : "bg-gray-100 text-gray-800"
                )}>
                  {test.status}
                </span>
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                {test.champion_model_id} vs {test.challenger_model_id}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">
                Traffic Split: {(test.traffic_split * 100).toFixed(0)}% to challenger
              </div>
            </div>
          ))}
        </div>
      )}
    </Card>
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            Model Performance Monitoring
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-1">
            Enterprise-grade model monitoring, A/B testing, and business impact analysis
          </p>
        </div>
        <div className="flex items-center space-x-3">
          <Button
            variant="outline"
            onClick={() => selectedModel && loadModelData(selectedModel)}
            className="flex items-center space-x-2"
          >
            <RefreshCw className="h-4 w-4" />
            <span>Refresh</span>
          </Button>
          <Button
            onClick={() => setShowRegistration(true)}
            className="flex items-center space-x-2"
          >
            <Upload className="h-4 w-4" />
            <span>Register Model</span>
          </Button>
        </div>
      </div>

      {/* Model Selection */}
      <Card className="p-6">
        <h2 className="text-lg font-semibold mb-4">Select Model</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {Object.entries(models).map(([modelId, model]) => (
            <button
              key={modelId}
              onClick={() => setSelectedModel(modelId)}
              className={cn(
                "p-4 border-2 rounded-lg text-left transition-colors",
                selectedModel === modelId
                  ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20"
                  : "border-gray-200 dark:border-gray-700 hover:border-gray-300"
              )}
            >
              <div className="flex items-center space-x-3 mb-2">
                <Database className="h-5 w-5 text-blue-600" />
                <span className="font-medium">{modelId}</span>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400">{model.model_type}</p>
              <div className="flex items-center space-x-2 mt-2">
                <span className={cn(
                  "inline-flex items-center px-2 py-1 rounded-full text-xs",
                  model.status === 'active' 
                    ? "bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-300"
                    : "bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-300"
                )}>
                  {model.status}
                </span>
                {model.has_tracking_data && (
                  <span className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-300">
                    tracking
                  </span>
                )}
              </div>
            </button>
          ))}
        </div>
      </Card>

      {/* Tabs */}
      {selectedModel && (
        <Card className="p-6">
          <div className="border-b border-gray-200 dark:border-gray-700">
            <nav className="-mb-px flex space-x-8">
              {[
                { id: 'overview', label: 'Overview', icon: Eye },
                { id: 'performance', label: 'Performance', icon: Gauge },
                { id: 'business', label: 'Business Impact', icon: DollarSign },
                { id: 'explainability', label: 'Explainability', icon: Eye },
                { id: 'alerts', label: 'Alerts', icon: Bell },
                { id: 'ab-tests', label: 'A/B Tests', icon: GitBranch }
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={cn(
                    "flex items-center space-x-2 py-2 px-1 border-b-2 font-medium text-sm",
                    activeTab === tab.id
                      ? "border-blue-500 text-blue-600 dark:text-blue-400"
                      : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
                  )}
                >
                  <tab.icon className="h-4 w-4" />
                  <span>{tab.label}</span>
                </button>
              ))}
            </nav>
          </div>

          <div className="mt-6">
            {activeTab === 'overview' && performanceData && (
              <div className="space-y-6">
                {/* Overview Metrics */}
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  <MetricCard
                    title="Model Health"
                    value={performanceData.performance_summary?.performance_trend || 'Stable'}
                    icon={Activity}
                    color="green"
                  />
                  <MetricCard
                    title="Data Quality"
                    value={performanceData.data_quality?.drift_status || 'Stable'}
                    icon={Shield}
                    color="blue"
                  />
                  <MetricCard
                    title="Business Impact"
                    value={businessMetrics ? `$${businessMetrics.total_revenue.toFixed(0)}` : 'N/A'}
                    icon={DollarSign}
                    color="purple"
                  />
                  <MetricCard
                    title="Active Alerts"
                    value={performanceData.alerts?.recent_alerts || 0}
                    icon={Bell}
                    color="orange"
                  />
                </div>

                {/* Recommendations */}
                {performanceData.recommendations && (
                  <Card className="p-6">
                    <h3 className="text-lg font-semibold mb-4">Recommendations</h3>
                    <div className="space-y-3">
                      {performanceData.recommendations.map((rec: any, index: number) => (
                        <div
                          key={index}
                          className={cn(
                            "p-4 border rounded-lg",
                            rec.type === 'critical' && "border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-900/20",
                            rec.type === 'warning' && "border-yellow-200 bg-yellow-50 dark:border-yellow-800 dark:bg-yellow-900/20",
                            rec.type === 'success' && "border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-900/20"
                          )}
                        >
                          <h4 className="font-medium">{rec.title}</h4>
                          <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{rec.description}</p>
                          <p className="text-sm font-medium mt-2">{rec.action}</p>
                        </div>
                      ))}
                    </div>
                  </Card>
                )}
              </div>
            )}

            {activeTab === 'performance' && (
              <div className="space-y-6">
                <PerformanceChart data={performanceData} />
                
                {performanceData?.performance_summary && (
                  <Card className="p-6">
                    <h3 className="text-lg font-semibold mb-4">Current Performance Metrics</h3>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      {Object.entries(performanceData.performance_summary.current_performance || {}).map(([metric, value]) => (
                        <div key={metric} className="text-center">
                          <p className="text-sm text-gray-600 dark:text-gray-400">{metric.replace('_', ' ').toUpperCase()}</p>
                          <p className="text-lg font-semibold">{typeof value === 'number' ? value.toFixed(3) : value}</p>
                        </div>
                      ))}
                    </div>
                  </Card>
                )}
              </div>
            )}

            {activeTab === 'business' && businessMetrics && (
              <div className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  <MetricCard
                    title="Total Revenue"
                    value={`$${businessMetrics.total_revenue.toFixed(0)}`}
                    icon={DollarSign}
                    color="green"
                  />
                  <MetricCard
                    title="Conversion Rate"
                    value={`${(businessMetrics.average_conversion_rate * 100).toFixed(1)}%`}
                    icon={Target}
                    color="blue"
                  />
                  <MetricCard
                    title="Total Predictions"
                    value={businessMetrics.total_predictions.toLocaleString()}
                    icon={BarChart3}
                    color="purple"
                  />
                  <MetricCard
                    title="Revenue per Prediction"
                    value={`$${businessMetrics.revenue_per_prediction.toFixed(2)}`}
                    icon={TrendingUp}
                    color="orange"
                  />
                </div>
              </div>
            )}

            {activeTab === 'explainability' && (
              <Card className="p-6">
                <h3 className="text-lg font-semibold mb-4">Feature Importance</h3>
                {performanceData?.feature_importance?.current_importance ? (
                  <div className="space-y-3">
                    {Object.entries(performanceData.feature_importance.current_importance).slice(0, 10).map(([feature, importance]) => (
                      <div key={feature} className="flex items-center justify-between">
                        <span className="text-sm font-medium">{feature}</span>
                        <div className="flex items-center space-x-2">
                          <div className="w-32 bg-gray-200 rounded-full h-2">
                            <div
                              className="bg-blue-600 h-2 rounded-full"
                              style={{ width: `${(Number(importance) * 100)}%` }}
                            />
                          </div>
                          <span className="text-sm text-gray-600">{Number(importance).toFixed(3)}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-8 text-gray-500">
                    No explainability data available
                  </div>
                )}
              </Card>
            )}

            {activeTab === 'alerts' && (
              <AlertsList alerts={alerts} />
            )}

            {activeTab === 'ab-tests' && (
              <ABTestsCard tests={abTests} />
            )}
          </div>
        </Card>
      )}

      {/* Model Registration Modal */}
      <AnimatePresence>
        {showRegistration && (
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
              className="bg-white dark:bg-gray-800 rounded-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto"
            >
              <div className="p-6">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-xl font-bold text-gray-900 dark:text-white">Register Model for Monitoring</h2>
                  <Button
                    onClick={() => setShowRegistration(false)}
                    variant="outline"
                    size="sm"
                  >
                    Cancel
                  </Button>
                </div>

                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium mb-1">Model ID</label>
                    <input
                      type="text"
                      value={registrationData.model_id}
                      onChange={(e) => setRegistrationData(prev => ({ ...prev, model_id: e.target.value }))}
                      className="w-full p-2 border rounded-md dark:bg-gray-700 dark:border-gray-600"
                      placeholder="e.g., fraud_detection_v1"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium mb-1">Model Type</label>
                    <select
                      value={registrationData.model_type}
                      onChange={(e) => setRegistrationData(prev => ({ ...prev, model_type: e.target.value }))}
                      className="w-full p-2 border rounded-md dark:bg-gray-700 dark:border-gray-600"
                    >
                      <option value="classification">Classification</option>
                      <option value="regression">Regression</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium mb-1">Performance Thresholds</label>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="block text-xs text-gray-600 mb-1">Accuracy</label>
                        <input
                          type="number"
                          min="0"
                          max="1"
                          step="0.01"
                          value={registrationData.performance_thresholds.accuracy}
                          onChange={(e) => setRegistrationData(prev => ({
                            ...prev,
                            performance_thresholds: {
                              ...prev.performance_thresholds,
                              accuracy: parseFloat(e.target.value)
                            }
                          }))}
                          className="w-full p-2 border rounded-md dark:bg-gray-700 dark:border-gray-600"
                        />
                      </div>
                      <div>
                        <label className="block text-xs text-gray-600 mb-1">F1 Score</label>
                        <input
                          type="number"
                          min="0"
                          max="1"
                          step="0.01"
                          value={registrationData.performance_thresholds.f1_score}
                          onChange={(e) => setRegistrationData(prev => ({
                            ...prev,
                            performance_thresholds: {
                              ...prev.performance_thresholds,
                              f1_score: parseFloat(e.target.value)
                            }
                          }))}
                          className="w-full p-2 border rounded-md dark:bg-gray-700 dark:border-gray-600"
                        />
                      </div>
                    </div>
                  </div>

                  <div className="flex justify-end space-x-3 pt-4">
                    <Button
                      variant="outline"
                      onClick={() => setShowRegistration(false)}
                    >
                      Cancel
                    </Button>
                    <Button
                      onClick={registerModel}
                      disabled={!registrationData.model_id}
                    >
                      Register Model
                    </Button>
                  </div>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Error Display */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
          >
            <Card className="p-4 border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-900/20">
              <div className="flex items-center space-x-3">
                <AlertCircle className="h-5 w-5 text-red-600 dark:text-red-400" />
                <p className="text-red-700 dark:text-red-300">{error}</p>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setError(null)}
                >
                  Dismiss
                </Button>
              </div>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Loading State */}
      {loading && (
        <div className="fixed inset-0 bg-black bg-opacity-25 flex items-center justify-center z-40">
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg flex items-center space-x-3">
            <LoadingSpinner size="sm" />
            <span>Loading model data...</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelPerformance;