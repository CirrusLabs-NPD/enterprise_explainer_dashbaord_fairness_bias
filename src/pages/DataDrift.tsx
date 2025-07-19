import React, { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Upload, 
  FileText, 
  AlertCircle, 
  CheckCircle, 
  TrendingUp, 
  TrendingDown,
  Activity,
  BarChart3,
  Calendar,
  Settings,
  Download,
  Info,
  Play,
  Pause,
  Database,
  Clock,
  Monitor,
  RefreshCw,
  Link
} from 'lucide-react';
import Button from '@/components/common/Button';
import Card from '@/components/common/Card';
import LoadingSpinner from '@/components/common/LoadingSpinner';
import DataDriftCharts from '@/components/charts/DataDriftCharts';
import { cn } from '@/utils';

interface DriftAnalysisResult {
  status: string;
  analysis_id: string;
  html_report: string;
  drift_summary: {
    overall_drift_detected: boolean;
    total_columns: number;
    drifted_columns: number;
    drift_percentage: number;
    column_results: Array<{
      column: string;
      drift_detected: boolean;
      drift_score: number;
      threshold: number;
      stattest_name: string;
    }>;
  };
  data_preview: {
    reference_preview: any[];
    current_preview: any[];
    reference_info: any;
    current_info: any;
  };
  metadata: {
    reference_shape: [number, number];
    current_shape: [number, number];
    columns_analyzed: string[];
    timestamp: string;
  };
}

interface ModelDriftResult {
  status: string;
  analysis_id: string;
  html_report: string;
  model_drift_summary: {
    prediction_drift_detected: boolean;
    prediction_drift_score: number;
    prediction_distribution_change: any;
    performance_change: any;
    drift_explanation: string;
  };
  prediction_distribution_changes: {
    reference_stats: any;
    current_stats: any;
    distribution_tests: any;
    shift_metrics: any;
    interpretation: string;
  };
  performance_degradation: {
    available: boolean;
    task_type?: string;
    reference_performance?: any;
    current_performance?: any;
    degradation?: any;
    interpretation?: string;
  };
  metadata: {
    task_type: string;
    reference_predictions_count: number;
    current_predictions_count: number;
    timestamp: string;
  };
}

interface FileUploadState {
  reference: File | null;
  current: File | null;
  single: File | null;
  referencePredictions: File | null;
  currentPredictions: File | null;
  referenceLabels: File | null;
  currentLabels: File | null;
}

interface AnalysisSettings {
  splitRatio: number;
  dropColumns: string[];
  timeColumn: string;
  referenceStart: string;
  referenceEnd: string;
  currentStart: string;
  currentEnd: string;
  useTimeBasedSplit: boolean;
  taskType: 'classification' | 'regression';
}

const DataDrift: React.FC = () => {
  const [files, setFiles] = useState<FileUploadState>({
    reference: null,
    current: null,
    single: null,
    referencePredictions: null,
    currentPredictions: null,
    referenceLabels: null,
    currentLabels: null
  });
  const [analysisMode, setAnalysisMode] = useState<'dual' | 'single' | 'model' | 'monitor'>('dual');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<DriftAnalysisResult | null>(null);
  const [modelResult, setModelResult] = useState<ModelDriftResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [showReport, setShowReport] = useState(false);
  const [settings, setSettings] = useState<AnalysisSettings>({
    splitRatio: 0.7,
    dropColumns: [],
    timeColumn: '',
    referenceStart: '',
    referenceEnd: '',
    currentStart: '',
    currentEnd: '',
    useTimeBasedSplit: false,
    taskType: 'classification'
  });

  // Monitoring state
  const [dataSources] = useState([
    {
      id: 'postgres-prod',
      name: 'Production Database',
      type: 'PostgreSQL',
      status: 'connected',
      icon: <Database className="w-5 h-5" />,
      connection: 'postgresql://prod:5432/ml_data',
      table: 'model_inputs'
    },
    {
      id: 'api-stream',
      name: 'ML API Stream',
      type: 'REST API',
      status: 'connected',
      icon: <Link className="w-5 h-5" />,
      connection: 'https://api.company.com/ml/predictions',
      table: 'predictions_log'
    },
    {
      id: 'kafka-events',
      name: 'Event Stream',
      type: 'Kafka',
      status: 'connected',
      icon: <Activity className="w-5 h-5" />,
      connection: 'kafka://localhost:9092',
      table: 'user_events'
    }
  ]);
  
  const [selectedDataSource, setSelectedDataSource] = useState<string>('');
  const [monitoringConfig, setMonitoringConfig] = useState({
    referenceDataSource: '',
    referencePeriod: '7d',
    monitoringInterval: '1h',
    driftThreshold: 0.1,
    alertEnabled: true,
    autoRetraining: false
  });
  
  const [monitoringHistory] = useState([
    {
      id: 'monitor-1',
      timestamp: '2024-01-15T10:30:00Z',
      driftDetected: true,
      driftScore: 0.15,
      affectedFeatures: ['age', 'income', 'location'],
      status: 'alert_sent'
    },
    {
      id: 'monitor-2',
      timestamp: '2024-01-15T09:30:00Z',
      driftDetected: false,
      driftScore: 0.05,
      affectedFeatures: [],
      status: 'normal'
    }
  ]);

  // Mock data for the new DataDriftCharts component
  const mockDriftTimeSeries = [
    { timestamp: '2024-01-15T08:00:00Z', value: 0.03, threshold: 0.05, severity: 'low' as const, driftType: 'feature' as const },
    { timestamp: '2024-01-15T09:00:00Z', value: 0.05, threshold: 0.05, severity: 'medium' as const, driftType: 'feature' as const },
    { timestamp: '2024-01-15T10:00:00Z', value: 0.15, threshold: 0.05, severity: 'high' as const, driftType: 'feature' as const, featureName: 'age' },
    { timestamp: '2024-01-15T11:00:00Z', value: 0.08, threshold: 0.05, severity: 'medium' as const, driftType: 'prediction' as const }
  ];

  const mockFeatureDrifts = [
    {
      feature: 'age',
      currentValue: 45.2,
      baselineValue: 42.1,
      driftScore: 0.15,
      pValue: 0.001,
      alertLevel: 'critical' as const,
      trend: 'increasing' as const,
      lastUpdated: '2024-01-15T10:30:00Z'
    },
    {
      feature: 'income',
      currentValue: 65000,
      baselineValue: 63000,
      driftScore: 0.08,
      pValue: 0.02,
      alertLevel: 'warning' as const,
      trend: 'increasing' as const,
      lastUpdated: '2024-01-15T10:30:00Z'
    },
    {
      feature: 'location',
      currentValue: 0.7,
      baselineValue: 0.8,
      driftScore: 0.12,
      pValue: 0.005,
      alertLevel: 'warning' as const,
      trend: 'decreasing' as const,
      lastUpdated: '2024-01-15T10:30:00Z'
    },
    {
      feature: 'education',
      currentValue: 3.2,
      baselineValue: 3.1,
      driftScore: 0.02,
      pValue: 0.45,
      alertLevel: 'none' as const,
      trend: 'stable' as const,
      lastUpdated: '2024-01-15T10:30:00Z'
    }
  ];

  const mockAlerts = [
    {
      id: 'alert-1',
      feature: 'age',
      alertType: 'feature_drift' as const,
      severity: 'critical' as const,
      message: 'Significant drift detected in age feature - distribution has shifted by 15%',
      timestamp: '2024-01-15T10:30:00Z',
      acknowledged: false,
      suggestedAction: 'Review data collection process and consider model retraining'
    },
    {
      id: 'alert-2',
      feature: 'income',
      alertType: 'feature_drift' as const,
      severity: 'high' as const,
      message: 'Income distribution shows moderate drift - monitoring required',
      timestamp: '2024-01-15T10:25:00Z',
      acknowledged: false,
      suggestedAction: 'Monitor for additional drift in related features'
    }
  ];

  const handleFileChange = useCallback((type: keyof FileUploadState, file: File | null) => {
    setFiles(prev => ({ ...prev, [type]: file }));
    setError(null);
    setResult(null);
    setModelResult(null);
  }, []);

  const validateFiles = useCallback(() => {
    if (analysisMode === 'dual') {
      if (!files.reference || !files.current) {
        setError('Please upload both reference and current datasets');
        return false;
      }
    } else if (analysisMode === 'single') {
      if (!files.single) {
        setError('Please upload a dataset to analyze');
        return false;
      }
    } else if (analysisMode === 'model') {
      if (!files.reference || !files.current || !files.referencePredictions || !files.currentPredictions) {
        setError('Please upload reference dataset, current dataset, and prediction files for both periods');
        return false;
      }
    } else if (analysisMode === 'monitor') {
      if (!selectedDataSource) {
        setError('Please select a data source for monitoring');
        return false;
      }
      if (!monitoringConfig.referenceDataSource) {
        setError('Please configure reference data source');
        return false;
      }
    }
    return true;
  }, [analysisMode, files, selectedDataSource, monitoringConfig]);

  const runAnalysis = async () => {
    if (!validateFiles()) return;

    setLoading(true);
    setError(null);

    try {
      let response;
      
      if (analysisMode === 'dual') {
        const formData = new FormData();
        formData.append('reference_file', files.reference!);
        formData.append('current_file', files.current!);
        formData.append('request_data', JSON.stringify({
          drop_columns: settings.dropColumns
        }));

        response = await fetch('/api/v1/data-drift/analyze/upload-datasets', {
          method: 'POST',
          body: formData
        });
      } else if (analysisMode === 'single') {
        const formData = new FormData();
        formData.append('dataset_file', files.single!);
        
        const requestData = {
          split_ratio: settings.splitRatio,
          drop_columns: settings.dropColumns,
          ...(settings.useTimeBasedSplit && {
            time_column: settings.timeColumn,
            reference_start: settings.referenceStart,
            reference_end: settings.referenceEnd,
            current_start: settings.currentStart,
            current_end: settings.currentEnd
          })
        };
        
        formData.append('request_data', JSON.stringify(requestData));

        response = await fetch('/api/v1/data-drift/analyze/single-dataset', {
          method: 'POST',
          body: formData
        });
      } else if (analysisMode === 'model') {
        const formData = new FormData();
        formData.append('reference_file', files.reference!);
        formData.append('current_file', files.current!);
        formData.append('reference_predictions_file', files.referencePredictions!);
        formData.append('current_predictions_file', files.currentPredictions!);
        
        if (files.referenceLabels) {
          formData.append('reference_labels_file', files.referenceLabels);
        }
        if (files.currentLabels) {
          formData.append('current_labels_file', files.currentLabels);
        }
        
        formData.append('request_data', JSON.stringify({
          task_type: settings.taskType,
          drop_columns: settings.dropColumns
        }));

        response = await fetch('/api/v1/data-drift/analyze/model-drift', {
          method: 'POST',
          body: formData
        });
      } else if (analysisMode === 'monitor') {
        // Start monitoring setup
        const monitoringData = {
          data_source_id: selectedDataSource,
          reference_data_source: monitoringConfig.referenceDataSource,
          reference_period: monitoringConfig.referencePeriod,
          monitoring_interval: monitoringConfig.monitoringInterval,
          drift_threshold: monitoringConfig.driftThreshold,
          alert_enabled: monitoringConfig.alertEnabled,
          auto_retraining: monitoringConfig.autoRetraining
        };

        response = await fetch('/api/v1/data-drift/setup-monitoring', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(monitoringData)
        });
      }

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Analysis failed');
      }

      const data = await response.json();
      
      if (analysisMode === 'model') {
        setModelResult(data);
      } else {
        setResult(data);
      }
      setShowReport(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred during analysis');
    } finally {
      setLoading(false);
    }
  };

  const FileUploadArea = ({ 
    type, 
    label, 
    accept = ".csv",
    file,
    onFileChange
  }: {
    type: string;
    label: string;
    accept?: string;
    file: File | null;
    onFileChange: (file: File | null) => void;
  }) => (
    <div className="space-y-2">
      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
        {label}
      </label>
      <div
        className={cn(
          "border-2 border-dashed rounded-lg p-6 text-center transition-colors cursor-pointer",
          file 
            ? "border-green-300 bg-green-50 dark:bg-green-900/20 dark:border-green-600"
            : "border-gray-300 hover:border-gray-400 dark:border-gray-600 dark:hover:border-gray-500"
        )}
        onClick={() => document.getElementById(`file-${type}`)?.click()}
      >
        <input
          id={`file-${type}`}
          type="file"
          accept={accept}
          onChange={(e) => onFileChange(e.target.files?.[0] || null)}
          className="hidden"
        />
        <div className="space-y-2">
          {file ? (
            <>
              <CheckCircle className="mx-auto h-8 w-8 text-green-500" />
              <p className="text-sm font-medium text-green-700 dark:text-green-300">
                {file.name}
              </p>
              <p className="text-xs text-green-600 dark:text-green-400">
                {(file.size / 1024 / 1024).toFixed(2)} MB
              </p>
            </>
          ) : (
            <>
              <Upload className="mx-auto h-8 w-8 text-gray-400" />
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Click to upload CSV file
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-500">
                Supports CSV files up to 100MB
              </p>
            </>
          )}
        </div>
      </div>
    </div>
  );

  const DriftSummaryCard = ({ summary }: { summary: DriftAnalysisResult['drift_summary'] }) => (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6"
    >
      <Card className="p-4">
        <div className="flex items-center space-x-3">
          <div className={cn(
            "p-2 rounded-full",
            summary.overall_drift_detected 
              ? "bg-red-100 dark:bg-red-900/20" 
              : "bg-green-100 dark:bg-green-900/20"
          )}>
            {summary.overall_drift_detected ? (
              <AlertCircle className="h-5 w-5 text-red-600 dark:text-red-400" />
            ) : (
              <CheckCircle className="h-5 w-5 text-green-600 dark:text-green-400" />
            )}
          </div>
          <div>
            <p className="text-sm text-gray-600 dark:text-gray-400">Overall Drift</p>
            <p className={cn(
              "font-semibold",
              summary.overall_drift_detected 
                ? "text-red-600 dark:text-red-400" 
                : "text-green-600 dark:text-green-400"
            )}>
              {summary.overall_drift_detected ? 'Detected' : 'Not Detected'}
            </p>
          </div>
        </div>
      </Card>

      <Card className="p-4">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-blue-100 dark:bg-blue-900/20 rounded-full">
            <BarChart3 className="h-5 w-5 text-blue-600 dark:text-blue-400" />
          </div>
          <div>
            <p className="text-sm text-gray-600 dark:text-gray-400">Total Columns</p>
            <p className="font-semibold text-gray-900 dark:text-gray-100">
              {summary.total_columns}
            </p>
          </div>
        </div>
      </Card>

      <Card className="p-4">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-orange-100 dark:bg-orange-900/20 rounded-full">
            <TrendingUp className="h-5 w-5 text-orange-600 dark:text-orange-400" />
          </div>
          <div>
            <p className="text-sm text-gray-600 dark:text-gray-400">Drifted Columns</p>
            <p className="font-semibold text-gray-900 dark:text-gray-100">
              {summary.drifted_columns}
            </p>
          </div>
        </div>
      </Card>

      <Card className="p-4">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-purple-100 dark:bg-purple-900/20 rounded-full">
            <Activity className="h-5 w-5 text-purple-600 dark:text-purple-400" />
          </div>
          <div>
            <p className="text-sm text-gray-600 dark:text-gray-400">Drift Percentage</p>
            <p className="font-semibold text-gray-900 dark:text-gray-100">
              {summary.drift_percentage.toFixed(1)}%
            </p>
          </div>
        </div>
      </Card>
    </motion.div>
  );

  const ColumnResultsTable = ({ columns }: { columns: DriftAnalysisResult['drift_summary']['column_results'] }) => (
    <Card className="p-6">
      <h3 className="text-lg font-semibold mb-4">Column-wise Drift Analysis</h3>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-200 dark:border-gray-700">
              <th className="text-left py-2 font-medium">Column</th>
              <th className="text-left py-2 font-medium">Status</th>
              <th className="text-left py-2 font-medium">Method</th>
              <th className="text-left py-2 font-medium">Score</th>
              <th className="text-left py-2 font-medium">Threshold</th>
            </tr>
          </thead>
          <tbody>
            {columns.map((col, index) => (
              <tr key={index} className="border-b border-gray-100 dark:border-gray-800">
                <td className="py-2 font-mono text-sm">{col.column}</td>
                <td className="py-2">
                  <span className={cn(
                    "inline-flex items-center px-2 py-1 rounded-full text-xs font-medium",
                    col.drift_detected
                      ? "bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-300"
                      : "bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-300"
                  )}>
                    {col.drift_detected ? 'Drift Detected' : 'No Drift'}
                  </span>
                </td>
                <td className="py-2 text-gray-600 dark:text-gray-400">{col.stattest_name}</td>
                <td className="py-2 font-mono">{col.drift_score?.toFixed(4) || 'N/A'}</td>
                <td className="py-2 font-mono">{col.threshold?.toFixed(4) || 'N/A'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );

  const ModelDriftSummaryCard = ({ summary }: { summary: ModelDriftResult['model_drift_summary'] }) => (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6"
    >
      <Card className="p-4">
        <div className="flex items-center space-x-3">
          <div className={cn(
            "p-2 rounded-full",
            summary.prediction_drift_detected 
              ? "bg-red-100 dark:bg-red-900/20" 
              : "bg-green-100 dark:bg-green-900/20"
          )}>
            {summary.prediction_drift_detected ? (
              <AlertCircle className="h-5 w-5 text-red-600 dark:text-red-400" />
            ) : (
              <CheckCircle className="h-5 w-5 text-green-600 dark:text-green-400" />
            )}
          </div>
          <div>
            <p className="text-sm text-gray-600 dark:text-gray-400">Model Drift</p>
            <p className={cn(
              "font-semibold",
              summary.prediction_drift_detected 
                ? "text-red-600 dark:text-red-400" 
                : "text-green-600 dark:text-green-400"
            )}>
              {summary.prediction_drift_detected ? 'Detected' : 'Not Detected'}
            </p>
          </div>
        </div>
      </Card>

      <Card className="p-4">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-blue-100 dark:bg-blue-900/20 rounded-full">
            <Activity className="h-5 w-5 text-blue-600 dark:text-blue-400" />
          </div>
          <div>
            <p className="text-sm text-gray-600 dark:text-gray-400">Drift Score</p>
            <p className="font-semibold text-gray-900 dark:text-gray-100">
              {summary.prediction_drift_score?.toFixed(4) || 'N/A'}
            </p>
          </div>
        </div>
      </Card>

      <Card className="p-4">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-purple-100 dark:bg-purple-900/20 rounded-full">
            <Info className="h-5 w-5 text-purple-600 dark:text-purple-400" />
          </div>
          <div>
            <p className="text-sm text-gray-600 dark:text-gray-400">Status</p>
            <p className="font-semibold text-gray-900 dark:text-gray-100">
              Model Analysis
            </p>
          </div>
        </div>
      </Card>
    </motion.div>
  );

  const PredictionDistributionCard = ({ changes }: { changes: ModelDriftResult['prediction_distribution_changes'] }) => (
    <Card className="p-6">
      <h3 className="text-lg font-semibold mb-4">Prediction Distribution Changes</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h4 className="font-medium mb-3">Reference Statistics</h4>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span>Mean:</span>
              <span className="font-mono">{changes.reference_stats?.mean?.toFixed(4)}</span>
            </div>
            <div className="flex justify-between">
              <span>Std Dev:</span>
              <span className="font-mono">{changes.reference_stats?.std?.toFixed(4)}</span>
            </div>
            <div className="flex justify-between">
              <span>Min:</span>
              <span className="font-mono">{changes.reference_stats?.min?.toFixed(4)}</span>
            </div>
            <div className="flex justify-between">
              <span>Max:</span>
              <span className="font-mono">{changes.reference_stats?.max?.toFixed(4)}</span>
            </div>
          </div>
        </div>
        
        <div>
          <h4 className="font-medium mb-3">Current Statistics</h4>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span>Mean:</span>
              <span className="font-mono">{changes.current_stats?.mean?.toFixed(4)}</span>
            </div>
            <div className="flex justify-between">
              <span>Std Dev:</span>
              <span className="font-mono">{changes.current_stats?.std?.toFixed(4)}</span>
            </div>
            <div className="flex justify-between">
              <span>Min:</span>
              <span className="font-mono">{changes.current_stats?.min?.toFixed(4)}</span>
            </div>
            <div className="flex justify-between">
              <span>Max:</span>
              <span className="font-mono">{changes.current_stats?.max?.toFixed(4)}</span>
            </div>
          </div>
        </div>
      </div>
      
      <div className="mt-6">
        <h4 className="font-medium mb-3">Distribution Test Results</h4>
        <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <div className="flex justify-between">
              <span>Kolmogorov-Smirnov Statistic:</span>
              <span className="font-mono">{changes.distribution_tests?.kolmogorov_smirnov?.statistic?.toFixed(4)}</span>
            </div>
            <div className="flex justify-between">
              <span>P-value:</span>
              <span className="font-mono">{changes.distribution_tests?.kolmogorov_smirnov?.p_value?.toFixed(4)}</span>
            </div>
          </div>
          <div className="mt-3">
            <p className="text-sm text-gray-600 dark:text-gray-400">
              <strong>Interpretation:</strong> {changes.interpretation}
            </p>
          </div>
        </div>
      </div>
    </Card>
  );

  const PerformanceDegradationCard = ({ performance }: { performance: ModelDriftResult['performance_degradation'] }) => {
    if (!performance.available) {
      return (
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">Performance Analysis</h3>
          <div className="text-center py-8">
            <Info className="mx-auto h-12 w-12 text-gray-400 mb-3" />
            <p className="text-gray-500 dark:text-gray-400">
              {performance.message || 'Performance analysis not available without true labels'}
            </p>
          </div>
        </Card>
      );
    }

    return (
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">Performance Degradation Analysis</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium mb-3">Reference Performance</h4>
            <div className="space-y-2 text-sm">
              {performance.task_type === 'classification' ? (
                <>
                  <div className="flex justify-between">
                    <span>Accuracy:</span>
                    <span className="font-mono">{performance.reference_performance?.accuracy?.toFixed(4)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>F1 Score:</span>
                    <span className="font-mono">{performance.reference_performance?.f1_score?.toFixed(4)}</span>
                  </div>
                </>
              ) : (
                <>
                  <div className="flex justify-between">
                    <span>MSE:</span>
                    <span className="font-mono">{performance.reference_performance?.mse?.toFixed(4)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>R² Score:</span>
                    <span className="font-mono">{performance.reference_performance?.r2_score?.toFixed(4)}</span>
                  </div>
                </>
              )}
            </div>
          </div>
          
          <div>
            <h4 className="font-medium mb-3">Current Performance</h4>
            <div className="space-y-2 text-sm">
              {performance.task_type === 'classification' ? (
                <>
                  <div className="flex justify-between">
                    <span>Accuracy:</span>
                    <span className="font-mono">{performance.current_performance?.accuracy?.toFixed(4)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>F1 Score:</span>
                    <span className="font-mono">{performance.current_performance?.f1_score?.toFixed(4)}</span>
                  </div>
                </>
              ) : (
                <>
                  <div className="flex justify-between">
                    <span>MSE:</span>
                    <span className="font-mono">{performance.current_performance?.mse?.toFixed(4)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>R² Score:</span>
                    <span className="font-mono">{performance.current_performance?.r2_score?.toFixed(4)}</span>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
        
        <div className="mt-6">
          <h4 className="font-medium mb-3">Performance Changes</h4>
          <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              {performance.task_type === 'classification' ? (
                <>
                  <div className="flex justify-between">
                    <span>Accuracy Change:</span>
                    <span className={cn(
                      "font-mono",
                      (performance.degradation?.accuracy_change || 0) < 0 ? "text-red-600" : "text-green-600"
                    )}>
                      {performance.degradation?.accuracy_change > 0 ? '+' : ''}{performance.degradation?.accuracy_change?.toFixed(4)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>F1 Score Change:</span>
                    <span className={cn(
                      "font-mono",
                      (performance.degradation?.f1_change || 0) < 0 ? "text-red-600" : "text-green-600"
                    )}>
                      {performance.degradation?.f1_change > 0 ? '+' : ''}{performance.degradation?.f1_change?.toFixed(4)}
                    </span>
                  </div>
                </>
              ) : (
                <>
                  <div className="flex justify-between">
                    <span>MSE Change:</span>
                    <span className={cn(
                      "font-mono",
                      (performance.degradation?.mse_change || 0) > 0 ? "text-red-600" : "text-green-600"
                    )}>
                      {performance.degradation?.mse_change > 0 ? '+' : ''}{performance.degradation?.mse_change?.toFixed(4)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>R² Score Change:</span>
                    <span className={cn(
                      "font-mono",
                      (performance.degradation?.r2_change || 0) < 0 ? "text-red-600" : "text-green-600"
                    )}>
                      {performance.degradation?.r2_change > 0 ? '+' : ''}{performance.degradation?.r2_change?.toFixed(4)}
                    </span>
                  </div>
                </>
              )}
            </div>
            <div className="mt-3">
              <p className="text-sm text-gray-600 dark:text-gray-400">
                <strong>Interpretation:</strong> {performance.interpretation}
              </p>
            </div>
          </div>
        </div>
      </Card>
    );
  };

  const MonitoringHistoryCard = () => (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">Monitoring History</h3>
        <Button
          variant="outline"
          size="sm"
          className="flex items-center space-x-2"
        >
          <RefreshCw className="h-4 w-4" />
          <span>Refresh</span>
        </Button>
      </div>
      
      <div className="space-y-4">
        {monitoringHistory.map((entry) => (
          <div
            key={entry.id}
            className={cn(
              "p-4 border rounded-lg",
              entry.driftDetected
                ? "border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-900/20"
                : "border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-900/20"
            )}
          >
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-3">
                {entry.driftDetected ? (
                  <AlertCircle className="h-5 w-5 text-red-600 dark:text-red-400" />
                ) : (
                  <CheckCircle className="h-5 w-5 text-green-600 dark:text-green-400" />
                )}
                <span className="font-medium">
                  {entry.driftDetected ? 'Drift Detected' : 'No Drift'}
                </span>
              </div>
              <span className="text-sm text-gray-500">
                {new Date(entry.timestamp).toLocaleString()}
              </span>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
              <div>
                <span className="text-gray-600 dark:text-gray-400">Drift Score:</span>
                <span className="ml-2 font-mono">{entry.driftScore.toFixed(3)}</span>
              </div>
              <div>
                <span className="text-gray-600 dark:text-gray-400">Status:</span>
                <span className={cn(
                  "ml-2 px-2 py-1 rounded-full text-xs",
                  entry.status === 'alert_sent'
                    ? "bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-300"
                    : "bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-300"
                )}>
                  {entry.status.replace('_', ' ')}
                </span>
              </div>
              <div>
                <span className="text-gray-600 dark:text-gray-400">Affected Features:</span>
                <span className="ml-2">{entry.affectedFeatures.length}</span>
              </div>
            </div>
            
            {entry.affectedFeatures.length > 0 && (
              <div className="mt-3">
                <span className="text-sm text-gray-600 dark:text-gray-400">Features: </span>
                <span className="text-sm">{entry.affectedFeatures.join(', ')}</span>
              </div>
            )}
          </div>
        ))}
      </div>
    </Card>
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            Data Drift Analysis
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-1">
            Detect statistical changes in your data over time using Evidently AI
          </p>
        </div>
        <div className="flex items-center space-x-3">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowSettings(!showSettings)}
            className="flex items-center space-x-2"
          >
            <Settings className="h-4 w-4" />
            <span>Settings</span>
          </Button>
          {(result || modelResult) && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowReport(!showReport)}
              className="flex items-center space-x-2"
            >
              <FileText className="h-4 w-4" />
              <span>View Report</span>
            </Button>
          )}
        </div>
      </div>

      {/* Analysis Mode Selection */}
      <Card className="p-6">
        <h2 className="text-lg font-semibold mb-4">Analysis Mode</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <button
            onClick={() => setAnalysisMode('dual')}
            className={cn(
              "p-4 border-2 rounded-lg text-left transition-colors",
              analysisMode === 'dual'
                ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20"
                : "border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600"
            )}
          >
            <div className="flex items-center space-x-3 mb-2">
              <FileText className="h-5 w-5 text-blue-600" />
              <span className="font-medium">Data Drift Analysis</span>
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Compare two separate datasets (reference vs current)
            </p>
          </button>

          <button
            onClick={() => setAnalysisMode('single')}
            className={cn(
              "p-4 border-2 rounded-lg text-left transition-colors",
              analysisMode === 'single'
                ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20"
                : "border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600"
            )}
          >
            <div className="flex items-center space-x-3 mb-2">
              <Calendar className="h-5 w-5 text-green-600" />
              <span className="font-medium">Single Dataset Analysis</span>
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Split one dataset by time or ratio for analysis
            </p>
          </button>

          <button
            onClick={() => setAnalysisMode('model')}
            className={cn(
              "p-4 border-2 rounded-lg text-left transition-colors",
              analysisMode === 'model'
                ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20"
                : "border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600"
            )}
          >
            <div className="flex items-center space-x-3 mb-2">
              <TrendingDown className="h-5 w-5 text-orange-600" />
              <span className="font-medium">Model Drift Analysis</span>
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Analyze how data drift affects model predictions
            </p>
          </button>

          <button
            onClick={() => setAnalysisMode('monitor')}
            className={cn(
              "p-4 border-2 rounded-lg text-left transition-colors",
              analysisMode === 'monitor'
                ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20"
                : "border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600"
            )}
          >
            <div className="flex items-center space-x-3 mb-2">
              <Monitor className="h-5 w-5 text-purple-600" />
              <span className="font-medium">Continuous Monitoring</span>
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Connect data sources for real-time drift monitoring
            </p>
          </button>
        </div>
      </Card>

      {/* File Upload */}
      <Card className="p-6">
        <h2 className="text-lg font-semibold mb-4">Upload Data</h2>
        
        {analysisMode === 'dual' ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <FileUploadArea
              type="reference"
              label="Reference Dataset"
              file={files.reference}
              onFileChange={(file) => handleFileChange('reference', file)}
            />
            <FileUploadArea
              type="current"
              label="Current Dataset"
              file={files.current}
              onFileChange={(file) => handleFileChange('current', file)}
            />
          </div>
        ) : analysisMode === 'single' ? (
          <div className="max-w-md">
            <FileUploadArea
              type="single"
              label="Dataset to Analyze"
              file={files.single}
              onFileChange={(file) => handleFileChange('single', file)}
            />
          </div>
        ) : analysisMode === 'model' ? (
          <div className="space-y-6">
            <div>
              <h3 className="text-md font-medium mb-3">Dataset Files</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <FileUploadArea
                  type="reference"
                  label="Reference Dataset"
                  file={files.reference}
                  onFileChange={(file) => handleFileChange('reference', file)}
                />
                <FileUploadArea
                  type="current"
                  label="Current Dataset"
                  file={files.current}
                  onFileChange={(file) => handleFileChange('current', file)}
                />
              </div>
            </div>
            
            <div>
              <h3 className="text-md font-medium mb-3">Prediction Files (Required)</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <FileUploadArea
                  type="referencePredictions"
                  label="Reference Predictions"
                  file={files.referencePredictions}
                  onFileChange={(file) => handleFileChange('referencePredictions', file)}
                />
                <FileUploadArea
                  type="currentPredictions"
                  label="Current Predictions"
                  file={files.currentPredictions}
                  onFileChange={(file) => handleFileChange('currentPredictions', file)}
                />
              </div>
            </div>
            
            <div>
              <h3 className="text-md font-medium mb-3">Label Files (Optional - for performance analysis)</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <FileUploadArea
                  type="referenceLabels"
                  label="Reference Labels"
                  file={files.referenceLabels}
                  onFileChange={(file) => handleFileChange('referenceLabels', file)}
                />
                <FileUploadArea
                  type="currentLabels"
                  label="Current Labels"
                  file={files.currentLabels}
                  onFileChange={(file) => handleFileChange('currentLabels', file)}
                />
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-6">
            {/* Data Source Selection */}
            <div>
              <h3 className="text-md font-medium mb-3">Select Data Source</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {dataSources.map((source) => (
                  <button
                    key={source.id}
                    onClick={() => setSelectedDataSource(source.id)}
                    className={cn(
                      "p-4 border-2 rounded-lg text-left transition-colors",
                      selectedDataSource === source.id
                        ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20"
                        : "border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600"
                    )}
                  >
                    <div className="flex items-center space-x-3 mb-2">
                      {source.icon}
                      <span className="font-medium">{source.name}</span>
                    </div>
                    <p className="text-sm text-gray-600 dark:text-gray-400">{source.type}</p>
                    <div className="mt-2">
                      <span className={cn(
                        "inline-flex items-center px-2 py-1 rounded-full text-xs",
                        source.status === 'connected' 
                          ? "bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-300"
                          : "bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-300"
                      )}>
                        {source.status}
                      </span>
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Monitoring Configuration */}
            {selectedDataSource && (
              <div>
                <h3 className="text-md font-medium mb-3">Monitoring Configuration</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium mb-1">Reference Data Source</label>
                    <select
                      value={monitoringConfig.referenceDataSource}
                      onChange={(e) => setMonitoringConfig(prev => ({ 
                        ...prev, 
                        referenceDataSource: e.target.value 
                      }))}
                      className="w-full p-2 border rounded-md dark:bg-gray-800 dark:border-gray-600"
                    >
                      <option value="">Select reference data source</option>
                      {dataSources.map((source) => (
                        <option key={source.id} value={source.id}>{source.name}</option>
                      ))}
                    </select>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium mb-1">Reference Period</label>
                    <select
                      value={monitoringConfig.referencePeriod}
                      onChange={(e) => setMonitoringConfig(prev => ({ 
                        ...prev, 
                        referencePeriod: e.target.value 
                      }))}
                      className="w-full p-2 border rounded-md dark:bg-gray-800 dark:border-gray-600"
                    >
                      <option value="1d">Last 1 day</option>
                      <option value="7d">Last 7 days</option>
                      <option value="30d">Last 30 days</option>
                      <option value="90d">Last 90 days</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium mb-1">Monitoring Interval</label>
                    <select
                      value={monitoringConfig.monitoringInterval}
                      onChange={(e) => setMonitoringConfig(prev => ({ 
                        ...prev, 
                        monitoringInterval: e.target.value 
                      }))}
                      className="w-full p-2 border rounded-md dark:bg-gray-800 dark:border-gray-600"
                    >
                      <option value="15m">Every 15 minutes</option>
                      <option value="1h">Every hour</option>
                      <option value="6h">Every 6 hours</option>
                      <option value="24h">Daily</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium mb-1">Drift Threshold</label>
                    <input
                      type="number"
                      min="0"
                      max="1"
                      step="0.01"
                      value={monitoringConfig.driftThreshold}
                      onChange={(e) => setMonitoringConfig(prev => ({ 
                        ...prev, 
                        driftThreshold: parseFloat(e.target.value) 
                      }))}
                      className="w-full p-2 border rounded-md dark:bg-gray-800 dark:border-gray-600"
                    />
                  </div>
                </div>
                
                <div className="mt-4 space-y-3">
                  <div className="flex items-center space-x-3">
                    <input
                      type="checkbox"
                      id="alert-enabled"
                      checked={monitoringConfig.alertEnabled}
                      onChange={(e) => setMonitoringConfig(prev => ({ 
                        ...prev, 
                        alertEnabled: e.target.checked 
                      }))}
                      className="rounded border-gray-300"
                    />
                    <label htmlFor="alert-enabled" className="text-sm font-medium">
                      Enable drift alerts
                    </label>
                  </div>
                  
                  <div className="flex items-center space-x-3">
                    <input
                      type="checkbox"
                      id="auto-retraining"
                      checked={monitoringConfig.autoRetraining}
                      onChange={(e) => setMonitoringConfig(prev => ({ 
                        ...prev, 
                        autoRetraining: e.target.checked 
                      }))}
                      className="rounded border-gray-300"
                    />
                    <label htmlFor="auto-retraining" className="text-sm font-medium">
                      Enable automatic model retraining
                    </label>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </Card>

      {/* Advanced Settings */}
      <AnimatePresence>
        {showSettings && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
          >
            <Card className="p-6">
              <h2 className="text-lg font-semibold mb-4">Advanced Settings</h2>
              
              {analysisMode === 'single' && (
                <div className="space-y-4">
                  <div className="flex items-center space-x-3">
                    <input
                      type="checkbox"
                      id="use-time-split"
                      checked={settings.useTimeBasedSplit}
                      onChange={(e) => setSettings(prev => ({ 
                        ...prev, 
                        useTimeBasedSplit: e.target.checked 
                      }))}
                      className="rounded border-gray-300"
                    />
                    <label htmlFor="use-time-split" className="text-sm font-medium">
                      Use time-based splitting
                    </label>
                  </div>

                  {settings.useTimeBasedSplit ? (
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div>
                        <label className="block text-sm font-medium mb-1">Time Column</label>
                        <input
                          type="text"
                          value={settings.timeColumn}
                          onChange={(e) => setSettings(prev => ({ 
                            ...prev, 
                            timeColumn: e.target.value 
                          }))}
                          className="w-full p-2 border rounded-md dark:bg-gray-800 dark:border-gray-600"
                          placeholder="e.g., timestamp"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium mb-1">Reference Period</label>
                        <div className="space-y-2">
                          <input
                            type="date"
                            value={settings.referenceStart}
                            onChange={(e) => setSettings(prev => ({ 
                              ...prev, 
                              referenceStart: e.target.value 
                            }))}
                            className="w-full p-2 border rounded-md dark:bg-gray-800 dark:border-gray-600"
                          />
                          <input
                            type="date"
                            value={settings.referenceEnd}
                            onChange={(e) => setSettings(prev => ({ 
                              ...prev, 
                              referenceEnd: e.target.value 
                            }))}
                            className="w-full p-2 border rounded-md dark:bg-gray-800 dark:border-gray-600"
                          />
                        </div>
                      </div>
                      <div>
                        <label className="block text-sm font-medium mb-1">Current Period</label>
                        <div className="space-y-2">
                          <input
                            type="date"
                            value={settings.currentStart}
                            onChange={(e) => setSettings(prev => ({ 
                              ...prev, 
                              currentStart: e.target.value 
                            }))}
                            className="w-full p-2 border rounded-md dark:bg-gray-800 dark:border-gray-600"
                          />
                          <input
                            type="date"
                            value={settings.currentEnd}
                            onChange={(e) => setSettings(prev => ({ 
                              ...prev, 
                              currentEnd: e.target.value 
                            }))}
                            className="w-full p-2 border rounded-md dark:bg-gray-800 dark:border-gray-600"
                          />
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div>
                      <label className="block text-sm font-medium mb-1">
                        Split Ratio (Reference/Current): {(settings.splitRatio * 100).toFixed(0)}% / {((1 - settings.splitRatio) * 100).toFixed(0)}%
                      </label>
                      <input
                        type="range"
                        min="0.1"
                        max="0.9"
                        step="0.1"
                        value={settings.splitRatio}
                        onChange={(e) => setSettings(prev => ({ 
                          ...prev, 
                          splitRatio: parseFloat(e.target.value) 
                        }))}
                        className="w-full"
                      />
                    </div>
                  )}
                </div>
              )}

              {analysisMode === 'model' && (
                <div className="mt-4">
                  <label className="block text-sm font-medium mb-1">Task Type</label>
                  <select
                    value={settings.taskType}
                    onChange={(e) => setSettings(prev => ({ 
                      ...prev, 
                      taskType: e.target.value as 'classification' | 'regression'
                    }))}
                    className="w-full p-2 border rounded-md dark:bg-gray-800 dark:border-gray-600"
                  >
                    <option value="classification">Classification</option>
                    <option value="regression">Regression</option>
                  </select>
                </div>
              )}

              <div className="mt-4">
                <label className="block text-sm font-medium mb-1">
                  Columns to Exclude (comma-separated)
                </label>
                <input
                  type="text"
                  value={settings.dropColumns.join(', ')}
                  onChange={(e) => setSettings(prev => ({ 
                    ...prev, 
                    dropColumns: e.target.value.split(',').map(s => s.trim()).filter(Boolean)
                  }))}
                  className="w-full p-2 border rounded-md dark:bg-gray-800 dark:border-gray-600"
                  placeholder="e.g., id, timestamp, metadata"
                />
              </div>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Action Buttons */}
      <div className="flex items-center justify-center space-x-4">
        <Button
          onClick={runAnalysis}
          disabled={loading || !validateFiles()}
          size="lg"
          className="flex items-center space-x-2"
        >
          {loading ? (
            <>
              <LoadingSpinner size="sm" />
              <span>Analyzing...</span>
            </>
          ) : (
            <>
              <Play className="h-5 w-5" />
              <span>
                {analysisMode === 'monitor' ? 'Start Monitoring' : 'Run Drift Analysis'}
              </span>
            </>
          )}
        </Button>
        
        {(result || modelResult) && (
          <Button
            variant="outline"
            size="lg"
            onClick={() => {
              const reportData = result || modelResult;
              const blob = new Blob([reportData!.html_report], { type: 'text/html' });
              const url = URL.createObjectURL(blob);
              const a = document.createElement('a');
              a.href = url;
              a.download = `${analysisMode}_report_${reportData!.analysis_id}.html`;
              a.click();
              URL.revokeObjectURL(url);
            }}
            className="flex items-center space-x-2"
          >
            <Download className="h-5 w-5" />
            <span>Download Report</span>
          </Button>
        )}
      </div>

      {/* Error Message */}
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
              </div>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Results */}
      <AnimatePresence>
        {(result || modelResult) && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            className="space-y-6"
          >
            <div className="flex items-center justify-between">
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
                {analysisMode === 'model' ? 'Model Drift Analysis Results' : 'Data Drift Analysis Results'}
              </h2>
              <div className="text-sm text-gray-500 dark:text-gray-400">
                Analysis ID: {result?.analysis_id || modelResult?.analysis_id}
              </div>
            </div>

            {/* Data Drift Results */}
            {result && analysisMode !== 'model' && (
              <>
                <DriftSummaryCard summary={result.drift_summary} />
                
                {result.drift_summary.column_results.length > 0 && (
                  <ColumnResultsTable columns={result.drift_summary.column_results} />
                )}

                {/* Data Preview */}
                <Card className="p-6">
                  <h3 className="text-lg font-semibold mb-4">Data Preview</h3>
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div>
                      <h4 className="font-medium mb-2">Reference Dataset</h4>
                      <div className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                        Shape: {result.data_preview.reference_info.shape.join(' × ')}
                      </div>
                      <div className="overflow-x-auto">
                        <table className="w-full text-xs border">
                          <thead>
                            <tr className="bg-gray-50 dark:bg-gray-800">
                              {Object.keys(result.data_preview.reference_preview[0] || {}).map(key => (
                                <th key={key} className="p-2 border text-left">{key}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {result.data_preview.reference_preview.map((row, idx) => (
                              <tr key={idx} className="border-t">
                                {Object.values(row).map((value, vidx) => (
                                  <td key={vidx} className="p-2 border">
                                    {value !== null ? String(value) : 'null'}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>

                    <div>
                      <h4 className="font-medium mb-2">Current Dataset</h4>
                      <div className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                        Shape: {result.data_preview.current_info.shape.join(' × ')}
                      </div>
                      <div className="overflow-x-auto">
                        <table className="w-full text-xs border">
                          <thead>
                            <tr className="bg-gray-50 dark:bg-gray-800">
                              {Object.keys(result.data_preview.current_preview[0] || {}).map(key => (
                                <th key={key} className="p-2 border text-left">{key}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {result.data_preview.current_preview.map((row, idx) => (
                              <tr key={idx} className="border-t">
                                {Object.values(row).map((value, vidx) => (
                                  <td key={vidx} className="p-2 border">
                                    {value !== null ? String(value) : 'null'}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>
                </Card>
              </>
            )}

            {/* Monitoring Results */}
            {analysisMode === 'monitor' && (
              <>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                  <Card className="p-4">
                    <div className="flex items-center space-x-3">
                      <div className="p-2 bg-blue-100 dark:bg-blue-900/20 rounded-full">
                        <Monitor className="h-5 w-5 text-blue-600 dark:text-blue-400" />
                      </div>
                      <div>
                        <p className="text-sm text-gray-600 dark:text-gray-400">Monitoring Status</p>
                        <p className="font-semibold text-green-600 dark:text-green-400">Active</p>
                      </div>
                    </div>
                  </Card>

                  <Card className="p-4">
                    <div className="flex items-center space-x-3">
                      <div className="p-2 bg-orange-100 dark:bg-orange-900/20 rounded-full">
                        <Clock className="h-5 w-5 text-orange-600 dark:text-orange-400" />
                      </div>
                      <div>
                        <p className="text-sm text-gray-600 dark:text-gray-400">Next Check</p>
                        <p className="font-semibold text-gray-900 dark:text-gray-100">
                          {monitoringConfig.monitoringInterval}
                        </p>
                      </div>
                    </div>
                  </Card>

                  <Card className="p-4">
                    <div className="flex items-center space-x-3">
                      <div className="p-2 bg-purple-100 dark:bg-purple-900/20 rounded-full">
                        <Activity className="h-5 w-5 text-purple-600 dark:text-purple-400" />
                      </div>
                      <div>
                        <p className="text-sm text-gray-600 dark:text-gray-400">Threshold</p>
                        <p className="font-semibold text-gray-900 dark:text-gray-100">
                          {monitoringConfig.driftThreshold}
                        </p>
                      </div>
                    </div>
                  </Card>
                </div>

                {/* New Enhanced Data Drift Charts */}
                <DataDriftCharts
                  driftTimeSeries={mockDriftTimeSeries}
                  featureDrifts={mockFeatureDrifts}
                  alerts={mockAlerts}
                  interactive={true}
                  onFeatureSelect={(feature) => console.log('Selected feature:', feature)}
                  onAlertAcknowledge={(alertId) => console.log('Acknowledged alert:', alertId)}
                  refreshInterval={30000}
                />

                <MonitoringHistoryCard />
              </>
            )}

            {/* Model Drift Results */}
            {modelResult && analysisMode === 'model' && (
              <>
                <ModelDriftSummaryCard summary={modelResult.model_drift_summary} />
                
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <PredictionDistributionCard changes={modelResult.prediction_distribution_changes} />
                  <PerformanceDegradationCard performance={modelResult.performance_degradation} />
                </div>

                {/* Model Drift Explanation */}
                <Card className="p-6">
                  <h3 className="text-lg font-semibold mb-4">Drift Analysis Explanation</h3>
                  <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg border border-blue-200 dark:border-blue-800">
                    <p className="text-gray-700 dark:text-gray-300">
                      {modelResult.model_drift_summary.drift_explanation}
                    </p>
                  </div>
                </Card>

                {/* Metadata */}
                <Card className="p-6">
                  <h3 className="text-lg font-semibold mb-4">Analysis Metadata</h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                    <div>
                      <span className="font-medium">Task Type:</span>
                      <span className="ml-2 capitalize">{modelResult.metadata.task_type}</span>
                    </div>
                    <div>
                      <span className="font-medium">Reference Predictions:</span>
                      <span className="ml-2">{modelResult.metadata.reference_predictions_count}</span>
                    </div>
                    <div>
                      <span className="font-medium">Current Predictions:</span>
                      <span className="ml-2">{modelResult.metadata.current_predictions_count}</span>
                    </div>
                  </div>
                </Card>
              </>
            )}

            {/* Full Report */}
            <AnimatePresence>
              {showReport && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                >
                  <Card className="p-6">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="text-lg font-semibold">Detailed Report</h3>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setShowReport(false)}
                      >
                        <Pause className="h-4 w-4" />
                      </Button>
                    </div>
                    <div 
                      className="w-full border rounded-lg overflow-hidden"
                      style={{ height: '600px' }}
                    >
                      <iframe
                        srcDoc={(result || modelResult)?.html_report}
                        className="w-full h-full"
                        title="Drift Analysis Report"
                      />
                    </div>
                  </Card>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default DataDrift;