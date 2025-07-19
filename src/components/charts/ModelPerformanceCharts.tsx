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
  Area,
  AreaChart,
  Cell,
  BarChart,
  Bar
} from 'recharts';
import {
  Activity,
  Target,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Download,
  Maximize2,
  RotateCcw,
  Info
} from 'lucide-react';
import { cn } from '../../utils';

interface ROCPoint {
  fpr: number;
  tpr: number;
  threshold: number;
}

interface ConfusionMatrixData {
  truePositives: number;
  falsePositives: number;
  trueNegatives: number;
  falseNegatives: number;
  classNames?: string[];
}

interface PrecisionRecallPoint {
  precision: number;
  recall: number;
  threshold: number;
}

interface ModelPerformanceChartsProps {
  rocData: ROCPoint[];
  confusionMatrix: ConfusionMatrixData;
  precisionRecallData?: PrecisionRecallPoint[];
  aucScore?: number;
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1Score?: number;
  className?: string;
  interactive?: boolean;
  onThresholdChange?: (threshold: number) => void;
}

const ModelPerformanceCharts: React.FC<ModelPerformanceChartsProps> = ({
  rocData,
  confusionMatrix,
  precisionRecallData,
  aucScore = 0.85,
  accuracy = 0.92,
  precision = 0.88,
  recall = 0.91,
  f1Score = 0.89,
  className,
  interactive = true,
  onThresholdChange
}) => {
  const [selectedThreshold, setSelectedThreshold] = useState<number>(0.5);
  const [activeChart, setActiveChart] = useState<'roc' | 'pr' | 'confusion'>('roc');
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Prepare confusion matrix visualization data
  const confusionMatrixViz = useMemo(() => {
    const { truePositives: tp, falsePositives: fp, trueNegatives: tn, falseNegatives: fn } = confusionMatrix;
    const total = tp + fp + tn + fn;
    
    return [
      [
        { 
          label: 'True Positive', 
          value: tp, 
          percentage: ((tp / total) * 100).toFixed(1),
          type: 'tp',
          row: 0,
          col: 0
        },
        { 
          label: 'False Negative', 
          value: fn, 
          percentage: ((fn / total) * 100).toFixed(1),
          type: 'fn',
          row: 0,
          col: 1
        }
      ],
      [
        { 
          label: 'False Positive', 
          value: fp, 
          percentage: ((fp / total) * 100).toFixed(1),
          type: 'fp',
          row: 1,
          col: 0
        },
        { 
          label: 'True Negative', 
          value: tn, 
          percentage: ((tn / total) * 100).toFixed(1),
          type: 'tn',
          row: 1,
          col: 1
        }
      ]
    ];
  }, [confusionMatrix]);

  const getCellColor = (type: string, intensity: number = 0.8) => {
    const alpha = intensity;
    switch (type) {
      case 'tp':
        return `rgba(34, 197, 94, ${alpha})`;
      case 'tn':
        return `rgba(34, 197, 94, ${alpha * 0.7})`;
      case 'fp':
        return `rgba(239, 68, 68, ${alpha})`;
      case 'fn':
        return `rgba(239, 68, 68, ${alpha * 0.7})`;
      default:
        return `rgba(107, 114, 128, ${alpha})`;
    }
  };

  const ROCTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="bg-white dark:bg-neutral-800 p-4 rounded-lg shadow-lg border border-neutral-200 dark:border-neutral-700"
        >
          <div className="font-semibold text-neutral-900 dark:text-neutral-100 mb-2">
            ROC Curve Point
          </div>
          <div className="space-y-1 text-sm">
            <div>FPR: {data.fpr.toFixed(3)}</div>
            <div>TPR: {data.tpr.toFixed(3)}</div>
            <div>Threshold: {data.threshold.toFixed(3)}</div>
          </div>
        </motion.div>
      );
    }
    return null;
  };

  const handleThresholdSelect = (data: any) => {
    if (data && data.threshold !== undefined) {
      setSelectedThreshold(data.threshold);
      onThresholdChange?.(data.threshold);
    }
  };

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
              <Activity className="w-5 h-5 text-purple-500" />
              Model Performance Analysis
            </h3>
            <p className="text-sm text-neutral-600 dark:text-neutral-400 mt-1">
              Interactive ROC curves, confusion matrix, and performance metrics
            </p>
          </div>
          
          <div className="flex items-center gap-2">
            <button
              onClick={() => setIsFullscreen(!isFullscreen)}
              className="p-2 text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100 transition-colors"
            >
              <Maximize2 className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Performance Metrics Summary */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mt-4">
          <div className="bg-gradient-to-r from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded-lg p-3">
            <div className="text-xs font-medium text-blue-600 dark:text-blue-400 uppercase tracking-wider">
              AUC Score
            </div>
            <div className="text-xl font-bold text-blue-900 dark:text-blue-100">
              {aucScore.toFixed(3)}
            </div>
          </div>
          
          <div className="bg-gradient-to-r from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 rounded-lg p-3">
            <div className="text-xs font-medium text-green-600 dark:text-green-400 uppercase tracking-wider">
              Accuracy
            </div>
            <div className="text-xl font-bold text-green-900 dark:text-green-100">
              {(accuracy * 100).toFixed(1)}%
            </div>
          </div>
          
          <div className="bg-gradient-to-r from-orange-50 to-orange-100 dark:from-orange-900/20 dark:to-orange-800/20 rounded-lg p-3">
            <div className="text-xs font-medium text-orange-600 dark:text-orange-400 uppercase tracking-wider">
              Precision
            </div>
            <div className="text-xl font-bold text-orange-900 dark:text-orange-100">
              {(precision * 100).toFixed(1)}%
            </div>
          </div>
          
          <div className="bg-gradient-to-r from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 rounded-lg p-3">
            <div className="text-xs font-medium text-purple-600 dark:text-purple-400 uppercase tracking-wider">
              F1 Score
            </div>
            <div className="text-xl font-bold text-purple-900 dark:text-purple-100">
              {f1Score.toFixed(3)}
            </div>
          </div>
        </div>

        {/* Chart Selection */}
        <div className="flex gap-2 mt-4">
          <button
            onClick={() => setActiveChart('roc')}
            className={cn(
              'px-4 py-2 rounded-lg text-sm font-medium transition-all',
              activeChart === 'roc'
                ? 'bg-blue-500 text-white shadow-lg'
                : 'bg-neutral-100 dark:bg-neutral-800 text-neutral-700 dark:text-neutral-300 hover:bg-neutral-200 dark:hover:bg-neutral-700'
            )}
          >
            ROC Curve
          </button>
          {precisionRecallData && (
            <button
              onClick={() => setActiveChart('pr')}
              className={cn(
                'px-4 py-2 rounded-lg text-sm font-medium transition-all',
                activeChart === 'pr'
                  ? 'bg-blue-500 text-white shadow-lg'
                  : 'bg-neutral-100 dark:bg-neutral-800 text-neutral-700 dark:text-neutral-300 hover:bg-neutral-200 dark:hover:bg-neutral-700'
              )}
            >
              Precision-Recall
            </button>
          )}
          <button
            onClick={() => setActiveChart('confusion')}
            className={cn(
              'px-4 py-2 rounded-lg text-sm font-medium transition-all',
              activeChart === 'confusion'
                ? 'bg-blue-500 text-white shadow-lg'
                : 'bg-neutral-100 dark:bg-neutral-800 text-neutral-700 dark:text-neutral-300 hover:bg-neutral-200 dark:hover:bg-neutral-700'
            )}
          >
            Confusion Matrix
          </button>
        </div>
      </div>

      {/* Chart Content */}
      <div className="p-6">
        <AnimatePresence mode="wait">
          {activeChart === 'roc' && (
            <motion.div
              key="roc"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              style={{ width: '100%', height: isFullscreen ? 'calc(100vh - 400px)' : '400px' }}
            >
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={rocData} onClick={handleThresholdSelect}>
                  <defs>
                    <linearGradient id="rocGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.05}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.1} />
                  <XAxis 
                    dataKey="fpr"
                    domain={[0, 1]}
                    type="number"
                    label={{ value: 'False Positive Rate', position: 'insideBottom', offset: -5 }}
                    stroke="#6b7280"
                  />
                  <YAxis 
                    domain={[0, 1]}
                    label={{ value: 'True Positive Rate', angle: -90, position: 'insideLeft' }}
                    stroke="#6b7280"
                  />
                  <Tooltip content={<ROCTooltip />} />
                  <Area
                    type="monotone"
                    dataKey="tpr"
                    stroke="#3b82f6"
                    strokeWidth={3}
                    fill="url(#rocGradient)"
                  />
                  {/* Diagonal reference line */}
                  <Line
                    type="linear"
                    dataKey={() => Math.random()}
                    stroke="#9ca3af"
                    strokeDasharray="5 5"
                    dot={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
              
              <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                <div className="text-sm text-blue-800 dark:text-blue-200">
                  <div className="font-medium">ROC Curve Analysis</div>
                  <div className="mt-1">
                    AUC = {aucScore.toFixed(3)} - {aucScore > 0.9 ? 'Excellent' : aucScore > 0.8 ? 'Good' : aucScore > 0.7 ? 'Fair' : 'Poor'} performance
                  </div>
                  {interactive && (
                    <div className="mt-1 text-xs">
                      Click on the curve to select different thresholds (Current: {selectedThreshold.toFixed(3)})
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          )}

          {activeChart === 'pr' && precisionRecallData && (
            <motion.div
              key="pr"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              style={{ width: '100%', height: isFullscreen ? 'calc(100vh - 400px)' : '400px' }}
            >
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={precisionRecallData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.1} />
                  <XAxis 
                    dataKey="recall"
                    domain={[0, 1]}
                    type="number"
                    label={{ value: 'Recall', position: 'insideBottom', offset: -5 }}
                    stroke="#6b7280"
                  />
                  <YAxis 
                    domain={[0, 1]}
                    label={{ value: 'Precision', angle: -90, position: 'insideLeft' }}
                    stroke="#6b7280"
                  />
                  <Tooltip />
                  <Line
                    type="monotone"
                    dataKey="precision"
                    stroke="#10b981"
                    strokeWidth={3}
                    dot={{ fill: '#10b981', strokeWidth: 2, r: 3 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </motion.div>
          )}

          {activeChart === 'confusion' && (
            <motion.div
              key="confusion"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              className="space-y-6"
            >
              {/* Confusion Matrix Heatmap */}
              <div className="flex justify-center">
                <div className="inline-block">
                  <div className="text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-4 text-center">
                    Confusion Matrix
                  </div>
                  
                  {/* Labels */}
                  <div className="grid grid-cols-3 gap-1 mb-2">
                    <div></div>
                    <div className="text-center text-xs font-medium text-neutral-600 dark:text-neutral-400">
                      Predicted Positive
                    </div>
                    <div className="text-center text-xs font-medium text-neutral-600 dark:text-neutral-400">
                      Predicted Negative
                    </div>
                  </div>
                  
                  {/* Matrix */}
                  <div className="grid grid-cols-3 gap-1">
                    <div className="flex items-center justify-end pr-2 text-xs font-medium text-neutral-600 dark:text-neutral-400">
                      Actual Positive
                    </div>
                    {confusionMatrixViz[0].map((cell, colIdx) => (
                      <motion.div
                        key={`row0-col${colIdx}`}
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        transition={{ delay: colIdx * 0.1 }}
                        className="w-24 h-24 rounded-lg flex flex-col items-center justify-center text-white font-semibold relative overflow-hidden"
                        style={{ backgroundColor: getCellColor(cell.type, 0.9) }}
                      >
                        <div className="text-2xl">{cell.value}</div>
                        <div className="text-xs opacity-90">{cell.percentage}%</div>
                        <div className="text-xs opacity-75 mt-1">{cell.label}</div>
                      </motion.div>
                    ))}
                    
                    <div className="flex items-center justify-end pr-2 text-xs font-medium text-neutral-600 dark:text-neutral-400">
                      Actual Negative
                    </div>
                    {confusionMatrixViz[1].map((cell, colIdx) => (
                      <motion.div
                        key={`row1-col${colIdx}`}
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        transition={{ delay: (colIdx + 2) * 0.1 }}
                        className="w-24 h-24 rounded-lg flex flex-col items-center justify-center text-white font-semibold relative overflow-hidden"
                        style={{ backgroundColor: getCellColor(cell.type, 0.9) }}
                      >
                        <div className="text-2xl">{cell.value}</div>
                        <div className="text-xs opacity-90">{cell.percentage}%</div>
                        <div className="text-xs opacity-75 mt-1">{cell.label}</div>
                      </motion.div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Performance Breakdown */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-400" />
                    <span className="font-medium text-green-900 dark:text-green-100">Correct Predictions</span>
                  </div>
                  <div className="text-2xl font-bold text-green-900 dark:text-green-100">
                    {confusionMatrix.truePositives + confusionMatrix.trueNegatives}
                  </div>
                  <div className="text-sm text-green-700 dark:text-green-300">
                    {(((confusionMatrix.truePositives + confusionMatrix.trueNegatives) / 
                       (confusionMatrix.truePositives + confusionMatrix.trueNegatives + 
                        confusionMatrix.falsePositives + confusionMatrix.falseNegatives)) * 100).toFixed(1)}% accuracy
                  </div>
                </div>
                
                <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <XCircle className="w-5 h-5 text-red-600 dark:text-red-400" />
                    <span className="font-medium text-red-900 dark:text-red-100">Incorrect Predictions</span>
                  </div>
                  <div className="text-2xl font-bold text-red-900 dark:text-red-100">
                    {confusionMatrix.falsePositives + confusionMatrix.falseNegatives}
                  </div>
                  <div className="text-sm text-red-700 dark:text-red-300">
                    {(((confusionMatrix.falsePositives + confusionMatrix.falseNegatives) / 
                       (confusionMatrix.truePositives + confusionMatrix.trueNegatives + 
                        confusionMatrix.falsePositives + confusionMatrix.falseNegatives)) * 100).toFixed(1)}% error rate
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  );
};

export default ModelPerformanceCharts;