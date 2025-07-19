import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Target, TrendingUp, BarChart3, Download, Settings, Activity } from 'lucide-react';
import Card from '@/components/common/Card';
import Button from '@/components/common/Button';

interface ClassificationMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  auc: number;
  logLoss: number;
}

interface ConfusionMatrix {
  truePositive: number;
  falsePositive: number;
  falseNegative: number;
  trueNegative: number;
}

const ClassificationStats: React.FC = () => {
  const [selectedThreshold, setSelectedThreshold] = useState(0.5);
  // Mock data
  const metrics: ClassificationMetrics = {
    accuracy: 0.847,
    precision: 0.821,
    recall: 0.758,
    f1Score: 0.788,
    auc: 0.912,
    logLoss: 0.234,
  };

  const confusionMatrix: ConfusionMatrix = {
    truePositive: 1521,
    falsePositive: 329,
    falseNegative: 487,
    trueNegative: 2163,
  };
  const total = confusionMatrix.truePositive + confusionMatrix.falsePositive + 
                confusionMatrix.falseNegative + confusionMatrix.trueNegative;

  // ROC curve data points
  const rocData = [
    { fpr: 0.0, tpr: 0.0 },
    { fpr: 0.05, tpr: 0.32 },
    { fpr: 0.12, tpr: 0.58 },
    { fpr: 0.18, tpr: 0.71 },
    { fpr: 0.23, tpr: 0.82 },
    { fpr: 0.31, tpr: 0.89 },
    { fpr: 0.42, tpr: 0.94 },
    { fpr: 0.58, tpr: 0.97 },
    { fpr: 0.73, tpr: 0.99 },
    { fpr: 1.0, tpr: 1.0 },
  ];

  const containerVariants = {
    initial: { opacity: 0 },
    animate: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
      },
    },
  };

  const itemVariants = {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
  };

  const getMetricColor = (value: number) => {
    if (value >= 0.9) return 'text-green-600 dark:text-green-400';
    if (value >= 0.8) return 'text-blue-600 dark:text-blue-400';
    if (value >= 0.7) return 'text-amber-600 dark:text-amber-400';
    return 'text-red-600 dark:text-red-400';
  };

  const getMetricBg = (value: number) => {
    if (value >= 0.9) return 'bg-green-100 dark:bg-green-900/20';
    if (value >= 0.8) return 'bg-blue-100 dark:bg-blue-900/20';
    if (value >= 0.7) return 'bg-amber-100 dark:bg-amber-900/20';
    return 'bg-red-100 dark:bg-red-900/20';
  };

  return (
    <motion.div
      variants={containerVariants}
      initial="initial"
      animate="animate"
      className="space-y-6"
    >
      <motion.div variants={itemVariants} className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-neutral-900 dark:text-neutral-100">
            Classification Statistics
          </h1>
          <p className="text-neutral-600 dark:text-neutral-400 mt-1">
            Detailed analysis of classification model performance and metrics
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            size="sm"
            leftIcon={<Settings className="w-4 h-4" />}
          >
            Settings
          </Button>
          <Button
            variant="outline"
            size="sm"
            leftIcon={<Download className="w-4 h-4" />}
          >
            Export
          </Button>
        </div>
      </motion.div>

      {/* Performance Metrics */}
      <motion.div variants={itemVariants}>
        <Card
          title="Performance Metrics"
          icon={<Activity className="w-5 h-5 text-primary-600 dark:text-primary-400" />}
        >
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            {Object.entries(metrics).map(([key, value]) => (
              <motion.div
                key={key}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: Object.keys(metrics).indexOf(key) * 0.1 }}
                className={`p-4 rounded-lg ${getMetricBg(value)} transition-all hover:scale-105`}
              >
                <div className="text-center">
                  <div className={`text-2xl font-bold ${getMetricColor(value)}`}>
                    {value.toFixed(3)}
                  </div>
                  <div className="text-sm text-neutral-600 dark:text-neutral-400 mt-1 capitalize">
                    {key.replace(/([A-Z])/g, ' $1').trim()}
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </Card>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Confusion Matrix */}
        <motion.div variants={itemVariants}>
          <Card
            title="Confusion Matrix"
            icon={<Target className="w-5 h-5 text-primary-600 dark:text-primary-400" />}
          >
            <div className="space-y-6">
              <div className="flex items-center justify-center">
                <div className="grid grid-cols-3 gap-2 text-center">
                  {/* Headers */}
                  <div></div>
                  <div className="text-sm font-medium text-neutral-600 dark:text-neutral-400">
                    Predicted
                  </div>
                  <div></div>
                  <div></div>
                  <div className="text-xs text-neutral-500 dark:text-neutral-400">Negative</div>
                  <div className="text-xs text-neutral-500 dark:text-neutral-400">Positive</div>
                  
                  {/* Actual row */}
                  <div className="flex items-center justify-center">
                    <div className="text-sm font-medium text-neutral-600 dark:text-neutral-400 transform -rotate-90">
                      Actual
                    </div>
                  </div>
                  <div></div>
                  <div></div>
                  
                  {/* Negative row */}
                  <div className="text-xs text-neutral-500 dark:text-neutral-400">Negative</div>
                  <motion.div
                    initial={{ opacity: 0, scale: 0 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.2 }}
                    className="w-16 h-16 bg-green-100 dark:bg-green-900/20 rounded-lg flex items-center justify-center cursor-pointer hover:bg-green-200 dark:hover:bg-green-900/40 transition-colors"
                  >
                    <div className="text-center">
                      <div className="text-lg font-bold text-green-700 dark:text-green-400">
                        {confusionMatrix.trueNegative}
                      </div>
                      <div className="text-xs text-green-600 dark:text-green-500">
                        {((confusionMatrix.trueNegative / total) * 100).toFixed(1)}%
                      </div>
                    </div>
                  </motion.div>
                  <motion.div
                    initial={{ opacity: 0, scale: 0 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.3 }}
                    className="w-16 h-16 bg-red-100 dark:bg-red-900/20 rounded-lg flex items-center justify-center cursor-pointer hover:bg-red-200 dark:hover:bg-red-900/40 transition-colors"
                  >
                    <div className="text-center">
                      <div className="text-lg font-bold text-red-700 dark:text-red-400">
                        {confusionMatrix.falsePositive}
                      </div>
                      <div className="text-xs text-red-600 dark:text-red-500">
                        {((confusionMatrix.falsePositive / total) * 100).toFixed(1)}%
                      </div>
                    </div>
                  </motion.div>
                  
                  {/* Positive row */}
                  <div className="text-xs text-neutral-500 dark:text-neutral-400">Positive</div>
                  <motion.div
                    initial={{ opacity: 0, scale: 0 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.4 }}
                    className="w-16 h-16 bg-red-100 dark:bg-red-900/20 rounded-lg flex items-center justify-center cursor-pointer hover:bg-red-200 dark:hover:bg-red-900/40 transition-colors"
                  >
                    <div className="text-center">
                      <div className="text-lg font-bold text-red-700 dark:text-red-400">
                        {confusionMatrix.falseNegative}
                      </div>
                      <div className="text-xs text-red-600 dark:text-red-500">
                        {((confusionMatrix.falseNegative / total) * 100).toFixed(1)}%
                      </div>
                    </div>
                  </motion.div>
                  <motion.div
                    initial={{ opacity: 0, scale: 0 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.5 }}
                    className="w-16 h-16 bg-green-100 dark:bg-green-900/20 rounded-lg flex items-center justify-center cursor-pointer hover:bg-green-200 dark:hover:bg-green-900/40 transition-colors"
                  >
                    <div className="text-center">
                      <div className="text-lg font-bold text-green-700 dark:text-green-400">
                        {confusionMatrix.truePositive}
                      </div>
                      <div className="text-xs text-green-600 dark:text-green-500">
                        {((confusionMatrix.truePositive / total) * 100).toFixed(1)}%
                      </div>
                    </div>
                  </motion.div>
                </div>
              </div>
              
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-neutral-600 dark:text-neutral-400">Total Samples:</span>
                    <span className="font-medium">{total.toLocaleString()}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-neutral-600 dark:text-neutral-400">Correct:</span>
                    <span className="font-medium text-green-600 dark:text-green-400">
                      {(confusionMatrix.truePositive + confusionMatrix.trueNegative).toLocaleString()}
                    </span>
                  </div>
                </div>
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-neutral-600 dark:text-neutral-400">Errors:</span>
                    <span className="font-medium text-red-600 dark:text-red-400">
                      {(confusionMatrix.falsePositive + confusionMatrix.falseNegative).toLocaleString()}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-neutral-600 dark:text-neutral-400">Error Rate:</span>
                    <span className="font-medium text-red-600 dark:text-red-400">
                      {(((confusionMatrix.falsePositive + confusionMatrix.falseNegative) / total) * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </Card>
        </motion.div>

        {/* ROC Curve */}
        <motion.div variants={itemVariants}>
          <Card
            title="ROC Curve"
            icon={<TrendingUp className="w-5 h-5 text-primary-600 dark:text-primary-400" />}
          >
            <div className="space-y-4">
              <div className="relative h-64 bg-neutral-50 dark:bg-neutral-900 rounded-lg p-4">
                <svg viewBox="0 0 300 300" className="w-full h-full">
                  {/* Grid lines */}
                  {[0, 0.2, 0.4, 0.6, 0.8, 1.0].map((tick) => (
                    <g key={tick}>
                      <line
                        x1={tick * 300}
                        y1={0}
                        x2={tick * 300}
                        y2={300}
                        stroke="currentColor"
                        strokeWidth="0.5"
                        className="text-neutral-300 dark:text-neutral-600"
                      />
                      <line
                        x1={0}
                        y1={300 - tick * 300}
                        x2={300}
                        y2={300 - tick * 300}
                        stroke="currentColor"
                        strokeWidth="0.5"
                        className="text-neutral-300 dark:text-neutral-600"
                      />
                    </g>
                  ))}
                  
                  {/* Diagonal line (random classifier) */}
                  <line
                    x1={0}
                    y1={300}
                    x2={300}
                    y2={0}
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeDasharray="5,5"
                    className="text-neutral-400 dark:text-neutral-500"
                  />
                  
                  {/* ROC curve */}
                  <motion.path
                    initial={{ pathLength: 0 }}
                    animate={{ pathLength: 1 }}
                    transition={{ duration: 2, delay: 0.5 }}
                    d={`M ${rocData.map((point, i) => 
                      `${i === 0 ? 'M' : 'L'} ${point.fpr * 300} ${300 - point.tpr * 300}`
                    ).join(' ')}`}
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="3"
                    className="text-primary-500"
                  />
                  
                  {/* Data points */}
                  {rocData.map((point, index) => (
                    <motion.circle
                      key={index}
                      initial={{ opacity: 0, scale: 0 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: 0.5 + index * 0.1 }}
                      cx={point.fpr * 300}
                      cy={300 - point.tpr * 300}
                      r="4"
                      fill="currentColor"
                      className="text-primary-600 hover:text-primary-700 cursor-pointer"
                    />
                  ))}
                </svg>
                
                {/* Axis labels */}
                <div className="absolute bottom-0 left-1/2 transform -translate-x-1/2 text-xs text-neutral-600 dark:text-neutral-400">
                  False Positive Rate
                </div>
                <div className="absolute top-1/2 left-0 transform -translate-y-1/2 -rotate-90 text-xs text-neutral-600 dark:text-neutral-400">
                  True Positive Rate
                </div>
              </div>
              
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-neutral-600 dark:text-neutral-400">AUC Score:</span>
                    <span className="font-bold text-primary-600 dark:text-primary-400">
                      {metrics.auc.toFixed(3)}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-neutral-600 dark:text-neutral-400">Threshold:</span>
                    <span className="font-medium">{selectedThreshold.toFixed(2)}</span>
                  </div>
                </div>
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-neutral-600 dark:text-neutral-400">Sensitivity:</span>
                    <span className="font-medium">{metrics.recall.toFixed(3)}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-neutral-600 dark:text-neutral-400">Specificity:</span>
                    <span className="font-medium">
                      {(confusionMatrix.trueNegative / (confusionMatrix.trueNegative + confusionMatrix.falsePositive)).toFixed(3)}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </Card>
        </motion.div>
      </div>

      {/* Threshold Analysis */}
      <motion.div variants={itemVariants}>
        <Card
          title="Threshold Analysis"
          icon={<BarChart3 className="w-5 h-5 text-primary-600 dark:text-primary-400" />}
        >
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                Classification Threshold: {selectedThreshold.toFixed(2)}
              </h3>
              <div className="flex items-center space-x-2">
                <span className="text-sm text-neutral-600 dark:text-neutral-400">0.0</span>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.01"
                  value={selectedThreshold}
                  onChange={(e) => setSelectedThreshold(parseFloat(e.target.value))}
                  className="w-32"
                />
                <span className="text-sm text-neutral-600 dark:text-neutral-400">1.0</span>
              </div>
            </div>
            
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {[
                { label: 'Precision', value: metrics.precision, color: 'blue' },
                { label: 'Recall', value: metrics.recall, color: 'green' },
                { label: 'F1 Score', value: metrics.f1Score, color: 'purple' },
                { label: 'Accuracy', value: metrics.accuracy, color: 'orange' },
              ].map((metric, index) => (
                <div key={metric.label} className="text-center">
                  <div className="relative w-20 h-20 mx-auto mb-2">
                    <svg className="w-full h-full transform -rotate-90">
                      <circle
                        cx="40"
                        cy="40"
                        r="35"
                        stroke="currentColor"
                        strokeWidth="4"
                        fill="none"
                        className="text-neutral-200 dark:text-neutral-700"
                      />
                      <motion.circle
                        initial={{ strokeDasharray: "0 220" }}
                        animate={{ strokeDasharray: `${metric.value * 220} 220` }}
                        transition={{ delay: index * 0.2, duration: 1 }}
                        cx="40"
                        cy="40"
                        r="35"
                        stroke="currentColor"
                        strokeWidth="4"
                        fill="none"
                        className={`text-${metric.color}-500`}
                      />
                    </svg>
                    <div className="absolute inset-0 flex items-center justify-center">
                      <span className="text-sm font-bold text-neutral-900 dark:text-neutral-100">
                        {(metric.value * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                  <div className="text-sm text-neutral-600 dark:text-neutral-400">
                    {metric.label}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </Card>
      </motion.div>
    </motion.div>
  );
};

export default ClassificationStats;