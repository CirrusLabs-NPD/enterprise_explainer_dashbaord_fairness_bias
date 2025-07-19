import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Activity, BarChart3, Download, Settings, Target } from 'lucide-react';
import Card from '@/components/common/Card';
import Button from '@/components/common/Button';

interface RegressionMetrics {
  r2Score: number;
  mse: number;
  rmse: number;
  mae: number;
  mape: number;
  adjustedR2: number;
}

interface ResidualData {
  fitted: number;
  residual: number;
  actual: number;
  predicted: number;
}

const RegressionStats: React.FC = () => {
  const [selectedPlot, setSelectedPlot] = useState<'residual' | 'qq' | 'predicted' | 'histogram'>('residual');

  // Mock data
  const metrics: RegressionMetrics = {
    r2Score: 0.847,
    mse: 142.3,
    rmse: 11.9,
    mae: 8.7,
    mape: 12.4,
    adjustedR2: 0.841,
  };

  // Generate mock residual data
  const residualData: ResidualData[] = Array.from({ length: 100 }, () => {
    const fitted = 20 + Math.random() * 100;
    const residual = (Math.random() - 0.5) * 30;
    const actual = fitted + residual;
    return {
      fitted,
      residual,
      actual,
      predicted: fitted,
    };
  });

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

  const getMetricColor = (metric: keyof RegressionMetrics, value: number) => {
    if (metric === 'r2Score' || metric === 'adjustedR2') {
      if (value >= 0.9) return 'text-green-600 dark:text-green-400';
      if (value >= 0.8) return 'text-blue-600 dark:text-blue-400';
      if (value >= 0.7) return 'text-amber-600 dark:text-amber-400';
      return 'text-red-600 dark:text-red-400';
    }
    // For error metrics (lower is better)
    if (value <= 5) return 'text-green-600 dark:text-green-400';
    if (value <= 15) return 'text-blue-600 dark:text-blue-400';
    if (value <= 25) return 'text-amber-600 dark:text-amber-400';
    return 'text-red-600 dark:text-red-400';
  };

  const getMetricBg = (metric: keyof RegressionMetrics, value: number) => {
    if (metric === 'r2Score' || metric === 'adjustedR2') {
      if (value >= 0.9) return 'bg-green-100 dark:bg-green-900/20';
      if (value >= 0.8) return 'bg-blue-100 dark:bg-blue-900/20';
      if (value >= 0.7) return 'bg-amber-100 dark:bg-amber-900/20';
      return 'bg-red-100 dark:bg-red-900/20';
    }
    if (value <= 5) return 'bg-green-100 dark:bg-green-900/20';
    if (value <= 15) return 'bg-blue-100 dark:bg-blue-900/20';
    if (value <= 25) return 'bg-amber-100 dark:bg-amber-900/20';
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
            Regression Statistics
          </h1>
          <p className="text-neutral-600 dark:text-neutral-400 mt-1">
            Comprehensive analysis of regression model performance and residuals
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
          title="Regression Metrics"
          icon={<Activity className="w-5 h-5 text-primary-600 dark:text-primary-400" />}
        >
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            {Object.entries(metrics).map(([key, value]) => (
              <motion.div
                key={key}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: Object.keys(metrics).indexOf(key) * 0.1 }}
                className={`p-4 rounded-lg ${getMetricBg(key as keyof RegressionMetrics, value)} transition-all hover:scale-105`}
              >
                <div className="text-center">
                  <div className={`text-2xl font-bold ${getMetricColor(key as keyof RegressionMetrics, value)}`}>
                    {key.includes('r2') || key.includes('R2') ? 
                      value.toFixed(3) : 
                      key === 'mape' ? 
                        `${value.toFixed(1)}%` : 
                        value.toFixed(1)
                    }
                  </div>
                  <div className="text-sm text-neutral-600 dark:text-neutral-400 mt-1">
                    {key === 'r2Score' ? 'R² Score' :
                     key === 'adjustedR2' ? 'Adj. R²' :
                     key === 'mse' ? 'MSE' :
                     key === 'rmse' ? 'RMSE' :
                     key === 'mae' ? 'MAE' :
                     key === 'mape' ? 'MAPE' : key}
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </Card>
      </motion.div>

      {/* Plot Selection */}
      <motion.div variants={itemVariants}>
        <Card
          title="Diagnostic Plots"
          icon={<BarChart3 className="w-5 h-5 text-primary-600 dark:text-primary-400" />}
        >
          <div className="space-y-6">
            <div className="flex flex-wrap gap-2">
              {[
                { id: 'residual', label: 'Residual Plot', desc: 'Residuals vs Fitted' },
                { id: 'qq', label: 'Q-Q Plot', desc: 'Normal Q-Q' },
                { id: 'predicted', label: 'Predicted vs Actual', desc: 'Scatter Plot' },
                { id: 'histogram', label: 'Residual Distribution', desc: 'Histogram' },
              ].map((plot) => (
                <Button
                  key={plot.id}
                  variant={selectedPlot === plot.id ? 'primary' : 'outline'}
                  size="sm"
                  onClick={() => setSelectedPlot(plot.id as any)}
                  className="flex flex-col items-start h-auto p-3"
                >
                  <div className="font-medium text-sm">{plot.label}</div>
                  <div className="text-xs opacity-70">{plot.desc}</div>
                </Button>
              ))}
            </div>

            <div className="relative h-96 bg-neutral-50 dark:bg-neutral-900 rounded-lg p-4">
              {selectedPlot === 'residual' && (
                <div className="h-full">
                  <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
                    Residual vs Fitted Plot
                  </h3>
                  <svg viewBox="0 0 400 300" className="w-full h-full">
                    {/* Grid lines */}
                    {[0, 0.25, 0.5, 0.75, 1.0].map((tick) => (
                      <g key={tick}>
                        <line
                          x1={tick * 400}
                          y1={0}
                          x2={tick * 400}
                          y2={300}
                          stroke="currentColor"
                          strokeWidth="0.5"
                          className="text-neutral-300 dark:text-neutral-600"
                        />
                        <line
                          x1={0}
                          y1={tick * 300}
                          x2={400}
                          y2={tick * 300}
                          stroke="currentColor"
                          strokeWidth="0.5"
                          className="text-neutral-300 dark:text-neutral-600"
                        />
                      </g>
                    ))}
                    
                    {/* Zero line */}
                    <line
                      x1={0}
                      y1={150}
                      x2={400}
                      y2={150}
                      stroke="currentColor"
                      strokeWidth="2"
                      className="text-red-500"
                    />
                    
                    {/* Data points */}
                    {residualData.map((point, index) => (
                      <motion.circle
                        key={index}
                        initial={{ opacity: 0, scale: 0 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: index * 0.01 }}
                        cx={(point.fitted / 120) * 400}
                        cy={150 - (point.residual / 30) * 100}
                        r="3"
                        fill="currentColor"
                        className="text-blue-500 opacity-70 hover:opacity-100"
                      />
                    ))}
                  </svg>
                </div>
              )}

              {selectedPlot === 'predicted' && (
                <div className="h-full">
                  <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
                    Predicted vs Actual Values
                  </h3>
                  <svg viewBox="0 0 400 300" className="w-full h-full">
                    {/* Grid lines */}
                    {[0, 0.25, 0.5, 0.75, 1.0].map((tick) => (
                      <g key={tick}>
                        <line
                          x1={tick * 400}
                          y1={0}
                          x2={tick * 400}
                          y2={300}
                          stroke="currentColor"
                          strokeWidth="0.5"
                          className="text-neutral-300 dark:text-neutral-600"
                        />
                        <line
                          x1={0}
                          y1={tick * 300}
                          x2={400}
                          y2={tick * 300}
                          stroke="currentColor"
                          strokeWidth="0.5"
                          className="text-neutral-300 dark:text-neutral-600"
                        />
                      </g>
                    ))}
                    
                    {/* Perfect prediction line */}
                    <line
                      x1={0}
                      y1={300}
                      x2={400}
                      y2={0}
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeDasharray="5,5"
                      className="text-red-500"
                    />
                    
                    {/* Data points */}
                    {residualData.map((point, index) => (
                      <motion.circle
                        key={index}
                        initial={{ opacity: 0, scale: 0 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: index * 0.01 }}
                        cx={(point.predicted / 120) * 400}
                        cy={300 - (point.actual / 120) * 300}
                        r="3"
                        fill="currentColor"
                        className="text-green-500 opacity-70 hover:opacity-100"
                      />
                    ))}
                  </svg>
                </div>
              )}

              {selectedPlot === 'histogram' && (
                <div className="h-full">
                  <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
                    Residual Distribution
                  </h3>
                  <div className="flex items-end justify-center h-64 space-x-1">
                    {Array.from({ length: 20 }, (_, i) => {
                      const height = Math.max(10, Math.random() * 200);
                      return (
                        <motion.div
                          key={i}
                          initial={{ height: 0 }}
                          animate={{ height: `${height}px` }}
                          transition={{ delay: i * 0.05, duration: 0.5 }}
                          className="w-4 bg-primary-500 rounded-t"
                        />
                      );
                    })}
                  </div>
                </div>
              )}

              {selectedPlot === 'qq' && (
                <div className="h-full">
                  <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
                    Q-Q Plot (Normal Distribution)
                  </h3>
                  <svg viewBox="0 0 400 300" className="w-full h-full">
                    {/* Grid lines */}
                    {[0, 0.25, 0.5, 0.75, 1.0].map((tick) => (
                      <g key={tick}>
                        <line
                          x1={tick * 400}
                          y1={0}
                          x2={tick * 400}
                          y2={300}
                          stroke="currentColor"
                          strokeWidth="0.5"
                          className="text-neutral-300 dark:text-neutral-600"
                        />
                        <line
                          x1={0}
                          y1={tick * 300}
                          x2={400}
                          y2={tick * 300}
                          stroke="currentColor"
                          strokeWidth="0.5"
                          className="text-neutral-300 dark:text-neutral-600"
                        />
                      </g>
                    ))}
                    
                    {/* Normal line */}
                    <line
                      x1={0}
                      y1={300}
                      x2={400}
                      y2={0}
                      stroke="currentColor"
                      strokeWidth="2"
                      className="text-red-500"
                    />
                    
                    {/* Q-Q points */}
                    {Array.from({ length: 50 }, (_, i) => {
                      const x = (i / 50) * 400;
                      const y = 300 - (i / 50) * 300 + (Math.random() - 0.5) * 40;
                      return (
                        <motion.circle
                          key={i}
                          initial={{ opacity: 0, scale: 0 }}
                          animate={{ opacity: 1, scale: 1 }}
                          transition={{ delay: i * 0.02 }}
                          cx={x}
                          cy={y}
                          r="3"
                          fill="currentColor"
                          className="text-purple-500 opacity-70 hover:opacity-100"
                        />
                      );
                    })}
                  </svg>
                </div>
              )}
            </div>
          </div>
        </Card>
      </motion.div>

      {/* Model Performance Summary */}
      <motion.div variants={itemVariants}>
        <Card
          title="Performance Summary"
          icon={<Target className="w-5 h-5 text-primary-600 dark:text-primary-400" />}
        >
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                Model Quality
              </h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-neutral-600 dark:text-neutral-400">Explained Variance:</span>
                  <span className="font-bold text-primary-600 dark:text-primary-400">
                    {(metrics.r2Score * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-neutral-600 dark:text-neutral-400">Model Fit:</span>
                  <span className={`font-medium ${
                    metrics.r2Score > 0.8 ? 'text-green-600 dark:text-green-400' : 
                    metrics.r2Score > 0.6 ? 'text-amber-600 dark:text-amber-400' : 
                    'text-red-600 dark:text-red-400'
                  }`}>
                    {metrics.r2Score > 0.8 ? 'Excellent' : 
                     metrics.r2Score > 0.6 ? 'Good' : 'Poor'}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-neutral-600 dark:text-neutral-400">Overfitting Risk:</span>
                  <span className="font-medium text-green-600 dark:text-green-400">
                    {Math.abs(metrics.r2Score - metrics.adjustedR2) < 0.02 ? 'Low' : 'Medium'}
                  </span>
                </div>
              </div>
            </div>
            
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                Error Analysis
              </h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-neutral-600 dark:text-neutral-400">Average Error:</span>
                  <span className="font-bold text-blue-600 dark:text-blue-400">
                    {metrics.mae.toFixed(1)}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-neutral-600 dark:text-neutral-400">Prediction Spread:</span>
                  <span className="font-medium text-neutral-900 dark:text-neutral-100">
                    ±{metrics.rmse.toFixed(1)}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-neutral-600 dark:text-neutral-400">Relative Error:</span>
                  <span className="font-medium text-amber-600 dark:text-amber-400">
                    {metrics.mape.toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>
          </div>
        </Card>
      </motion.div>
    </motion.div>
  );
};

export default RegressionStats;