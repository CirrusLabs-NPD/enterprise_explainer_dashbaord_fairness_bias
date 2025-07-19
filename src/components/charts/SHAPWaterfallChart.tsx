import React, { useMemo, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  ReferenceLine,
  Cell
} from 'recharts';
import { 
  ArrowUpRight, 
  ArrowDownRight, 
  Target, 
  TrendingUp, 
  Info,
  Download,
  Settings,
  Maximize2
} from 'lucide-react';
import { cn } from '../../utils';

interface SHAPValue {
  feature: string;
  value: number;
  baseValue?: number;
  displayValue: string;
  impact: 'positive' | 'negative' | 'neutral';
  originalFeatureValue?: number | string;
  confidence?: number;
}

interface SHAPWaterfallChartProps {
  shapValues: SHAPValue[];
  prediction: number;
  baseValue: number;
  modelType?: 'classification' | 'regression';
  title?: string;
  subtitle?: string;
  className?: string;
  interactive?: boolean;
  showConfidence?: boolean;
  onFeatureClick?: (feature: string) => void;
  exportable?: boolean;
}

const SHAPWaterfallChart: React.FC<SHAPWaterfallChartProps> = ({
  shapValues,
  prediction,
  baseValue,
  modelType = 'classification',
  title = 'SHAP Explanation',
  subtitle,
  className,
  interactive = true,
  showConfidence = true,
  onFeatureClick,
  exportable = true
}) => {
  const [selectedFeature, setSelectedFeature] = useState<string | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [sortBy, setSortBy] = useState<'impact' | 'feature'>('impact');

  // Process data for waterfall visualization
  const waterfallData = useMemo(() => {
    let cumulativeValue = baseValue;
    const data = [];

    // Add base value
    data.push({
      name: 'Base Value',
      value: baseValue,
      cumulative: baseValue,
      type: 'base',
      impact: 'neutral' as const,
      displayValue: baseValue.toFixed(3),
      originalValue: null
    });

    // Sort features by impact if selected
    const sortedValues = [...shapValues].sort((a, b) => {
      if (sortBy === 'impact') {
        return Math.abs(b.value) - Math.abs(a.value);
      }
      return a.feature.localeCompare(b.feature);
    });

    // Add each SHAP value
    sortedValues.forEach((shap, index) => {
      const prevCumulative = cumulativeValue;
      cumulativeValue += shap.value;

      data.push({
        name: shap.feature,
        value: shap.value,
        cumulative: cumulativeValue,
        prevCumulative: prevCumulative,
        type: 'feature',
        impact: shap.impact,
        displayValue: shap.displayValue,
        originalValue: shap.originalFeatureValue,
        confidence: shap.confidence,
        index
      });
    });

    // Add final prediction
    data.push({
      name: 'Prediction',
      value: prediction,
      cumulative: prediction,
      type: 'prediction',
      impact: prediction > baseValue ? 'positive' : 'negative' as const,
      displayValue: prediction.toFixed(3),
      originalValue: null
    });

    return data;
  }, [shapValues, baseValue, prediction, sortBy]);

  const maxAbsValue = useMemo(() => {
    return Math.max(...waterfallData.map(d => Math.abs(d.value || 0)));
  }, [waterfallData]);

  const getBarColor = (impact: string, isSelected: boolean = false) => {
    const opacity = isSelected ? 1 : 0.8;
    switch (impact) {
      case 'positive':
        return isSelected ? '#10b981' : `rgba(16, 185, 129, ${opacity})`;
      case 'negative':
        return isSelected ? '#ef4444' : `rgba(239, 68, 68, ${opacity})`;
      default:
        return isSelected ? '#6b7280' : `rgba(107, 114, 128, ${opacity})`;
    }
  };

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      
      return (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="bg-white dark:bg-neutral-800 p-4 rounded-lg shadow-lg border border-neutral-200 dark:border-neutral-700 min-w-64"
        >
          <div className="font-semibold text-neutral-900 dark:text-neutral-100 mb-2">
            {data.name}
          </div>
          
          <div className="space-y-2 text-sm">
            <div className="flex justify-between items-center">
              <span className="text-neutral-600 dark:text-neutral-400">SHAP Value:</span>
              <span className={cn(
                "font-medium",
                data.impact === 'positive' ? 'text-green-600 dark:text-green-400' :
                data.impact === 'negative' ? 'text-red-600 dark:text-red-400' :
                'text-neutral-600 dark:text-neutral-400'
              )}>
                {data.impact === 'positive' ? '+' : ''}{data.value?.toFixed(4) || 'N/A'}
              </span>
            </div>
            
            {data.originalValue !== null && (
              <div className="flex justify-between items-center">
                <span className="text-neutral-600 dark:text-neutral-400">Feature Value:</span>
                <span className="font-medium text-neutral-900 dark:text-neutral-100">
                  {data.originalValue}
                </span>
              </div>
            )}
            
            <div className="flex justify-between items-center">
              <span className="text-neutral-600 dark:text-neutral-400">Cumulative:</span>
              <span className="font-medium text-neutral-900 dark:text-neutral-100">
                {data.cumulative?.toFixed(4)}
              </span>
            </div>
            
            {showConfidence && data.confidence && (
              <div className="flex justify-between items-center">
                <span className="text-neutral-600 dark:text-neutral-400">Confidence:</span>
                <span className="font-medium text-blue-600 dark:text-blue-400">
                  {(data.confidence * 100).toFixed(1)}%
                </span>
              </div>
            )}
          </div>
          
          {data.type === 'feature' && (
            <div className="mt-3 pt-2 border-t border-neutral-200 dark:border-neutral-600">
              <div className="text-xs text-neutral-500 dark:text-neutral-400">
                Click to explore feature details
              </div>
            </div>
          )}
        </motion.div>
      );
    }
    return null;
  };

  const handleBarClick = (data: any) => {
    if (data.type === 'feature' && interactive) {
      setSelectedFeature(data.name);
      onFeatureClick?.(data.name);
    }
  };

  const handleExport = () => {
    // Implementation for exporting chart
    const exportData = {
      title,
      baseValue,
      prediction,
      shapValues: waterfallData,
      timestamp: new Date().toISOString()
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: 'application/json'
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `shap-explanation-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
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
              <Target className="w-5 h-5 text-blue-500" />
              {title}
            </h3>
            {subtitle && (
              <p className="text-sm text-neutral-600 dark:text-neutral-400 mt-1">
                {subtitle}
              </p>
            )}
          </div>
          
          <div className="flex items-center gap-2">
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as 'impact' | 'feature')}
              className="px-3 py-1 text-sm border border-neutral-300 dark:border-neutral-600 rounded-md bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
            >
              <option value="impact">Sort by Impact</option>
              <option value="feature">Sort by Feature</option>
            </select>
            
            {exportable && (
              <button
                onClick={handleExport}
                className="p-2 text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100 transition-colors"
                title="Export data"
              >
                <Download className="w-4 h-4" />
              </button>
            )}
            
            <button
              onClick={() => setIsFullscreen(!isFullscreen)}
              className="p-2 text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100 transition-colors"
              title="Toggle fullscreen"
            >
              <Maximize2 className="w-4 h-4" />
            </button>
          </div>
        </div>
        
        {/* Summary Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
          <div className="bg-neutral-50 dark:bg-neutral-800 rounded-lg p-3">
            <div className="text-xs font-medium text-neutral-500 dark:text-neutral-400 uppercase tracking-wider">
              Base Value
            </div>
            <div className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
              {baseValue.toFixed(4)}
            </div>
          </div>
          
          <div className="bg-neutral-50 dark:bg-neutral-800 rounded-lg p-3">
            <div className="text-xs font-medium text-neutral-500 dark:text-neutral-400 uppercase tracking-wider">
              Final Prediction
            </div>
            <div className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
              {prediction.toFixed(4)}
            </div>
          </div>
          
          <div className="bg-neutral-50 dark:bg-neutral-800 rounded-lg p-3">
            <div className="text-xs font-medium text-neutral-500 dark:text-neutral-400 uppercase tracking-wider">
              Total Impact
            </div>
            <div className={cn(
              "text-lg font-semibold",
              prediction - baseValue > 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
            )}>
              {prediction - baseValue > 0 ? '+' : ''}{(prediction - baseValue).toFixed(4)}
            </div>
          </div>
        </div>
      </div>

      {/* Chart */}
      <div className="p-6">
        <div style={{ width: '100%', height: isFullscreen ? 'calc(100vh - 280px)' : '400px' }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={waterfallData}
              margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.1} />
              <XAxis 
                dataKey="name"
                angle={-45}
                textAnchor="end"
                height={80}
                fontSize={12}
                stroke="#6b7280"
              />
              <YAxis 
                fontSize={12}
                stroke="#6b7280"
                domain={[
                  Math.min(baseValue, prediction) - maxAbsValue * 0.1,
                  Math.max(baseValue, prediction) + maxAbsValue * 0.1
                ]}
              />
              <Tooltip 
                content={<CustomTooltip />}
                cursor={false}
              />
              <ReferenceLine 
                y={baseValue} 
                stroke="#6b7280" 
                strokeDasharray="5 5"
                label={{ value: "Base", position: "topLeft" }}
              />
              <Bar
                dataKey="cumulative"
                onClick={handleBarClick}
                style={{ cursor: interactive ? 'pointer' : 'default' }}
              >
                {waterfallData.map((entry, index) => (
                  <Cell 
                    key={`cell-${index}`} 
                    fill={getBarColor(entry.impact, selectedFeature === entry.name)}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Feature Details Panel */}
      <AnimatePresence>
        {selectedFeature && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="border-t border-neutral-200 dark:border-neutral-700 p-6 bg-neutral-50 dark:bg-neutral-800"
          >
            <div className="flex items-center justify-between mb-4">
              <h4 className="font-semibold text-neutral-900 dark:text-neutral-100">
                Feature Details: {selectedFeature}
              </h4>
              <button
                onClick={() => setSelectedFeature(null)}
                className="text-neutral-500 hover:text-neutral-700 dark:hover:text-neutral-300"
              >
                ×
              </button>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div>
                <span className="font-medium text-neutral-700 dark:text-neutral-300">
                  Impact on Prediction:
                </span>
                <span className="ml-2 text-neutral-900 dark:text-neutral-100">
                  Detailed analysis would go here
                </span>
              </div>
              <div>
                <span className="font-medium text-neutral-700 dark:text-neutral-300">
                  Feature Importance Rank:
                </span>
                <span className="ml-2 text-neutral-900 dark:text-neutral-100">
                  #{waterfallData.findIndex(d => d.name === selectedFeature)}
                </span>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default SHAPWaterfallChart;