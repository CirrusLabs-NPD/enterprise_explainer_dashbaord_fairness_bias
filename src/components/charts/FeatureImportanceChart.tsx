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
  Cell,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Treemap,
  LineChart,
  Line,
  Area,
  AreaChart
} from 'recharts';
import {
  BarChart3,
  Target,
  Filter,
  Download,
  Maximize2,
  Eye,
  EyeOff,
  Settings,
  Info,
  TrendingUp,
  Zap
} from 'lucide-react';
import { cn } from '../../utils';

interface FeatureImportance {
  feature: string;
  importance: number;
  confidenceInterval?: [number, number];
  pValue?: number;
  method: 'shap' | 'lime' | 'permutation' | 'gain' | 'cover' | 'frequency';
  rank: number;
  category?: string;
  description?: string;
}

interface FeatureImportanceChartProps {
  importanceData: FeatureImportance[];
  title?: string;
  subtitle?: string;
  className?: string;
  interactive?: boolean;
  showConfidenceIntervals?: boolean;
  defaultView?: 'bar' | 'radar' | 'treemap' | 'timeline';
  maxFeatures?: number;
  onFeatureSelect?: (feature: string) => void;
  allowMethodComparison?: boolean;
  exportable?: boolean;
}

const FeatureImportanceChart: React.FC<FeatureImportanceChartProps> = ({
  importanceData,
  title = 'Feature Importance Analysis',
  subtitle,
  className,
  interactive = true,
  showConfidenceIntervals = true,
  defaultView = 'bar',
  maxFeatures = 20,
  onFeatureSelect,
  allowMethodComparison = true,
  exportable = true
}) => {
  const [currentView, setCurrentView] = useState(defaultView);
  const [selectedMethod, setSelectedMethod] = useState<string>('all');
  const [selectedFeature, setSelectedFeature] = useState<string | null>(null);
  const [displayCount, setDisplayCount] = useState(maxFeatures);
  const [sortBy, setSortBy] = useState<'importance' | 'alphabetical' | 'category'>('importance');
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [hiddenFeatures, setHiddenFeatures] = useState<Set<string>>(new Set());

  // Get available methods
  const availableMethods = useMemo(() => {
    const methods = ['all', ...Array.from(new Set(importanceData.map(d => d.method)))];
    return methods;
  }, [importanceData]);

  // Filter and sort data
  const processedData = useMemo(() => {
    let filtered = importanceData;

    // Filter by method
    if (selectedMethod !== 'all') {
      filtered = filtered.filter(d => d.method === selectedMethod);
    }

    // Filter out hidden features
    filtered = filtered.filter(d => !hiddenFeatures.has(d.feature));

    // Sort data
    filtered = [...filtered].sort((a, b) => {
      switch (sortBy) {
        case 'importance':
          return Math.abs(b.importance) - Math.abs(a.importance);
        case 'alphabetical':
          return a.feature.localeCompare(b.feature);
        case 'category':
          return (a.category || 'uncategorized').localeCompare(b.category || 'uncategorized');
        default:
          return 0;
      }
    });

    // Limit to display count
    return filtered.slice(0, displayCount);
  }, [importanceData, selectedMethod, hiddenFeatures, sortBy, displayCount]);

  // Prepare data for different visualizations
  const barData = useMemo(() => {
    return processedData.map((item, index) => ({
      ...item,
      absImportance: Math.abs(item.importance),
      color: item.importance >= 0 ? '#10b981' : '#ef4444',
      index
    }));
  }, [processedData]);

  const radarData = useMemo(() => {
    const topFeatures = processedData.slice(0, Math.min(8, displayCount));
    return topFeatures.map(item => ({
      feature: item.feature.length > 10 ? item.feature.substring(0, 10) + '...' : item.feature,
      importance: Math.abs(item.importance) * 100,
      fullName: item.feature
    }));
  }, [processedData, displayCount]);

  const treemapData = useMemo(() => {
    return processedData.map((item, index) => ({
      name: item.feature,
      size: Math.abs(item.importance) * 1000,
      importance: item.importance,
      category: item.category || 'uncategorized',
      color: item.importance >= 0 ? `hsl(${120 + index * 10}, 70%, 60%)` : `hsl(${0 + index * 10}, 70%, 60%)`
    }));
  }, [processedData]);

  const getFeatureColor = (importance: number, index: number) => {
    if (importance >= 0) {
      return `hsl(120, 70%, ${60 - (index * 2)}%)`;
    } else {
      return `hsl(0, 70%, ${60 - (index * 2)}%)`;
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
            {data.feature}
          </div>
          
          <div className="space-y-2 text-sm">
            <div className="flex justify-between items-center">
              <span className="text-neutral-600 dark:text-neutral-400">Importance:</span>
              <span className={cn(
                "font-medium",
                data.importance >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
              )}>
                {data.importance.toFixed(4)}
              </span>
            </div>
            
            <div className="flex justify-between items-center">
              <span className="text-neutral-600 dark:text-neutral-400">Method:</span>
              <span className="font-medium text-neutral-900 dark:text-neutral-100 uppercase">
                {data.method}
              </span>
            </div>
            
            <div className="flex justify-between items-center">
              <span className="text-neutral-600 dark:text-neutral-400">Rank:</span>
              <span className="font-medium text-blue-600 dark:text-blue-400">
                #{data.rank}
              </span>
            </div>
            
            {data.category && (
              <div className="flex justify-between items-center">
                <span className="text-neutral-600 dark:text-neutral-400">Category:</span>
                <span className="font-medium text-neutral-900 dark:text-neutral-100">
                  {data.category}
                </span>
              </div>
            )}
            
            {showConfidenceIntervals && data.confidenceInterval && (
              <div className="flex justify-between items-center">
                <span className="text-neutral-600 dark:text-neutral-400">95% CI:</span>
                <span className="font-medium text-purple-600 dark:text-purple-400">
                  [{data.confidenceInterval[0].toFixed(3)}, {data.confidenceInterval[1].toFixed(3)}]
                </span>
              </div>
            )}
            
            {data.pValue && (
              <div className="flex justify-between items-center">
                <span className="text-neutral-600 dark:text-neutral-400">p-value:</span>
                <span className={cn(
                  "font-medium",
                  data.pValue < 0.01 ? 'text-green-600 dark:text-green-400' :
                  data.pValue < 0.05 ? 'text-yellow-600 dark:text-yellow-400' :
                  'text-red-600 dark:text-red-400'
                )}>
                  {data.pValue.toFixed(4)}
                </span>
              </div>
            )}
          </div>
          
          {data.description && (
            <div className="mt-3 pt-2 border-t border-neutral-200 dark:border-neutral-600">
              <div className="text-xs text-neutral-500 dark:text-neutral-400">
                {data.description}
              </div>
            </div>
          )}
          
          {interactive && (
            <div className="mt-2 text-xs text-blue-600 dark:text-blue-400">
              Click to explore feature details
            </div>
          )}
        </motion.div>
      );
    }
    return null;
  };

  const handleFeatureClick = (data: any) => {
    if (interactive) {
      setSelectedFeature(data.feature);
      onFeatureSelect?.(data.feature);
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

  const renderVisualization = () => {
    const height = isFullscreen ? 'calc(100vh - 350px)' : '500px';

    switch (currentView) {
      case 'bar':
        return (
          <ResponsiveContainer width="100%" height={height}>
            <BarChart
              data={barData}
              layout="horizontal"
              margin={{ top: 20, right: 30, left: 100, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.1} />
              <XAxis 
                type="number"
                stroke="#6b7280"
                fontSize={12}
              />
              <YAxis 
                type="category"
                dataKey="feature"
                stroke="#6b7280"
                fontSize={11}
                width={90}
              />
              <Tooltip content={<CustomTooltip />} />
              <Bar
                dataKey="importance"
                onClick={handleFeatureClick}
                style={{ cursor: interactive ? 'pointer' : 'default' }}
              >
                {barData.map((entry, index) => (
                  <Cell 
                    key={`cell-${index}`} 
                    fill={getFeatureColor(entry.importance, index)}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        );

      case 'radar':
        return (
          <ResponsiveContainer width="100%" height={height}>
            <RadarChart data={radarData} margin={{ top: 40, right: 40, bottom: 40, left: 40 }}>
              <PolarGrid stroke="#374151" />
              <PolarAngleAxis 
                dataKey="feature" 
                tick={{ fontSize: 12, fill: '#6b7280' }}
              />
              <PolarRadiusAxis 
                angle={90} 
                domain={[0, 'dataMax']}
                tick={{ fontSize: 10, fill: '#6b7280' }}
              />
              <Radar
                name="Importance"
                dataKey="importance"
                stroke="#3b82f6"
                fill="#3b82f6"
                fillOpacity={0.3}
                strokeWidth={2}
              />
              <Tooltip content={<CustomTooltip />} />
            </RadarChart>
          </ResponsiveContainer>
        );

      case 'treemap':
        return (
          <ResponsiveContainer width="100%" height={height}>
            <Treemap
              data={treemapData}
              dataKey="size"
              stroke="#374151"
              fill="#3b82f6"
            >
              <Tooltip content={<CustomTooltip />} />
            </Treemap>
          </ResponsiveContainer>
        );

      default:
        return null;
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
              <BarChart3 className="w-5 h-5 text-orange-500" />
              {title}
            </h3>
            {subtitle && (
              <p className="text-sm text-neutral-600 dark:text-neutral-400 mt-1">
                {subtitle}
              </p>
            )}
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

        {/* Controls */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mt-4">
          <div>
            <label className="block text-xs font-medium text-neutral-700 dark:text-neutral-300 mb-1">
              Visualization
            </label>
            <select
              value={currentView}
              onChange={(e) => setCurrentView(e.target.value as any)}
              className="w-full px-3 py-2 text-sm border border-neutral-300 dark:border-neutral-600 rounded-md bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
            >
              <option value="bar">Bar Chart</option>
              <option value="radar">Radar Chart</option>
              <option value="treemap">Treemap</option>
            </select>
          </div>

          {allowMethodComparison && (
            <div>
              <label className="block text-xs font-medium text-neutral-700 dark:text-neutral-300 mb-1">
                Method
              </label>
              <select
                value={selectedMethod}
                onChange={(e) => setSelectedMethod(e.target.value)}
                className="w-full px-3 py-2 text-sm border border-neutral-300 dark:border-neutral-600 rounded-md bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
              >
                {availableMethods.map(method => (
                  <option key={method} value={method}>
                    {method === 'all' ? 'All Methods' : method.toUpperCase()}
                  </option>
                ))}
              </select>
            </div>
          )}

          <div>
            <label className="block text-xs font-medium text-neutral-700 dark:text-neutral-300 mb-1">
              Sort By
            </label>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as any)}
              className="w-full px-3 py-2 text-sm border border-neutral-300 dark:border-neutral-600 rounded-md bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
            >
              <option value="importance">Importance</option>
              <option value="alphabetical">Alphabetical</option>
              <option value="category">Category</option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-medium text-neutral-700 dark:text-neutral-300 mb-1">
              Show Features
            </label>
            <input
              type="range"
              min="5"
              max={Math.min(50, importanceData.length)}
              value={displayCount}
              onChange={(e) => setDisplayCount(parseInt(e.target.value))}
              className="w-full"
            />
            <div className="text-xs text-neutral-500 dark:text-neutral-400 mt-1">
              {displayCount} features
            </div>
          </div>
        </div>

        {/* Summary Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
          <div className="bg-gradient-to-r from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded-lg p-3">
            <div className="text-xs font-medium text-blue-600 dark:text-blue-400 uppercase tracking-wider">
              Total Features
            </div>
            <div className="text-xl font-bold text-blue-900 dark:text-blue-100">
              {processedData.length}
            </div>
          </div>
          
          <div className="bg-gradient-to-r from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 rounded-lg p-3">
            <div className="text-xs font-medium text-green-600 dark:text-green-400 uppercase tracking-wider">
              Positive Impact
            </div>
            <div className="text-xl font-bold text-green-900 dark:text-green-100">
              {processedData.filter(d => d.importance > 0).length}
            </div>
          </div>
          
          <div className="bg-gradient-to-r from-red-50 to-red-100 dark:from-red-900/20 dark:to-red-800/20 rounded-lg p-3">
            <div className="text-xs font-medium text-red-600 dark:text-red-400 uppercase tracking-wider">
              Negative Impact
            </div>
            <div className="text-xl font-bold text-red-900 dark:text-red-100">
              {processedData.filter(d => d.importance < 0).length}
            </div>
          </div>
        </div>
      </div>

      {/* Chart */}
      <div className="p-6">
        {renderVisualization()}
      </div>

      {/* Feature List */}
      {interactive && (
        <div className="border-t border-neutral-200 dark:border-neutral-700 p-6">
          <div className="flex items-center justify-between mb-4">
            <h4 className="font-semibold text-neutral-900 dark:text-neutral-100">
              Feature Controls
            </h4>
            <div className="text-sm text-neutral-500 dark:text-neutral-400">
              Click eye icons to show/hide features
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2 max-h-40 overflow-y-auto">
            {importanceData.slice(0, 20).map((feature) => (
              <div
                key={feature.feature}
                className="flex items-center justify-between p-2 bg-neutral-50 dark:bg-neutral-800 rounded-lg"
              >
                <span className={cn(
                  "text-sm",
                  hiddenFeatures.has(feature.feature) 
                    ? 'text-neutral-400 dark:text-neutral-600 line-through' 
                    : 'text-neutral-900 dark:text-neutral-100'
                )}>
                  {feature.feature}
                </span>
                <button
                  onClick={() => toggleFeatureVisibility(feature.feature)}
                  className="text-neutral-500 hover:text-neutral-700 dark:hover:text-neutral-300"
                >
                  {hiddenFeatures.has(feature.feature) ? (
                    <EyeOff className="w-4 h-4" />
                  ) : (
                    <Eye className="w-4 h-4" />
                  )}
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Selected Feature Details */}
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
                Feature Analysis: {selectedFeature}
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
                  Statistical Significance:
                </span>
                <span className="ml-2 text-neutral-900 dark:text-neutral-100">
                  {/* Feature analysis would go here */}
                  Detailed statistical analysis for {selectedFeature}
                </span>
              </div>
              <div>
                <span className="font-medium text-neutral-700 dark:text-neutral-300">
                  Correlation Analysis:
                </span>
                <span className="ml-2 text-neutral-900 dark:text-neutral-100">
                  Correlation data and interaction effects
                </span>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default FeatureImportanceChart;