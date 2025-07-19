import React, { useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import { TrendingUp, BarChart3, Activity, Download, Settings, Filter, Search } from 'lucide-react';
import Card from '@/components/common/Card';
import Button from '@/components/common/Button';

interface FeatureDependenceProps {
  modelType: 'classification' | 'regression';
}

interface FeatureData {
  name: string;
  type: 'numerical' | 'categorical';
  description: string;
  importance: number;
}

interface DependenceData {
  value: number;
  prediction: number;
  confidence: number;
  shapValue: number;
}

const FeatureDependence: React.FC<FeatureDependenceProps> = ({ modelType }) => {
  const [selectedFeature, setSelectedFeature] = useState<string>('Annual_Income');
  const [plotType, setPlotType] = useState<'partial' | 'shap' | 'ice'>('partial');
  const [searchTerm, setSearchTerm] = useState('');

  // Mock feature data
  const features: FeatureData[] = [
    { name: 'Annual_Income', type: 'numerical', description: 'Annual income in thousands', importance: 0.289 },
    { name: 'Credit_Score', type: 'numerical', description: 'Credit score (300-850)', importance: 0.156 },
    { name: 'Customer_Age', type: 'numerical', description: 'Age of the customer in years', importance: 0.342 },
    { name: 'Account_Balance', type: 'numerical', description: 'Current account balance', importance: 0.134 },
    { name: 'Loan_Amount', type: 'numerical', description: 'Requested loan amount', importance: 0.087 },
    { name: 'Employment_Type', type: 'categorical', description: 'Type of employment', importance: 0.098 },
    { name: 'Location', type: 'categorical', description: 'Geographic location', importance: 0.076 },
    { name: 'Education_Level', type: 'categorical', description: 'Highest education level', importance: 0.065 },
  ];

  const filteredFeatures = useMemo(() => {
    return features.filter(feature => 
      feature.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      feature.description.toLowerCase().includes(searchTerm.toLowerCase())
    ).sort((a, b) => b.importance - a.importance);
  }, [searchTerm]);

  const selectedFeatureData = features.find(f => f.name === selectedFeature);

  // Generate mock dependence data
  const dependenceData: DependenceData[] = useMemo(() => {
    if (!selectedFeatureData) return [];
    
    if (selectedFeatureData.type === 'numerical') {
      return Array.from({ length: 50 }, (_, i) => {
        const value = (i / 49) * 100 + 20; // Range from 20 to 120
        const baseline = 0.5;
        const trend = selectedFeatureData.importance * (value - 60) / 60;
        const noise = (Math.random() - 0.5) * 0.1;
        const prediction = Math.max(0, Math.min(1, baseline + trend + noise));
        
        return {
          value,
          prediction,
          confidence: 0.7 + Math.random() * 0.3,
          shapValue: trend + noise * 0.5,
        };
      });
    } else {
      // Categorical data
      const categories = ['Entry Level', 'Mid Level', 'Senior Level', 'Executive', 'Freelance'];
      return categories.map((_, i) => {
        const baseline = 0.5;
        const effect = (i - 2) * 0.15; // Centered effect
        const noise = (Math.random() - 0.5) * 0.05;
        
        return {
          value: i,
          prediction: Math.max(0, Math.min(1, baseline + effect + noise)),
          confidence: 0.8 + Math.random() * 0.2,
          shapValue: effect + noise,
        };
      });
    }
  }, [selectedFeature, selectedFeatureData]);

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

  const getFeatureColor = (importance: number) => {
    if (importance > 0.2) return 'bg-red-500';
    if (importance > 0.1) return 'bg-amber-500';
    if (importance > 0.05) return 'bg-blue-500';
    return 'bg-green-500';
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
            Feature Dependence
          </h1>
          <p className="text-neutral-600 dark:text-neutral-400 mt-1">
            Explore how individual features affect model predictions across their value ranges
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

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Feature Selection */}
        <motion.div variants={itemVariants}>
          <Card
            title="Select Feature"
            icon={<Filter className="w-5 h-5 text-primary-600 dark:text-primary-400" />}
          >
            <div className="space-y-4">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-neutral-400" />
                <input
                  type="text"
                  placeholder="Search features..."
                  className="w-full pl-10 pr-4 py-2 border border-neutral-200 dark:border-neutral-700 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100 focus:outline-none focus:ring-2 focus:ring-primary-500 text-sm"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                />
              </div>
              
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {filteredFeatures.map((feature) => (
                  <div
                    key={feature.name}
                    onClick={() => setSelectedFeature(feature.name)}
                    className={`p-3 rounded-lg border cursor-pointer transition-all duration-200 hover:shadow-md ${
                      selectedFeature === feature.name
                        ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20 shadow-sm'
                        : 'border-neutral-200 dark:border-neutral-700 hover:bg-neutral-50 dark:hover:bg-neutral-800'
                    }`}
                  >
                    <div className="flex items-start space-x-3">
                      <div className={`w-3 h-3 rounded-full mt-1 ${getFeatureColor(feature.importance)}`} />
                      <div className="flex-1">
                        <div className="font-medium text-sm text-neutral-900 dark:text-neutral-100">
                          {feature.name.replace('_', ' ')}
                        </div>
                        <div className="text-xs text-neutral-500 dark:text-neutral-400 mt-1">
                          {feature.description}
                        </div>
                        <div className="flex items-center justify-between mt-2">
                          <span className={`px-2 py-1 text-xs rounded-full ${
                            feature.type === 'numerical' 
                              ? 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200'
                              : 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                          }`}>
                            {feature.type}
                          </span>
                          <span className="text-xs font-medium text-neutral-600 dark:text-neutral-400">
                            {((feature.importance || 0) * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </Card>
        </motion.div>

        {/* Dependence Plots */}
        <motion.div variants={itemVariants} className="lg:col-span-3 space-y-6">
          {/* Plot Type Selection */}
          <Card
            title="Plot Type"
            icon={<BarChart3 className="w-5 h-5 text-primary-600 dark:text-primary-400" />}
          >
            <div className="flex flex-wrap gap-2">
              {[
                { id: 'partial', label: 'Partial Dependence', desc: 'Average effect of feature' },
                { id: 'shap', label: 'SHAP Dependence', desc: 'Feature interaction effects' },
                { id: 'ice', label: 'Individual ICE', desc: 'Individual conditional expectation' },
              ].map((plot) => (
                <Button
                  key={plot.id}
                  variant={plotType === plot.id ? 'primary' : 'outline'}
                  size="sm"
                  onClick={() => setPlotType(plot.id as any)}
                  className="flex flex-col items-start h-auto p-3"
                >
                  <div className="font-medium text-sm">{plot.label}</div>
                  <div className="text-xs opacity-70">{plot.desc}</div>
                </Button>
              ))}
            </div>
          </Card>

          {/* Main Plot */}
          <Card
            title={`${plotType.toUpperCase()} Plot: ${selectedFeatureData?.name.replace('_', ' ')}`}
            icon={<TrendingUp className="w-5 h-5 text-primary-600 dark:text-primary-400" />}
          >
            <div className="space-y-4">
              <div className="text-sm text-neutral-600 dark:text-neutral-400">
                {selectedFeatureData?.description}
              </div>
              
              <div className="relative h-96 bg-neutral-50 dark:bg-neutral-900 rounded-lg p-4">
                <svg viewBox="0 0 600 400" className="w-full h-full">
                  {/* Grid lines */}
                  {[0, 0.25, 0.5, 0.75, 1.0].map((tick) => (
                    <g key={tick}>
                      <line
                        x1={60 + tick * 520}
                        y1={50}
                        x2={60 + tick * 520}
                        y2={350}
                        stroke="currentColor"
                        strokeWidth="0.5"
                        className="text-neutral-300 dark:text-neutral-600"
                      />
                      <line
                        x1={60}
                        y1={50 + tick * 300}
                        x2={580}
                        y2={50 + tick * 300}
                        stroke="currentColor"
                        strokeWidth="0.5"
                        className="text-neutral-300 dark:text-neutral-600"
                      />
                    </g>
                  ))}
                  
                  {/* Axes */}
                  <line x1={60} y1={350} x2={580} y2={350} stroke="currentColor" strokeWidth="2" className="text-neutral-700 dark:text-neutral-300" />
                  <line x1={60} y1={50} x2={60} y2={350} stroke="currentColor" strokeWidth="2" className="text-neutral-700 dark:text-neutral-300" />
                  
                  {/* Data points and line */}
                  {selectedFeatureData?.type === 'numerical' ? (
                    <>
                      {/* Line */}
                      <motion.path
                        initial={{ pathLength: 0 }}
                        animate={{ pathLength: 1 }}
                        transition={{ duration: 1.5, delay: 0.5 }}
                        d={`M ${dependenceData.map((point, i) => 
                          `${60 + (i / (dependenceData.length - 1)) * 520} ${350 - point.prediction * 300}`
                        ).join(' L ')}`}
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="3"
                        className="text-primary-500"
                      />
                      
                      {/* Points */}
                      {dependenceData.map((point, index) => (
                        <motion.circle
                          key={index}
                          initial={{ opacity: 0, scale: 0 }}
                          animate={{ opacity: 1, scale: 1 }}
                          transition={{ delay: 0.5 + index * 0.02 }}
                          cx={60 + (index / (dependenceData.length - 1)) * 520}
                          cy={350 - point.prediction * 300}
                          r="4"
                          fill="currentColor"
                          className="text-primary-600 hover:text-primary-700 cursor-pointer"
                          style={{ opacity: point.confidence }}
                        />
                      ))}
                    </>
                  ) : (
                    /* Categorical bars */
                    dependenceData.map((point, index) => (
                      <motion.rect
                        key={index}
                        initial={{ height: 0, y: 350 }}
                        animate={{ height: point.prediction * 300, y: 350 - point.prediction * 300 }}
                        transition={{ delay: index * 0.2, duration: 0.8 }}
                        x={60 + index * 104}
                        width="80"
                        fill="currentColor"
                        className="text-primary-500 hover:text-primary-600 cursor-pointer"
                        style={{ opacity: point.confidence }}
                      />
                    ))
                  )}
                  
                  {/* Axis labels */}
                  <text x={320} y={390} textAnchor="middle" className="text-sm fill-neutral-600 dark:fill-neutral-400">
                    {selectedFeatureData?.name.replace('_', ' ')}
                  </text>
                  <text x={30} y={200} textAnchor="middle" className="text-sm fill-neutral-600 dark:fill-neutral-400" transform="rotate(-90 30 200)">
                    {modelType === 'classification' ? 'Prediction Probability' : 'Prediction Value'}
                  </text>
                  
                  {/* Y-axis labels */}
                  {[0, 0.25, 0.5, 0.75, 1.0].map((tick) => (
                    <text key={tick} x={50} y={355 - tick * 300} textAnchor="end" className="text-xs fill-neutral-500 dark:fill-neutral-400">
                      {tick.toFixed(2)}
                    </text>
                  ))}
                </svg>
              </div>
              
              {/* Statistics */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-4 border-t border-neutral-200 dark:border-neutral-700">
                <div className="text-center">
                  <div className="text-lg font-bold text-neutral-900 dark:text-neutral-100">
                    {Math.min(...dependenceData.map(d => d.prediction)).toFixed(3)}
                  </div>
                  <div className="text-sm text-neutral-600 dark:text-neutral-400">
                    Min Effect
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-lg font-bold text-neutral-900 dark:text-neutral-100">
                    {Math.max(...dependenceData.map(d => d.prediction)).toFixed(3)}
                  </div>
                  <div className="text-sm text-neutral-600 dark:text-neutral-400">
                    Max Effect
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-lg font-bold text-neutral-900 dark:text-neutral-100">
                    {(Math.max(...dependenceData.map(d => d.prediction)) - Math.min(...dependenceData.map(d => d.prediction))).toFixed(3)}
                  </div>
                  <div className="text-sm text-neutral-600 dark:text-neutral-400">
                    Range
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-lg font-bold text-neutral-900 dark:text-neutral-100">
                    {(dependenceData.reduce((sum, d) => sum + d.confidence, 0) / dependenceData.length * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm text-neutral-600 dark:text-neutral-400">
                    Avg Confidence
                  </div>
                </div>
              </div>
            </div>
          </Card>

          {/* Feature Impact Analysis */}
          <Card
            title="Feature Impact Analysis"
            icon={<Activity className="w-5 h-5 text-primary-600 dark:text-primary-400" />}
          >
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-3">
                  <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                    Impact Summary
                  </h3>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-neutral-600 dark:text-neutral-400">Feature Type:</span>
                      <span className={`px-2 py-1 text-xs rounded-full font-medium ${
                        selectedFeatureData?.type === 'numerical' 
                          ? 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200'
                          : 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                      }`}>
                        {selectedFeatureData?.type}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-neutral-600 dark:text-neutral-400">Importance:</span>
                      <span className="font-bold text-primary-600 dark:text-primary-400">
                        {((selectedFeatureData?.importance || 0) * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-neutral-600 dark:text-neutral-400">Effect Range:</span>
                      <span className="font-medium text-neutral-900 dark:text-neutral-100">
                        {(Math.max(...dependenceData.map(d => d.prediction)) - Math.min(...dependenceData.map(d => d.prediction))).toFixed(3)}
                      </span>
                    </div>
                  </div>
                </div>
                
                <div className="space-y-3">
                  <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                    Trend Analysis
                  </h3>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-neutral-600 dark:text-neutral-400">Trend Direction:</span>
                      <span className={`font-medium ${
                        dependenceData[dependenceData.length - 1]?.prediction > dependenceData[0]?.prediction
                          ? 'text-green-600 dark:text-green-400'
                          : 'text-red-600 dark:text-red-400'
                      }`}>
                        {dependenceData[dependenceData.length - 1]?.prediction > dependenceData[0]?.prediction ? 'Increasing' : 'Decreasing'}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-neutral-600 dark:text-neutral-400">Variability:</span>
                      <span className="font-medium text-neutral-900 dark:text-neutral-100">
                        {selectedFeatureData?.type === 'numerical' ? 'Continuous' : 'Discrete'}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-neutral-600 dark:text-neutral-400">Confidence:</span>
                      <span className="font-medium text-blue-600 dark:text-blue-400">
                        {(dependenceData.reduce((sum, d) => sum + d.confidence, 0) / dependenceData.length * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </Card>
        </motion.div>
      </div>
    </motion.div>
  );
};

export default FeatureDependence;