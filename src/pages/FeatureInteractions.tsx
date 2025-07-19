import React, { useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import { Network, Grid, TrendingUp, Download, Settings, Filter, Search, BarChart3 } from 'lucide-react';
import Card from '@/components/common/Card';
import Button from '@/components/common/Button';

interface FeatureInteractionsProps {
  modelType: 'classification' | 'regression';
}

interface FeatureData {
  name: string;
  type: 'numerical' | 'categorical';
  importance: number;
}

interface InteractionData {
  feature1: string;
  feature2: string;
  strength: number;
  type: 'synergy' | 'redundancy' | 'independent';
  shapInteraction: number;
  rank: number;
}

const FeatureInteractions: React.FC<FeatureInteractionsProps> = ({ modelType: _modelType }) => {
  const [selectedFeatures, setSelectedFeatures] = useState<[string, string]>(['Annual_Income', 'Credit_Score']);
  const [viewMode, setViewMode] = useState<'heatmap' | 'network' | 'pairwise'>('heatmap');
  const [searchTerm, setSearchTerm] = useState('');
  const [minInteractionStrength, setMinInteractionStrength] = useState(0.1);

  // Mock feature data
  const features: FeatureData[] = [
    { name: 'Annual_Income', type: 'numerical', importance: 0.289 },
    { name: 'Credit_Score', type: 'numerical', importance: 0.156 },
    { name: 'Customer_Age', type: 'numerical', importance: 0.342 },
    { name: 'Account_Balance', type: 'numerical', importance: 0.134 },
    { name: 'Loan_Amount', type: 'numerical', importance: 0.087 },
    { name: 'Employment_Type', type: 'categorical', importance: 0.098 },
    { name: 'Location', type: 'categorical', importance: 0.076 },
    { name: 'Education_Level', type: 'categorical', importance: 0.065 },
  ];

  // Generate mock interaction data
  const interactionData: InteractionData[] = useMemo(() => {
    const interactions: InteractionData[] = [];
    let rank = 1;
    
    for (let i = 0; i < features.length; i++) {
      for (let j = i + 1; j < features.length; j++) {
        const feature1 = features[i];
        const feature2 = features[j];
        
        // Calculate interaction strength based on feature importance
        const baseStrength = (feature1.importance + feature2.importance) / 2;
        const randomFactor = Math.random() * 0.5 + 0.5;
        const strength = Math.min(1, baseStrength * randomFactor);
        
        // Determine interaction type
        let type: 'synergy' | 'redundancy' | 'independent' = 'independent';
        if (strength > 0.3) type = 'synergy';
        else if (strength > 0.15) type = 'redundancy';
        
        const shapInteraction = (Math.random() - 0.5) * strength;
        
        interactions.push({
          feature1: feature1.name,
          feature2: feature2.name,
          strength,
          type,
          shapInteraction,
          rank: rank++,
        });
      }
    }
    
    return interactions.sort((a, b) => b.strength - a.strength);
  }, [features]);

  const filteredInteractions = useMemo(() => {
    return interactionData.filter(interaction => {
      const matchesSearch = searchTerm === '' || 
        interaction.feature1.toLowerCase().includes(searchTerm.toLowerCase()) ||
        interaction.feature2.toLowerCase().includes(searchTerm.toLowerCase());
      const meetsStrength = interaction.strength >= minInteractionStrength;
      return matchesSearch && meetsStrength;
    });
  }, [interactionData, searchTerm, minInteractionStrength]);

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

  const getInteractionColor = (type: string) => {
    switch (type) {
      case 'synergy': return 'text-green-600 dark:text-green-400';
      case 'redundancy': return 'text-amber-600 dark:text-amber-400';
      default: return 'text-neutral-600 dark:text-neutral-400';
    }
  };

  const getInteractionBgColor = (type: string, strength: number) => {
    const alpha = strength;
    switch (type) {
      case 'synergy': return `rgba(34, 197, 94, ${alpha})`;
      case 'redundancy': return `rgba(245, 158, 11, ${alpha})`;
      default: return `rgba(107, 114, 128, ${alpha})`;
    }
  };

  const getStrengthColor = (strength: number) => {
    if (strength > 0.5) return 'text-red-600 dark:text-red-400';
    if (strength > 0.3) return 'text-amber-600 dark:text-amber-400';
    if (strength > 0.15) return 'text-blue-600 dark:text-blue-400';
    return 'text-green-600 dark:text-green-400';
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
            Feature Interactions
          </h1>
          <p className="text-neutral-600 dark:text-neutral-400 mt-1">
            Discover how features interact with each other to influence predictions
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

      {/* Controls */}
      <motion.div variants={itemVariants}>
        <Card
          title="Analysis Controls"
          icon={<Filter className="w-5 h-5 text-primary-600 dark:text-primary-400" />}
        >
          <div className="space-y-4">
            <div className="flex flex-wrap gap-2">
              {[
                { id: 'heatmap', label: 'Interaction Heatmap', icon: <Grid className="w-4 h-4" /> },
                { id: 'network', label: 'Network Graph', icon: <Network className="w-4 h-4" /> },
                { id: 'pairwise', label: 'Pairwise Analysis', icon: <BarChart3 className="w-4 h-4" /> },
              ].map((mode) => (
                <Button
                  key={mode.id}
                  variant={viewMode === mode.id ? 'primary' : 'outline'}
                  size="sm"
                  leftIcon={mode.icon}
                  onClick={() => setViewMode(mode.id as any)}
                >
                  {mode.label}
                </Button>
              ))}
            </div>
            
            <div className="flex flex-col sm:flex-row gap-4 items-center">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-neutral-400" />
                <input
                  type="text"
                  placeholder="Search feature pairs..."
                  className="w-full pl-10 pr-4 py-2 border border-neutral-200 dark:border-neutral-700 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100 focus:outline-none focus:ring-2 focus:ring-primary-500"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                />
              </div>
              <div className="flex items-center space-x-2">
                <span className="text-sm text-neutral-600 dark:text-neutral-400">Min Strength:</span>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={minInteractionStrength}
                  onChange={(e) => setMinInteractionStrength(parseFloat(e.target.value))}
                  className="w-24"
                />
                <span className="text-sm font-medium text-neutral-900 dark:text-neutral-100 min-w-12">
                  {minInteractionStrength.toFixed(2)}
                </span>
              </div>
            </div>
          </div>
        </Card>
      </motion.div>

      {/* Main Visualization */}
      <motion.div variants={itemVariants}>
        <Card
          title={`${viewMode === 'heatmap' ? 'Interaction Heatmap' : viewMode === 'network' ? 'Network Graph' : 'Pairwise Analysis'}`}
          icon={<TrendingUp className="w-5 h-5 text-primary-600 dark:text-primary-400" />}
        >
          {viewMode === 'heatmap' && (
            <div className="space-y-4">
              <div className="text-sm text-neutral-600 dark:text-neutral-400">
                Color intensity represents interaction strength between feature pairs
              </div>
              
              <div className="overflow-x-auto">
                <div className="inline-block min-w-full">
                  <div className="grid gap-1 p-4 bg-neutral-50 dark:bg-neutral-900 rounded-lg" style={{ gridTemplateColumns: `repeat(${features.length + 1}, minmax(0, 1fr))` }}>
                    {/* Header row */}
                    <div className="p-2"></div>
                    {features.map((feature) => (
                      <div key={feature.name} className="p-2 text-xs text-center text-neutral-600 dark:text-neutral-400 transform -rotate-45 h-16 flex items-end justify-center">
                        <span className="whitespace-nowrap">{feature.name.substring(0, 10)}</span>
                      </div>
                    ))}
                    
                    {/* Data rows */}
                    {features.map((rowFeature, rowIndex) => (
                      <React.Fragment key={rowFeature.name}>
                        <div className="p-2 text-xs text-neutral-600 dark:text-neutral-400 text-right flex items-center justify-end">
                          <span className="whitespace-nowrap">{rowFeature.name.substring(0, 10)}</span>
                        </div>
                        {features.map((colFeature, colIndex) => {
                          const interaction = interactionData.find(i => 
                            (i.feature1 === rowFeature.name && i.feature2 === colFeature.name) ||
                            (i.feature1 === colFeature.name && i.feature2 === rowFeature.name)
                          );
                          
                          const strength = rowIndex === colIndex ? 1 : (interaction?.strength || 0);
                          const type = rowIndex === colIndex ? 'synergy' : (interaction?.type || 'independent');
                          
                          return (
                            <motion.div
                              key={`${rowFeature.name}-${colFeature.name}`}
                              initial={{ opacity: 0, scale: 0 }}
                              animate={{ opacity: 1, scale: 1 }}
                              transition={{ delay: (rowIndex + colIndex) * 0.02 }}
                              className="aspect-square rounded flex items-center justify-center text-xs font-medium text-white cursor-pointer hover:scale-110 transition-transform"
                              style={{
                                backgroundColor: getInteractionBgColor(type, strength),
                                minHeight: '40px',
                              }}
                              title={`${rowFeature.name} vs ${colFeature.name}: ${strength.toFixed(2)}`}
                            >
                              {strength.toFixed(2)}
                            </motion.div>
                          );
                        })}
                      </React.Fragment>
                    ))}
                  </div>
                </div>
              </div>
              
              {/* Legend */}
              <div className="flex items-center justify-center space-x-6 pt-4 border-t border-neutral-200 dark:border-neutral-700">
                <div className="flex items-center space-x-2">
                  <div className="w-4 h-4 rounded bg-green-500"></div>
                  <span className="text-sm text-neutral-600 dark:text-neutral-400">Synergy</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-4 h-4 rounded bg-amber-500"></div>
                  <span className="text-sm text-neutral-600 dark:text-neutral-400">Redundancy</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-4 h-4 rounded bg-neutral-500"></div>
                  <span className="text-sm text-neutral-600 dark:text-neutral-400">Independent</span>
                </div>
              </div>
            </div>
          )}
          
          {viewMode === 'network' && (
            <div className="space-y-4">
              <div className="text-sm text-neutral-600 dark:text-neutral-400">
                Network visualization of feature interactions (nodes represent features, edges represent interactions)
              </div>
              
              <div className="relative h-96 bg-neutral-50 dark:bg-neutral-900 rounded-lg p-4">
                <svg viewBox="0 0 600 400" className="w-full h-full">
                  {/* Feature nodes */}
                  {features.map((feature, index) => {
                    const angle = (index / features.length) * 2 * Math.PI;
                    const x = 300 + Math.cos(angle) * 150;
                    const y = 200 + Math.sin(angle) * 150;
                    
                    return (
                      <motion.g key={feature.name}>
                        <motion.circle
                          initial={{ opacity: 0, scale: 0 }}
                          animate={{ opacity: 1, scale: 1 }}
                          transition={{ delay: index * 0.1 }}
                          cx={x}
                          cy={y}
                          r={20 + feature.importance * 30}
                          fill="currentColor"
                          className="text-primary-500 hover:text-primary-600 cursor-pointer"
                        />
                        <text
                          x={x}
                          y={y + 5}
                          textAnchor="middle"
                          className="text-xs font-medium fill-white pointer-events-none"
                        >
                          {feature.name.substring(0, 8)}
                        </text>
                      </motion.g>
                    );
                  })}
                  
                  {/* Interaction edges */}
                  {filteredInteractions.map((interaction, index) => {
                    const feature1Index = features.findIndex(f => f.name === interaction.feature1);
                    const feature2Index = features.findIndex(f => f.name === interaction.feature2);
                    
                    const angle1 = (feature1Index / features.length) * 2 * Math.PI;
                    const angle2 = (feature2Index / features.length) * 2 * Math.PI;
                    
                    const x1 = 300 + Math.cos(angle1) * 150;
                    const y1 = 200 + Math.sin(angle1) * 150;
                    const x2 = 300 + Math.cos(angle2) * 150;
                    const y2 = 200 + Math.sin(angle2) * 150;
                    
                    return (
                      <motion.line
                        key={`${interaction.feature1}-${interaction.feature2}`}
                        initial={{ opacity: 0, pathLength: 0 }}
                        animate={{ opacity: interaction.strength, pathLength: 1 }}
                        transition={{ delay: 0.5 + index * 0.05 }}
                        x1={x1}
                        y1={y1}
                        x2={x2}
                        y2={y2}
                        stroke={interaction.type === 'synergy' ? '#22c55e' : interaction.type === 'redundancy' ? '#f59e0b' : '#6b7280'}
                        strokeWidth={2 + interaction.strength * 4}
                        className="hover:stroke-primary-500 cursor-pointer"
                      />
                    );
                  })}
                </svg>
              </div>
            </div>
          )}
          
          {viewMode === 'pairwise' && (
            <div className="space-y-4">
              <div className="text-sm text-neutral-600 dark:text-neutral-400">
                Select two features to analyze their interaction pattern
              </div>
              
              <div className="flex flex-wrap gap-2">
                {features.map((feature) => (
                  <Button
                    key={feature.name}
                    variant={selectedFeatures.includes(feature.name) ? 'primary' : 'outline'}
                    size="sm"
                    onClick={() => {
                      const newSelection = selectedFeatures.includes(feature.name)
                        ? selectedFeatures.filter(f => f !== feature.name)
                        : [...selectedFeatures.slice(-1), feature.name];
                      setSelectedFeatures(newSelection.slice(0, 2) as [string, string]);
                    }}
                  >
                    {feature.name}
                  </Button>
                ))}
              </div>
              
              {selectedFeatures.length === 2 && (
                <div className="relative h-64 bg-neutral-50 dark:bg-neutral-900 rounded-lg p-4">
                  <div className="text-center text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
                    {selectedFeatures[0]} vs {selectedFeatures[1]}
                  </div>
                  
                  <svg viewBox="0 0 400 200" className="w-full h-full">
                    {/* Grid */}
                    {[0, 0.25, 0.5, 0.75, 1.0].map((tick) => (
                      <g key={tick}>
                        <line
                          x1={tick * 400}
                          y1={0}
                          x2={tick * 400}
                          y2={200}
                          stroke="currentColor"
                          strokeWidth="0.5"
                          className="text-neutral-300 dark:text-neutral-600"
                        />
                        <line
                          x1={0}
                          y1={tick * 200}
                          x2={400}
                          y2={tick * 200}
                          stroke="currentColor"
                          strokeWidth="0.5"
                          className="text-neutral-300 dark:text-neutral-600"
                        />
                      </g>
                    ))}
                    
                    {/* Scatter plot */}
                    {Array.from({ length: 50 }, (_, i) => {
                      const x = Math.random() * 400;
                      const y = Math.random() * 200;
                      const interaction = interactionData.find(int => 
                        (int.feature1 === selectedFeatures[0] && int.feature2 === selectedFeatures[1]) ||
                        (int.feature1 === selectedFeatures[1] && int.feature2 === selectedFeatures[0])
                      );
                      
                      return (
                        <motion.circle
                          key={i}
                          initial={{ opacity: 0, scale: 0 }}
                          animate={{ opacity: 0.7, scale: 1 }}
                          transition={{ delay: i * 0.02 }}
                          cx={x}
                          cy={y}
                          r="3"
                          fill={interaction?.type === 'synergy' ? '#22c55e' : interaction?.type === 'redundancy' ? '#f59e0b' : '#6b7280'}
                          className="hover:opacity-100 cursor-pointer"
                        />
                      );
                    })}
                  </svg>
                </div>
              )}
            </div>
          )}
        </Card>
      </motion.div>

      {/* Interaction Rankings */}
      <motion.div variants={itemVariants}>
        <Card
          title="Top Feature Interactions"
          icon={<TrendingUp className="w-5 h-5 text-primary-600 dark:text-primary-400" />}
        >
          <div className="space-y-3">
            {filteredInteractions.slice(0, 10).map((interaction, index) => (
              <motion.div
                key={`${interaction.feature1}-${interaction.feature2}`}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.05 }}
                className="flex items-center justify-between p-3 bg-neutral-50 dark:bg-neutral-800 rounded-lg hover:bg-neutral-100 dark:hover:bg-neutral-700 transition-colors"
              >
                <div className="flex items-center space-x-4">
                  <div className="w-8 h-8 bg-primary-100 dark:bg-primary-900 rounded-full flex items-center justify-center">
                    <span className="text-sm font-bold text-primary-600 dark:text-primary-400">
                      {index + 1}
                    </span>
                  </div>
                  <div>
                    <div className="font-medium text-neutral-900 dark:text-neutral-100">
                      {interaction.feature1} ↔ {interaction.feature2}
                    </div>
                    <div className="text-sm text-neutral-600 dark:text-neutral-400">
                      SHAP Interaction: {interaction.shapInteraction > 0 ? '+' : ''}{interaction.shapInteraction.toFixed(3)}
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <div className={`text-lg font-bold ${getStrengthColor(interaction.strength)}`}>
                    {(interaction.strength * 100).toFixed(1)}%
                  </div>
                  <div className={`text-sm font-medium ${getInteractionColor(interaction.type)}`}>
                    {interaction.type}
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </Card>
      </motion.div>

      {/* Summary Statistics */}
      <motion.div variants={itemVariants}>
        <Card
          title="Interaction Summary"
          icon={<BarChart3 className="w-5 h-5 text-primary-600 dark:text-primary-400" />}
        >
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
                {interactionData.filter(i => i.type === 'synergy').length}
              </div>
              <div className="text-sm text-neutral-600 dark:text-neutral-400">
                Synergistic
              </div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
                {interactionData.filter(i => i.type === 'redundancy').length}
              </div>
              <div className="text-sm text-neutral-600 dark:text-neutral-400">
                Redundant
              </div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
                {interactionData.filter(i => i.type === 'independent').length}
              </div>
              <div className="text-sm text-neutral-600 dark:text-neutral-400">
                Independent
              </div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
                {(interactionData.reduce((sum, i) => sum + i.strength, 0) / interactionData.length * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-neutral-600 dark:text-neutral-400">
                Avg Strength
              </div>
            </div>
          </div>
        </Card>
      </motion.div>
    </motion.div>
  );
};

export default FeatureInteractions;