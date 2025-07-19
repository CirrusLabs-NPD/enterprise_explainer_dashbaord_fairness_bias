import React, { useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import { Target, TrendingUp, Search, Filter, BarChart3, Download, RefreshCw } from 'lucide-react';
import Card from '@/components/common/Card';
import Button from '@/components/common/Button';

interface IndividualPredictionsProps {
  modelType: 'classification' | 'regression';
}

interface Instance {
  id: string;
  features: Record<string, number | string>;
  prediction: number;
  actualValue?: number;
  shapValues: Record<string, number>;
  baseValue: number;
  confidence: number;
}

interface ShapContribution {
  feature: string;
  value: number | string;
  shapValue: number;
  contribution: number;
  rank: number;
}

const IndividualPredictions: React.FC<IndividualPredictionsProps> = ({ modelType }) => {
  const [selectedInstanceId, setSelectedInstanceId] = useState<string>('inst-1');
  const [searchTerm, setSearchTerm] = useState('');
  const [sortBy, setSortBy] = useState<'prediction' | 'confidence'>('prediction');

  // Mock instances data
  const mockInstances: Instance[] = [
    {
      id: 'inst-1',
      features: {
        'Customer_Age': 35,
        'Annual_Income': 75000,
        'Credit_Score': 720,
        'Employment_Type': 'Full-time',
        'Account_Balance': 15000,
        'Loan_Amount': 50000,
        'Education_Level': 'Bachelor',
        'Marital_Status': 'Married',
      },
      prediction: 0.76,
      actualValue: 1,
      shapValues: {
        'Customer_Age': 0.08,
        'Annual_Income': 0.15,
        'Credit_Score': 0.12,
        'Employment_Type': 0.09,
        'Account_Balance': 0.06,
        'Loan_Amount': -0.04,
        'Education_Level': 0.03,
        'Marital_Status': 0.02,
      },
      baseValue: 0.51,
      confidence: 0.89,
    },
    {
      id: 'inst-2',
      features: {
        'Customer_Age': 28,
        'Annual_Income': 45000,
        'Credit_Score': 650,
        'Employment_Type': 'Part-time',
        'Account_Balance': 3000,
        'Loan_Amount': 25000,
        'Education_Level': 'High School',
        'Marital_Status': 'Single',
      },
      prediction: 0.34,
      actualValue: 0,
      shapValues: {
        'Customer_Age': -0.05,
        'Annual_Income': -0.08,
        'Credit_Score': -0.06,
        'Employment_Type': -0.07,
        'Account_Balance': -0.03,
        'Loan_Amount': -0.02,
        'Education_Level': -0.04,
        'Marital_Status': -0.01,
      },
      baseValue: 0.51,
      confidence: 0.78,
    },
    {
      id: 'inst-3',
      features: {
        'Customer_Age': 42,
        'Annual_Income': 95000,
        'Credit_Score': 780,
        'Employment_Type': 'Full-time',
        'Account_Balance': 25000,
        'Loan_Amount': 75000,
        'Education_Level': 'Master',
        'Marital_Status': 'Married',
      },
      prediction: 0.89,
      actualValue: 1,
      shapValues: {
        'Customer_Age': 0.12,
        'Annual_Income': 0.18,
        'Credit_Score': 0.15,
        'Employment_Type': 0.08,
        'Account_Balance': 0.09,
        'Loan_Amount': -0.03,
        'Education_Level': 0.05,
        'Marital_Status': 0.02,
      },
      baseValue: 0.51,
      confidence: 0.94,
    },
  ];

  const selectedInstance = mockInstances.find(inst => inst.id === selectedInstanceId)!;

  const shapContributions: ShapContribution[] = useMemo(() => {
    if (!selectedInstance) return [];
    
    const contributions = Object.entries(selectedInstance.shapValues).map(([feature, shapValue]) => ({
      feature,
      value: selectedInstance.features[feature],
      shapValue,
      contribution: Math.abs(shapValue),
      rank: 0,
    }));

    contributions.sort((a, b) => b.contribution - a.contribution);
    contributions.forEach((contrib, index) => {
      contrib.rank = index + 1;
    });

    return contributions;
  }, [selectedInstance]);

  const filteredInstances = useMemo(() => {
    const filtered = mockInstances.filter(inst => 
      inst.id.includes(searchTerm) || 
      Object.values(inst.features).some(val => 
        String(val).toLowerCase().includes(searchTerm.toLowerCase())
      )
    );

    filtered.sort((a, b) => {
      if (sortBy === 'prediction') {
        return b.prediction - a.prediction;
      } else {
        return b.confidence - a.confidence;
      }
    });

    return filtered;
  }, [searchTerm, sortBy]);

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

  const getContributionColor = (shapValue: number) => {
    if (shapValue > 0) return 'text-green-600 dark:text-green-400';
    if (shapValue < 0) return 'text-red-600 dark:text-red-400';
    return 'text-neutral-600 dark:text-neutral-400';
  };

  const getContributionBgColor = (shapValue: number) => {
    if (shapValue > 0) return 'bg-green-500';
    if (shapValue < 0) return 'bg-red-500';
    return 'bg-neutral-500';
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
            Individual Predictions
          </h1>
          <p className="text-neutral-600 dark:text-neutral-400 mt-1">
            Analyze individual prediction explanations using SHAP values
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            size="sm"
            leftIcon={<RefreshCw className="w-4 h-4" />}
          >
            Refresh
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

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Instance Selection */}
        <motion.div variants={itemVariants}>
          <Card
            title="Select Instance"
            icon={<Filter className="w-5 h-5 text-primary-600 dark:text-primary-400" />}
          >
            <div className="space-y-4">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-neutral-400" />
                <input
                  type="text"
                  placeholder="Search instances..."
                  className="w-full pl-10 pr-4 py-2 border border-neutral-200 dark:border-neutral-700 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100 focus:outline-none focus:ring-2 focus:ring-primary-500 text-sm"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                />
              </div>
              
              <div className="flex items-center space-x-2">
                <Button
                  variant={sortBy === 'prediction' ? 'primary' : 'outline'}
                  size="sm"
                  onClick={() => setSortBy('prediction')}
                >
                  By Prediction
                </Button>
                <Button
                  variant={sortBy === 'confidence' ? 'primary' : 'outline'}
                  size="sm"
                  onClick={() => setSortBy('confidence')}
                >
                  By Confidence
                </Button>
              </div>

              <div className="space-y-2 max-h-96 overflow-y-auto">
                {filteredInstances.map((instance) => (
                  <div
                    key={instance.id}
                    onClick={() => setSelectedInstanceId(instance.id)}
                    className={`p-3 rounded-lg border cursor-pointer transition-colors ${
                      selectedInstanceId === instance.id
                        ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                        : 'border-neutral-200 dark:border-neutral-700 hover:bg-neutral-50 dark:hover:bg-neutral-800'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="font-medium text-sm text-neutral-900 dark:text-neutral-100">
                          Instance {instance.id.split('-')[1]}
                        </div>
                        <div className="text-xs text-neutral-500 dark:text-neutral-400">
                          Age: {instance.features.Customer_Age}, Income: ${instance.features.Annual_Income}
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-sm font-bold text-neutral-900 dark:text-neutral-100">
                          {modelType === 'classification' 
                            ? `${(instance.prediction * 100).toFixed(1)}%`
                            : instance.prediction.toFixed(3)
                          }
                        </div>
                        <div className="text-xs text-neutral-500 dark:text-neutral-400">
                          {(instance.confidence * 100).toFixed(1)}% conf
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </Card>
        </motion.div>

        {/* SHAP Analysis */}
        <motion.div variants={itemVariants} className="lg:col-span-2 space-y-6">
          {/* Prediction Summary */}
          <Card
            title="Prediction Summary"
            icon={<Target className="w-5 h-5 text-primary-600 dark:text-primary-400" />}
          >
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
                  {modelType === 'classification' 
                    ? `${(selectedInstance.prediction * 100).toFixed(1)}%`
                    : selectedInstance.prediction.toFixed(3)
                  }
                </div>
                <div className="text-sm text-neutral-600 dark:text-neutral-400">
                  Prediction
                </div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
                  {selectedInstance.actualValue !== undefined ? selectedInstance.actualValue : 'N/A'}
                </div>
                <div className="text-sm text-neutral-600 dark:text-neutral-400">
                  Actual
                </div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
                  {(selectedInstance.confidence * 100).toFixed(1)}%
                </div>
                <div className="text-sm text-neutral-600 dark:text-neutral-400">
                  Confidence
                </div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
                  {selectedInstance.baseValue.toFixed(3)}
                </div>
                <div className="text-sm text-neutral-600 dark:text-neutral-400">
                  Base Value
                </div>
              </div>
            </div>
          </Card>

          {/* SHAP Waterfall Chart */}
          <Card
            title="SHAP Waterfall Chart"
            icon={<BarChart3 className="w-5 h-5 text-primary-600 dark:text-primary-400" />}
          >
            <div className="space-y-4">
              <div className="flex items-center justify-between text-sm text-neutral-600 dark:text-neutral-400">
                <span>Base Value</span>
                <span className="font-mono">{selectedInstance.baseValue.toFixed(3)}</span>
              </div>
              
              <div className="space-y-3">
                {shapContributions.map((contrib, index) => (
                  <motion.div
                    key={contrib.feature}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className="flex items-center space-x-4"
                  >
                    <div className="w-32 text-sm text-neutral-900 dark:text-neutral-100 font-medium">
                      {contrib.feature}
                    </div>
                    <div className="flex-1 flex items-center space-x-2">
                      <div className="relative flex-1 h-8 bg-neutral-200 dark:bg-neutral-700 rounded-lg overflow-hidden">
                        <div
                          className={`absolute top-0 h-full ${getContributionBgColor(contrib.shapValue)} transition-all duration-500`}
                          style={{
                            width: `${Math.abs(contrib.shapValue) * 400}px`,
                            maxWidth: '100%',
                            left: contrib.shapValue > 0 ? '50%' : 'auto',
                            right: contrib.shapValue < 0 ? '50%' : 'auto',
                          }}
                        />
                        <div className="absolute inset-0 flex items-center justify-center">
                          <span className="text-xs font-medium text-white">
                            {contrib.shapValue > 0 ? '+' : ''}{contrib.shapValue.toFixed(3)}
                          </span>
                        </div>
                      </div>
                      <div className="w-20 text-sm text-neutral-600 dark:text-neutral-400">
                        {typeof contrib.value === 'number' ? contrib.value.toFixed(0) : contrib.value}
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
              
              <div className="flex items-center justify-between text-sm font-bold text-neutral-900 dark:text-neutral-100 pt-2 border-t border-neutral-200 dark:border-neutral-700">
                <span>Final Prediction</span>
                <span className="font-mono">{selectedInstance.prediction.toFixed(3)}</span>
              </div>
            </div>
          </Card>

          {/* Feature Contributions */}
          <Card
            title="Feature Contributions"
            icon={<TrendingUp className="w-5 h-5 text-primary-600 dark:text-primary-400" />}
          >
            <div className="space-y-3">
              {shapContributions.map((contrib) => (
                <div
                  key={contrib.feature}
                  className="flex items-center justify-between p-3 bg-neutral-50 dark:bg-neutral-800 rounded-lg"
                >
                  <div className="flex items-center space-x-3">
                    <div className="w-6 h-6 bg-primary-100 dark:bg-primary-900 rounded-full flex items-center justify-center">
                      <span className="text-xs font-bold text-primary-600 dark:text-primary-400">
                        {contrib.rank}
                      </span>
                    </div>
                    <div>
                      <div className="font-medium text-neutral-900 dark:text-neutral-100">
                        {contrib.feature}
                      </div>
                      <div className="text-sm text-neutral-600 dark:text-neutral-400">
                        Value: {typeof contrib.value === 'number' ? contrib.value.toFixed(0) : contrib.value}
                      </div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className={`text-lg font-bold ${getContributionColor(contrib.shapValue)}`}>
                      {contrib.shapValue > 0 ? '+' : ''}{contrib.shapValue.toFixed(3)}
                    </div>
                    <div className="text-sm text-neutral-500 dark:text-neutral-400">
                      {contrib.shapValue > 0 ? 'increases' : 'decreases'}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </motion.div>
      </div>
    </motion.div>
  );
};

export default IndividualPredictions;