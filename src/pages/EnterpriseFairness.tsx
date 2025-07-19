import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  Shield,
  AlertTriangle,
  CheckCircle,
  BarChart3,
  Users,
  Scale,
  TrendingUp,
  TrendingDown,
  Eye,
  FileText,
  Download,
  Settings,
} from 'lucide-react';
import Card from '../components/common/Card';
import MetricCard from '../components/common/MetricCard';

const EnterpriseFairness: React.FC = () => {
  const [selectedModel, setSelectedModel] = useState('model_v2.3');
  const [selectedFramework, setSelectedFramework] = useState('EU_AI_Act');
  
  // Mock fairness metrics data
  const fairnessOverview = {
    overallFairnessScore: 82.4,
    biasViolations: 2,
    complianceStatus: 'Partially Compliant',
    attributesAnalyzed: 4,
    metricsCalculated: 85,
  };

  const biasMetrics = [
    {
      metric: 'Demographic Parity',
      value: 0.12,
      threshold: 0.10,
      status: 'violation',
      description: 'Difference in positive prediction rates',
      complianceFrameworks: ['EU AI Act', 'US EEOC'],
    },
    {
      metric: 'Equalized Odds',
      value: 0.08,
      threshold: 0.10,
      status: 'compliant',
      description: 'Equal TPR and FPR across groups',
      complianceFrameworks: ['EU AI Act'],
    },
    {
      metric: 'Equality of Opportunity',
      value: 0.06,
      threshold: 0.10,
      status: 'compliant',
      description: 'Equal TPR across groups',
      complianceFrameworks: ['US EEOC'],
    },
    {
      metric: 'Calibration',
      value: 0.04,
      threshold: 0.05,
      status: 'compliant',
      description: 'Predicted vs actual positive rates',
      complianceFrameworks: ['FDA'],
    },
    {
      metric: 'Disparate Impact Ratio',
      value: 0.76,
      threshold: 0.80,
      status: 'violation',
      description: '4/5ths rule for adverse impact',
      complianceFrameworks: ['US EEOC'],
    },
  ];

  const sensitiveAttributes = [
    {
      attribute: 'Gender',
      groups: ['Male', 'Female', 'Non-binary'],
      biasDetected: true,
      severity: 'Medium',
      affectedMetrics: ['Demographic Parity', 'Disparate Impact'],
    },
    {
      attribute: 'Race',
      groups: ['White', 'Black', 'Hispanic', 'Asian', 'Other'],
      biasDetected: false,
      severity: 'Low',
      affectedMetrics: [],
    },
    {
      attribute: 'Age Group',
      groups: ['18-25', '26-35', '36-50', '51+'],
      biasDetected: false,
      severity: 'Low',
      affectedMetrics: [],
    },
    {
      attribute: 'Geography',
      groups: ['Urban', 'Suburban', 'Rural'],
      biasDetected: true,
      severity: 'High',
      affectedMetrics: ['Demographic Parity', 'Equalized Odds'],
    },
  ];

  const intersectionalAnalysis = [
    {
      intersection: 'Female × Rural',
      biasScore: 0.23,
      riskLevel: 'High',
      description: 'Significant underrepresentation in positive predictions',
    },
    {
      intersection: 'Male × Urban',
      biasScore: 0.08,
      riskLevel: 'Low',
      description: 'Within acceptable fairness thresholds',
    },
    {
      intersection: 'Non-binary × Suburban',
      biasScore: 0.15,
      riskLevel: 'Medium',
      description: 'Moderate bias detected, requires monitoring',
    },
  ];

  const complianceFrameworks = [
    {
      name: 'EU AI Act',
      status: 'Partially Compliant',
      score: 78,
      violations: 2,
      requirements: [
        'Risk assessment documentation',
        'Bias monitoring and mitigation',
        'Transparency and explainability',
        'Human oversight mechanisms',
      ],
    },
    {
      name: 'US EEOC Guidelines',
      status: 'Non-Compliant',
      score: 65,
      violations: 3,
      requirements: [
        '4/5ths rule compliance',
        'Adverse impact analysis',
        'Job-relatedness validation',
        'Alternative selection procedures',
      ],
    },
    {
      name: 'GDPR Article 22',
      status: 'Compliant',
      score: 92,
      violations: 0,
      requirements: [
        'Automated decision-making transparency',
        'Right to explanation',
        'Data protection by design',
        'Consent management',
      ],
    },
  ];

  const mitigationStrategies = [
    {
      strategy: 'Data Resampling',
      effectiveness: 'High',
      implementation: 'Pre-processing',
      description: 'Balance training data representation across groups',
      estimatedImprovement: '+12% fairness score',
    },
    {
      strategy: 'Fairness Constraints',
      effectiveness: 'Medium',
      implementation: 'In-processing',
      description: 'Add fairness constraints during model training',
      estimatedImprovement: '+8% fairness score',
    },
    {
      strategy: 'Threshold Optimization',
      effectiveness: 'Medium',
      implementation: 'Post-processing',
      description: 'Optimize decision thresholds per group',
      estimatedImprovement: '+6% fairness score',
    },
  ];

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-neutral-900 dark:text-neutral-100">
            Enterprise Fairness Analysis
          </h1>
          <p className="text-neutral-600 dark:text-neutral-400 mt-1">
            Comprehensive bias detection with 80+ metrics and regulatory compliance
          </p>
        </div>
        
        <div className="flex items-center space-x-4">
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="px-4 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
          >
            <option value="model_v2.3">Credit Risk Model v2.3</option>
            <option value="model_v2.2">Credit Risk Model v2.2</option>
            <option value="fraud_model">Fraud Detection Model</option>
          </select>
          
          <select
            value={selectedFramework}
            onChange={(e) => setSelectedFramework(e.target.value)}
            className="px-4 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
          >
            <option value="EU_AI_Act">EU AI Act</option>
            <option value="US_EEOC">US EEOC</option>
            <option value="GDPR">GDPR</option>
            <option value="All">All Frameworks</option>
          </select>
          
          <button className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors">
            <Download className="w-4 h-4" />
            <span>Export Report</span>
          </button>
        </div>
      </div>

      {/* Fairness Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
        <MetricCard
          title="Overall Fairness Score"
          value={`${fairnessOverview.overallFairnessScore}%`}
          change={fairnessOverview.overallFairnessScore > 80 ? '+2.3%' : '-1.2%'}
          changeType={fairnessOverview.overallFairnessScore > 80 ? 'positive' : 'negative'}
          icon={Scale}
          description="Composite fairness across all metrics"
        />
        
        <MetricCard
          title="Bias Violations"
          value={fairnessOverview.biasViolations.toString()}
          change={fairnessOverview.biasViolations === 0 ? '0' : '-1'}
          changeType={fairnessOverview.biasViolations === 0 ? 'neutral' : 'negative'}
          icon={AlertTriangle}
          description="Metrics exceeding thresholds"
        />
        
        <MetricCard
          title="Compliance Status"
          value={fairnessOverview.complianceStatus}
          change="Improving"
          changeType="positive"
          icon={Shield}
          description="Regulatory compliance level"
        />
        
        <MetricCard
          title="Sensitive Attributes"
          value={fairnessOverview.attributesAnalyzed.toString()}
          change="+1"
          changeType="positive"
          icon={Users}
          description="Protected characteristics analyzed"
        />
        
        <MetricCard
          title="Metrics Calculated"
          value={fairnessOverview.metricsCalculated.toString()}
          change="+5"
          changeType="positive"
          icon={BarChart3}
          description="Total fairness metrics computed"
        />
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Bias Metrics */}
        <Card className="lg:col-span-2">
          <div className="p-6">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                Bias Metrics Analysis
              </h3>
              <div className="flex items-center space-x-2">
                <Eye className="w-4 h-4 text-neutral-500" />
                <span className="text-sm text-neutral-500">Real-time monitoring</span>
              </div>
            </div>
            
            <div className="space-y-4">
              {biasMetrics.map((metric, index) => (
                <motion.div
                  key={metric.metric}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className={`p-4 rounded-lg border-l-4 ${
                    metric.status === 'violation'
                      ? 'border-red-500 bg-red-50 dark:bg-red-900/20'
                      : 'border-green-500 bg-green-50 dark:bg-green-900/20'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-3">
                      <span className="font-medium text-neutral-900 dark:text-neutral-100">
                        {metric.metric}
                      </span>
                      <span className={`px-2 py-1 rounded text-xs font-medium ${
                        metric.status === 'violation'
                          ? 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300'
                          : 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300'
                      }`}>
                        {metric.status === 'violation' ? 'Violation' : 'Compliant'}
                      </span>
                    </div>
                    
                    <div className="text-right">
                      <span className="font-bold text-lg">
                        {metric.value.toFixed(3)}
                      </span>
                      <div className="text-xs text-neutral-500">
                        Threshold: {metric.threshold.toFixed(2)}
                      </div>
                    </div>
                  </div>
                  
                  <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-2">
                    {metric.description}
                  </p>
                  
                  <div className="flex items-center space-x-2">
                    <span className="text-xs text-neutral-500">Frameworks:</span>
                    {metric.complianceFrameworks.map((framework) => (
                      <span
                        key={framework}
                        className="px-2 py-1 bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300 rounded text-xs"
                      >
                        {framework}
                      </span>
                    ))}
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </Card>

        {/* Sensitive Attributes */}
        <Card>
          <div className="p-6">
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
              Sensitive Attributes
            </h3>
            
            <div className="space-y-4">
              {sensitiveAttributes.map((attr, index) => (
                <motion.div
                  key={attr.attribute}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="p-4 border border-neutral-200 dark:border-neutral-700 rounded-lg"
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium text-neutral-900 dark:text-neutral-100">
                      {attr.attribute}
                    </span>
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      attr.biasDetected
                        ? attr.severity === 'High'
                          ? 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300'
                          : 'bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300'
                        : 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300'
                    }`}>
                      {attr.biasDetected ? attr.severity : 'Clean'}
                    </span>
                  </div>
                  
                  <div className="text-xs text-neutral-500 dark:text-neutral-400 mb-2">
                    Groups: {attr.groups.join(', ')}
                  </div>
                  
                  {attr.affectedMetrics.length > 0 && (
                    <div className="text-xs">
                      <span className="text-neutral-500">Affected metrics:</span>
                      <div className="mt-1 space-x-1">
                        {attr.affectedMetrics.map((metric) => (
                          <span
                            key={metric}
                            className="inline-block px-2 py-1 bg-neutral-100 dark:bg-neutral-800 text-neutral-700 dark:text-neutral-300 rounded text-xs"
                          >
                            {metric}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </motion.div>
              ))}
            </div>
          </div>
        </Card>
      </div>

      {/* Intersectional Analysis */}
      <Card>
        <div className="p-6">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                Intersectional Bias Analysis
              </h3>
              <p className="text-sm text-neutral-600 dark:text-neutral-400">
                Multi-dimensional bias detection across attribute combinations
              </p>
            </div>
            <Settings className="w-5 h-5 text-neutral-500" />
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {intersectionalAnalysis.map((intersection, index) => (
              <motion.div
                key={intersection.intersection}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                className={`p-4 rounded-lg border ${
                  intersection.riskLevel === 'High'
                    ? 'border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-900/20'
                    : intersection.riskLevel === 'Medium'
                    ? 'border-amber-200 bg-amber-50 dark:border-amber-800 dark:bg-amber-900/20'
                    : 'border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-900/20'
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium text-neutral-900 dark:text-neutral-100">
                    {intersection.intersection}
                  </span>
                  <span className={`px-2 py-1 rounded text-xs font-medium ${
                    intersection.riskLevel === 'High'
                      ? 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300'
                      : intersection.riskLevel === 'Medium'
                      ? 'bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300'
                      : 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300'
                  }`}>
                    {intersection.riskLevel}
                  </span>
                </div>
                
                <div className="text-lg font-bold text-neutral-900 dark:text-neutral-100 mb-2">
                  {intersection.biasScore.toFixed(3)}
                </div>
                
                <p className="text-sm text-neutral-600 dark:text-neutral-400">
                  {intersection.description}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </Card>

      {/* Compliance and Mitigation */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        
        {/* Compliance Status */}
        <Card>
          <div className="p-6">
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
              Regulatory Compliance
            </h3>
            
            <div className="space-y-4">
              {complianceFrameworks.map((framework, index) => (
                <motion.div
                  key={framework.name}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="p-4 border border-neutral-200 dark:border-neutral-700 rounded-lg"
                >
                  <div className="flex items-center justify-between mb-3">
                    <span className="font-medium text-neutral-900 dark:text-neutral-100">
                      {framework.name}
                    </span>
                    <div className="flex items-center space-x-2">
                      <span className={`px-2 py-1 rounded text-xs font-medium ${
                        framework.status === 'Compliant'
                          ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300'
                          : framework.status === 'Partially Compliant'
                          ? 'bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300'
                          : 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300'
                      }`}>
                        {framework.status}
                      </span>
                      <span className="text-sm font-medium text-neutral-900 dark:text-neutral-100">
                        {framework.score}%
                      </span>
                    </div>
                  </div>
                  
                  <div className="w-full bg-neutral-200 dark:bg-neutral-700 rounded-full h-2 mb-3">
                    <div
                      className={`h-2 rounded-full transition-all duration-500 ${
                        framework.score >= 90
                          ? 'bg-green-500'
                          : framework.score >= 70
                          ? 'bg-amber-500'
                          : 'bg-red-500'
                      }`}
                      style={{ width: `${framework.score}%` }}
                    />
                  </div>
                  
                  {framework.violations > 0 && (
                    <div className="text-sm text-red-600 dark:text-red-400 mb-2">
                      {framework.violations} violation{framework.violations !== 1 ? 's' : ''} detected
                    </div>
                  )}
                  
                  <div className="text-xs text-neutral-500 dark:text-neutral-400">
                    Key requirements: {framework.requirements.slice(0, 2).join(', ')}
                    {framework.requirements.length > 2 && '...'}
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </Card>

        {/* Mitigation Strategies */}
        <Card>
          <div className="p-6">
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
              Bias Mitigation Strategies
            </h3>
            
            <div className="space-y-4">
              {mitigationStrategies.map((strategy, index) => (
                <motion.div
                  key={strategy.strategy}
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="p-4 border border-neutral-200 dark:border-neutral-700 rounded-lg hover:border-blue-300 dark:hover:border-blue-600 transition-colors cursor-pointer"
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium text-neutral-900 dark:text-neutral-100">
                      {strategy.strategy}
                    </span>
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      strategy.effectiveness === 'High'
                        ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300'
                        : 'bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300'
                    }`}>
                      {strategy.effectiveness} Impact
                    </span>
                  </div>
                  
                  <div className="text-sm text-neutral-600 dark:text-neutral-400 mb-2">
                    {strategy.description}
                  </div>
                  
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-neutral-500">
                      Type: {strategy.implementation}
                    </span>
                    <span className="text-green-600 dark:text-green-400 font-medium">
                      {strategy.estimatedImprovement}
                    </span>
                  </div>
                </motion.div>
              ))}
            </div>
            
            <button className="w-full mt-4 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors">
              Generate Mitigation Plan
            </button>
          </div>
        </Card>
      </div>
    </div>
  );
};

export default EnterpriseFairness;