import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  Search,
  AlertTriangle,
  TrendingDown,
  Clock,
  BarChart3,
  ArrowRight,
  CheckCircle,
  XCircle,
  Lightbulb,
  Target,
  GitBranch,
  Database,
  Settings,
  Zap,
  FileText,
  Play,
  Filter,
} from 'lucide-react';
import Card from '../components/common/Card';
import MetricCard from '../components/common/MetricCard';

const RootCauseAnalysis: React.FC = () => {
  const [selectedIssue, setSelectedIssue] = useState('issue_001');
  const [analysisStatus, setAnalysisStatus] = useState('completed');

  // Mock root cause analysis data
  const analysisOverview = {
    activeInvestigations: 3,
    resolvedIssues: 15,
    averageResolutionTime: '2.4 hours',
    automatedHypotheses: 28,
    successRate: 87.5,
    criticalIssues: 1,
  };

  const activeIssues = [
    {
      id: 'issue_001',
      title: 'Model Accuracy Degradation - Credit Risk Model',
      severity: 'High',
      model: 'credit_risk_v2.3',
      detected: '2024-01-19T09:15:00Z',
      status: 'investigating',
      impact: 'High',
      metrics: {
        accuracy: { current: 0.78, baseline: 0.85, change: -8.2 },
        precision: { current: 0.72, baseline: 0.82, change: -12.2 },
        recall: { current: 0.84, baseline: 0.88, change: -4.5 },
      },
      assignee: 'ML Team',
    },
    {
      id: 'issue_002',
      title: 'Data Drift Alert - Fraud Detection',
      severity: 'Medium',
      model: 'fraud_detection_v1.8',
      detected: '2024-01-19T11:30:00Z',
      status: 'analyzing',
      impact: 'Medium',
      metrics: {
        jsd: { current: 0.23, baseline: 0.08, change: 187.5 },
        psi: { current: 0.15, baseline: 0.05, change: 200.0 },
      },
      assignee: 'Data Science Team',
    },
    {
      id: 'issue_003',
      title: 'Bias Threshold Violation - Loan Approval',
      severity: 'Critical',
      model: 'loan_approval_v1.5',
      detected: '2024-01-19T14:45:00Z',
      status: 'hypothesis_generated',
      impact: 'Critical',
      metrics: {
        demographic_parity: { current: 0.18, baseline: 0.08, change: 125.0 },
        equalized_odds: { current: 0.12, baseline: 0.06, change: 100.0 },
      },
      assignee: 'Fairness Team',
    },
  ];

  const hypotheses = [
    {
      id: 'hyp_001',
      hypothesis: 'Recent data distribution shift due to seasonal patterns',
      confidence: 0.85,
      evidence: [
        'Feature correlation analysis shows 23% change in income distribution',
        'Temporal analysis indicates alignment with Q4 seasonal trends',
        'Similar patterns observed in historical data from previous years',
      ],
      category: 'Data Drift',
      impact: 'High',
      likelihood: 'Very Likely',
      supportingData: {
        correlationChange: 0.23,
        temporalAlignment: 0.91,
        historicalSimilarity: 0.88,
      },
    },
    {
      id: 'hyp_002',
      hypothesis: 'Training data bias in recent model update',
      confidence: 0.72,
      evidence: [
        'Model version 2.3 deployed 3 days before issue detection',
        'Training dataset shows underrepresentation of certain demographics',
        'Bias metrics degraded specifically after model deployment',
      ],
      category: 'Model Bias',
      impact: 'Critical',
      likelihood: 'Likely',
      supportingData: {
        deploymentAlignment: 0.95,
        datasetBias: 0.68,
        metricCorrelation: 0.89,
      },
    },
    {
      id: 'hyp_003',
      hypothesis: 'Infrastructure latency affecting feature computation',
      confidence: 0.45,
      evidence: [
        'Increased feature computation time observed',
        'Database query performance degraded by 15%',
        'Feature engineering pipeline showing timeout errors',
      ],
      category: 'Infrastructure',
      impact: 'Medium',
      likelihood: 'Possible',
      supportingData: {
        latencyIncrease: 0.15,
        queryPerformance: 0.85,
        timeoutFrequency: 0.08,
      },
    },
  ];

  const investigationSteps = [
    {
      step: 1,
      title: 'Issue Detection & Classification',
      status: 'completed',
      timestamp: '2024-01-19T09:15:00Z',
      details: 'Automated monitoring detected accuracy degradation below threshold',
      evidence: ['Performance metrics', 'Alert logs', 'Model metadata'],
      duration: '2 minutes',
    },
    {
      step: 2,
      title: 'Data Collection & Analysis',
      status: 'completed',
      timestamp: '2024-01-19T09:17:00Z',
      details: 'Gathered recent prediction data, feature distributions, and system logs',
      evidence: ['Prediction logs', 'Feature analysis', 'System metrics'],
      duration: '15 minutes',
    },
    {
      step: 3,
      title: 'Hypothesis Generation',
      status: 'completed',
      timestamp: '2024-01-19T09:32:00Z',
      details: 'AI-powered analysis generated 3 potential root causes with confidence scores',
      evidence: ['Statistical analysis', 'Pattern recognition', 'Historical comparisons'],
      duration: '8 minutes',
    },
    {
      step: 4,
      title: 'Evidence Validation',
      status: 'in_progress',
      timestamp: '2024-01-19T09:40:00Z',
      details: 'Validating hypotheses through controlled experiments and data analysis',
      evidence: ['A/B testing', 'Feature importance', 'Correlation analysis'],
      duration: '45 minutes (ongoing)',
    },
    {
      step: 5,
      title: 'Root Cause Confirmation',
      status: 'pending',
      timestamp: null,
      details: 'Final validation and confirmation of primary root cause',
      evidence: [],
      duration: 'Estimated 20 minutes',
    },
    {
      step: 6,
      title: 'Remediation Planning',
      status: 'pending',
      timestamp: null,
      details: 'Generate actionable remediation steps and implementation plan',
      evidence: [],
      duration: 'Estimated 15 minutes',
    },
  ];

  const recommendations = [
    {
      id: 'rec_001',
      title: 'Immediate Model Rollback',
      priority: 'Critical',
      category: 'Emergency Response',
      description: 'Rollback to previous model version (v2.2) to restore baseline performance',
      estimatedImpact: 'Restore 85% accuracy within 15 minutes',
      effort: 'Low',
      timeline: '15 minutes',
      confidence: 0.95,
      steps: [
        'Deploy model version 2.2 to production',
        'Update routing configuration',
        'Validate performance restoration',
        'Monitor for 2 hours',
      ],
    },
    {
      id: 'rec_002',
      title: 'Retrain with Balanced Dataset',
      priority: 'High',
      category: 'Model Improvement',
      description: 'Address training data bias by rebalancing demographic representation',
      estimatedImpact: 'Improve fairness metrics by 60-80%',
      effort: 'Medium',
      timeline: '2-3 days',
      confidence: 0.82,
      steps: [
        'Audit training dataset for bias',
        'Apply sampling techniques for balance',
        'Retrain model with fairness constraints',
        'Validate on holdout test set',
        'Deploy with gradual rollout',
      ],
    },
    {
      id: 'rec_003',
      title: 'Enhanced Data Monitoring',
      priority: 'Medium',
      category: 'Prevention',
      description: 'Implement proactive monitoring for data distribution changes',
      estimatedImpact: 'Detect drift 70% earlier',
      effort: 'Medium',
      timeline: '1 week',
      confidence: 0.78,
      steps: [
        'Deploy real-time drift detection',
        'Set up automated alerts',
        'Create drift visualization dashboard',
        'Train team on new monitoring tools',
      ],
    },
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'text-green-600 bg-green-100 dark:bg-green-900/30 dark:text-green-300';
      case 'in_progress':
        return 'text-blue-600 bg-blue-100 dark:bg-blue-900/30 dark:text-blue-300';
      case 'pending':
        return 'text-gray-600 bg-gray-100 dark:bg-gray-900/30 dark:text-gray-300';
      default:
        return 'text-gray-600 bg-gray-100 dark:bg-gray-900/30 dark:text-gray-300';
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'Critical':
        return 'text-red-600 bg-red-100 dark:bg-red-900/30 dark:text-red-300';
      case 'High':
        return 'text-orange-600 bg-orange-100 dark:bg-orange-900/30 dark:text-orange-300';
      case 'Medium':
        return 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900/30 dark:text-yellow-300';
      case 'Low':
        return 'text-blue-600 bg-blue-100 dark:bg-blue-900/30 dark:text-blue-300';
      default:
        return 'text-gray-600 bg-gray-100 dark:bg-gray-900/30 dark:text-gray-300';
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'Critical':
        return 'text-red-600 bg-red-100 dark:bg-red-900/30 dark:text-red-300';
      case 'High':
        return 'text-orange-600 bg-orange-100 dark:bg-orange-900/30 dark:text-orange-300';
      case 'Medium':
        return 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900/30 dark:text-yellow-300';
      case 'Low':
        return 'text-blue-600 bg-blue-100 dark:bg-blue-900/30 dark:text-blue-300';
      default:
        return 'text-gray-600 bg-gray-100 dark:bg-gray-900/30 dark:text-gray-300';
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-neutral-900 dark:text-neutral-100">
            Root Cause Analysis
          </h1>
          <p className="text-neutral-600 dark:text-neutral-400 mt-1">
            AI-powered investigation and automated hypothesis generation for ML issues
          </p>
        </div>
        
        <div className="flex items-center space-x-4">
          <button className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors">
            <Play className="w-4 h-4" />
            <span>Start Investigation</span>
          </button>
          
          <button className="flex items-center space-x-2 px-4 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg hover:bg-neutral-50 dark:hover:bg-neutral-800 transition-colors">
            <Settings className="w-4 h-4" />
            <span>Settings</span>
          </button>
        </div>
      </div>

      {/* Analysis Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-6">
        <MetricCard
          title="Active Investigations"
          value={analysisOverview.activeInvestigations.toString()}
          change="+1"
          changeType="neutral"
          icon={Search}
          description="Currently analyzing issues"
        />
        
        <MetricCard
          title="Resolved Issues"
          value={analysisOverview.resolvedIssues.toString()}
          change="+3"
          changeType="positive"
          icon={CheckCircle}
          description="Issues resolved this week"
        />
        
        <MetricCard
          title="Avg Resolution Time"
          value={analysisOverview.averageResolutionTime}
          change="-0.3 hours"
          changeType="positive"
          icon={Clock}
          description="Time to identify root cause"
        />
        
        <MetricCard
          title="AI Hypotheses"
          value={analysisOverview.automatedHypotheses.toString()}
          change="+5"
          changeType="positive"
          icon={Lightbulb}
          description="Generated this week"
        />
        
        <MetricCard
          title="Success Rate"
          value={`${analysisOverview.successRate}%`}
          change="+2.1%"
          changeType="positive"
          icon={Target}
          description="Hypothesis accuracy rate"
        />
        
        <MetricCard
          title="Critical Issues"
          value={analysisOverview.criticalIssues.toString()}
          change="0"
          changeType="neutral"
          icon={AlertTriangle}
          description="High priority investigations"
        />
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Active Issues */}
        <Card>
          <div className="p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                Active Issues
              </h3>
              <Filter className="w-4 h-4 text-neutral-500" />
            </div>
            
            <div className="space-y-4">
              {activeIssues.map((issue, index) => (
                <motion.div
                  key={issue.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className={`p-4 border-l-4 rounded-lg cursor-pointer transition-all ${
                    selectedIssue === issue.id
                      ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                      : issue.severity === 'Critical'
                      ? 'border-red-500 bg-red-50 dark:bg-red-900/20'
                      : issue.severity === 'High'
                      ? 'border-orange-500 bg-orange-50 dark:bg-orange-900/20'
                      : 'border-yellow-500 bg-yellow-50 dark:bg-yellow-900/20'
                  }`}
                  onClick={() => setSelectedIssue(issue.id)}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className={`px-2 py-1 rounded text-xs font-medium ${getSeverityColor(issue.severity)}`}>
                      {issue.severity}
                    </span>
                    <span className={`px-2 py-1 rounded text-xs font-medium ${getStatusColor(issue.status)}`}>
                      {issue.status.replace('_', ' ')}
                    </span>
                  </div>
                  
                  <h4 className="font-medium text-neutral-900 dark:text-neutral-100 mb-2">
                    {issue.title}
                  </h4>
                  
                  <div className="text-sm text-neutral-600 dark:text-neutral-400 space-y-1">
                    <div>Model: {issue.model}</div>
                    <div>Detected: {new Date(issue.detected).toLocaleString()}</div>
                    <div>Assignee: {issue.assignee}</div>
                  </div>
                  
                  <div className="mt-3 pt-3 border-t border-neutral-200 dark:border-neutral-700">
                    <div className="text-xs text-neutral-500 mb-1">Key Metrics:</div>
                    <div className="space-y-1">
                      {Object.entries(issue.metrics).map(([metric, data]) => (
                        <div key={metric} className="flex items-center justify-between text-xs">
                          <span className="capitalize">{metric}:</span>
                          <span className={`font-medium ${
                            data.change < 0 ? 'text-red-600' : 'text-green-600'
                          }`}>
                            {data.current} ({data.change > 0 ? '+' : ''}{data.change.toFixed(1)}%)
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </Card>

        {/* Investigation Timeline */}
        <Card className="lg:col-span-2">
          <div className="p-6">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                Investigation Timeline
              </h3>
              <div className="text-sm text-neutral-500">
                Issue: {selectedIssue}
              </div>
            </div>
            
            <div className="space-y-6">
              {investigationSteps.map((step, index) => (
                <motion.div
                  key={step.step}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="flex items-start space-x-4"
                >
                  <div className="flex flex-col items-center">
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                      step.status === 'completed'
                        ? 'bg-green-500 text-white'
                        : step.status === 'in_progress'
                        ? 'bg-blue-500 text-white'
                        : 'bg-gray-300 text-gray-600'
                    }`}>
                      {step.status === 'completed' ? (
                        <CheckCircle className="w-4 h-4" />
                      ) : step.status === 'in_progress' ? (
                        <Clock className="w-4 h-4" />
                      ) : (
                        <span className="text-sm font-medium">{step.step}</span>
                      )}
                    </div>
                    {index < investigationSteps.length - 1 && (
                      <div className={`w-0.5 h-12 mt-2 ${
                        step.status === 'completed' ? 'bg-green-500' : 'bg-gray-300'
                      }`} />
                    )}
                  </div>
                  
                  <div className="flex-1">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="font-medium text-neutral-900 dark:text-neutral-100">
                        {step.title}
                      </h4>
                      <span className={`px-2 py-1 rounded text-xs font-medium ${getStatusColor(step.status)}`}>
                        {step.status.replace('_', ' ')}
                      </span>
                    </div>
                    
                    <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-2">
                      {step.details}
                    </p>
                    
                    <div className="flex items-center justify-between text-xs text-neutral-500">
                      <div>
                        {step.timestamp && (
                          <span>Started: {new Date(step.timestamp).toLocaleString()}</span>
                        )}
                      </div>
                      <div>Duration: {step.duration}</div>
                    </div>
                    
                    {step.evidence.length > 0 && (
                      <div className="mt-2">
                        <div className="text-xs text-neutral-500 mb-1">Evidence:</div>
                        <div className="flex flex-wrap gap-1">
                          {step.evidence.map((evidence) => (
                            <span
                              key={evidence}
                              className="px-2 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded text-xs"
                            >
                              {evidence}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </Card>
      </div>

      {/* Hypotheses and Recommendations */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        
        {/* Generated Hypotheses */}
        <Card>
          <div className="p-6">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                AI-Generated Hypotheses
              </h3>
              <Lightbulb className="w-5 h-5 text-yellow-500" />
            </div>
            
            <div className="space-y-4">
              {hypotheses.map((hypothesis, index) => (
                <motion.div
                  key={hypothesis.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="p-4 border border-neutral-200 dark:border-neutral-700 rounded-lg"
                >
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-sm font-medium text-neutral-900 dark:text-neutral-100">
                      {hypothesis.category}
                    </span>
                    <div className="flex items-center space-x-2">
                      <span className={`px-2 py-1 rounded text-xs font-medium ${getPriorityColor(hypothesis.impact)}`}>
                        {hypothesis.impact} Impact
                      </span>
                      <span className="text-sm font-medium text-neutral-900 dark:text-neutral-100">
                        {(hypothesis.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                  
                  <h4 className="font-medium text-neutral-900 dark:text-neutral-100 mb-2">
                    {hypothesis.hypothesis}
                  </h4>
                  
                  <div className="mb-3">
                    <div className="text-xs text-neutral-500 mb-1">Confidence Level</div>
                    <div className="w-full bg-neutral-200 dark:bg-neutral-700 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full ${
                          hypothesis.confidence >= 0.8
                            ? 'bg-green-500'
                            : hypothesis.confidence >= 0.6
                            ? 'bg-yellow-500'
                            : 'bg-red-500'
                        }`}
                        style={{ width: `${hypothesis.confidence * 100}%` }}
                      />
                    </div>
                    <div className="text-xs text-neutral-500 mt-1">{hypothesis.likelihood}</div>
                  </div>
                  
                  <div className="text-xs text-neutral-600 dark:text-neutral-400">
                    <div className="font-medium mb-1">Supporting Evidence:</div>
                    <ul className="space-y-1">
                      {hypothesis.evidence.slice(0, 2).map((evidence, idx) => (
                        <li key={idx} className="flex items-start space-x-2">
                          <span className="text-neutral-400 mt-1">•</span>
                          <span>{evidence}</span>
                        </li>
                      ))}
                      {hypothesis.evidence.length > 2 && (
                        <li className="text-blue-600 dark:text-blue-400 cursor-pointer">
                          +{hypothesis.evidence.length - 2} more evidence items
                        </li>
                      )}
                    </ul>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </Card>

        {/* Recommendations */}
        <Card>
          <div className="p-6">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                Recommended Actions
              </h3>
              <Target className="w-5 h-5 text-green-500" />
            </div>
            
            <div className="space-y-4">
              {recommendations.map((rec, index) => (
                <motion.div
                  key={rec.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="p-4 border border-neutral-200 dark:border-neutral-700 rounded-lg hover:border-blue-300 dark:hover:border-blue-600 transition-colors cursor-pointer"
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      <span className={`px-2 py-1 rounded text-xs font-medium ${getPriorityColor(rec.priority)}`}>
                        {rec.priority}
                      </span>
                      <span className="text-xs text-neutral-500">{rec.category}</span>
                    </div>
                    <span className="text-sm font-medium text-neutral-900 dark:text-neutral-100">
                      {(rec.confidence * 100).toFixed(0)}% confidence
                    </span>
                  </div>
                  
                  <h4 className="font-medium text-neutral-900 dark:text-neutral-100 mb-2">
                    {rec.title}
                  </h4>
                  
                  <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-3">
                    {rec.description}
                  </p>
                  
                  <div className="grid grid-cols-2 gap-4 text-xs mb-3">
                    <div>
                      <span className="text-neutral-500">Impact:</span>
                      <div className="font-medium">{rec.estimatedImpact}</div>
                    </div>
                    <div>
                      <span className="text-neutral-500">Timeline:</span>
                      <div className="font-medium">{rec.timeline}</div>
                    </div>
                  </div>
                  
                  <div className="text-xs">
                    <div className="text-neutral-500 mb-1">Implementation Steps:</div>
                    <ol className="space-y-1">
                      {rec.steps.slice(0, 2).map((step, idx) => (
                        <li key={idx} className="flex items-start space-x-2">
                          <span className="text-neutral-400">{idx + 1}.</span>
                          <span>{step}</span>
                        </li>
                      ))}
                      {rec.steps.length > 2 && (
                        <li className="text-blue-600 dark:text-blue-400">
                          +{rec.steps.length - 2} more steps
                        </li>
                      )}
                    </ol>
                  </div>
                  
                  <div className="mt-3 pt-3 border-t border-neutral-200 dark:border-neutral-700">
                    <button className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors text-sm font-medium">
                      Implement Solution
                    </button>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </Card>
      </div>

      {/* Analysis Insights */}
      <Card>
        <div className="p-6">
          <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
            Analysis Insights & Historical Patterns
          </h3>
          
          {/* Mock Chart Placeholder */}
          <div className="h-64 bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg flex items-center justify-center border-2 border-dashed border-purple-200 dark:border-purple-700">
            <div className="text-center">
              <BarChart3 className="w-12 h-12 text-purple-400 mx-auto mb-2" />
              <p className="text-purple-600 dark:text-purple-400 font-medium">
                Historical Root Cause Analysis
              </p>
              <p className="text-sm text-purple-500 dark:text-purple-300 mt-1">
                Patterns, trends, and correlation analysis
              </p>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default RootCauseAnalysis;