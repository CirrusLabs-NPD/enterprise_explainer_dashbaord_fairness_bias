import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  FlaskConical,
  TrendingUp,
  Users,
  BarChart3,
  Calendar,
  AlertCircle,
  CheckCircle,
  XCircle,
  Play,
  Pause,
  Settings,
  Download,
  RefreshCw,
  Filter,
  Search,
  Plus,
  Eye,
  Edit,
  Trash2
} from 'lucide-react';
import { cn } from '../utils';

interface ABTest {
  id: string;
  name: string;
  description: string;
  status: 'draft' | 'running' | 'completed' | 'paused' | 'cancelled';
  modelA: {
    id: string;
    name: string;
    version: string;
  };
  modelB: {
    id: string;
    name: string;
    version: string;
  };
  trafficSplit: {
    modelA: number;
    modelB: number;
  };
  metrics: {
    accuracy: { modelA: number; modelB: number };
    precision: { modelA: number; modelB: number };
    recall: { modelA: number; modelB: number };
    f1Score: { modelA: number; modelB: number };
    latency: { modelA: number; modelB: number };
  };
  startDate: string;
  endDate?: string;
  sampleSize: number;
  currentSamples: number;
  significanceLevel: number;
  statisticalSignificance: boolean;
  winner?: 'modelA' | 'modelB' | 'inconclusive';
  createdBy: string;
  createdAt: string;
  businessMetrics?: {
    revenue: { modelA: number; modelB: number };
    conversions: { modelA: number; modelB: number };
    userSatisfaction: { modelA: number; modelB: number };
  };
}

interface NewTestConfig {
  name: string;
  description: string;
  modelA: string;
  modelB: string;
  trafficSplit: number;
  duration: number;
  sampleSize: number;
  significanceLevel: number;
  metrics: string[];
}

const mockABTests: ABTest[] = [
  {
    id: 'test-001',
    name: 'Credit Risk Model v2.1 vs v2.0',
    description: 'Testing improved credit risk model with additional features',
    status: 'running',
    modelA: { id: 'model-v20', name: 'Credit Risk v2.0', version: '2.0.1' },
    modelB: { id: 'model-v21', name: 'Credit Risk v2.1', version: '2.1.0' },
    trafficSplit: { modelA: 50, modelB: 50 },
    metrics: {
      accuracy: { modelA: 0.876, modelB: 0.891 },
      precision: { modelA: 0.823, modelB: 0.847 },
      recall: { modelA: 0.901, modelB: 0.912 },
      f1Score: { modelA: 0.860, modelB: 0.878 },
      latency: { modelA: 45, modelB: 52 }
    },
    startDate: '2024-01-10T08:00:00Z',
    sampleSize: 10000,
    currentSamples: 7834,
    significanceLevel: 0.05,
    statisticalSignificance: true,
    winner: 'modelB',
    createdBy: 'Data Science Team',
    createdAt: '2024-01-09T14:30:00Z',
    businessMetrics: {
      revenue: { modelA: 1250000, modelB: 1287500 },
      conversions: { modelA: 0.156, modelB: 0.162 },
      userSatisfaction: { modelA: 4.2, modelB: 4.5 }
    }
  },
  {
    id: 'test-002',
    name: 'Fraud Detection Enhanced Features',
    description: 'Testing enhanced fraud detection with transaction patterns',
    status: 'completed',
    modelA: { id: 'fraud-v10', name: 'Fraud Detection v1.0', version: '1.0.3' },
    modelB: { id: 'fraud-v11', name: 'Fraud Detection v1.1', version: '1.1.0' },
    trafficSplit: { modelA: 30, modelB: 70 },
    metrics: {
      accuracy: { modelA: 0.943, modelB: 0.967 },
      precision: { modelA: 0.889, modelB: 0.923 },
      recall: { modelA: 0.756, modelB: 0.834 },
      f1Score: { modelA: 0.817, modelB: 0.876 },
      latency: { modelA: 23, modelB: 28 }
    },
    startDate: '2024-01-01T00:00:00Z',
    endDate: '2024-01-08T23:59:59Z',
    sampleSize: 50000,
    currentSamples: 50000,
    significanceLevel: 0.01,
    statisticalSignificance: true,
    winner: 'modelB',
    createdBy: 'ML Engineering Team',
    createdAt: '2023-12-28T16:45:00Z',
    businessMetrics: {
      revenue: { modelA: 2100000, modelB: 2234000 },
      conversions: { modelA: 0.234, modelB: 0.267 },
      userSatisfaction: { modelA: 4.1, modelB: 4.3 }
    }
  },
  {
    id: 'test-003',
    name: 'Customer Churn Prediction A/B',
    description: 'Comparing gradient boosting vs neural network approaches',
    status: 'draft',
    modelA: { id: 'churn-gb', name: 'Churn Gradient Boosting', version: '1.0.0' },
    modelB: { id: 'churn-nn', name: 'Churn Neural Network', version: '1.0.0' },
    trafficSplit: { modelA: 50, modelB: 50 },
    metrics: {
      accuracy: { modelA: 0, modelB: 0 },
      precision: { modelA: 0, modelB: 0 },
      recall: { modelA: 0, modelB: 0 },
      f1Score: { modelA: 0, modelB: 0 },
      latency: { modelA: 0, modelB: 0 }
    },
    startDate: '2024-01-20T09:00:00Z',
    sampleSize: 25000,
    currentSamples: 0,
    significanceLevel: 0.05,
    statisticalSignificance: false,
    createdBy: 'Product Team',
    createdAt: '2024-01-15T11:20:00Z'
  }
];

const ABTestingDashboard: React.FC = () => {
  const [tests, setTests] = useState<ABTest[]>(mockABTests);
  const [selectedTest, setSelectedTest] = useState<ABTest | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'running' | 'completed' | 'create'>('overview');
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<'all' | 'running' | 'completed' | 'draft'>('all');
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newTestConfig, setNewTestConfig] = useState<NewTestConfig>({
    name: '',
    description: '',
    modelA: '',
    modelB: '',
    trafficSplit: 50,
    duration: 7,
    sampleSize: 10000,
    significanceLevel: 0.05,
    metrics: ['accuracy', 'precision', 'recall']
  });

  const filteredTests = tests.filter(test => {
    const matchesSearch = test.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         test.description.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStatus = statusFilter === 'all' || test.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  const testStats = {
    total: tests.length,
    running: tests.filter(t => t.status === 'running').length,
    completed: tests.filter(t => t.status === 'completed').length,
    draft: tests.filter(t => t.status === 'draft').length,
    significant: tests.filter(t => t.statisticalSignificance).length
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running':
        return <Play className="w-4 h-4 text-green-500" />;
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-blue-500" />;
      case 'paused':
        return <Pause className="w-4 h-4 text-yellow-500" />;
      case 'cancelled':
        return <XCircle className="w-4 h-4 text-red-500" />;
      default:
        return <Settings className="w-4 h-4 text-gray-500" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
        return 'text-green-600 bg-green-50 dark:bg-green-900/20 dark:text-green-400';
      case 'completed':
        return 'text-blue-600 bg-blue-50 dark:bg-blue-900/20 dark:text-blue-400';
      case 'paused':
        return 'text-yellow-600 bg-yellow-50 dark:bg-yellow-900/20 dark:text-yellow-400';
      case 'cancelled':
        return 'text-red-600 bg-red-50 dark:bg-red-900/20 dark:text-red-400';
      default:
        return 'text-gray-600 bg-gray-50 dark:bg-gray-900/20 dark:text-gray-400';
    }
  };

  const calculateProgress = (test: ABTest) => {
    return Math.min((test.currentSamples / test.sampleSize) * 100, 100);
  };

  const createNewTest = async () => {
    // API call would go here
    const newTest: ABTest = {
      id: `test-${Date.now()}`,
      name: newTestConfig.name,
      description: newTestConfig.description,
      status: 'draft',
      modelA: { id: newTestConfig.modelA, name: `Model A (${newTestConfig.modelA})`, version: '1.0.0' },
      modelB: { id: newTestConfig.modelB, name: `Model B (${newTestConfig.modelB})`, version: '1.0.0' },
      trafficSplit: { modelA: newTestConfig.trafficSplit, modelB: 100 - newTestConfig.trafficSplit },
      metrics: {
        accuracy: { modelA: 0, modelB: 0 },
        precision: { modelA: 0, modelB: 0 },
        recall: { modelA: 0, modelB: 0 },
        f1Score: { modelA: 0, modelB: 0 },
        latency: { modelA: 0, modelB: 0 }
      },
      startDate: new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString(),
      sampleSize: newTestConfig.sampleSize,
      currentSamples: 0,
      significanceLevel: newTestConfig.significanceLevel,
      statisticalSignificance: false,
      createdBy: 'Current User',
      createdAt: new Date().toISOString()
    };
    
    setTests(prev => [...prev, newTest]);
    setShowCreateModal(false);
    setNewTestConfig({
      name: '',
      description: '',
      modelA: '',
      modelB: '',
      trafficSplit: 50,
      duration: 7,
      sampleSize: 10000,
      significanceLevel: 0.05,
      metrics: ['accuracy', 'precision', 'recall']
    });
  };

  const startTest = async (testId: string) => {
    // API call to /api/v1/model-monitoring/setup-ab-test would go here
    setTests(prev => prev.map(test => 
      test.id === testId ? { ...test, status: 'running' as const } : test
    ));
  };

  const pauseTest = async (testId: string) => {
    setTests(prev => prev.map(test => 
      test.id === testId ? { ...test, status: 'paused' as const } : test
    ));
  };

  const deleteTest = async (testId: string) => {
    setTests(prev => prev.filter(test => test.id !== testId));
  };

  return (
    <div className="min-h-screen bg-neutral-50 dark:bg-neutral-900">
      {/* Header */}
      <div className="bg-white dark:bg-neutral-800 border-b border-neutral-200 dark:border-neutral-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-neutral-900 dark:text-neutral-100 flex items-center gap-3">
                <FlaskConical className="w-8 h-8 text-purple-500" />
                A/B Testing Dashboard
              </h1>
              <p className="text-sm text-neutral-600 dark:text-neutral-400 mt-1">
                Design, monitor, and analyze A/B tests for model performance comparison
              </p>
            </div>
            
            <div className="flex items-center gap-3">
              <button
                onClick={() => setShowCreateModal(true)}
                className="flex items-center gap-2 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
              >
                <Plus className="w-4 h-4" />
                New A/B Test
              </button>
              <button className="p-2 text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100">
                <RefreshCw className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Stats Overview */}
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4 mt-6">
            <div className="bg-gradient-to-r from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded-lg p-4">
              <div className="text-xs font-medium text-blue-600 dark:text-blue-400 uppercase tracking-wider">
                Total Tests
              </div>
              <div className="text-2xl font-bold text-blue-900 dark:text-blue-100">
                {testStats.total}
              </div>
            </div>
            
            <div className="bg-gradient-to-r from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 rounded-lg p-4">
              <div className="text-xs font-medium text-green-600 dark:text-green-400 uppercase tracking-wider">
                Running
              </div>
              <div className="text-2xl font-bold text-green-900 dark:text-green-100">
                {testStats.running}
              </div>
            </div>
            
            <div className="bg-gradient-to-r from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 rounded-lg p-4">
              <div className="text-xs font-medium text-purple-600 dark:text-purple-400 uppercase tracking-wider">
                Completed
              </div>
              <div className="text-2xl font-bold text-purple-900 dark:text-purple-100">
                {testStats.completed}
              </div>
            </div>
            
            <div className="bg-gradient-to-r from-yellow-50 to-yellow-100 dark:from-yellow-900/20 dark:to-yellow-800/20 rounded-lg p-4">
              <div className="text-xs font-medium text-yellow-600 dark:text-yellow-400 uppercase tracking-wider">
                Draft
              </div>
              <div className="text-2xl font-bold text-yellow-900 dark:text-yellow-100">
                {testStats.draft}
              </div>
            </div>
            
            <div className="bg-gradient-to-r from-indigo-50 to-indigo-100 dark:from-indigo-900/20 dark:to-indigo-800/20 rounded-lg p-4">
              <div className="text-xs font-medium text-indigo-600 dark:text-indigo-400 uppercase tracking-wider">
                Significant
              </div>
              <div className="text-2xl font-bold text-indigo-900 dark:text-indigo-100">
                {testStats.significant}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="bg-white dark:bg-neutral-800 rounded-lg border border-neutral-200 dark:border-neutral-700 p-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="relative">
              <Search className="absolute left-3 top-2.5 w-4 h-4 text-neutral-400" />
              <input
                type="text"
                placeholder="Search tests..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10 w-full px-3 py-2 text-sm border border-neutral-300 dark:border-neutral-600 rounded-md bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
              />
            </div>

            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value as any)}
              className="px-3 py-2 text-sm border border-neutral-300 dark:border-neutral-600 rounded-md bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
            >
              <option value="all">All Statuses</option>
              <option value="running">Running</option>
              <option value="completed">Completed</option>
              <option value="draft">Draft</option>
            </select>

            <div className="flex items-center gap-2">
              <Filter className="w-4 h-4 text-neutral-500" />
              <span className="text-sm text-neutral-600 dark:text-neutral-400">
                {filteredTests.length} tests found
              </span>
            </div>
          </div>
        </div>

        {/* A/B Tests List */}
        <div className="mt-6 space-y-4">
          {filteredTests.map((test) => (
            <motion.div
              key={test.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-white dark:bg-neutral-800 rounded-lg border border-neutral-200 dark:border-neutral-700 p-6 hover:shadow-lg transition-shadow"
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                      {test.name}
                    </h3>
                    <div className="flex items-center gap-1">
                      {getStatusIcon(test.status)}
                      <span className={cn(
                        "px-2 py-1 rounded-full text-xs font-medium",
                        getStatusColor(test.status)
                      )}>
                        {test.status}
                      </span>
                    </div>
                    {test.statisticalSignificance && (
                      <span className="px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400">
                        Significant
                      </span>
                    )}
                  </div>
                  
                  <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-4">
                    {test.description}
                  </p>

                  {/* Models Comparison */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                    <div className="bg-neutral-50 dark:bg-neutral-700 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="font-medium text-neutral-900 dark:text-neutral-100">
                          Model A: {test.modelA.name}
                        </h4>
                        <span className="text-sm text-neutral-600 dark:text-neutral-400">
                          {test.trafficSplit.modelA}% traffic
                        </span>
                      </div>
                      <div className="grid grid-cols-2 gap-2 text-sm">
                        <div>
                          <span className="text-neutral-600 dark:text-neutral-400">Accuracy:</span>
                          <span className="ml-2 font-medium">{(test.metrics.accuracy.modelA * 100).toFixed(1)}%</span>
                        </div>
                        <div>
                          <span className="text-neutral-600 dark:text-neutral-400">Latency:</span>
                          <span className="ml-2 font-medium">{test.metrics.latency.modelA}ms</span>
                        </div>
                      </div>
                    </div>

                    <div className="bg-neutral-50 dark:bg-neutral-700 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="font-medium text-neutral-900 dark:text-neutral-100">
                          Model B: {test.modelB.name}
                        </h4>
                        <span className="text-sm text-neutral-600 dark:text-neutral-400">
                          {test.trafficSplit.modelB}% traffic
                        </span>
                      </div>
                      <div className="grid grid-cols-2 gap-2 text-sm">
                        <div>
                          <span className="text-neutral-600 dark:text-neutral-400">Accuracy:</span>
                          <span className="ml-2 font-medium">{(test.metrics.accuracy.modelB * 100).toFixed(1)}%</span>
                        </div>
                        <div>
                          <span className="text-neutral-600 dark:text-neutral-400">Latency:</span>
                          <span className="ml-2 font-medium">{test.metrics.latency.modelB}ms</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Progress Bar */}
                  {test.status === 'running' && (
                    <div className="mb-4">
                      <div className="flex justify-between text-sm text-neutral-600 dark:text-neutral-400 mb-1">
                        <span>Progress</span>
                        <span>{test.currentSamples.toLocaleString()} / {test.sampleSize.toLocaleString()} samples</span>
                      </div>
                      <div className="w-full bg-neutral-200 dark:bg-neutral-700 rounded-full h-2">
                        <div 
                          className="bg-blue-500 h-2 rounded-full transition-all duration-500"
                          style={{ width: `${calculateProgress(test)}%` }}
                        />
                      </div>
                    </div>
                  )}

                  {/* Winner Badge */}
                  {test.winner && (
                    <div className="mb-4">
                      <span className="inline-flex items-center gap-2 px-3 py-1 bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400 rounded-full text-sm font-medium">
                        <CheckCircle className="w-4 h-4" />
                        Winner: {test.winner === 'modelA' ? test.modelA.name : test.modelB.name}
                      </span>
                    </div>
                  )}
                </div>

                {/* Actions */}
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setSelectedTest(test)}
                    className="p-2 text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100"
                  >
                    <Eye className="w-4 h-4" />
                  </button>
                  
                  {test.status === 'draft' && (
                    <button
                      onClick={() => startTest(test.id)}
                      className="p-2 text-green-600 hover:text-green-700"
                    >
                      <Play className="w-4 h-4" />
                    </button>
                  )}
                  
                  {test.status === 'running' && (
                    <button
                      onClick={() => pauseTest(test.id)}
                      className="p-2 text-yellow-600 hover:text-yellow-700"
                    >
                      <Pause className="w-4 h-4" />
                    </button>
                  )}
                  
                  <button className="p-2 text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100">
                    <Edit className="w-4 h-4" />
                  </button>
                  
                  <button
                    onClick={() => deleteTest(test.id)}
                    className="p-2 text-red-600 hover:text-red-700"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Create New Test Modal */}
      <AnimatePresence>
        {showCreateModal && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
            onClick={() => setShowCreateModal(false)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="bg-white dark:bg-neutral-800 rounded-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="p-6 border-b border-neutral-200 dark:border-neutral-700">
                <h2 className="text-xl font-bold text-neutral-900 dark:text-neutral-100">
                  Create New A/B Test
                </h2>
              </div>

              <div className="p-6 space-y-4">
                <div>
                  <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                    Test Name
                  </label>
                  <input
                    type="text"
                    value={newTestConfig.name}
                    onChange={(e) => setNewTestConfig(prev => ({ ...prev, name: e.target.value }))}
                    className="w-full px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-md bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                    Description
                  </label>
                  <textarea
                    value={newTestConfig.description}
                    onChange={(e) => setNewTestConfig(prev => ({ ...prev, description: e.target.value }))}
                    rows={3}
                    className="w-full px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-md bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
                  />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                      Model A (Control)
                    </label>
                    <input
                      type="text"
                      value={newTestConfig.modelA}
                      onChange={(e) => setNewTestConfig(prev => ({ ...prev, modelA: e.target.value }))}
                      className="w-full px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-md bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                      Model B (Variant)
                    </label>
                    <input
                      type="text"
                      value={newTestConfig.modelB}
                      onChange={(e) => setNewTestConfig(prev => ({ ...prev, modelB: e.target.value }))}
                      className="w-full px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-md bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                    Traffic Split (Model A): {newTestConfig.trafficSplit}%
                  </label>
                  <input
                    type="range"
                    min="10"
                    max="90"
                    value={newTestConfig.trafficSplit}
                    onChange={(e) => setNewTestConfig(prev => ({ ...prev, trafficSplit: parseInt(e.target.value) }))}
                    className="w-full"
                  />
                  <div className="flex justify-between text-xs text-neutral-500 mt-1">
                    <span>Model A: {newTestConfig.trafficSplit}%</span>
                    <span>Model B: {100 - newTestConfig.trafficSplit}%</span>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                      Sample Size
                    </label>
                    <input
                      type="number"
                      value={newTestConfig.sampleSize}
                      onChange={(e) => setNewTestConfig(prev => ({ ...prev, sampleSize: parseInt(e.target.value) }))}
                      className="w-full px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-md bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                      Significance Level
                    </label>
                    <select
                      value={newTestConfig.significanceLevel}
                      onChange={(e) => setNewTestConfig(prev => ({ ...prev, significanceLevel: parseFloat(e.target.value) }))}
                      className="w-full px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-md bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
                    >
                      <option value={0.01}>0.01 (99% confidence)</option>
                      <option value={0.05}>0.05 (95% confidence)</option>
                      <option value={0.1}>0.1 (90% confidence)</option>
                    </select>
                  </div>
                </div>
              </div>

              <div className="p-6 border-t border-neutral-200 dark:border-neutral-700 flex justify-end gap-3">
                <button
                  onClick={() => setShowCreateModal(false)}
                  className="px-4 py-2 text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100"
                >
                  Cancel
                </button>
                <button
                  onClick={createNewTest}
                  disabled={!newTestConfig.name || !newTestConfig.modelA || !newTestConfig.modelB}
                  className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Create Test
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Test Details Modal */}
      <AnimatePresence>
        {selectedTest && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
            onClick={() => setSelectedTest(null)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="bg-white dark:bg-neutral-800 rounded-lg max-w-4xl w-full max-h-[90vh] overflow-y-auto"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="p-6 border-b border-neutral-200 dark:border-neutral-700">
                <div className="flex items-center justify-between">
                  <h2 className="text-xl font-bold text-neutral-900 dark:text-neutral-100">
                    {selectedTest.name}
                  </h2>
                  <button
                    onClick={() => setSelectedTest(null)}
                    className="text-neutral-500 hover:text-neutral-700 dark:hover:text-neutral-300"
                  >
                    ×
                  </button>
                </div>
              </div>

              <div className="p-6">
                {/* Detailed metrics would go here */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h3 className="font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
                      Model Performance Comparison
                    </h3>
                    <div className="space-y-3">
                      <div className="flex justify-between">
                        <span>Accuracy:</span>
                        <span>
                          A: {(selectedTest.metrics.accuracy.modelA * 100).toFixed(1)}% | 
                          B: {(selectedTest.metrics.accuracy.modelB * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>Precision:</span>
                        <span>
                          A: {(selectedTest.metrics.precision.modelA * 100).toFixed(1)}% | 
                          B: {(selectedTest.metrics.precision.modelB * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>Recall:</span>
                        <span>
                          A: {(selectedTest.metrics.recall.modelA * 100).toFixed(1)}% | 
                          B: {(selectedTest.metrics.recall.modelB * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  </div>

                  {selectedTest.businessMetrics && (
                    <div>
                      <h3 className="font-semibold text-neutral-900 dark:text-neutral-100 mb-4">
                        Business Impact
                      </h3>
                      <div className="space-y-3">
                        <div className="flex justify-between">
                          <span>Revenue:</span>
                          <span>
                            A: ${selectedTest.businessMetrics.revenue.modelA.toLocaleString()} | 
                            B: ${selectedTest.businessMetrics.revenue.modelB.toLocaleString()}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span>Conversion Rate:</span>
                          <span>
                            A: {(selectedTest.businessMetrics.conversions.modelA * 100).toFixed(1)}% | 
                            B: {(selectedTest.businessMetrics.conversions.modelB * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default ABTestingDashboard;