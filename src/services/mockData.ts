/**
 * Mock Data Service for ML Explainer Dashboard
 * Provides fallback data when backend is unavailable
 */

import { ApiResponse, ModelMetadata, DatasetMetadata, DriftReport } from './api';

export const mockModels: ModelMetadata[] = [
  {
    model_id: 'model_1',
    name: 'Credit Risk Classifier',
    model_type: 'classification',
    framework: 'sklearn',
    version: 'v1.2.3',
    description: 'Advanced credit risk assessment model with 87% accuracy',
    feature_names: ['income', 'age', 'credit_score', 'employment_length', 'debt_ratio', 'payment_history', 'credit_utilization', 'account_age', 'recent_inquiries', 'loan_amount'],
    target_names: ['low_risk', 'high_risk'],
    training_metrics: {
      accuracy: 0.873,
      precision: 0.851,
      recall: 0.892,
      f1_score: 0.871,
      auc_roc: 0.934
    },
    validation_metrics: {
      accuracy: 0.869,
      precision: 0.847,
      recall: 0.888,
      f1_score: 0.867,
      auc_roc: 0.931
    },
    created_at: '2024-01-15T10:30:00Z',
    status: 'active'
  },
  {
    model_id: 'model_2',
    name: 'Fraud Detection System',
    model_type: 'classification',
    framework: 'xgboost',
    version: 'v2.1.0',
    description: 'Real-time fraud detection with 94% precision',
    feature_names: ['transaction_amount', 'merchant_category', 'time_of_day', 'location_risk', 'card_type', 'velocity_score', 'device_fingerprint', 'ip_reputation', 'spending_pattern', 'account_age'],
    target_names: ['legitimate', 'fraud'],
    training_metrics: {
      accuracy: 0.921,
      precision: 0.943,
      recall: 0.887,
      f1_score: 0.914,
      auc_roc: 0.965
    },
    validation_metrics: {
      accuracy: 0.918,
      precision: 0.940,
      recall: 0.884,
      f1_score: 0.911,
      auc_roc: 0.962
    },
    created_at: '2024-02-20T14:15:00Z',
    status: 'active'
  },
  {
    model_id: 'model_3',
    name: 'Customer Churn Predictor',
    model_type: 'classification',
    framework: 'tensorflow',
    version: 'v1.0.5',
    description: 'Predicts customer churn with 89% accuracy',
    feature_names: ['tenure', 'monthly_charges', 'total_charges', 'contract_type', 'payment_method', 'internet_service', 'online_security', 'tech_support', 'streaming_tv', 'paperless_billing'],
    target_names: ['retain', 'churn'],
    training_metrics: {
      accuracy: 0.892,
      precision: 0.876,
      recall: 0.903,
      f1_score: 0.889,
      auc_roc: 0.945
    },
    validation_metrics: {
      accuracy: 0.889,
      precision: 0.872,
      recall: 0.900,
      f1_score: 0.886,
      auc_roc: 0.942
    },
    created_at: '2024-03-10T09:45:00Z',
    status: 'active'
  },
  {
    model_id: 'model_4',
    name: 'House Price Predictor',
    model_type: 'regression',
    framework: 'sklearn',
    version: 'v1.1.2',
    description: 'Real estate price prediction with R² = 0.85',
    feature_names: ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'yr_built'],
    target_names: ['price'],
    training_metrics: {
      r2_score: 0.853,
      mse: 15623.45,
      mae: 98.76,
      mape: 0.123
    },
    validation_metrics: {
      r2_score: 0.847,
      mse: 16234.78,
      mae: 102.34,
      mape: 0.127
    },
    created_at: '2024-01-25T16:20:00Z',
    status: 'active'
  },
  {
    model_id: 'model_5',
    name: 'Recommendation Engine',
    model_type: 'clustering',
    framework: 'pytorch',
    version: 'v3.0.1',
    description: 'Product recommendation system with collaborative filtering',
    feature_names: ['user_age', 'user_income', 'purchase_history', 'browsing_time', 'category_preference', 'price_sensitivity', 'brand_loyalty', 'seasonal_pattern', 'device_type', 'location'],
    target_names: ['cluster_id'],
    training_metrics: {
      silhouette_score: 0.724,
      inertia: 45623.12,
      davies_bouldin_score: 0.892,
      calinski_harabasz_score: 3456.78
    },
    validation_metrics: {
      silhouette_score: 0.718,
      inertia: 47123.45,
      davies_bouldin_score: 0.897,
      calinski_harabasz_score: 3398.45
    },
    created_at: '2024-03-05T11:30:00Z',
    status: 'active'
  }
];

export const mockDatasets: DatasetMetadata[] = [
  {
    dataset_id: 'dataset_1',
    name: 'Credit Applications Training Data',
    description: 'Historical credit application data with risk labels',
    format: 'CSV',
    size_bytes: 52428800,
    num_rows: 50000,
    num_columns: 25,
    column_names: ['income', 'age', 'credit_score', 'employment_length', 'debt_ratio'],
    column_types: {
      'income': 'float64',
      'age': 'int64',
      'credit_score': 'int64',
      'employment_length': 'float64',
      'debt_ratio': 'float64'
    },
    created_at: '2024-01-10T08:00:00Z',
    tags: ['credit', 'risk', 'training']
  },
  {
    dataset_id: 'dataset_2',
    name: 'Transaction Data for Fraud Detection',
    description: 'Real-time transaction data with fraud labels',
    format: 'Parquet',
    size_bytes: 157286400,
    num_rows: 125000,
    num_columns: 18,
    column_names: ['transaction_amount', 'merchant_category', 'time_of_day', 'location_risk'],
    column_types: {
      'transaction_amount': 'float64',
      'merchant_category': 'str',
      'time_of_day': 'int64',
      'location_risk': 'float64'
    },
    created_at: '2024-02-15T12:30:00Z',
    tags: ['fraud', 'transactions', 'realtime']
  }
];

export const mockPerformanceMetrics = {
  accuracy: 0.873,
  precision: 0.851,
  recall: 0.892,
  f1_score: 0.871,
  auc_roc: 0.934,
  confusion_matrix: [[8520, 480], [324, 4676]],
  feature_importance: [
    { feature: 'credit_score', importance: 0.245 },
    { feature: 'income', importance: 0.189 },
    { feature: 'debt_ratio', importance: 0.167 },
    { feature: 'employment_length', importance: 0.134 },
    { feature: 'age', importance: 0.098 },
    { feature: 'payment_history', importance: 0.087 },
    { feature: 'credit_utilization', importance: 0.052 },
    { feature: 'account_age', importance: 0.028 }
  ]
};

export const mockDriftData = {
  drift_detected: true,
  drift_score: 0.156,
  threshold: 0.1,
  last_check: '2024-03-15T14:30:00Z',
  features_with_drift: ['income', 'credit_score'],
  feature_drifts: {
    'income': 0.234,
    'age': 0.045,
    'credit_score': 0.187,
    'employment_length': 0.076,
    'debt_ratio': 0.098
  }
};

export const mockBiasMetrics = {
  overall_fairness_score: 0.824,
  demographic_parity: 0.127,
  equalized_odds: 0.089,
  equality_of_opportunity: 0.067,
  calibration: 0.043,
  disparate_impact_ratio: 0.762,
  protected_groups: ['gender', 'race', 'age'],
  bias_violations: 2,
  intersectional_analysis: {
    'female_rural': 0.234,
    'male_urban': 0.087,
    'nonbinary_suburban': 0.156
  }
};

export const mockAlerts = [
  {
    alert_id: 'alert_1',
    model_id: 'model_1',
    alert_type: 'bias_detected',
    severity: 'critical',
    title: 'High Bias Detected in Credit Model',
    message: 'Demographic parity violation detected (0.127 > 0.10 threshold)',
    status: 'active',
    created_at: '2024-03-15T14:25:00Z',
    data: {
      metric: 'demographic_parity',
      value: 0.127,
      threshold: 0.10,
      affected_groups: ['gender']
    }
  },
  {
    alert_id: 'alert_2',
    model_id: 'model_2',
    alert_type: 'drift_detected',
    severity: 'high',
    title: 'Data Drift Detected in Fraud Model',
    message: 'Significant drift in transaction_amount feature distribution',
    status: 'active',
    created_at: '2024-03-15T13:45:00Z',
    data: {
      feature: 'transaction_amount',
      drift_score: 0.198,
      threshold: 0.10
    }
  },
  {
    alert_id: 'alert_3',
    model_id: 'model_3',
    alert_type: 'performance_degradation',
    severity: 'medium',
    title: 'Performance Drop in Churn Model',
    message: 'Accuracy dropped from 89% to 85% over the last 24 hours',
    status: 'acknowledged',
    created_at: '2024-03-15T10:30:00Z',
    data: {
      previous_accuracy: 0.89,
      current_accuracy: 0.85,
      degradation: 0.04
    }
  }
];

export const mockComplianceData = {
  overall_score: 94.2,
  frameworks: {
    'GDPR': { score: 98.5, status: 'compliant', violations: 0 },
    'EU AI Act': { score: 95.2, status: 'compliant', violations: 0 },
    'SOC 2 Type II': { score: 96.8, status: 'compliant', violations: 0 },
    'ISO 27001': { score: 87.3, status: 'in_progress', violations: 2 }
  },
  risk_assessment: {
    'data_privacy': 95,
    'algorithmic_bias': 92,
    'model_transparency': 85,
    'data_security': 98
  },
  last_audit: '2024-01-15T00:00:00Z',
  next_audit: '2024-04-15T00:00:00Z'
};

export const mockRootCauseData = {
  active_investigations: 3,
  success_rate: 87.5,
  avg_resolution_time: 144, // minutes
  current_investigation: {
    id: 'rca_1',
    title: 'Model Accuracy Degradation',
    status: 'evidence_validation',
    progress: 75,
    timeline: [
      { stage: 'issue_detection', status: 'completed', duration: 2 },
      { stage: 'data_collection', status: 'completed', duration: 15 },
      { stage: 'hypothesis_generation', status: 'completed', duration: 8 },
      { stage: 'evidence_validation', status: 'in_progress', duration: 45 },
      { stage: 'root_cause_confirmation', status: 'pending', duration: null }
    ]
  },
  hypotheses: [
    { id: 1, description: 'Data distribution shift', confidence: 85, status: 'investigating' },
    { id: 2, description: 'Training data bias', confidence: 72, status: 'investigating' },
    { id: 3, description: 'Infrastructure latency', confidence: 45, status: 'low_priority' }
  ],
  recommendations: [
    { priority: 'critical', action: 'Immediate model rollback', timeline: '15 min' },
    { priority: 'high', action: 'Retrain with balanced dataset', timeline: '2-3 days' },
    { priority: 'medium', action: 'Enhanced data monitoring', timeline: '1 week' }
  ]
};

export const mockDataConnectors = [
  {
    id: 'snowflake_prod',
    name: 'Snowflake Production',
    type: 'snowflake',
    status: 'connected',
    data_transferred: '458.2 GB',
    health_score: 98,
    last_sync: '2024-03-15T14:25:00Z'
  },
  {
    id: 'aws_s3_training',
    name: 'AWS S3 ML Training',
    type: 'aws_s3',
    status: 'connected',
    data_transferred: '1.2 TB',
    health_score: 99,
    last_sync: '2024-03-15T14:20:00Z'
  },
  {
    id: 'databricks_analytics',
    name: 'Databricks Analytics',
    type: 'databricks',
    status: 'connected',
    data_transferred: '234.5 GB',
    health_score: 95,
    last_sync: '2024-03-15T14:15:00Z'
  },
  {
    id: 'bigquery_compliance',
    name: 'BigQuery Compliance',
    type: 'bigquery',
    status: 'connected',
    data_transferred: '89.3 GB',
    health_score: 97,
    last_sync: '2024-03-15T14:10:00Z'
  },
  {
    id: 'postgres_customer',
    name: 'PostgreSQL Customer DB',
    type: 'postgresql',
    status: 'error',
    data_transferred: '45.7 GB',
    health_score: 0,
    last_sync: '2024-03-15T10:30:00Z'
  },
  {
    id: 'aws_s3_features',
    name: 'AWS S3 Feature Store',
    type: 'aws_s3',
    status: 'syncing',
    data_transferred: '567.8 GB',
    health_score: 85,
    last_sync: '2024-03-15T14:00:00Z'
  }
];

export const mockDataQualityMetrics = {
  completeness: 94.2,
  accuracy: 96.8,
  consistency: 91.5,
  timeliness: 98.1,
  validity: 93.7,
  uniqueness: 99.2
};

// Mock API service that returns the mock data
export class MockApiService {
  private simulateNetworkDelay(min = 200, max = 800): Promise<void> {
    const delay = Math.random() * (max - min) + min;
    return new Promise(resolve => setTimeout(resolve, delay));
  }

  async listModels(): Promise<ApiResponse<ModelMetadata[]>> {
    await this.simulateNetworkDelay();
    return { success: true, data: mockModels };
  }

  async getModel(modelId: string): Promise<ApiResponse<ModelMetadata>> {
    await this.simulateNetworkDelay();
    const model = mockModels.find(m => m.model_id === modelId);
    if (model) {
      return { success: true, data: model };
    }
    return { success: false, error: 'Model not found' };
  }

  async getModelPerformance(modelId: string): Promise<ApiResponse<any>> {
    await this.simulateNetworkDelay();
    return { success: true, data: mockPerformanceMetrics };
  }

  async getDriftData(): Promise<ApiResponse<any>> {
    await this.simulateNetworkDelay();
    return { success: true, data: mockDriftData };
  }

  async getBiasMetrics(): Promise<ApiResponse<any>> {
    await this.simulateNetworkDelay();
    return { success: true, data: mockBiasMetrics };
  }

  async getAlerts(): Promise<ApiResponse<any[]>> {
    await this.simulateNetworkDelay();
    return { success: true, data: mockAlerts };
  }

  async getComplianceData(): Promise<ApiResponse<any>> {
    await this.simulateNetworkDelay();
    return { success: true, data: mockComplianceData };
  }

  async getRootCauseData(): Promise<ApiResponse<any>> {
    await this.simulateNetworkDelay();
    return { success: true, data: mockRootCauseData };
  }

  async getDataConnectors(): Promise<ApiResponse<any[]>> {
    await this.simulateNetworkDelay();
    return { success: true, data: mockDataConnectors };
  }

  async getDataQualityMetrics(): Promise<ApiResponse<any>> {
    await this.simulateNetworkDelay();
    return { success: true, data: mockDataQualityMetrics };
  }

  async getGlobalExplanations(modelId: string): Promise<ApiResponse<any>> {
    await this.simulateNetworkDelay();
    return {
      success: true,
      data: {
        feature_importance: mockPerformanceMetrics.feature_importance,
        partial_dependence: {
          credit_score: Array.from({ length: 50 }, (_, i) => Math.sin(i / 5) * 0.3),
          income: Array.from({ length: 50 }, (_, i) => Math.cos(i / 7) * 0.4),
          debt_ratio: Array.from({ length: 50 }, (_, i) => Math.tan(i / 10) * 0.2)
        }
      }
    };
  }

  async getLocalExplanations(modelId: string, data: any): Promise<ApiResponse<any>> {
    await this.simulateNetworkDelay();
    return {
      success: true,
      data: {
        prediction: Math.random(),
        shap_values: Array.from({ length: 10 }, () => (Math.random() - 0.5) * 0.2),
        lime_explanation: [
          { feature: 'credit_score', value: 0.089 },
          { feature: 'income', value: -0.034 },
          { feature: 'debt_ratio', value: 0.156 },
          { feature: 'employment_length', value: 0.023 },
          { feature: 'age', value: -0.012 }
        ]
      }
    };
  }

  // Data Quality Assessment methods
  async getDataQuality(datasetId: string): Promise<ApiResponse<any>> {
    await this.simulateNetworkDelay();
    return {
      success: true,
      data: {
        dataset_id: datasetId,
        dataset_name: 'Customer Demographics',
        total_rows: 125000,
        total_columns: 28,
        metrics: {
          completeness: 87.3,
          accuracy: 91.2,
          consistency: 85.7,
          validity: 93.8,
          uniqueness: 96.1,
          timeliness: 78.9,
          overall_score: 88.8
        },
        issues: [
          {
            id: 'iss_001',
            type: 'missing_values',
            severity: 'high',
            description: 'High rate of missing values in income field',
            affected_rows: 15600,
            affected_columns: ['annual_income'],
            suggested_action: 'Implement income estimation model or require field completion',
            detected_at: '2024-01-20T10:30:00Z'
          },
          {
            id: 'iss_002',
            type: 'duplicates',
            severity: 'medium',
            description: 'Duplicate customer records based on email',
            affected_rows: 3200,
            affected_columns: ['email', 'customer_id'],
            suggested_action: 'Merge duplicate records and establish unique constraints',
            detected_at: '2024-01-20T10:45:00Z'
          }
        ],
        last_assessed: '2024-01-20T12:00:00Z',
        trend: 'improving'
      }
    };
  }

  async runQualityAssessment(datasetId: string): Promise<ApiResponse<any>> {
    await this.simulateNetworkDelay(3000); // Longer delay for assessment
    return {
      success: true,
      data: {
        assessment_id: `assess_${Date.now()}`,
        dataset_id: datasetId,
        status: 'completed',
        duration_ms: 2845,
        timestamp: new Date().toISOString()
      }
    };
  }

  // A/B Testing methods
  async setupABTest(testConfig: any): Promise<ApiResponse<any>> {
    await this.simulateNetworkDelay();
    return {
      success: true,
      data: {
        test_id: `test_${Date.now()}`,
        name: testConfig.name,
        status: 'draft',
        created_at: new Date().toISOString()
      }
    };
  }

  async getABTests(): Promise<ApiResponse<any[]>> {
    await this.simulateNetworkDelay();
    return {
      success: true,
      data: [
        {
          id: 'test_001',
          name: 'Credit Model v2.3 vs v2.4',
          status: 'running',
          modelA: { id: 'model_1', name: 'Credit Risk v2.3', version: 'v2.3' },
          modelB: { id: 'model_2', name: 'Credit Risk v2.4', version: 'v2.4' },
          trafficSplit: { modelA: 50, modelB: 50 },
          metrics: {
            accuracy: { modelA: 87.3, modelB: 89.1 },
            precision: { modelA: 85.2, modelB: 87.8 },
            recall: { modelA: 88.9, modelB: 90.2 }
          },
          statisticalSignificance: true,
          winner: 'modelB',
          startDate: '2024-01-15T00:00:00Z',
          endDate: '2024-01-29T23:59:59Z',
          duration: 14,
          createdAt: '2024-01-14T10:00:00Z'
        }
      ]
    };
  }

  async updateABTest(testId: string, updates: any): Promise<ApiResponse<any>> {
    await this.simulateNetworkDelay();
    return {
      success: true,
      data: {
        test_id: testId,
        ...updates,
        updated_at: new Date().toISOString()
      }
    };
  }

  async deleteABTest(testId: string): Promise<ApiResponse<any>> {
    await this.simulateNetworkDelay();
    return {
      success: true,
      data: { message: 'A/B test deleted successfully' }
    };
  }
}

export const mockApiService = new MockApiService();