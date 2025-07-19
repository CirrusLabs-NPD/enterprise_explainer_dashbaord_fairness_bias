/**
 * API Client for ML Explainer Dashboard Backend
 * Handles all communication with the FastAPI backend
 */

// API Configuration
const API_BASE_URL = (import.meta as any).env?.VITE_API_URL || 'http://localhost:8000';
const API_VERSION = '/api/v1';

// Types
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

export interface ModelMetadata {
  model_id: string;
  name: string;
  model_type: 'classification' | 'regression' | 'clustering';
  framework: string;
  version: string;
  description: string;
  feature_names: string[];
  target_names: string[];
  training_metrics: Record<string, number>;
  validation_metrics: Record<string, number>;
  created_at: string;
  status: string;
}

export interface DatasetMetadata {
  dataset_id: string;
  name: string;
  description: string;
  format: string;
  size_bytes: number;
  num_rows: number;
  num_columns: number;
  column_names: string[];
  column_types: Record<string, string>;
  created_at: string;
  tags: string[];
}

export interface DataQualityReport {
  dataset_id: string;
  total_rows: number;
  total_features: number;
  missing_values: Record<string, number>;
  duplicate_rows: number;
  outliers: Record<string, number>;
  data_types: Record<string, string>;
  quality_score: number;
  issues: string[];
  timestamp: string;
}

export interface PredictionRequest {
  model_id: string;
  data: number[][];
  feature_names: string[];
  return_probabilities?: boolean;
  return_explanations?: boolean;
  explanation_method?: string;
}

export interface PredictionResult {
  predictions: number[];
  probabilities?: number[][];
  explanations?: any[];
  prediction_time_ms: number;
  model_version: string;
  timestamp: string;
}

export interface ExplanationRequest {
  model_id: string;
  method: 'shap' | 'lime' | 'permutation';
  data: number[][];
  feature_names: string[];
  target_names?: string[];
  parameters?: Record<string, any>;
}

export interface ExplanationResult {
  explanation_id: string;
  model_id: string;
  method: string;
  feature_names: string[];
  explanation: any;
  metadata: Record<string, any>;
  created_at: string;
  execution_time_ms: number;
}

export interface DriftReport {
  model_id: string;
  feature_drifts: Record<string, number>;
  overall_drift_score: number;
  drift_threshold: number;
  drift_detected: boolean;
  timestamp: string;
  detection_method: string;
}

// API Client Class
class ApiClient {
  private baseUrl: string;
  private defaultHeaders: HeadersInit;

  constructor() {
    this.baseUrl = `${API_BASE_URL}${API_VERSION}`;
    this.defaultHeaders = {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    };
  }

  protected async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    try {
      const url = `${this.baseUrl}${endpoint}`;
      const config: RequestInit = {
        ...options,
        headers: {
          ...this.defaultHeaders,
          ...options.headers,
        },
      };

      const response = await fetch(url, config);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      return { success: true, data };
    } catch (error) {
      console.error('API Request Error:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  // Health Check
  async healthCheck(): Promise<ApiResponse> {
    return this.request('/health', { method: 'GET' });
  }

  // Model Management
  async uploadModel(
    modelFile: File,
    metadata: {
      model_name: string;
      model_type: string;
      description?: string;
      feature_names?: string[];
      target_names?: string[];
    }
  ): Promise<ApiResponse<{ model_id: string }>> {
    const formData = new FormData();
    formData.append('model_file', modelFile);
    formData.append('model_name', metadata.model_name);
    formData.append('model_type', metadata.model_type);
    formData.append('description', metadata.description || '');
    formData.append('feature_names', JSON.stringify(metadata.feature_names || []));
    formData.append('target_names', JSON.stringify(metadata.target_names || []));

    return this.request('/models/upload', {
      method: 'POST',
      body: formData,
      headers: {}, // Let browser set content-type for FormData
    });
  }

  async listModels(): Promise<ApiResponse<ModelMetadata[]>> {
    return this.request('/models', { method: 'GET' });
  }

  async getModel(modelId: string): Promise<ApiResponse<ModelMetadata>> {
    return this.request(`/models/${modelId}`, { method: 'GET' });
  }

  async deleteModel(modelId: string): Promise<ApiResponse> {
    return this.request(`/models/${modelId}`, { method: 'DELETE' });
  }

  async predict(request: PredictionRequest): Promise<ApiResponse<PredictionResult>> {
    return this.request(`/models/${request.model_id}/predict`, {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async evaluateModel(
    modelId: string,
    testData: number[][],
    testLabels: number[]
  ): Promise<ApiResponse<{ metrics: Record<string, number> }>> {
    return this.request(`/models/${modelId}/evaluate`, {
      method: 'POST',
      body: JSON.stringify({
        test_data: testData,
        test_labels: testLabels,
      }),
    });
  }

  async getModelStats(modelId: string): Promise<ApiResponse<Record<string, any>>> {
    return this.request(`/models/${modelId}/stats`, { method: 'GET' });
  }

  async getFeatureImportance(
    modelId: string,
    method: string = 'shap'
  ): Promise<ApiResponse<any>> {
    return this.request(`/models/${modelId}/feature-importance?method=${method}`, {
      method: 'GET',
    });
  }

  async getFeatureInteractions(
    modelId: string,
    method: string = 'shap'
  ): Promise<ApiResponse<any>> {
    return this.request(`/models/${modelId}/interactions?method=${method}`, {
      method: 'GET',
    });
  }

  // Explanations
  async explainPrediction(request: ExplanationRequest): Promise<ApiResponse<ExplanationResult>> {
    return this.request(`/models/${request.model_id}/explain`, {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  // Data Management
  async uploadDataset(
    dataFile: File,
    metadata: {
      dataset_name: string;
      description?: string;
    }
  ): Promise<ApiResponse<{ dataset_id: string }>> {
    const formData = new FormData();
    formData.append('data_file', dataFile);
    formData.append('dataset_name', metadata.dataset_name);
    formData.append('description', metadata.description || '');

    return this.request('/data/upload', {
      method: 'POST',
      body: formData,
      headers: {}, // Let browser set content-type for FormData
    });
  }

  async listDatasets(): Promise<ApiResponse<DatasetMetadata[]>> {
    return this.request('/data', { method: 'GET' });
  }

  async getDataset(datasetId: string): Promise<ApiResponse<DatasetMetadata>> {
    return this.request(`/data/${datasetId}`, { method: 'GET' });
  }

  async deleteDataset(datasetId: string): Promise<ApiResponse> {
    return this.request(`/data/${datasetId}`, { method: 'DELETE' });
  }

  async getDatasetSample(
    datasetId: string,
    nRows: number = 100
  ): Promise<ApiResponse<{ sample: Record<string, any>[] }>> {
    return this.request(`/data/${datasetId}/sample?n_rows=${nRows}`, {
      method: 'GET',
    });
  }

  async getDatasetStatistics(
    datasetId: string
  ): Promise<ApiResponse<Record<string, any>>> {
    return this.request(`/data/${datasetId}/statistics`, { method: 'GET' });
  }

  async assessDataQuality(
    datasetId: string
  ): Promise<ApiResponse<DataQualityReport>> {
    return this.request(`/data/${datasetId}/quality`, { method: 'GET' });
  }

  async preprocessDataset(
    datasetId: string,
    preprocessingSteps: Array<{
      type: string;
      params: Record<string, any>;
    }>
  ): Promise<ApiResponse> {
    return this.request(`/data/${datasetId}/preprocess`, {
      method: 'POST',
      body: JSON.stringify(preprocessingSteps),
    });
  }

  async splitDataset(
    datasetId: string,
    options: {
      train_ratio?: number;
      validation_ratio?: number;
      test_ratio?: number;
      random_state?: number;
    } = {}
  ): Promise<ApiResponse<Record<string, string>>> {
    const params = new URLSearchParams();
    Object.entries(options).forEach(([key, value]) => {
      if (value !== undefined) {
        params.append(key, value.toString());
      }
    });

    return this.request(`/data/${datasetId}/split?${params.toString()}`, {
      method: 'POST',
    });
  }

  async profileDataset(
    datasetId: string
  ): Promise<ApiResponse<Record<string, any>>> {
    return this.request(`/data/${datasetId}/profile`, { method: 'GET' });
  }

  async validateDataset(
    datasetId: string,
    validationRules: Array<{
      type: string;
      params: Record<string, any>;
    }>
  ): Promise<ApiResponse<Record<string, any>>> {
    return this.request(`/data/${datasetId}/validate`, {
      method: 'POST',
      body: JSON.stringify(validationRules),
    });
  }

  // Monitoring
  async configureMonitoring(
    modelId: string,
    config: {
      monitoring_enabled?: boolean;
      drift_detection_enabled?: boolean;
      performance_monitoring_enabled?: boolean;
      drift_threshold?: number;
      performance_threshold?: number;
    }
  ): Promise<ApiResponse> {
    return this.request(`/monitoring/configure/${modelId}`, {
      method: 'POST',
      body: JSON.stringify(config),
    });
  }

  async setReferenceData(
    modelId: string,
    referenceData: number[][],
    featureNames: string[],
    referencePredictions?: number[]
  ): Promise<ApiResponse> {
    return this.request(`/monitoring/reference-data/${modelId}`, {
      method: 'POST',
      body: JSON.stringify({
        reference_data: referenceData,
        feature_names: featureNames,
        reference_predictions: referencePredictions,
      }),
    });
  }

  async detectDataDrift(
    modelId: string,
    currentData: number[][],
    featureNames: string[],
    method: string = 'kolmogorov_smirnov'
  ): Promise<ApiResponse<DriftReport>> {
    return this.request(`/monitoring/drift/detect/${modelId}`, {
      method: 'POST',
      body: JSON.stringify({
        current_data: currentData,
        feature_names: featureNames,
        method,
      }),
    });
  }

  async detectModelDrift(
    modelId: string,
    currentData: number[][],
    currentLabels: number[]
  ): Promise<ApiResponse<DriftReport>> {
    return this.request(`/monitoring/drift/model/${modelId}`, {
      method: 'POST',
      body: JSON.stringify({
        current_data: currentData,
        current_labels: currentLabels,
      }),
    });
  }

  async getModelAlerts(
    modelId: string,
    limit: number = 100
  ): Promise<ApiResponse<any[]>> {
    return this.request(`/monitoring/alerts/${modelId}?limit=${limit}`, {
      method: 'GET',
    });
  }

  async acknowledgeAlert(alertId: string): Promise<ApiResponse> {
    return this.request(`/monitoring/alerts/${alertId}/acknowledge`, {
      method: 'POST',
    });
  }

  async resolveAlert(alertId: string): Promise<ApiResponse> {
    return this.request(`/monitoring/alerts/${alertId}/resolve`, {
      method: 'POST',
    });
  }

  async getMonitoringHealth(): Promise<ApiResponse<Record<string, any>>> {
    return this.request('/monitoring/health', { method: 'GET' });
  }

  async getMonitoringStats(): Promise<ApiResponse<Record<string, any>>> {
    return this.request('/monitoring/stats', { method: 'GET' });
  }

  // WebSocket Stats
  async getWebSocketStats(): Promise<ApiResponse<Record<string, any>>> {
    return this.request('/ws/stats', { method: 'GET' });
  }

  // Streaming Data
  async connectStream(streamConfig: Record<string, any>): Promise<ApiResponse<{ stream_id: string }>> {
    return this.request('/data/stream/connect', {
      method: 'POST',
      body: JSON.stringify(streamConfig),
    });
  }

  async disconnectStream(streamId: string): Promise<ApiResponse> {
    return this.request(`/data/stream/${streamId}`, { method: 'DELETE' });
  }

  async getStreamStatus(streamId: string): Promise<ApiResponse<Record<string, any>>> {
    return this.request(`/data/stream/${streamId}/status`, { method: 'GET' });
  }
}

// Import mock service
import { mockApiService } from './mockData';

// Smart API client that falls back to mock data
class SmartApiClient extends ApiClient {
  private useMockData = false;

  private async detectBackendAvailability(): Promise<boolean> {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 2000); // 2 second timeout
      
      const response = await fetch(`${API_BASE_URL}/health`, {
        signal: controller.signal,
        method: 'GET'
      });
      
      clearTimeout(timeoutId);
      return response.ok;
    } catch (error) {
      console.log('Backend not available, using mock data');
      return false;
    }
  }

  private async smartRequest<T>(
    endpoint: string,
    options: RequestInit = {},
    mockMethod?: () => Promise<ApiResponse<T>>
  ): Promise<ApiResponse<T>> {
    // Try real API first if we haven't detected it's unavailable
    if (!this.useMockData) {
      try {
        const result = await this.request<T>(endpoint, options);
        if (result.success) {
          return result;
        }
      } catch (error) {
        console.log(`API request failed for ${endpoint}, falling back to mock data`);
        this.useMockData = true;
      }
    }

    // Fall back to mock data if available
    if (mockMethod) {
      try {
        return await mockMethod();
      } catch (error) {
        return {
          success: false,
          error: 'Both API and mock data failed'
        };
      }
    }

    return {
      success: false,
      error: 'Backend unavailable and no mock data provided'
    };
  }

  // Override key methods to use smart fallback
  async listModels(): Promise<ApiResponse<ModelMetadata[]>> {
    return this.smartRequest(
      '/models',
      { method: 'GET' },
      () => mockApiService.listModels()
    );
  }

  async getModel(modelId: string): Promise<ApiResponse<ModelMetadata>> {
    return this.smartRequest(
      `/models/${modelId}`,
      { method: 'GET' },
      () => mockApiService.getModel(modelId)
    );
  }

  async getModelPerformance(modelId: string): Promise<ApiResponse<any>> {
    return this.smartRequest(
      `/models/${modelId}/performance`,
      { method: 'GET' },
      () => mockApiService.getModelPerformance(modelId)
    );
  }

  async getDriftData(): Promise<ApiResponse<any>> {
    return this.smartRequest(
      '/monitoring/drift',
      { method: 'GET' },
      () => mockApiService.getDriftData()
    );
  }

  async getBiasMetrics(): Promise<ApiResponse<any>> {
    return this.smartRequest(
      '/fairness/bias',
      { method: 'GET' },
      () => mockApiService.getBiasMetrics()
    );
  }

  async getAlerts(): Promise<ApiResponse<any[]>> {
    return this.smartRequest(
      '/monitoring/alerts',
      { method: 'GET' },
      () => mockApiService.getAlerts()
    );
  }

  async getComplianceData(): Promise<ApiResponse<any>> {
    return this.smartRequest(
      '/compliance/reports',
      { method: 'GET' },
      () => mockApiService.getComplianceData()
    );
  }

  async getRootCauseData(): Promise<ApiResponse<any>> {
    return this.smartRequest(
      '/monitoring/root-cause',
      { method: 'GET' },
      () => mockApiService.getRootCauseData()
    );
  }

  async getDataConnectors(): Promise<ApiResponse<any[]>> {
    return this.smartRequest(
      '/data/connectors',
      { method: 'GET' },
      () => mockApiService.getDataConnectors()
    );
  }

  async getGlobalExplanations(modelId: string): Promise<ApiResponse<any>> {
    return this.smartRequest(
      `/explanations/${modelId}/global`,
      { method: 'GET' },
      () => mockApiService.getGlobalExplanations(modelId)
    );
  }

  async getLocalExplanations(modelId: string, data: any): Promise<ApiResponse<any>> {
    return this.smartRequest(
      `/explanations/${modelId}/local`,
      { method: 'POST', body: JSON.stringify(data) },
      () => mockApiService.getLocalExplanations(modelId, data)
    );
  }

  // Data Quality Assessment methods
  async getDataQuality(datasetId: string): Promise<ApiResponse<any>> {
    return this.smartRequest(
      `/data/${datasetId}/quality`,
      { method: 'GET' },
      () => mockApiService.getDataQuality(datasetId)
    );
  }

  async runQualityAssessment(datasetId: string): Promise<ApiResponse<any>> {
    return this.smartRequest(
      `/monitoring/quality/assess/${datasetId}`,
      { method: 'POST' },
      () => mockApiService.runQualityAssessment(datasetId)
    );
  }

  // A/B Testing methods
  async setupABTest(testConfig: any): Promise<ApiResponse<any>> {
    return this.smartRequest(
      '/model-monitoring/setup-ab-test',
      { method: 'POST', body: JSON.stringify(testConfig) },
      () => mockApiService.setupABTest(testConfig)
    );
  }

  async getABTests(): Promise<ApiResponse<any[]>> {
    return this.smartRequest(
      '/model-monitoring/ab-tests',
      { method: 'GET' },
      () => mockApiService.getABTests()
    );
  }

  async updateABTest(testId: string, updates: any): Promise<ApiResponse<any>> {
    return this.smartRequest(
      `/model-monitoring/ab-tests/${testId}`,
      { method: 'PUT', body: JSON.stringify(updates) },
      () => mockApiService.updateABTest(testId, updates)
    );
  }

  async deleteABTest(testId: string): Promise<ApiResponse<any>> {
    return this.smartRequest(
      `/model-monitoring/ab-tests/${testId}`,
      { method: 'DELETE' },
      () => mockApiService.deleteABTest(testId)
    );
  }
}

// Export singleton instance
export const apiClient = new SmartApiClient();
export default apiClient;