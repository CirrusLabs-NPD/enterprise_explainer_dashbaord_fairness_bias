// Core application types
export interface ModelData {
  id: string;
  name: string;
  type: 'classification' | 'regression';
  algorithm: string;
  framework: string;
  features: Feature[];
  target: string;
  targetType: 'binary' | 'multiclass' | 'regression';
  classes?: string[];
  performanceMetrics: PerformanceMetrics;
  trainingMetrics: TrainingMetrics;
  datasetInfo: DatasetInfo;
  shapValues?: ShapValues;
  permutationImportance?: PermutationImportance;
  createdAt: string;
  updatedAt: string;
}

export interface Feature {
  name: string;
  type: 'numerical' | 'categorical' | 'boolean';
  description?: string;
  importance?: number;
  range?: [number, number];
  categories?: string[];
  isTarget?: boolean;
  nullable?: boolean;
  dataType: string;
}

export interface PerformanceMetrics {
  // Classification metrics
  accuracy?: number;
  precision?: number | number[];
  recall?: number | number[];
  f1Score?: number | number[];
  rocAuc?: number | number[];
  prAuc?: number | number[];
  logLoss?: number;
  confusionMatrix?: number[][];
  
  // Regression metrics
  mae?: number;
  mse?: number;
  rmse?: number;
  r2?: number;
  mape?: number;
  medianAbsoluteError?: number;
  
  // Common metrics
  meanAbsoluteError?: number;
  meanSquaredError?: number;
  crossValidationScore?: number[];
}

export interface TrainingMetrics {
  trainScore: number;
  validationScore: number;
  testScore?: number;
  trainingTime: number;
  hyperparameters: Record<string, any>;
  modelSize: number;
  overfitting: boolean;
}

export interface DatasetInfo {
  name: string;
  size: number;
  features: number;
  samples: number;
  trainSize: number;
  testSize: number;
  validationSize?: number;
  missingValues: number;
  duplicates: number;
  imbalance?: number;
  description?: string;
}

export interface ShapValues {
  globalImportance: FeatureImportance[];
  localExplanations: LocalExplanation[];
  baseValue: number;
  expectedValue: number;
  summary: ShapSummary;
}

export interface FeatureImportance {
  feature: string;
  importance: number;
  confidenceInterval?: [number, number];
  pValue?: number;
  ranking: number;
}

export interface LocalExplanation {
  index: number;
  features: Record<string, number>;
  shapValues: Record<string, number>;
  prediction: number | number[];
  probability?: number[];
  expectedValue: number;
  featureValues: Record<string, any>;
}

export interface ShapSummary {
  topFeatures: string[];
  meanAbsoluteShap: Record<string, number>;
  maxShap: Record<string, number>;
  minShap: Record<string, number>;
  featureInteractions: FeatureInteraction[];
}

export interface FeatureInteraction {
  feature1: string;
  feature2: string;
  interactionStrength: number;
  pValue?: number;
}

export interface PermutationImportance {
  features: FeatureImportance[];
  baselineScore: number;
  randomState: number;
  nRepeats: number;
  scoringMetric: string;
}

export interface PredictionData {
  index: number;
  features: Record<string, any>;
  prediction: number | number[];
  probability?: number[];
  actualValue?: number | string;
  explanation?: LocalExplanation;
  residual?: number;
  outlierScore?: number;
}

export interface WhatIfScenario {
  id: string;
  name: string;
  baselineIndex: number;
  modifications: Record<string, any>;
  prediction: number | number[];
  probability?: number[];
  explanation?: LocalExplanation;
  createdAt: string;
  description?: string;
}

export interface ComparisonData {
  original: PredictionData;
  modified: PredictionData;
  featureChanges: FeatureChange[];
  predictionChange: number;
  explanationChange: Record<string, number>;
}

export interface FeatureChange {
  feature: string;
  originalValue: any;
  newValue: any;
  changeType: 'increased' | 'decreased' | 'changed';
  magnitude: number;
}

export interface ChartData {
  labels: string[];
  datasets: ChartDataset[];
}

export interface ChartDataset {
  label: string;
  data: number[];
  backgroundColor?: string | string[];
  borderColor?: string | string[];
  borderWidth?: number;
  fill?: boolean;
  tension?: number;
  pointRadius?: number;
  pointHoverRadius?: number;
}

export interface ChartOptions {
  responsive: boolean;
  maintainAspectRatio?: boolean;
  plugins?: {
    legend?: {
      display: boolean;
      position?: 'top' | 'bottom' | 'left' | 'right';
    };
    tooltip?: {
      enabled: boolean;
      mode?: 'nearest' | 'index' | 'dataset';
      intersect?: boolean;
    };
  };
  scales?: {
    x?: ScaleOptions;
    y?: ScaleOptions;
  };
  animation?: {
    duration?: number;
    easing?: string;
  };
  interaction?: {
    mode?: 'nearest' | 'index' | 'dataset';
    intersect?: boolean;
  };
}

export interface ScaleOptions {
  display?: boolean;
  title?: {
    display: boolean;
    text?: string;
  };
  min?: number;
  max?: number;
  ticks?: {
    beginAtZero?: boolean;
    callback?: (value: any) => string;
  };
}

export interface FilterOptions {
  features?: string[];
  importance?: [number, number];
  correlation?: [number, number];
  pValue?: number;
  sortBy?: 'importance' | 'name' | 'correlation';
  sortOrder?: 'asc' | 'desc';
  showTop?: number;
}

export interface ExportOptions {
  format: 'png' | 'svg' | 'pdf' | 'csv' | 'json';
  quality?: 'low' | 'medium' | 'high';
  dimensions?: {
    width: number;
    height: number;
  };
  includeData?: boolean;
  filename?: string;
}

export interface ThemeSettings {
  mode: 'light' | 'dark' | 'system';
  primaryColor: string;
  secondaryColor: string;
  accentColor: string;
  customColors?: Record<string, string>;
  animations: boolean;
  reducedMotion: boolean;
}

export interface UserPreferences {
  theme: ThemeSettings;
  defaultChartType: string;
  showTooltips: boolean;
  showConfidenceIntervals: boolean;
  defaultExportFormat: string;
  autoSave: boolean;
  notifications: boolean;
  language: string;
}

export interface AppError {
  id: string;
  message: string;
  type: 'error' | 'warning' | 'info';
  timestamp: string;
  details?: any;
  stack?: string;
  component?: string;
  action?: string;
}

export interface LoadingState {
  isLoading: boolean;
  message?: string;
  progress?: number;
  error?: AppError;
}

export interface TabItem {
  id: string;
  label: string;
  icon: string;
  component: React.ComponentType<any>;
  disabled?: boolean;
  badge?: string | number;
  description?: string;
}

export interface NavigationItem {
  id: string;
  label: string;
  icon: string;
  path: string;
  children?: NavigationItem[];
  disabled?: boolean;
  badge?: string | number;
  external?: boolean;
}

export interface ComponentState {
  isVisible: boolean;
  isExpanded: boolean;
  isLoading: boolean;
  error?: AppError;
  lastUpdated?: string;
}

export interface APIResponse<T> {
  data: T;
  success: boolean;
  message?: string;
  error?: string;
  timestamp: string;
  pagination?: {
    page: number;
    pageSize: number;
    total: number;
    totalPages: number;
  };
}

export interface PaginationOptions {
  page: number;
  pageSize: number;
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
  filters?: Record<string, any>;
}

export interface ValidationError {
  field: string;
  message: string;
  value?: any;
}

export interface FormState {
  values: Record<string, any>;
  errors: ValidationError[];
  touched: Record<string, boolean>;
  isSubmitting: boolean;
  isValid: boolean;
}

// Utility types
export type ModelType = 'classification' | 'regression';
export type FeatureType = 'numerical' | 'categorical' | 'boolean';
export type ChartType = 'bar' | 'line' | 'scatter' | 'heatmap' | 'box' | 'violin' | 'histogram';
export type ExportFormat = 'png' | 'svg' | 'pdf' | 'csv' | 'json';
export type ThemeMode = 'light' | 'dark' | 'system';
export type SortOrder = 'asc' | 'desc';
export type LoadingStatus = 'idle' | 'loading' | 'success' | 'error';

// Event types
export interface ChartEvent {
  type: 'click' | 'hover' | 'select';
  data: any;
  element?: HTMLElement;
  index?: number;
}

export interface FeatureEvent {
  type: 'select' | 'deselect' | 'modify';
  feature: string;
  value?: any;
  previousValue?: any;
}

export interface PredictionEvent {
  type: 'predict' | 'explain' | 'compare';
  data: any;
  result?: any;
  error?: AppError;
}

// Configuration types
export interface ChartConfig {
  type: ChartType;
  options: ChartOptions;
  data: ChartData;
  responsive?: boolean;
  interactive?: boolean;
  exportable?: boolean;
}

export interface FeatureConfig {
  displayName?: string;
  description?: string;
  format?: (value: any) => string;
  validate?: (value: any) => boolean;
  options?: any[];
  min?: number;
  max?: number;
  step?: number;
  placeholder?: string;
  disabled?: boolean;
  required?: boolean;
}

export interface ViewportSize {
  width: number;
  height: number;
  isMobile: boolean;
  isTablet: boolean;
  isDesktop: boolean;
}

export interface PerformanceMetric {
  name: string;
  value: number;
  unit?: string;
  description?: string;
  target?: number;
  status?: 'good' | 'warning' | 'poor';
}

export interface Notification {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message: string;
  duration?: number;
  persistent?: boolean;
  actions?: NotificationAction[];
  timestamp: string;
}

export interface NotificationAction {
  label: string;
  action: () => void;
  style?: 'primary' | 'secondary' | 'ghost';
}

export interface KeyboardShortcut {
  key: string;
  modifiers?: ('ctrl' | 'alt' | 'shift' | 'meta')[];
  action: () => void;
  description: string;
  enabled?: boolean;
}

export interface AccessibilityOptions {
  screenReader: boolean;
  highContrast: boolean;
  reducedMotion: boolean;
  keyboardNavigation: boolean;
  focusIndicators: boolean;
  skipLinks: boolean;
}