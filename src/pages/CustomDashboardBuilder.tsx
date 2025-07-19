import React, { useState, useEffect, useRef } from 'react';
import { motion, useDrag } from 'framer-motion';
import {
  Plus,
  Save,
  Download,
  Upload,
  Settings,
  BarChart3,
  LineChart,
  PieChart,
  Target,
  TrendingUp,
  Users,
  Clock,
  AlertTriangle,
  Grid,
  Layout,
  Edit3,
  Trash2,
  Copy,
  Eye,
  RefreshCw,
  Monitor,
  Maximize2,
  Minimize2,
  Filter,
  Search,
  ChevronDown,
  ChevronRight,
  Play,
  Palette,
  Type,
  Image,
  MoreVertical
} from 'lucide-react';
import { Responsive, WidthProvider, Layout as GridLayout } from 'react-grid-layout';
import Card from '../components/common/Card';
import MetricCard from '../components/common/MetricCard';
import { apiClient } from '../services/api';

const ResponsiveGridLayout = WidthProvider(Responsive);

interface DashboardWidget {
  id: string;
  type: 'metric' | 'chart' | 'table' | 'text' | 'image' | 'kpi';
  title: string;
  config: {
    dataSource?: string;
    metric?: string;
    aggregation?: string;
    filters?: Record<string, any>;
    visualization?: string;
    styling?: {
      backgroundColor?: string;
      textColor?: string;
      borderColor?: string;
      fontSize?: string;
    };
    customQuery?: string;
    refreshInterval?: number;
  };
  layout: {
    x: number;
    y: number;
    w: number;
    h: number;
    minW?: number;
    minH?: number;
    maxW?: number;
    maxH?: number;
  };
  data?: any;
  lastUpdated?: string;
}

interface Dashboard {
  id: string;
  name: string;
  description: string;
  widgets: DashboardWidget[];
  layout: GridLayout[];
  settings: {
    refreshInterval: number;
    theme: 'light' | 'dark';
    columns: number;
    autoRefresh: boolean;
  };
  permissions: {
    viewers: string[];
    editors: string[];
    isPublic: boolean;
  };
  tags: string[];
  createdAt: string;
  updatedAt: string;
  createdBy: string;
}

interface WidgetTemplate {
  id: string;
  name: string;
  type: DashboardWidget['type'];
  icon: React.ReactNode;
  description: string;
  defaultConfig: DashboardWidget['config'];
  defaultLayout: { w: number; h: number };
}

const CustomDashboardBuilder: React.FC = () => {
  const [dashboards, setDashboards] = useState<Dashboard[]>([]);
  const [currentDashboard, setCurrentDashboard] = useState<Dashboard | null>(null);
  const [isEditing, setIsEditing] = useState(false);
  const [selectedWidget, setSelectedWidget] = useState<DashboardWidget | null>(null);
  const [showWidgetLibrary, setShowWidgetLibrary] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [loading, setLoading] = useState(true);
  const [previewMode, setPreviewMode] = useState(false);
  const [activeTab, setActiveTab] = useState<'builder' | 'dashboards' | 'templates'>('dashboards');

  const widgetTemplates: WidgetTemplate[] = [
    {
      id: 'metric-card',
      name: 'Metric Card',
      type: 'metric',
      icon: <Target className="w-5 h-5" />,
      description: 'Display key metrics with trend indicators',
      defaultConfig: {
        dataSource: 'models',
        metric: 'accuracy',
        aggregation: 'avg',
        refreshInterval: 300
      },
      defaultLayout: { w: 3, h: 2 }
    },
    {
      id: 'line-chart',
      name: 'Line Chart',
      type: 'chart',
      icon: <LineChart className="w-5 h-5" />,
      description: 'Time series visualization',
      defaultConfig: {
        dataSource: 'predictions',
        visualization: 'line',
        refreshInterval: 300
      },
      defaultLayout: { w: 6, h: 4 }
    },
    {
      id: 'bar-chart',
      name: 'Bar Chart',
      type: 'chart',
      icon: <BarChart3 className="w-5 h-5" />,
      description: 'Compare values across categories',
      defaultConfig: {
        dataSource: 'models',
        visualization: 'bar',
        refreshInterval: 300
      },
      defaultLayout: { w: 6, h: 4 }
    },
    {
      id: 'pie-chart',
      name: 'Pie Chart',
      type: 'chart',
      icon: <PieChart className="w-5 h-5" />,
      description: 'Show proportional data',
      defaultConfig: {
        dataSource: 'alerts',
        visualization: 'pie',
        refreshInterval: 300
      },
      defaultLayout: { w: 4, h: 4 }
    },
    {
      id: 'kpi-widget',
      name: 'KPI Widget',
      type: 'kpi',
      icon: <TrendingUp className="w-5 h-5" />,
      description: 'Key performance indicator with goals',
      defaultConfig: {
        dataSource: 'custom',
        refreshInterval: 300
      },
      defaultLayout: { w: 4, h: 3 }
    },
    {
      id: 'data-table',
      name: 'Data Table',
      type: 'table',
      icon: <Grid className="w-5 h-5" />,
      description: 'Tabular data display',
      defaultConfig: {
        dataSource: 'predictions',
        refreshInterval: 300
      },
      defaultLayout: { w: 8, h: 6 }
    },
    {
      id: 'text-widget',
      name: 'Text Widget',
      type: 'text',
      icon: <Type className="w-5 h-5" />,
      description: 'Custom text and markdown content',
      defaultConfig: {
        refreshInterval: 0
      },
      defaultLayout: { w: 4, h: 2 }
    }
  ];

  const mockDashboards: Dashboard[] = [
    {
      id: 'dash_001',
      name: 'Model Performance Overview',
      description: 'Real-time monitoring of all model performance metrics',
      widgets: [
        {
          id: 'widget_001',
          type: 'metric',
          title: 'Overall Accuracy',
          config: {
            dataSource: 'models',
            metric: 'accuracy',
            aggregation: 'avg',
            refreshInterval: 300
          },
          layout: { x: 0, y: 0, w: 3, h: 2 },
          data: { value: 0.867, change: 2.3, trend: 'up' }
        },
        {
          id: 'widget_002',
          type: 'chart',
          title: 'Prediction Volume',
          config: {
            dataSource: 'predictions',
            visualization: 'line',
            refreshInterval: 300
          },
          layout: { x: 3, y: 0, w: 6, h: 4 },
          data: {
            labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            datasets: [{
              label: 'Predictions',
              data: [1200, 1900, 3000, 5000, 2300, 3200, 4100]
            }]
          }
        },
        {
          id: 'widget_003',
          type: 'metric',
          title: 'Active Models',
          config: {
            dataSource: 'models',
            metric: 'count',
            filters: { status: 'active' },
            refreshInterval: 300
          },
          layout: { x: 9, y: 0, w: 3, h: 2 },
          data: { value: 12, change: 1, trend: 'up' }
        }
      ],
      layout: [
        { i: 'widget_001', x: 0, y: 0, w: 3, h: 2 },
        { i: 'widget_002', x: 3, y: 0, w: 6, h: 4 },
        { i: 'widget_003', x: 9, y: 0, w: 3, h: 2 }
      ],
      settings: {
        refreshInterval: 300,
        theme: 'light',
        columns: 12,
        autoRefresh: true
      },
      permissions: {
        viewers: ['analyst', 'data_scientist'],
        editors: ['admin'],
        isPublic: false
      },
      tags: ['performance', 'monitoring', 'models'],
      createdAt: '2024-01-15T10:00:00Z',
      updatedAt: '2024-01-20T14:30:00Z',
      createdBy: 'admin'
    },
    {
      id: 'dash_002',
      name: 'Executive Summary',
      description: 'High-level KPIs for executive stakeholders',
      widgets: [],
      layout: [],
      settings: {
        refreshInterval: 600,
        theme: 'light',
        columns: 12,
        autoRefresh: true
      },
      permissions: {
        viewers: ['executive', 'admin'],
        editors: ['admin'],
        isPublic: false
      },
      tags: ['executive', 'kpi', 'summary'],
      createdAt: '2024-01-18T09:00:00Z',
      updatedAt: '2024-01-20T12:00:00Z',
      createdBy: 'admin'
    }
  ];

  useEffect(() => {
    loadDashboards();
  }, []);

  const loadDashboards = async () => {
    try {
      setLoading(true);
      
      // Try to fetch from API, fallback to mock data
      const response = await apiClient.getCustomDashboards();
      
      if (response.success && response.data) {
        setDashboards(response.data);
      } else {
        // Use mock data as fallback
        setTimeout(() => {
          setDashboards(mockDashboards);
          setCurrentDashboard(mockDashboards[0]);
          setLoading(false);
        }, 1000);
        return;
      }
    } catch (err) {
      console.warn('API call failed, using mock data:', err);
      // Use mock data as fallback
      setTimeout(() => {
        setDashboards(mockDashboards);
        setCurrentDashboard(mockDashboards[0]);
        setLoading(false);
      }, 1000);
      return;
    }
    
    setLoading(false);
  };

  const createNewDashboard = () => {
    const newDashboard: Dashboard = {
      id: `dash_${Date.now()}`,
      name: 'New Dashboard',
      description: 'Custom dashboard description',
      widgets: [],
      layout: [],
      settings: {
        refreshInterval: 300,
        theme: 'light',
        columns: 12,
        autoRefresh: true
      },
      permissions: {
        viewers: [],
        editors: [],
        isPublic: false
      },
      tags: [],
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      createdBy: 'current_user'
    };
    
    setCurrentDashboard(newDashboard);
    setIsEditing(true);
    setActiveTab('builder');
  };

  const addWidget = (template: WidgetTemplate) => {
    if (!currentDashboard) return;
    
    const newWidget: DashboardWidget = {
      id: `widget_${Date.now()}`,
      type: template.type,
      title: template.name,
      config: { ...template.defaultConfig },
      layout: {
        x: 0,
        y: 0,
        w: template.defaultLayout.w,
        h: template.defaultLayout.h
      }
    };
    
    const newLayout: GridLayout = {
      i: newWidget.id,
      x: 0,
      y: 0,
      w: template.defaultLayout.w,
      h: template.defaultLayout.h
    };
    
    setCurrentDashboard({
      ...currentDashboard,
      widgets: [...currentDashboard.widgets, newWidget],
      layout: [...currentDashboard.layout, newLayout],
      updatedAt: new Date().toISOString()
    });
    
    setShowWidgetLibrary(false);
  };

  const updateWidget = (widgetId: string, updates: Partial<DashboardWidget>) => {
    if (!currentDashboard) return;
    
    setCurrentDashboard({
      ...currentDashboard,
      widgets: currentDashboard.widgets.map(widget =>
        widget.id === widgetId ? { ...widget, ...updates } : widget
      ),
      updatedAt: new Date().toISOString()
    });
  };

  const deleteWidget = (widgetId: string) => {
    if (!currentDashboard) return;
    
    setCurrentDashboard({
      ...currentDashboard,
      widgets: currentDashboard.widgets.filter(widget => widget.id !== widgetId),
      layout: currentDashboard.layout.filter(item => item.i !== widgetId),
      updatedAt: new Date().toISOString()
    });
    
    if (selectedWidget?.id === widgetId) {
      setSelectedWidget(null);
    }
  };

  const duplicateWidget = (widgetId: string) => {
    if (!currentDashboard) return;
    
    const widget = currentDashboard.widgets.find(w => w.id === widgetId);
    if (!widget) return;
    
    const newWidget: DashboardWidget = {
      ...widget,
      id: `widget_${Date.now()}`,
      title: `${widget.title} (Copy)`,
      layout: {
        ...widget.layout,
        x: widget.layout.x + 1,
        y: widget.layout.y + 1
      }
    };
    
    const newLayout: GridLayout = {
      i: newWidget.id,
      x: newWidget.layout.x,
      y: newWidget.layout.y,
      w: newWidget.layout.w,
      h: newWidget.layout.h
    };
    
    setCurrentDashboard({
      ...currentDashboard,
      widgets: [...currentDashboard.widgets, newWidget],
      layout: [...currentDashboard.layout, newLayout],
      updatedAt: new Date().toISOString()
    });
  };

  const onLayoutChange = (layout: GridLayout[]) => {
    if (!currentDashboard || !isEditing) return;
    
    setCurrentDashboard({
      ...currentDashboard,
      layout,
      updatedAt: new Date().toISOString()
    });
  };

  const saveDashboard = async () => {
    if (!currentDashboard) return;
    
    try {
      const response = await apiClient.saveDashboard(currentDashboard);
      
      if (response.success) {
        setDashboards(prev => {
          const existingIndex = prev.findIndex(d => d.id === currentDashboard.id);
          if (existingIndex >= 0) {
            const updated = [...prev];
            updated[existingIndex] = currentDashboard;
            return updated;
          } else {
            return [...prev, currentDashboard];
          }
        });
        
        setIsEditing(false);
      }
    } catch (err) {
      console.error('Failed to save dashboard:', err);
      // For demo, just update local state
      setDashboards(prev => {
        const existingIndex = prev.findIndex(d => d.id === currentDashboard.id);
        if (existingIndex >= 0) {
          const updated = [...prev];
          updated[existingIndex] = currentDashboard;
          return updated;
        } else {
          return [...prev, currentDashboard];
        }
      });
      setIsEditing(false);
    }
  };

  const renderWidget = (widget: DashboardWidget) => {
    const isSelected = selectedWidget?.id === widget.id;
    
    return (
      <div
        key={widget.id}
        className={`relative group ${isSelected ? 'ring-2 ring-primary-500' : ''}`}
        onClick={() => isEditing && setSelectedWidget(widget)}
      >
        <Card className="h-full">
          <div className="p-4 h-full">
            {/* Widget Header */}
            <div className="flex items-center justify-between mb-3">
              <h4 className="font-medium text-neutral-900 dark:text-neutral-100 truncate">
                {widget.title}
              </h4>
              
              {isEditing && (
                <div className="opacity-0 group-hover:opacity-100 transition-opacity flex items-center space-x-1">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      setSelectedWidget(widget);
                    }}
                    className="p-1 text-neutral-500 hover:text-primary-600 transition-colors"
                  >
                    <Settings className="w-4 h-4" />
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      duplicateWidget(widget.id);
                    }}
                    className="p-1 text-neutral-500 hover:text-blue-600 transition-colors"
                  >
                    <Copy className="w-4 h-4" />
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      deleteWidget(widget.id);
                    }}
                    className="p-1 text-neutral-500 hover:text-red-600 transition-colors"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              )}
            </div>
            
            {/* Widget Content */}
            <div className="flex-1">
              {widget.type === 'metric' && (
                <div className="text-center">
                  <div className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
                    {widget.data?.value || '---'}
                  </div>
                  {widget.data?.change && (
                    <div className={`text-sm ${
                      widget.data.trend === 'up' ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {widget.data.change > 0 ? '+' : ''}{widget.data.change}%
                    </div>
                  )}
                </div>
              )}
              
              {widget.type === 'chart' && (
                <div className="h-32 bg-neutral-100 dark:bg-neutral-800 rounded-lg flex items-center justify-center">
                  <div className="text-center text-neutral-500">
                    {widget.config.visualization === 'line' && <LineChart className="w-8 h-8 mx-auto mb-2" />}
                    {widget.config.visualization === 'bar' && <BarChart3 className="w-8 h-8 mx-auto mb-2" />}
                    {widget.config.visualization === 'pie' && <PieChart className="w-8 h-8 mx-auto mb-2" />}
                    <div className="text-sm">Chart Preview</div>
                  </div>
                </div>
              )}
              
              {widget.type === 'kpi' && (
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm text-neutral-600 dark:text-neutral-400">Current:</span>
                    <span className="font-medium">85.2%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-neutral-600 dark:text-neutral-400">Target:</span>
                    <span className="font-medium">90.0%</span>
                  </div>
                  <div className="w-full bg-neutral-200 dark:bg-neutral-700 rounded-full h-2">
                    <div className="bg-primary-500 h-2 rounded-full" style={{ width: '85%' }} />
                  </div>
                </div>
              )}
              
              {widget.type === 'table' && (
                <div className="space-y-2">
                  <div className="grid grid-cols-3 gap-2 text-xs font-medium text-neutral-600 dark:text-neutral-400">
                    <div>Model</div>
                    <div>Accuracy</div>
                    <div>Status</div>
                  </div>
                  <div className="space-y-1">
                    {[1, 2, 3].map(i => (
                      <div key={i} className="grid grid-cols-3 gap-2 text-xs">
                        <div>Model {i}</div>
                        <div>0.{85 + i}%</div>
                        <div className="text-green-600">Active</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              
              {widget.type === 'text' && (
                <div className="text-sm text-neutral-600 dark:text-neutral-400">
                  {widget.config.customQuery || 'Custom text content...'}
                </div>
              )}
            </div>
            
            {/* Last Updated */}
            {widget.lastUpdated && (
              <div className="text-xs text-neutral-400 mt-2">
                Updated: {new Date(widget.lastUpdated).toLocaleTimeString()}
              </div>
            )}
          </div>
        </Card>
      </div>
    );
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <RefreshCw className="w-8 h-8 animate-spin text-primary-600" />
        <span className="ml-3 text-lg text-neutral-600 dark:text-neutral-400">
          Loading dashboards...
        </span>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-neutral-900 dark:text-neutral-100">
            Custom Dashboard Builder
          </h1>
          <p className="text-neutral-600 dark:text-neutral-400 mt-1">
            Create and manage custom dashboards with drag-and-drop widgets
          </p>
        </div>
        
        <div className="flex items-center space-x-4">
          {currentDashboard && (
            <>
              <button
                onClick={() => setPreviewMode(!previewMode)}
                className="flex items-center space-x-2 px-4 py-2 bg-neutral-600 hover:bg-neutral-700 text-white rounded-lg transition-colors"
              >
                {previewMode ? <Edit3 className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                <span>{previewMode ? 'Edit' : 'Preview'}</span>
              </button>
              
              {isEditing && (
                <button
                  onClick={saveDashboard}
                  className="flex items-center space-x-2 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors"
                >
                  <Save className="w-4 h-4" />
                  <span>Save</span>
                </button>
              )}
            </>
          )}
          
          <button
            onClick={createNewDashboard}
            className="flex items-center space-x-2 px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors"
          >
            <Plus className="w-4 h-4" />
            <span>New Dashboard</span>
          </button>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-neutral-200 dark:border-neutral-700">
        <nav className="flex space-x-8">
          {[
            { id: 'dashboards', label: 'My Dashboards', icon: Monitor },
            { id: 'builder', label: 'Dashboard Builder', icon: Layout },
            { id: 'templates', label: 'Templates', icon: Grid }
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                activeTab === tab.id
                  ? 'border-primary-500 text-primary-600 dark:text-primary-400'
                  : 'border-transparent text-neutral-500 hover:text-neutral-700 dark:hover:text-neutral-300'
              }`}
            >
              <tab.icon className="w-4 h-4" />
              <span>{tab.label}</span>
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      {activeTab === 'dashboards' && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {dashboards.map((dashboard) => (
            <Card key={dashboard.id} className="cursor-pointer hover:shadow-lg transition-shadow">
              <div className="p-6" onClick={() => {
                setCurrentDashboard(dashboard);
                setActiveTab('builder');
              }}>
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-semibold text-neutral-900 dark:text-neutral-100">
                    {dashboard.name}
                  </h3>
                  <div className="flex items-center space-x-1">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setCurrentDashboard(dashboard);
                        setIsEditing(true);
                        setActiveTab('builder');
                      }}
                      className="p-1 text-neutral-500 hover:text-primary-600 transition-colors"
                    >
                      <Edit3 className="w-4 h-4" />
                    </button>
                    <button className="p-1 text-neutral-500 hover:text-red-600 transition-colors">
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
                
                <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-4">
                  {dashboard.description}
                </p>
                
                <div className="flex items-center justify-between text-xs text-neutral-500">
                  <span>{dashboard.widgets.length} widgets</span>
                  <span>Updated {new Date(dashboard.updatedAt).toLocaleDateString()}</span>
                </div>
                
                {dashboard.tags.length > 0 && (
                  <div className="flex flex-wrap gap-1 mt-3">
                    {dashboard.tags.slice(0, 3).map((tag) => (
                      <span
                        key={tag}
                        className="px-2 py-1 bg-neutral-100 dark:bg-neutral-800 text-neutral-600 dark:text-neutral-400 rounded text-xs"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </Card>
          ))}
        </div>
      )}

      {activeTab === 'builder' && currentDashboard && (
        <div className="space-y-6">
          {/* Dashboard Header */}
          <div className="flex items-center justify-between">
            <div className="flex-1">
              {isEditing ? (
                <input
                  type="text"
                  value={currentDashboard.name}
                  onChange={(e) => setCurrentDashboard({
                    ...currentDashboard,
                    name: e.target.value,
                    updatedAt: new Date().toISOString()
                  })}
                  className="text-2xl font-bold bg-transparent border-b border-neutral-300 dark:border-neutral-600 text-neutral-900 dark:text-neutral-100 focus:outline-none focus:border-primary-500"
                />
              ) : (
                <h2 className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
                  {currentDashboard.name}
                </h2>
              )}
              
              {isEditing ? (
                <input
                  type="text"
                  value={currentDashboard.description}
                  onChange={(e) => setCurrentDashboard({
                    ...currentDashboard,
                    description: e.target.value,
                    updatedAt: new Date().toISOString()
                  })}
                  className="mt-1 text-neutral-600 dark:text-neutral-400 bg-transparent border-b border-neutral-300 dark:border-neutral-600 focus:outline-none focus:border-primary-500"
                  placeholder="Dashboard description..."
                />
              ) : (
                <p className="text-neutral-600 dark:text-neutral-400 mt-1">
                  {currentDashboard.description}
                </p>
              )}
            </div>
            
            <div className="flex items-center space-x-3">
              {isEditing && (
                <button
                  onClick={() => setShowWidgetLibrary(true)}
                  className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
                >
                  <Plus className="w-4 h-4" />
                  <span>Add Widget</span>
                </button>
              )}
              
              <button
                onClick={() => setIsEditing(!isEditing)}
                className="flex items-center space-x-2 px-4 py-2 bg-neutral-600 hover:bg-neutral-700 text-white rounded-lg transition-colors"
              >
                {isEditing ? <Eye className="w-4 h-4" /> : <Edit3 className="w-4 h-4" />}
                <span>{isEditing ? 'Preview' : 'Edit'}</span>
              </button>
            </div>
          </div>

          {/* Dashboard Grid */}
          <div className="min-h-96">
            {currentDashboard.widgets.length > 0 ? (
              <ResponsiveGridLayout
                className="layout"
                layouts={{
                  lg: currentDashboard.layout,
                  md: currentDashboard.layout,
                  sm: currentDashboard.layout,
                  xs: currentDashboard.layout,
                  xxs: currentDashboard.layout
                }}
                breakpoints={{ lg: 1200, md: 996, sm: 768, xs: 480, xxs: 0 }}
                cols={{ lg: 12, md: 10, sm: 6, xs: 4, xxs: 2 }}
                rowHeight={60}
                onLayoutChange={onLayoutChange}
                isDraggable={isEditing}
                isResizable={isEditing}
              >
                {currentDashboard.widgets.map(renderWidget)}
              </ResponsiveGridLayout>
            ) : (
              <div className="flex flex-col items-center justify-center h-64 border-2 border-dashed border-neutral-300 dark:border-neutral-600 rounded-lg">
                <Grid className="w-12 h-12 text-neutral-400 mb-4" />
                <h3 className="text-lg font-medium text-neutral-900 dark:text-neutral-100 mb-2">
                  No widgets yet
                </h3>
                <p className="text-neutral-600 dark:text-neutral-400 mb-4">
                  Add your first widget to get started building your dashboard
                </p>
                <button
                  onClick={() => setShowWidgetLibrary(true)}
                  className="flex items-center space-x-2 px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors"
                >
                  <Plus className="w-4 h-4" />
                  <span>Add Widget</span>
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {activeTab === 'templates' && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <Card className="cursor-pointer hover:shadow-lg transition-shadow">
            <div className="p-6">
              <h3 className="font-semibold text-neutral-900 dark:text-neutral-100 mb-2">
                Model Performance Template
              </h3>
              <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-4">
                Pre-built dashboard for monitoring ML model performance metrics
              </p>
              <button className="w-full px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors">
                Use Template
              </button>
            </div>
          </Card>
          
          <Card className="cursor-pointer hover:shadow-lg transition-shadow">
            <div className="p-6">
              <h3 className="font-semibold text-neutral-900 dark:text-neutral-100 mb-2">
                Executive Summary Template
              </h3>
              <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-4">
                High-level KPIs and metrics for executive stakeholders
              </p>
              <button className="w-full px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors">
                Use Template
              </button>
            </div>
          </Card>
          
          <Card className="cursor-pointer hover:shadow-lg transition-shadow">
            <div className="p-6">
              <h3 className="font-semibold text-neutral-900 dark:text-neutral-100 mb-2">
                Data Quality Template
              </h3>
              <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-4">
                Monitor data quality metrics and validation results
              </p>
              <button className="w-full px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors">
                Use Template
              </button>
            </div>
          </Card>
        </div>
      )}

      {/* Widget Library Modal */}
      {showWidgetLibrary && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-neutral-800 rounded-lg shadow-xl max-w-4xl w-full mx-4 max-h-[80vh] overflow-y-auto">
            <div className="p-6 border-b border-neutral-200 dark:border-neutral-700">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                  Widget Library
                </h3>
                <button
                  onClick={() => setShowWidgetLibrary(false)}
                  className="text-neutral-500 hover:text-neutral-700 dark:hover:text-neutral-300"
                >
                  ×
                </button>
              </div>
            </div>
            
            <div className="p-6">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {widgetTemplates.map((template) => (
                  <div
                    key={template.id}
                    className="p-4 border border-neutral-200 dark:border-neutral-700 rounded-lg hover:border-primary-300 dark:hover:border-primary-600 cursor-pointer transition-colors"
                    onClick={() => addWidget(template)}
                  >
                    <div className="flex items-center space-x-3 mb-3">
                      <div className="p-2 bg-primary-100 dark:bg-primary-900/20 rounded-lg">
                        {template.icon}
                      </div>
                      <div>
                        <h4 className="font-medium text-neutral-900 dark:text-neutral-100">
                          {template.name}
                        </h4>
                        <p className="text-xs text-neutral-600 dark:text-neutral-400">
                          {template.defaultLayout.w}×{template.defaultLayout.h}
                        </p>
                      </div>
                    </div>
                    <p className="text-sm text-neutral-600 dark:text-neutral-400">
                      {template.description}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Widget Configuration Panel */}
      {selectedWidget && isEditing && (
        <div className="fixed right-0 top-0 h-full w-80 bg-white dark:bg-neutral-800 shadow-xl border-l border-neutral-200 dark:border-neutral-700 z-40">
          <div className="p-6 border-b border-neutral-200 dark:border-neutral-700">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                Widget Settings
              </h3>
              <button
                onClick={() => setSelectedWidget(null)}
                className="text-neutral-500 hover:text-neutral-700 dark:hover:text-neutral-300"
              >
                ×
              </button>
            </div>
          </div>
          
          <div className="p-6 space-y-4">
            <div>
              <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-1">
                Widget Title
              </label>
              <input
                type="text"
                value={selectedWidget.title}
                onChange={(e) => updateWidget(selectedWidget.id, { title: e.target.value })}
                className="w-full px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-1">
                Data Source
              </label>
              <select
                value={selectedWidget.config.dataSource || ''}
                onChange={(e) => updateWidget(selectedWidget.id, {
                  config: { ...selectedWidget.config, dataSource: e.target.value }
                })}
                className="w-full px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
              >
                <option value="models">Models</option>
                <option value="predictions">Predictions</option>
                <option value="alerts">Alerts</option>
                <option value="custom">Custom Query</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-1">
                Refresh Interval (seconds)
              </label>
              <input
                type="number"
                value={selectedWidget.config.refreshInterval || 300}
                onChange={(e) => updateWidget(selectedWidget.id, {
                  config: { ...selectedWidget.config, refreshInterval: parseInt(e.target.value) }
                })}
                className="w-full px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
              />
            </div>
            
            {selectedWidget.type === 'chart' && (
              <div>
                <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-1">
                  Chart Type
                </label>
                <select
                  value={selectedWidget.config.visualization || ''}
                  onChange={(e) => updateWidget(selectedWidget.id, {
                    config: { ...selectedWidget.config, visualization: e.target.value }
                  })}
                  className="w-full px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
                >
                  <option value="line">Line Chart</option>
                  <option value="bar">Bar Chart</option>
                  <option value="pie">Pie Chart</option>
                  <option value="area">Area Chart</option>
                </select>
              </div>
            )}
            
            {(selectedWidget.type === 'text' || selectedWidget.config.dataSource === 'custom') && (
              <div>
                <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-1">
                  Custom Content/Query
                </label>
                <textarea
                  value={selectedWidget.config.customQuery || ''}
                  onChange={(e) => updateWidget(selectedWidget.id, {
                    config: { ...selectedWidget.config, customQuery: e.target.value }
                  })}
                  rows={4}
                  className="w-full px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-lg bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
                  placeholder="Enter custom content or SQL query..."
                />
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default CustomDashboardBuilder;