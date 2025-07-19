import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  Settings as SettingsIcon, Save, RefreshCw, Shield, Database,
  Bell, Palette, Monitor, Globe, Key, AlertTriangle, Check,
  Mail, Slack, Upload, Download, Trash2, Eye, EyeOff
} from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import Button from '../components/common/Button';
import Card from '../components/common/Card';
import { toast } from 'react-hot-toast';

interface SettingsData {
  // System Settings
  systemName: string;
  systemDescription: string;
  maintenanceMode: boolean;
  debugMode: boolean;
  logLevel: string;
  
  // Security Settings
  sessionTimeout: number;
  maxLoginAttempts: number;
  passwordMinLength: number;
  requireTwoFactor: boolean;
  
  // ML Settings
  defaultExplanationMethod: string;
  maxModelSize: number;
  autoRetraining: boolean;
  driftThreshold: number;
  
  // Notification Settings
  emailNotifications: boolean;
  slackNotifications: boolean;
  emailServer: string;
  slackWebhook: string;
  
  // API Settings
  rateLimitDefault: number;
  rateLimitStrict: number;
  corsOrigins: string;
  
  // Database Settings
  backupEnabled: boolean;
  backupRetentionDays: number;
  connectionPoolSize: number;
}

const Settings: React.FC = () => {
  const { hasPermission } = useAuth();
  const [activeTab, setActiveTab] = useState('system');
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setSaving] = useState(false);
  const [showPasswords, setShowPasswords] = useState(false);
  
  const [settings, setSettings] = useState<SettingsData>({
    // System Settings
    systemName: 'ML Explainer Dashboard',
    systemDescription: 'Enterprise Machine Learning Explainability Platform',
    maintenanceMode: false,
    debugMode: false,
    logLevel: 'INFO',
    
    // Security Settings
    sessionTimeout: 30,
    maxLoginAttempts: 5,
    passwordMinLength: 8,
    requireTwoFactor: false,
    
    // ML Settings
    defaultExplanationMethod: 'SHAP',
    maxModelSize: 500,
    autoRetraining: false,
    driftThreshold: 0.1,
    
    // Notification Settings
    emailNotifications: true,
    slackNotifications: false,
    emailServer: '',
    slackWebhook: '',
    
    // API Settings
    rateLimitDefault: 100,
    rateLimitStrict: 20,
    corsOrigins: '',
    
    // Database Settings
    backupEnabled: true,
    backupRetentionDays: 30,
    connectionPoolSize: 10,
  });

  const canManageSettings = hasPermission('system:config');

  useEffect(() => {
    if (canManageSettings) {
      loadSettings();
    }
  }, [canManageSettings]);

  const loadSettings = async () => {
    setIsLoading(true);
    try {
      // Mock API call - replace with actual implementation
      await new Promise(resolve => setTimeout(resolve, 1000));
      // Settings would be loaded from API here
    } catch (error) {
      toast.error('Failed to load settings');
    } finally {
      setIsLoading(false);
    }
  };

  const saveSettings = async () => {
    setSaving(true);
    try {
      // Mock API call - replace with actual implementation
      await new Promise(resolve => setTimeout(resolve, 1500));
      toast.success('Settings saved successfully');
    } catch (error) {
      toast.error('Failed to save settings');
    } finally {
      setSaving(false);
    }
  };

  const resetToDefaults = () => {
    if (confirm('Are you sure you want to reset all settings to defaults? This action cannot be undone.')) {
      // Reset logic here
      toast.success('Settings reset to defaults');
    }
  };

  const exportSettings = () => {
    const dataStr = JSON.stringify(settings, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'ml-explainer-settings.json';
    link.click();
    URL.revokeObjectURL(url);
    toast.success('Settings exported');
  };

  const tabs = [
    { id: 'system', label: 'System', icon: Monitor },
    { id: 'security', label: 'Security', icon: Shield },
    { id: 'ml', label: 'ML Engine', icon: Database },
    { id: 'notifications', label: 'Notifications', icon: Bell },
    { id: 'api', label: 'API', icon: Globe },
    { id: 'database', label: 'Database', icon: Database }
  ];

  if (!canManageSettings) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <Shield className="w-16 h-16 text-neutral-400 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-neutral-900 dark:text-white mb-2">
            Access Denied
          </h2>
          <p className="text-neutral-600 dark:text-neutral-400">
            You don't have permission to manage system settings.
          </p>
        </div>
      </div>
    );
  }

  const renderSystemSettings = () => (
    <div className="space-y-6">
      <div>
        <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
          System Name
        </label>
        <input
          type="text"
          value={settings.systemName}
          onChange={(e) => setSettings(prev => ({ ...prev, systemName: e.target.value }))}
          className="w-full px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white dark:bg-neutral-700 text-neutral-900 dark:text-white"
        />
      </div>
      
      <div>
        <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
          System Description
        </label>
        <textarea
          value={settings.systemDescription}
          onChange={(e) => setSettings(prev => ({ ...prev, systemDescription: e.target.value }))}
          rows={3}
          className="w-full px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white dark:bg-neutral-700 text-neutral-900 dark:text-white"
        />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
            Log Level
          </label>
          <select
            value={settings.logLevel}
            onChange={(e) => setSettings(prev => ({ ...prev, logLevel: e.target.value }))}
            className="w-full px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white dark:bg-neutral-700 text-neutral-900 dark:text-white"
          >
            <option value="DEBUG">Debug</option>
            <option value="INFO">Info</option>
            <option value="WARNING">Warning</option>
            <option value="ERROR">Error</option>
          </select>
        </div>
      </div>

      <div className="space-y-4">
        <div className="flex items-center justify-between p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg border border-yellow-200 dark:border-yellow-800">
          <div className="flex items-center space-x-3">
            <AlertTriangle className="w-5 h-5 text-yellow-600 dark:text-yellow-400" />
            <div>
              <h4 className="font-medium text-yellow-800 dark:text-yellow-200">Maintenance Mode</h4>
              <p className="text-sm text-yellow-700 dark:text-yellow-300">
                Prevents new user sessions and displays maintenance message
              </p>
            </div>
          </div>
          <label className="relative inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={settings.maintenanceMode}
              onChange={(e) => setSettings(prev => ({ ...prev, maintenanceMode: e.target.checked }))}
              className="sr-only peer"
            />
            <div className="w-11 h-6 bg-neutral-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary-300 dark:peer-focus:ring-primary-800 rounded-full peer dark:bg-neutral-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-neutral-600 peer-checked:bg-primary-600"></div>
          </label>
        </div>

        <div className="flex items-center justify-between p-4 bg-neutral-50 dark:bg-neutral-800 rounded-lg">
          <div>
            <h4 className="font-medium text-neutral-900 dark:text-white">Debug Mode</h4>
            <p className="text-sm text-neutral-600 dark:text-neutral-400">
              Enable detailed logging and debugging features
            </p>
          </div>
          <label className="relative inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={settings.debugMode}
              onChange={(e) => setSettings(prev => ({ ...prev, debugMode: e.target.checked }))}
              className="sr-only peer"
            />
            <div className="w-11 h-6 bg-neutral-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary-300 dark:peer-focus:ring-primary-800 rounded-full peer dark:bg-neutral-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-neutral-600 peer-checked:bg-primary-600"></div>
          </label>
        </div>
      </div>
    </div>
  );

  const renderSecuritySettings = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
            Session Timeout (minutes)
          </label>
          <input
            type="number"
            value={settings.sessionTimeout}
            onChange={(e) => setSettings(prev => ({ ...prev, sessionTimeout: parseInt(e.target.value) }))}
            min="5"
            max="480"
            className="w-full px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white dark:bg-neutral-700 text-neutral-900 dark:text-white"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
            Max Login Attempts
          </label>
          <input
            type="number"
            value={settings.maxLoginAttempts}
            onChange={(e) => setSettings(prev => ({ ...prev, maxLoginAttempts: parseInt(e.target.value) }))}
            min="3"
            max="10"
            className="w-full px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white dark:bg-neutral-700 text-neutral-900 dark:text-white"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
            Password Minimum Length
          </label>
          <input
            type="number"
            value={settings.passwordMinLength}
            onChange={(e) => setSettings(prev => ({ ...prev, passwordMinLength: parseInt(e.target.value) }))}
            min="6"
            max="20"
            className="w-full px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white dark:bg-neutral-700 text-neutral-900 dark:text-white"
          />
        </div>
      </div>

      <div className="flex items-center justify-between p-4 bg-neutral-50 dark:bg-neutral-800 rounded-lg">
        <div>
          <h4 className="font-medium text-neutral-900 dark:text-white">Two-Factor Authentication</h4>
          <p className="text-sm text-neutral-600 dark:text-neutral-400">
            Require 2FA for all user accounts
          </p>
        </div>
        <label className="relative inline-flex items-center cursor-pointer">
          <input
            type="checkbox"
            checked={settings.requireTwoFactor}
            onChange={(e) => setSettings(prev => ({ ...prev, requireTwoFactor: e.target.checked }))}
            className="sr-only peer"
          />
          <div className="w-11 h-6 bg-neutral-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary-300 dark:peer-focus:ring-primary-800 rounded-full peer dark:bg-neutral-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-neutral-600 peer-checked:bg-primary-600"></div>
        </label>
      </div>
    </div>
  );

  const renderMLSettings = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
            Default Explanation Method
          </label>
          <select
            value={settings.defaultExplanationMethod}
            onChange={(e) => setSettings(prev => ({ ...prev, defaultExplanationMethod: e.target.value }))}
            className="w-full px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white dark:bg-neutral-700 text-neutral-900 dark:text-white"
          >
            <option value="SHAP">SHAP</option>
            <option value="LIME">LIME</option>
            <option value="Integrated Gradients">Integrated Gradients</option>
            <option value="Permutation Importance">Permutation Importance</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
            Max Model Size (MB)
          </label>
          <input
            type="number"
            value={settings.maxModelSize}
            onChange={(e) => setSettings(prev => ({ ...prev, maxModelSize: parseInt(e.target.value) }))}
            min="50"
            max="2000"
            className="w-full px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white dark:bg-neutral-700 text-neutral-900 dark:text-white"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
            Drift Detection Threshold
          </label>
          <input
            type="number"
            value={settings.driftThreshold}
            onChange={(e) => setSettings(prev => ({ ...prev, driftThreshold: parseFloat(e.target.value) }))}
            min="0.01"
            max="0.5"
            step="0.01"
            className="w-full px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white dark:bg-neutral-700 text-neutral-900 dark:text-white"
          />
        </div>
      </div>

      <div className="flex items-center justify-between p-4 bg-neutral-50 dark:bg-neutral-800 rounded-lg">
        <div>
          <h4 className="font-medium text-neutral-900 dark:text-white">Auto Retraining</h4>
          <p className="text-sm text-neutral-600 dark:text-neutral-400">
            Automatically retrain models when drift is detected
          </p>
        </div>
        <label className="relative inline-flex items-center cursor-pointer">
          <input
            type="checkbox"
            checked={settings.autoRetraining}
            onChange={(e) => setSettings(prev => ({ ...prev, autoRetraining: e.target.checked }))}
            className="sr-only peer"
          />
          <div className="w-11 h-6 bg-neutral-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary-300 dark:peer-focus:ring-primary-800 rounded-full peer dark:bg-neutral-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-neutral-600 peer-checked:bg-primary-600"></div>
        </label>
      </div>
    </div>
  );

  const renderNotificationSettings = () => (
    <div className="space-y-6">
      <div className="space-y-4">
        <div className="flex items-center justify-between p-4 bg-neutral-50 dark:bg-neutral-800 rounded-lg">
          <div className="flex items-center space-x-3">
            <Mail className="w-5 h-5 text-blue-500" />
            <div>
              <h4 className="font-medium text-neutral-900 dark:text-white">Email Notifications</h4>
              <p className="text-sm text-neutral-600 dark:text-neutral-400">
                Send alerts and reports via email
              </p>
            </div>
          </div>
          <label className="relative inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={settings.emailNotifications}
              onChange={(e) => setSettings(prev => ({ ...prev, emailNotifications: e.target.checked }))}
              className="sr-only peer"
            />
            <div className="w-11 h-6 bg-neutral-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary-300 dark:peer-focus:ring-primary-800 rounded-full peer dark:bg-neutral-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-neutral-600 peer-checked:bg-primary-600"></div>
          </label>
        </div>

        <div className="flex items-center justify-between p-4 bg-neutral-50 dark:bg-neutral-800 rounded-lg">
          <div className="flex items-center space-x-3">
            <Slack className="w-5 h-5 text-purple-500" />
            <div>
              <h4 className="font-medium text-neutral-900 dark:text-white">Slack Notifications</h4>
              <p className="text-sm text-neutral-600 dark:text-neutral-400">
                Send alerts to Slack channels
              </p>
            </div>
          </div>
          <label className="relative inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={settings.slackNotifications}
              onChange={(e) => setSettings(prev => ({ ...prev, slackNotifications: e.target.checked }))}
              className="sr-only peer"
            />
            <div className="w-11 h-6 bg-neutral-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary-300 dark:peer-focus:ring-primary-800 rounded-full peer dark:bg-neutral-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-neutral-600 peer-checked:bg-primary-600"></div>
          </label>
        </div>
      </div>

      {settings.emailNotifications && (
        <div>
          <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
            Email Server Configuration
          </label>
          <input
            type="text"
            value={settings.emailServer}
            onChange={(e) => setSettings(prev => ({ ...prev, emailServer: e.target.value }))}
            placeholder="smtp.example.com"
            className="w-full px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white dark:bg-neutral-700 text-neutral-900 dark:text-white"
          />
        </div>
      )}

      {settings.slackNotifications && (
        <div>
          <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
            Slack Webhook URL
          </label>
          <div className="relative">
            <input
              type={showPasswords ? 'text' : 'password'}
              value={settings.slackWebhook}
              onChange={(e) => setSettings(prev => ({ ...prev, slackWebhook: e.target.value }))}
              placeholder="https://hooks.slack.com/services/..."
              className="w-full px-3 py-2 pr-10 border border-neutral-300 dark:border-neutral-600 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white dark:bg-neutral-700 text-neutral-900 dark:text-white"
            />
            <button
              type="button"
              onClick={() => setShowPasswords(!showPasswords)}
              className="absolute right-3 top-1/2 transform -translate-y-1/2 text-neutral-400 hover:text-neutral-600"
            >
              {showPasswords ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
            </button>
          </div>
        </div>
      )}
    </div>
  );

  const renderAPISettings = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
            Default Rate Limit (requests/minute)
          </label>
          <input
            type="number"
            value={settings.rateLimitDefault}
            onChange={(e) => setSettings(prev => ({ ...prev, rateLimitDefault: parseInt(e.target.value) }))}
            min="10"
            max="1000"
            className="w-full px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white dark:bg-neutral-700 text-neutral-900 dark:text-white"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
            Strict Rate Limit (requests/minute)
          </label>
          <input
            type="number"
            value={settings.rateLimitStrict}
            onChange={(e) => setSettings(prev => ({ ...prev, rateLimitStrict: parseInt(e.target.value) }))}
            min="5"
            max="100"
            className="w-full px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white dark:bg-neutral-700 text-neutral-900 dark:text-white"
          />
        </div>
      </div>

      <div>
        <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
          CORS Allowed Origins (comma-separated)
        </label>
        <textarea
          value={settings.corsOrigins}
          onChange={(e) => setSettings(prev => ({ ...prev, corsOrigins: e.target.value }))}
          placeholder="https://yourdomain.com, https://app.yourdomain.com"
          rows={3}
          className="w-full px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white dark:bg-neutral-700 text-neutral-900 dark:text-white"
        />
      </div>
    </div>
  );

  const renderDatabaseSettings = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
            Backup Retention (days)
          </label>
          <input
            type="number"
            value={settings.backupRetentionDays}
            onChange={(e) => setSettings(prev => ({ ...prev, backupRetentionDays: parseInt(e.target.value) }))}
            min="7"
            max="365"
            className="w-full px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white dark:bg-neutral-700 text-neutral-900 dark:text-white"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
            Connection Pool Size
          </label>
          <input
            type="number"
            value={settings.connectionPoolSize}
            onChange={(e) => setSettings(prev => ({ ...prev, connectionPoolSize: parseInt(e.target.value) }))}
            min="5"
            max="50"
            className="w-full px-3 py-2 border border-neutral-300 dark:border-neutral-600 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white dark:bg-neutral-700 text-neutral-900 dark:text-white"
          />
        </div>
      </div>

      <div className="flex items-center justify-between p-4 bg-neutral-50 dark:bg-neutral-800 rounded-lg">
        <div>
          <h4 className="font-medium text-neutral-900 dark:text-white">Automatic Backups</h4>
          <p className="text-sm text-neutral-600 dark:text-neutral-400">
            Enable daily database backups
          </p>
        </div>
        <label className="relative inline-flex items-center cursor-pointer">
          <input
            type="checkbox"
            checked={settings.backupEnabled}
            onChange={(e) => setSettings(prev => ({ ...prev, backupEnabled: e.target.checked }))}
            className="sr-only peer"
          />
          <div className="w-11 h-6 bg-neutral-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary-300 dark:peer-focus:ring-primary-800 rounded-full peer dark:bg-neutral-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-neutral-600 peer-checked:bg-primary-600"></div>
        </label>
      </div>
    </div>
  );

  const renderTabContent = () => {
    switch (activeTab) {
      case 'system':
        return renderSystemSettings();
      case 'security':
        return renderSecuritySettings();
      case 'ml':
        return renderMLSettings();
      case 'notifications':
        return renderNotificationSettings();
      case 'api':
        return renderAPISettings();
      case 'database':
        return renderDatabaseSettings();
      default:
        return renderSystemSettings();
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-neutral-900 dark:text-white">
            System Settings
          </h1>
          <p className="text-neutral-600 dark:text-neutral-400 mt-1">
            Configure system behavior, security, and integrations
          </p>
        </div>
        <div className="flex space-x-3">
          <Button
            variant="outline"
            onClick={exportSettings}
          >
            <Download className="w-4 h-4 mr-2" />
            Export
          </Button>
          <Button
            variant="outline"
            onClick={resetToDefaults}
          >
            <RefreshCw className="w-4 h-4 mr-2" />
            Reset
          </Button>
          <Button
            onClick={saveSettings}
            loading={isSaving}
          >
            <Save className="w-4 h-4 mr-2" />
            Save Changes
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Sidebar */}
        <Card className="p-4 h-fit">
          <nav className="space-y-2">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`w-full flex items-center space-x-3 px-3 py-2 rounded-md text-left transition-colors ${
                    activeTab === tab.id
                      ? 'bg-primary-100 text-primary-700 dark:bg-primary-900 dark:text-primary-300'
                      : 'text-neutral-600 dark:text-neutral-400 hover:bg-neutral-100 dark:hover:bg-neutral-700'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span className="text-sm font-medium">{tab.label}</span>
                </button>
              );
            })}
          </nav>
        </Card>

        {/* Content */}
        <div className="lg:col-span-3">
          <Card className="p-6">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.2 }}
            >
              {renderTabContent()}
            </motion.div>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default Settings;