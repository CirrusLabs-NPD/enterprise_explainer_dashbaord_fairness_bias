import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  Settings,
  Moon,
  Sun,
  Monitor,
  Download,
  User,
  Bell,
  Search,
  Menu,
  X,
  ChevronDown,
  LogOut,
  HelpCircle,
  Zap,
  Database,
} from 'lucide-react';
import { useTheme } from '../../contexts/ThemeContext';
import { useAuth } from '../../contexts/AuthContext';
import RateLimitStatus from '../common/RateLimitStatus';
import NotificationCenter from '../common/NotificationCenter';
import { cn } from '../../utils';

interface HeaderProps {
  onToggleSidebar: () => void;
  isSidebarOpen: boolean;
  modelType: 'classification' | 'regression';
  onModelTypeChange: (type: 'classification' | 'regression') => void;
  onShowHelp: () => void;
  onShowTour?: () => void;
  onTabChange?: (tabId: string) => void;
}

const Header: React.FC<HeaderProps> = ({
  onToggleSidebar,
  isSidebarOpen,
  modelType,
  onModelTypeChange,
  onShowHelp,
  onShowTour,
  onTabChange,
}) => {
  const { theme, setMode, isDark } = useTheme();
  const { user, logout } = useAuth();
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [showThemeMenu, setShowThemeMenu] = useState(false);
  const [showNotifications, setShowNotifications] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  const themeOptions = [
    { value: 'light', label: 'Light', icon: Sun },
    { value: 'dark', label: 'Dark', icon: Moon },
    { value: 'system', label: 'System', icon: Monitor },
  ];

  // Notification count for the indicator
  const notificationCount = 3; // This would come from your notification context/API

  const handleExport = () => {
    // TODO: Implement export functionality
    console.log('Export functionality to be implemented');
  };

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    // TODO: Implement search functionality
    console.log('Search query:', searchQuery);
  };

  return (
    <header className="bg-white/80 dark:bg-neutral-900/80 backdrop-blur-md border-b border-neutral-200 dark:border-neutral-800 sticky top-0 z-50">
      <div className="max-w-full mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Left Section */}
          <div className="flex items-center space-x-4">
            {/* Sidebar Toggle */}
            <button
              onClick={onToggleSidebar}
              className="p-2 rounded-lg hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors lg:hidden"
              aria-label="Toggle sidebar"
            >
              {isSidebarOpen ? (
                <X className="w-5 h-5 text-neutral-700 dark:text-neutral-300" />
              ) : (
                <Menu className="w-5 h-5 text-neutral-700 dark:text-neutral-300" />
              )}
            </button>

            {/* Logo */}
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="flex items-center space-x-2"
            >
              <div className="w-8 h-8 bg-gradient-to-br from-primary-500 to-accent-500 rounded-lg flex items-center justify-center">
                <Zap className="w-4 h-4 text-white" />
              </div>
              <div className="hidden sm:block">
                <h1 className="text-lg font-bold text-gradient">
                  ML Explainer
                </h1>
                <p className="text-xs text-neutral-600 dark:text-neutral-400">
                  Enterprise Dashboard
                </p>
              </div>
            </motion.div>

            {/* Model Type Selector */}
            <div className="hidden md:flex items-center space-x-2">
              <span className="text-sm text-neutral-600 dark:text-neutral-400">
                Model Type:
              </span>
              <div className="flex bg-neutral-100 dark:bg-neutral-800 rounded-lg p-1">
                {['classification', 'regression'].map((type) => (
                  <button
                    key={type}
                    onClick={() => onModelTypeChange(type as 'classification' | 'regression')}
                    className={cn(
                      'px-3 py-1 text-sm font-medium rounded-md transition-all duration-200',
                      modelType === type
                        ? 'bg-white dark:bg-neutral-700 text-primary-600 dark:text-primary-400 shadow-sm'
                        : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-200'
                    )}
                  >
                    {type.charAt(0).toUpperCase() + type.slice(1)}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Center Section - Search */}
          <div className="hidden md:flex flex-1 max-w-md mx-4">
            <form onSubmit={handleSearch} className="relative w-full">
              <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none">
                <Search className="w-4 h-4 text-neutral-400" />
              </div>
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search features, models, or data..."
                className="w-full pl-10 pr-4 py-2 border border-neutral-200 dark:border-neutral-700 rounded-lg 
                         bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100 
                         placeholder-neutral-500 dark:placeholder-neutral-400
                         focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent
                         transition-all duration-200"
              />
            </form>
          </div>

          {/* Right Section */}
          <div className="flex items-center space-x-2">
            {/* Data Connectivity Quick Access */}
            {onTabChange && (
              <button
                onClick={() => onTabChange('data-connectivity')}
                className="flex items-center space-x-2 px-3 py-2 rounded-lg bg-blue-50 hover:bg-blue-100 dark:bg-blue-900/20 dark:hover:bg-blue-900/30 text-blue-700 dark:text-blue-300 transition-all duration-200 border border-blue-200 dark:border-blue-700 hover:border-blue-300 dark:hover:border-blue-600"
                aria-label="Data Connectivity"
              >
                <Database className="w-4 h-4" />
                <span className="text-sm font-medium hidden sm:inline">Data</span>
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
              </button>
            )}

            {/* Export Button */}
            <button
              onClick={handleExport}
              className="p-2 rounded-lg hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors"
              aria-label="Export data"
            >
              <Download className="w-5 h-5 text-neutral-700 dark:text-neutral-300" />
            </button>

            {/* Rate Limit Status */}
            <div className="hidden sm:block">
              <RateLimitStatus className="mr-2" />
            </div>

            {/* Notifications */}
            <div className="relative">
              <button
                onClick={() => setShowNotifications(!showNotifications)}
                className="p-2 rounded-lg hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors relative"
                aria-label="Notifications"
              >
                <Bell className="w-5 h-5 text-neutral-700 dark:text-neutral-300" />
                {notificationCount > 0 && (
                  <span className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full"></span>
                )}
              </button>

              {/* NotificationCenter */}
              <NotificationCenter
                isOpen={showNotifications}
                onClose={() => setShowNotifications(false)}
                onNotificationClick={(notification) => {
                  // Handle navigation based on notification action
                  if (notification.actionUrl && onTabChange) {
                    if (notification.actionUrl === '/data-drift') {
                      onTabChange('data-drift');
                    } else if (notification.actionUrl === '/user-management') {
                      onTabChange('user-management');
                    } else if (notification.actionUrl === '/system-health') {
                      onTabChange('system-health');
                    }
                  }
                  setShowNotifications(false);
                }}
              />
            </div>

            {/* Tour Button */}
            {onShowTour && (
              <button
                onClick={onShowTour}
                className="p-2 rounded-lg hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors"
                aria-label="Show tour"
              >
                <Zap className="w-5 h-5 text-neutral-700 dark:text-neutral-300" />
              </button>
            )}
            
            {/* Help Button */}
            <button
              onClick={onShowHelp}
              className="p-2 rounded-lg hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors"
              aria-label="Show help guide"
            >
              <HelpCircle className="w-5 h-5 text-neutral-700 dark:text-neutral-300" />
            </button>

            {/* Theme Toggle */}
            <div className="relative">
              <button
                onClick={() => setShowThemeMenu(!showThemeMenu)}
                className="p-2 rounded-lg hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors"
                aria-label="Toggle theme"
              >
                {isDark ? (
                  <Moon className="w-5 h-5 text-neutral-700 dark:text-neutral-300" />
                ) : (
                  <Sun className="w-5 h-5 text-neutral-700 dark:text-neutral-300" />
                )}
              </button>

              {/* Theme Menu */}
              {showThemeMenu && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="absolute right-0 mt-2 w-48 bg-white dark:bg-neutral-800 rounded-lg shadow-lg border border-neutral-200 dark:border-neutral-700 z-50"
                >
                  <div className="p-2">
                    {themeOptions.map((option) => {
                      const Icon = option.icon;
                      return (
                        <button
                          key={option.value}
                          onClick={() => {
                            setMode(option.value as 'light' | 'dark' | 'system');
                            setShowThemeMenu(false);
                          }}
                          className={cn(
                            'w-full flex items-center space-x-2 px-3 py-2 text-sm rounded-lg transition-colors',
                            theme.mode === option.value
                              ? 'bg-primary-100 dark:bg-primary-900 text-primary-700 dark:text-primary-300'
                              : 'text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-700'
                          )}
                        >
                          <Icon className="w-4 h-4" />
                          <span>{option.label}</span>
                        </button>
                      );
                    })}
                  </div>
                </motion.div>
              )}
            </div>

            {/* User Menu */}
            <div className="relative">
              <button
                onClick={() => setShowUserMenu(!showUserMenu)}
                className="flex items-center space-x-2 p-2 rounded-lg hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors"
                aria-label="User menu"
              >
                <div className="w-8 h-8 bg-gradient-to-br from-primary-500 to-accent-500 rounded-full flex items-center justify-center">
                  <User className="w-4 h-4 text-white" />
                </div>
                <ChevronDown className="w-4 h-4 text-neutral-700 dark:text-neutral-300" />
              </button>

              {/* User Menu Dropdown */}
              {showUserMenu && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="absolute right-0 mt-2 w-48 bg-white dark:bg-neutral-800 rounded-lg shadow-lg border border-neutral-200 dark:border-neutral-700 z-50"
                >
                  <div className="p-4 border-b border-neutral-200 dark:border-neutral-700">
                    <p className="font-medium text-neutral-900 dark:text-neutral-100">
                      John Doe
                    </p>
                    <p className="text-sm text-neutral-600 dark:text-neutral-400">
                      john.doe@company.com
                    </p>
                  </div>
                  <div className="p-2">
                    <button className="w-full flex items-center space-x-2 px-3 py-2 text-sm text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-700 rounded-lg transition-colors">
                      <Settings className="w-4 h-4" />
                      <span>Settings</span>
                    </button>
                    <button className="w-full flex items-center space-x-2 px-3 py-2 text-sm text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-700 rounded-lg transition-colors">
                      <HelpCircle className="w-4 h-4" />
                      <span>Help</span>
                    </button>
                    <hr className="my-2 border-neutral-200 dark:border-neutral-700" />
                    <button className="w-full flex items-center space-x-2 px-3 py-2 text-sm text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg transition-colors">
                      <LogOut className="w-4 h-4" />
                      <span>Sign out</span>
                    </button>
                  </div>
                </motion.div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Mobile Search */}
      <div className="md:hidden px-4 pb-4">
        <form onSubmit={handleSearch} className="relative">
          <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none">
            <Search className="w-4 h-4 text-neutral-400" />
          </div>
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search..."
            className="w-full pl-10 pr-4 py-2 border border-neutral-200 dark:border-neutral-700 rounded-lg 
                     bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100 
                     placeholder-neutral-500 dark:placeholder-neutral-400
                     focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent
                     transition-all duration-200"
          />
        </form>
      </div>
    </header>
  );
};

export default Header;