import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { toast } from 'react-hot-toast';

interface User {
  user_id: string;
  username: string;
  email: string;
  role: string;
  is_active: boolean;
  permissions: string[];
}

interface AuthTokens {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

interface AuthContextType {
  user: User | null;
  tokens: AuthTokens | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
  refreshToken: () => Promise<void>;
  updatePassword: (currentPassword: string, newPassword: string) => Promise<void>;
  hasPermission: (permission: string) => boolean;
  hasRole: (role: string) => boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

const API_BASE_URL = 'http://localhost:8000/api/v1';

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [tokens, setTokens] = useState<AuthTokens | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const isAuthenticated = !!user && !!tokens;

  // Load saved auth data on mount
  useEffect(() => {
    const savedTokens = localStorage.getItem('auth_tokens');
    const savedUser = localStorage.getItem('auth_user');

    if (savedTokens && savedUser) {
      try {
        const parsedTokens = JSON.parse(savedTokens);
        const parsedUser = JSON.parse(savedUser);
        
        setTokens(parsedTokens);
        setUser(parsedUser);
        
        // Verify token is still valid
        verifyToken(parsedTokens.access_token);
      } catch (error) {
        console.error('Error loading saved auth data:', error);
        clearAuthData();
      }
    }
    
    setIsLoading(false);
  }, []);

  const saveAuthData = (authTokens: AuthTokens, userData: User) => {
    setTokens(authTokens);
    setUser(userData);
    localStorage.setItem('auth_tokens', JSON.stringify(authTokens));
    localStorage.setItem('auth_user', JSON.stringify(userData));
  };

  const clearAuthData = () => {
    setTokens(null);
    setUser(null);
    localStorage.removeItem('auth_tokens');
    localStorage.removeItem('auth_user');
  };

  const makeAuthenticatedRequest = async (url: string, options: RequestInit = {}) => {
    if (!tokens) {
      throw new Error('No authentication tokens available');
    }

    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${tokens.access_token}`,
        ...options.headers,
      },
    });

    if (response.status === 401) {
      // Token might be expired, try to refresh
      try {
        await refreshToken();
        // Retry the request with new token
        return await fetch(url, {
          ...options,
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${tokens?.access_token}`,
            ...options.headers,
          },
        });
      } catch (refreshError) {
        // Refresh failed, logout user
        logout();
        throw new Error('Session expired. Please log in again.');
      }
    }

    return response;
  };

  const verifyToken = async (token: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/auth/check-token`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (!response.ok) {
        throw new Error('Token verification failed');
      }

      const data = await response.json();
      if (!data.valid) {
        throw new Error('Token is invalid');
      }
    } catch (error) {
      console.error('Token verification failed:', error);
      clearAuthData();
    }
  };

  const login = async (username: string, password: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/auth/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username, password }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error?.message || 'Login failed');
      }

      const data = await response.json();
      saveAuthData(data, data.user);
    } catch (error) {
      console.error('Login error:', error);
      throw error;
    }
  };

  const logout = () => {
    clearAuthData();
    toast.success('Logged out successfully');
  };

  const refreshToken = async () => {
    if (!tokens?.refresh_token) {
      throw new Error('No refresh token available');
    }

    try {
      const response = await fetch(`${API_BASE_URL}/auth/refresh`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ refresh_token: tokens.refresh_token }),
      });

      if (!response.ok) {
        throw new Error('Token refresh failed');
      }

      const data = await response.json();
      saveAuthData(data, data.user);
    } catch (error) {
      console.error('Token refresh error:', error);
      clearAuthData();
      throw error;
    }
  };

  const updatePassword = async (currentPassword: string, newPassword: string) => {
    try {
      const response = await makeAuthenticatedRequest(`${API_BASE_URL}/auth/change-password`, {
        method: 'POST',
        body: JSON.stringify({
          current_password: currentPassword,
          new_password: newPassword,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error?.message || 'Password update failed');
      }
    } catch (error) {
      console.error('Password update error:', error);
      throw error;
    }
  };

  const hasPermission = (permission: string): boolean => {
    return user?.permissions?.includes(permission) || false;
  };

  const hasRole = (role: string): boolean => {
    return user?.role === role;
  };

  // Auto-refresh token before it expires
  useEffect(() => {
    if (!tokens || !isAuthenticated) return;

    const refreshInterval = setInterval(() => {
      const expiresAt = Date.now() + (tokens.expires_in * 1000);
      const shouldRefresh = expiresAt - Date.now() < 5 * 60 * 1000; // Refresh 5 minutes before expiry

      if (shouldRefresh) {
        refreshToken().catch(() => {
          // Ignore errors, user will be logged out on next request
        });
      }
    }, 60000); // Check every minute

    return () => clearInterval(refreshInterval);
  }, [tokens, isAuthenticated]);

  const value: AuthContextType = {
    user,
    tokens,
    isLoading,
    isAuthenticated,
    login,
    logout,
    refreshToken,
    updatePassword,
    hasPermission,
    hasRole,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

export default AuthContext;