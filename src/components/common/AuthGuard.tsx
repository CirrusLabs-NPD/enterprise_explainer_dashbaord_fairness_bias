import React from 'react';
import { useAuth } from '../../contexts/AuthContext';
import LoginForm from '../auth/LoginForm';
import LoadingSpinner from './LoadingSpinner';

interface AuthGuardProps {
  children: React.ReactNode;
  requireAuth?: boolean;
  requiredPermission?: string;
  requiredRole?: string;
  fallback?: React.ReactNode;
}

const AuthGuard: React.FC<AuthGuardProps> = ({
  children,
  requireAuth = true,
  requiredPermission,
  requiredRole,
  fallback
}) => {
  const { isAuthenticated, isLoading, hasPermission, hasRole } = useAuth();

  // Show loading spinner while checking auth status
  if (isLoading) {
    return <LoadingSpinner fullScreen size="xl" message="Checking authentication..." />;
  }

  // If authentication is required but user is not authenticated
  if (requireAuth && !isAuthenticated) {
    return (
      <div className="min-h-screen bg-neutral-50 dark:bg-neutral-900 flex items-center justify-center p-4">
        <LoginForm onSuccess={() => window.location.reload()} />
      </div>
    );
  }

  // Check for specific permission
  if (requiredPermission && !hasPermission(requiredPermission)) {
    return (
      fallback || (
        <div className="min-h-screen bg-neutral-50 dark:bg-neutral-900 flex items-center justify-center">
          <div className="text-center">
            <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <svg className="w-8 h-8 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.732-.833-2.5 0L4.268 18.5c-.77.833.192 2.5 1.732 2.5z" />
              </svg>
            </div>
            <h2 className="text-xl font-semibold text-neutral-900 dark:text-white mb-2">
              Access Denied
            </h2>
            <p className="text-neutral-600 dark:text-neutral-400">
              You don't have permission to access this page.
            </p>
            <p className="text-sm text-neutral-500 dark:text-neutral-500 mt-2">
              Required permission: {requiredPermission}
            </p>
          </div>
        </div>
      )
    );
  }

  // Check for specific role
  if (requiredRole && !hasRole(requiredRole)) {
    return (
      fallback || (
        <div className="min-h-screen bg-neutral-50 dark:bg-neutral-900 flex items-center justify-center">
          <div className="text-center">
            <div className="w-16 h-16 bg-yellow-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <svg className="w-8 h-8 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
              </svg>
            </div>
            <h2 className="text-xl font-semibold text-neutral-900 dark:text-white mb-2">
              Insufficient Role
            </h2>
            <p className="text-neutral-600 dark:text-neutral-400">
              Your current role doesn't allow access to this page.
            </p>
            <p className="text-sm text-neutral-500 dark:text-neutral-500 mt-2">
              Required role: {requiredRole}
            </p>
          </div>
        </div>
      )
    );
  }

  // All checks passed, render children
  return <>{children}</>;
};

export default AuthGuard;