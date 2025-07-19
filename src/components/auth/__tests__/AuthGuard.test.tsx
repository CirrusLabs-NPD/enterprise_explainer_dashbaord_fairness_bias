import { render, screen } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import AuthGuard from '../AuthGuard'
import React from 'react'

// Mock the AuthContext
const mockAuthContext = {
  user: null,
  isLoading: false,
  isAuthenticated: false,
  login: vi.fn(),
  logout: vi.fn(),
  refreshToken: vi.fn(),
  hasPermission: vi.fn(),
  hasRole: vi.fn(),
}

vi.mock('../../../contexts/AuthContext', () => ({
  useAuth: () => mockAuthContext,
}))

const TestComponent = () => <div>Protected Content</div>

describe('AuthGuard', () => {
  it('shows loading state when authentication is loading', () => {
    mockAuthContext.isLoading = true
    mockAuthContext.isAuthenticated = false
    
    render(
      <AuthGuard>
        <TestComponent />
      </AuthGuard>
    )
    
    expect(screen.getByText('Loading...')).toBeInTheDocument()
    expect(screen.queryByText('Protected Content')).not.toBeInTheDocument()
  })

  it('shows login form when user is not authenticated', () => {
    mockAuthContext.isLoading = false
    mockAuthContext.isAuthenticated = false
    mockAuthContext.user = null
    
    render(
      <AuthGuard>
        <TestComponent />
      </AuthGuard>
    )
    
    expect(screen.getByText('Access Denied')).toBeInTheDocument()
    expect(screen.getByText('You need to sign in to access this page.')).toBeInTheDocument()
    expect(screen.queryByText('Protected Content')).not.toBeInTheDocument()
  })

  it('renders children when user is authenticated', () => {
    mockAuthContext.isLoading = false
    mockAuthContext.isAuthenticated = true
    mockAuthContext.user = {
      id: '1',
      username: 'testuser',
      email: 'test@example.com',
      role: 'user',
      permissions: ['read:data'],
    }
    
    render(
      <AuthGuard>
        <TestComponent />
      </AuthGuard>
    )
    
    expect(screen.getByText('Protected Content')).toBeInTheDocument()
    expect(screen.queryByText('Access Denied')).not.toBeInTheDocument()
  })

  it('checks required permission when specified', () => {
    mockAuthContext.isLoading = false
    mockAuthContext.isAuthenticated = true
    mockAuthContext.user = {
      id: '1',
      username: 'testuser',
      email: 'test@example.com',
      role: 'user',
      permissions: ['read:data'],
    }
    mockAuthContext.hasPermission = vi.fn().mockReturnValue(false)
    
    render(
      <AuthGuard requiredPermission="admin:users">
        <TestComponent />
      </AuthGuard>
    )
    
    expect(mockAuthContext.hasPermission).toHaveBeenCalledWith('admin:users')
    expect(screen.getByText('Insufficient Permissions')).toBeInTheDocument()
    expect(screen.getByText('You do not have permission to access this page.')).toBeInTheDocument()
    expect(screen.queryByText('Protected Content')).not.toBeInTheDocument()
  })

  it('renders children when user has required permission', () => {
    mockAuthContext.isLoading = false
    mockAuthContext.isAuthenticated = true
    mockAuthContext.user = {
      id: '1',
      username: 'testuser',
      email: 'test@example.com',
      role: 'admin',
      permissions: ['admin:users'],
    }
    mockAuthContext.hasPermission = vi.fn().mockReturnValue(true)
    
    render(
      <AuthGuard requiredPermission="admin:users">
        <TestComponent />
      </AuthGuard>
    )
    
    expect(mockAuthContext.hasPermission).toHaveBeenCalledWith('admin:users')
    expect(screen.getByText('Protected Content')).toBeInTheDocument()
    expect(screen.queryByText('Insufficient Permissions')).not.toBeInTheDocument()
  })

  it('checks required role when specified', () => {
    mockAuthContext.isLoading = false
    mockAuthContext.isAuthenticated = true
    mockAuthContext.user = {
      id: '1',
      username: 'testuser',
      email: 'test@example.com',
      role: 'user',
      permissions: ['read:data'],
    }
    mockAuthContext.hasRole = vi.fn().mockReturnValue(false)
    
    render(
      <AuthGuard requiredRole="admin">
        <TestComponent />
      </AuthGuard>
    )
    
    expect(mockAuthContext.hasRole).toHaveBeenCalledWith('admin')
    expect(screen.getByText('Insufficient Permissions')).toBeInTheDocument()
    expect(screen.queryByText('Protected Content')).not.toBeInTheDocument()
  })

  it('renders children when user has required role', () => {
    mockAuthContext.isLoading = false
    mockAuthContext.isAuthenticated = true
    mockAuthContext.user = {
      id: '1',
      username: 'testuser',
      email: 'test@example.com',
      role: 'admin',
      permissions: ['admin:users'],
    }
    mockAuthContext.hasRole = vi.fn().mockReturnValue(true)
    
    render(
      <AuthGuard requiredRole="admin">
        <TestComponent />
      </AuthGuard>
    )
    
    expect(mockAuthContext.hasRole).toHaveBeenCalledWith('admin')
    expect(screen.getByText('Protected Content')).toBeInTheDocument()
    expect(screen.queryByText('Insufficient Permissions')).not.toBeInTheDocument()
  })

  it('checks both role and permission when both are specified', () => {
    mockAuthContext.isLoading = false
    mockAuthContext.isAuthenticated = true
    mockAuthContext.user = {
      id: '1',
      username: 'testuser',
      email: 'test@example.com',
      role: 'admin',
      permissions: ['admin:users'],
    }
    mockAuthContext.hasRole = vi.fn().mockReturnValue(true)
    mockAuthContext.hasPermission = vi.fn().mockReturnValue(true)
    
    render(
      <AuthGuard requiredRole="admin" requiredPermission="admin:users">
        <TestComponent />
      </AuthGuard>
    )
    
    expect(mockAuthContext.hasRole).toHaveBeenCalledWith('admin')
    expect(mockAuthContext.hasPermission).toHaveBeenCalledWith('admin:users')
    expect(screen.getByText('Protected Content')).toBeInTheDocument()
  })

  it('denies access if role matches but permission does not', () => {
    mockAuthContext.isLoading = false
    mockAuthContext.isAuthenticated = true
    mockAuthContext.user = {
      id: '1',
      username: 'testuser',
      email: 'test@example.com',
      role: 'admin',
      permissions: ['read:data'],
    }
    mockAuthContext.hasRole = vi.fn().mockReturnValue(true)
    mockAuthContext.hasPermission = vi.fn().mockReturnValue(false)
    
    render(
      <AuthGuard requiredRole="admin" requiredPermission="admin:users">
        <TestComponent />
      </AuthGuard>
    )
    
    expect(screen.getByText('Insufficient Permissions')).toBeInTheDocument()
    expect(screen.queryByText('Protected Content')).not.toBeInTheDocument()
  })

  it('shows custom fallback when provided', () => {
    mockAuthContext.isLoading = false
    mockAuthContext.isAuthenticated = false
    mockAuthContext.user = null
    
    const CustomFallback = () => <div>Custom Access Denied</div>
    
    render(
      <AuthGuard fallback={<CustomFallback />}>
        <TestComponent />
      </AuthGuard>
    )
    
    expect(screen.getByText('Custom Access Denied')).toBeInTheDocument()
    expect(screen.queryByText('Access Denied')).not.toBeInTheDocument()
    expect(screen.queryByText('Protected Content')).not.toBeInTheDocument()
  })
})