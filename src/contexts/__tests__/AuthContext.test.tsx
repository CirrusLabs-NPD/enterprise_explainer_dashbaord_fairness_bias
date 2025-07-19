import { renderHook, act, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { AuthProvider, useAuth } from '../AuthContext'
import React from 'react'

// Mock localStorage
const mockLocalStorage = {
  getItem: vi.fn(),
  setItem: vi.fn(),
  removeItem: vi.fn(),
  clear: vi.fn(),
}

Object.defineProperty(window, 'localStorage', {
  value: mockLocalStorage,
})

// Mock fetch
global.fetch = vi.fn()

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <AuthProvider>{children}</AuthProvider>
)

describe('AuthContext', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockLocalStorage.getItem.mockReturnValue(null)
  })

  afterEach(() => {
    vi.resetAllMocks()
  })

  describe('Initial State', () => {
    it('should initialize with no user and not loading', () => {
      const { result } = renderHook(() => useAuth(), { wrapper })
      
      expect(result.current.user).toBeNull()
      expect(result.current.isLoading).toBe(false)
      expect(result.current.isAuthenticated).toBe(false)
    })

    it('should restore user from localStorage on mount', async () => {
      const mockUser = {
        id: '1',
        username: 'testuser',
        email: 'test@example.com',
        role: 'user' as const,
        permissions: ['read:data'],
      }

      mockLocalStorage.getItem.mockImplementation((key) => {
        if (key === 'auth_user') return JSON.stringify(mockUser)
        if (key === 'auth_token') return 'mock-token'
        return null
      })

      const { result } = renderHook(() => useAuth(), { wrapper })
      
      await waitFor(() => {
        expect(result.current.user).toEqual(mockUser)
        expect(result.current.isAuthenticated).toBe(true)
      })
    })
  })

  describe('Login', () => {
    it('should login successfully with valid credentials', async () => {
      const mockUser = {
        id: '1',
        username: 'testuser',
        email: 'test@example.com',
        role: 'user' as const,
        permissions: ['read:data'],
      }

      const mockResponse = {
        user: mockUser,
        token: 'mock-token',
        refreshToken: 'mock-refresh-token',
      }

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      })

      const { result } = renderHook(() => useAuth(), { wrapper })

      await act(async () => {
        await result.current.login('testuser', 'password')
      })

      expect(result.current.user).toEqual(mockUser)
      expect(result.current.isAuthenticated).toBe(true)
      expect(mockLocalStorage.setItem).toHaveBeenCalledWith('auth_user', JSON.stringify(mockUser))
      expect(mockLocalStorage.setItem).toHaveBeenCalledWith('auth_token', 'mock-token')
      expect(mockLocalStorage.setItem).toHaveBeenCalledWith('auth_refresh_token', 'mock-refresh-token')
    })

    it('should throw error with invalid credentials', async () => {
      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: false,
        status: 401,
        json: () => Promise.resolve({ message: 'Invalid credentials' }),
      })

      const { result } = renderHook(() => useAuth(), { wrapper })

      await expect(async () => {
        await act(async () => {
          await result.current.login('testuser', 'wrongpassword')
        })
      }).rejects.toThrow('Invalid credentials')

      expect(result.current.user).toBeNull()
      expect(result.current.isAuthenticated).toBe(false)
    })

    it('should handle network errors', async () => {
      global.fetch = vi.fn().mockRejectedValueOnce(new Error('Network error'))

      const { result } = renderHook(() => useAuth(), { wrapper })

      await expect(async () => {
        await act(async () => {
          await result.current.login('testuser', 'password')
        })
      }).rejects.toThrow('Network error')
    })
  })

  describe('Logout', () => {
    it('should logout and clear stored data', async () => {
      const mockUser = {
        id: '1',
        username: 'testuser',
        email: 'test@example.com',
        role: 'user' as const,
        permissions: ['read:data'],
      }

      // Setup initial authenticated state
      mockLocalStorage.getItem.mockImplementation((key) => {
        if (key === 'auth_user') return JSON.stringify(mockUser)
        if (key === 'auth_token') return 'mock-token'
        return null
      })

      const { result } = renderHook(() => useAuth(), { wrapper })

      // Wait for initial authentication
      await waitFor(() => {
        expect(result.current.isAuthenticated).toBe(true)
      })

      // Mock logout API call
      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({}),
      })

      await act(async () => {
        await result.current.logout()
      })

      expect(result.current.user).toBeNull()
      expect(result.current.isAuthenticated).toBe(false)
      expect(mockLocalStorage.removeItem).toHaveBeenCalledWith('auth_user')
      expect(mockLocalStorage.removeItem).toHaveBeenCalledWith('auth_token')
      expect(mockLocalStorage.removeItem).toHaveBeenCalledWith('auth_refresh_token')
    })
  })

  describe('Token Refresh', () => {
    it('should refresh token when expired', async () => {
      const mockUser = {
        id: '1',
        username: 'testuser',
        email: 'test@example.com',
        role: 'user' as const,
        permissions: ['read:data'],
      }

      mockLocalStorage.getItem.mockImplementation((key) => {
        if (key === 'auth_user') return JSON.stringify(mockUser)
        if (key === 'auth_token') return 'expired-token'
        if (key === 'auth_refresh_token') return 'refresh-token'
        return null
      })

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          token: 'new-token',
          refreshToken: 'new-refresh-token',
        }),
      })

      const { result } = renderHook(() => useAuth(), { wrapper })

      await act(async () => {
        await result.current.refreshToken()
      })

      expect(mockLocalStorage.setItem).toHaveBeenCalledWith('auth_token', 'new-token')
      expect(mockLocalStorage.setItem).toHaveBeenCalledWith('auth_refresh_token', 'new-refresh-token')
    })

    it('should logout if refresh fails', async () => {
      const mockUser = {
        id: '1',
        username: 'testuser',
        email: 'test@example.com',
        role: 'user' as const,
        permissions: ['read:data'],
      }

      mockLocalStorage.getItem.mockImplementation((key) => {
        if (key === 'auth_user') return JSON.stringify(mockUser)
        if (key === 'auth_token') return 'expired-token'
        if (key === 'auth_refresh_token') return 'invalid-refresh-token'
        return null
      })

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: false,
        status: 401,
        json: () => Promise.resolve({ message: 'Invalid refresh token' }),
      })

      const { result } = renderHook(() => useAuth(), { wrapper })

      await act(async () => {
        try {
          await result.current.refreshToken()
        } catch (error) {
          // Expected to fail
        }
      })

      expect(result.current.user).toBeNull()
      expect(result.current.isAuthenticated).toBe(false)
    })
  })

  describe('Permissions', () => {
    it('should check permissions correctly', async () => {
      const mockUser = {
        id: '1',
        username: 'testuser',
        email: 'test@example.com',
        role: 'admin' as const,
        permissions: ['read:data', 'write:data', 'admin:users'],
      }

      mockLocalStorage.getItem.mockImplementation((key) => {
        if (key === 'auth_user') return JSON.stringify(mockUser)
        if (key === 'auth_token') return 'mock-token'
        return null
      })

      const { result } = renderHook(() => useAuth(), { wrapper })

      await waitFor(() => {
        expect(result.current.isAuthenticated).toBe(true)
      })

      expect(result.current.hasPermission('read:data')).toBe(true)
      expect(result.current.hasPermission('write:data')).toBe(true)
      expect(result.current.hasPermission('admin:users')).toBe(true)
      expect(result.current.hasPermission('delete:data')).toBe(false)
    })

    it('should handle role-based permissions', async () => {
      const mockUser = {
        id: '1',
        username: 'testuser',
        email: 'test@example.com',
        role: 'admin' as const,
        permissions: ['read:data'],
      }

      mockLocalStorage.getItem.mockImplementation((key) => {
        if (key === 'auth_user') return JSON.stringify(mockUser)
        if (key === 'auth_token') return 'mock-token'
        return null
      })

      const { result } = renderHook(() => useAuth(), { wrapper })

      await waitFor(() => {
        expect(result.current.isAuthenticated).toBe(true)
      })

      expect(result.current.hasRole('admin')).toBe(true)
      expect(result.current.hasRole('user')).toBe(false)
    })
  })

  describe('Error Handling', () => {
    it('should handle invalid JSON in localStorage', () => {
      mockLocalStorage.getItem.mockImplementation((key) => {
        if (key === 'auth_user') return 'invalid-json'
        return null
      })

      const { result } = renderHook(() => useAuth(), { wrapper })
      
      expect(result.current.user).toBeNull()
      expect(result.current.isAuthenticated).toBe(false)
    })

    it('should handle missing required fields in stored user', () => {
      mockLocalStorage.getItem.mockImplementation((key) => {
        if (key === 'auth_user') return JSON.stringify({ id: '1' }) // Missing required fields
        return null
      })

      const { result } = renderHook(() => useAuth(), { wrapper })
      
      expect(result.current.user).toBeNull()
      expect(result.current.isAuthenticated).toBe(false)
    })
  })
})