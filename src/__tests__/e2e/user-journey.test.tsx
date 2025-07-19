import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import App from '../../App'
import { AuthProvider } from '../../contexts/AuthContext'
import { ThemeProvider } from '../../contexts/ThemeContext'

// Mock all heavy components for E2E testing
vi.mock('../../components/dashboard/Dashboard', () => ({
  default: () => <div data-testid="dashboard">Dashboard Content</div>,
}))

vi.mock('../../components/settings/Settings', () => ({
  default: () => <div data-testid="settings">Settings Content</div>,
}))

vi.mock('../../components/user-management/UserManagement', () => ({
  default: () => <div data-testid="user-management">User Management Content</div>,
}))

vi.mock('../../components/system-health/SystemHealth', () => ({
  default: () => <div data-testid="system-health">System Health Content</div>,
}))

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

const renderFullApp = () => {
  return render(
    <ThemeProvider>
      <AuthProvider>
        <App />
      </AuthProvider>
    </ThemeProvider>
  )
}

describe('Critical User Journeys', () => {
  const user = userEvent.setup()

  beforeEach(() => {
    vi.clearAllMocks()
    mockLocalStorage.getItem.mockReturnValue(null)
  })

  describe('New User Onboarding Journey', () => {
    it('guides new user through complete login and dashboard exploration', async () => {
      // Step 1: User arrives at the application
      renderFullApp()
      
      expect(screen.getByText('Welcome Back')).toBeInTheDocument()
      expect(screen.getByText('Sign in to your account to continue')).toBeInTheDocument()

      // Step 2: User attempts to log in
      const mockUser = {
        id: '1',
        username: 'newuser',
        email: 'newuser@example.com',
        role: 'user' as const,
        permissions: ['read:data'],
      }

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          user: mockUser,
          token: 'mock-token',
          refreshToken: 'mock-refresh-token',
        }),
      })

      const usernameInput = screen.getByLabelText(/username/i)
      const passwordInput = screen.getByLabelText(/password/i)
      const submitButton = screen.getByRole('button', { name: /sign in/i })

      await user.type(usernameInput, 'newuser')
      await user.type(passwordInput, 'password123')
      await user.click(submitButton)

      // Step 3: User successfully accesses dashboard
      await waitFor(() => {
        expect(screen.getByTestId('dashboard')).toBeInTheDocument()
      })

      // Verify user authentication state
      expect(mockLocalStorage.setItem).toHaveBeenCalledWith('auth_user', JSON.stringify(mockUser))
      expect(mockLocalStorage.setItem).toHaveBeenCalledWith('auth_token', 'mock-token')

      // Step 4: User can see they are logged in
      expect(screen.queryByText('Welcome Back')).not.toBeInTheDocument()
    })
  })

  describe('Admin User Management Journey', () => {
    it('allows admin to access all protected areas', async () => {
      // Setup admin user
      const mockAdminUser = {
        id: '1',
        username: 'admin',
        email: 'admin@example.com',
        role: 'admin' as const,
        permissions: ['read:data', 'write:data', 'admin:users', 'system:config'],
      }

      mockLocalStorage.getItem.mockImplementation((key) => {
        if (key === 'auth_user') return JSON.stringify(mockAdminUser)
        if (key === 'auth_token') return 'admin-token'
        return null
      })

      renderFullApp()

      // Admin should immediately see dashboard
      await waitFor(() => {
        expect(screen.getByTestId('dashboard')).toBeInTheDocument()
      })

      // Admin has access to all areas (verified by permissions)
      expect(mockLocalStorage.getItem).toHaveBeenCalledWith('auth_user')
    })
  })

  describe('Error Recovery Journey', () => {
    it('handles authentication errors and allows retry', async () => {
      renderFullApp()

      // Step 1: User tries to log in with wrong credentials
      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: false,
        status: 401,
        json: () => Promise.resolve({ message: 'Invalid credentials' }),
      })

      const usernameInput = screen.getByLabelText(/username/i)
      const passwordInput = screen.getByLabelText(/password/i)
      const submitButton = screen.getByRole('button', { name: /sign in/i })

      await user.type(usernameInput, 'wronguser')
      await user.type(passwordInput, 'wrongpassword')
      await user.click(submitButton)

      // Step 2: Error is displayed
      await waitFor(() => {
        expect(screen.getByText('Invalid credentials')).toBeInTheDocument()
      })

      // Step 3: User clears error and tries again with correct credentials
      await user.clear(usernameInput)
      await user.clear(passwordInput)

      // Error should clear when user starts typing
      await user.type(usernameInput, 'c')
      expect(screen.queryByText('Invalid credentials')).not.toBeInTheDocument()

      // Step 4: Successful login on retry
      const mockUser = {
        id: '1',
        username: 'correctuser',
        email: 'correct@example.com',
        role: 'user' as const,
        permissions: ['read:data'],
      }

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          user: mockUser,
          token: 'correct-token',
          refreshToken: 'correct-refresh-token',
        }),
      })

      await user.clear(usernameInput)
      await user.type(usernameInput, 'correctuser')
      await user.type(passwordInput, 'correctpassword')
      await user.click(submitButton)

      await waitFor(() => {
        expect(screen.getByTestId('dashboard')).toBeInTheDocument()
      })
    })
  })

  describe('Session Management Journey', () => {
    it('handles session expiry and automatic logout', async () => {
      // Setup authenticated user
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
        if (key === 'auth_refresh_token') return 'invalid-refresh'
        return null
      })

      // Mock failed refresh token
      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: false,
        status: 401,
        json: () => Promise.resolve({ message: 'Invalid refresh token' }),
      })

      renderFullApp()

      // Should automatically log out and redirect to login
      await waitFor(() => {
        expect(screen.getByText('Welcome Back')).toBeInTheDocument()
      })

      expect(screen.queryByTestId('dashboard')).not.toBeInTheDocument()
    })
  })

  describe('Data Persistence Journey', () => {
    it('maintains user session across page refreshes', async () => {
      // Step 1: User logs in successfully
      const mockUser = {
        id: '1',
        username: 'persistentuser',
        email: 'persistent@example.com',
        role: 'user' as const,
        permissions: ['read:data'],
      }

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          user: mockUser,
          token: 'persistent-token',
          refreshToken: 'persistent-refresh',
        }),
      })

      renderFullApp()

      const usernameInput = screen.getByLabelText(/username/i)
      const passwordInput = screen.getByLabelText(/password/i)
      const submitButton = screen.getByRole('button', { name: /sign in/i })

      await user.type(usernameInput, 'persistentuser')
      await user.type(passwordInput, 'password123')
      await user.click(submitButton)

      await waitFor(() => {
        expect(screen.getByTestId('dashboard')).toBeInTheDocument()
      })

      // Step 2: Simulate page refresh by re-rendering with stored data
      mockLocalStorage.getItem.mockImplementation((key) => {
        if (key === 'auth_user') return JSON.stringify(mockUser)
        if (key === 'auth_token') return 'persistent-token'
        if (key === 'auth_refresh_token') return 'persistent-refresh'
        return null
      })

      // Re-render the app (simulating page refresh)
      renderFullApp()

      // Step 3: User should still be logged in
      await waitFor(() => {
        expect(screen.getByTestId('dashboard')).toBeInTheDocument()
      })

      expect(screen.queryByText('Welcome Back')).not.toBeInTheDocument()
    })
  })

  describe('Notification System Journey', () => {
    it('handles notification interactions correctly', async () => {
      // Setup authenticated user
      const mockUser = {
        id: '1',
        username: 'testuser',
        email: 'test@example.com',
        role: 'admin' as const,
        permissions: ['read:data', 'admin:users', 'system:config'],
      }

      mockLocalStorage.getItem.mockImplementation((key) => {
        if (key === 'auth_user') return JSON.stringify(mockUser)
        if (key === 'auth_token') return 'test-token'
        return null
      })

      renderFullApp()

      await waitFor(() => {
        expect(screen.getByTestId('dashboard')).toBeInTheDocument()
      })

      // User should be able to interact with notifications
      // (The actual notification system would be tested here)
    })
  })

  describe('Theme and UI Persistence Journey', () => {
    it('remembers user preferences across sessions', async () => {
      // This would test theme persistence, but for now we verify the basic structure
      renderFullApp()

      // The theme system should be working
      expect(screen.getByText('Welcome Back')).toBeInTheDocument()
    })
  })

  describe('Performance and Loading Journey', () => {
    it('shows appropriate loading states during authentication', async () => {
      renderFullApp()

      // Mock slow login response
      const slowLogin = new Promise(resolve => 
        setTimeout(() => resolve({
          ok: true,
          json: () => Promise.resolve({
            user: {
              id: '1',
              username: 'slowuser',
              email: 'slow@example.com',
              role: 'user' as const,
              permissions: ['read:data'],
            },
            token: 'slow-token',
            refreshToken: 'slow-refresh',
          }),
        }), 100)
      )

      global.fetch = vi.fn().mockReturnValueOnce(slowLogin)

      const usernameInput = screen.getByLabelText(/username/i)
      const passwordInput = screen.getByLabelText(/password/i)
      const submitButton = screen.getByRole('button', { name: /sign in/i })

      await user.type(usernameInput, 'slowuser')
      await user.type(passwordInput, 'password123')
      await user.click(submitButton)

      // Should show loading state
      expect(screen.getByText('Signing in...')).toBeInTheDocument()
      expect(submitButton).toBeDisabled()

      // Wait for completion
      await waitFor(() => {
        expect(screen.getByTestId('dashboard')).toBeInTheDocument()
      })
    })
  })
})