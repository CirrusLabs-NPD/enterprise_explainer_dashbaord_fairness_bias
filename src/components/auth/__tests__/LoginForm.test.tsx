import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import LoginForm from '../LoginForm'
import { AuthProvider } from '../../../contexts/AuthContext'
import React from 'react'

// Mock the AuthContext
const mockLogin = vi.fn()
const mockAuthContext = {
  user: null,
  isLoading: false,
  isAuthenticated: false,
  login: mockLogin,
  logout: vi.fn(),
  refreshToken: vi.fn(),
  hasPermission: vi.fn(),
  hasRole: vi.fn(),
}

vi.mock('../../../contexts/AuthContext', async () => {
  const actual = await vi.importActual('../../../contexts/AuthContext')
  return {
    ...actual,
    useAuth: () => mockAuthContext,
  }
})

// Mock react-hot-toast
vi.mock('react-hot-toast', () => ({
  toast: {
    success: vi.fn(),
    error: vi.fn(),
  },
}))

const renderWithAuth = (component: React.ReactElement) => {
  return render(
    <AuthProvider>
      {component}
    </AuthProvider>
  )
}

describe('LoginForm', () => {
  const user = userEvent.setup()

  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders login form with all elements', () => {
    renderWithAuth(<LoginForm />)
    
    expect(screen.getByText('Welcome Back')).toBeInTheDocument()
    expect(screen.getByText('Sign in to your account to continue')).toBeInTheDocument()
    expect(screen.getByLabelText(/username/i)).toBeInTheDocument()
    expect(screen.getByLabelText(/password/i)).toBeInTheDocument()
    expect(screen.getByRole('checkbox', { name: /remember me/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /sign in/i })).toBeInTheDocument()
    expect(screen.getByText('Forgot your password?')).toBeInTheDocument()
  })

  it('validates required fields', async () => {
    renderWithAuth(<LoginForm />)
    
    const submitButton = screen.getByRole('button', { name: /sign in/i })
    await user.click(submitButton)
    
    expect(await screen.findByText('Username is required')).toBeInTheDocument()
    expect(await screen.findByText('Password is required')).toBeInTheDocument()
  })

  it('validates username format', async () => {
    renderWithAuth(<LoginForm />)
    
    const usernameInput = screen.getByLabelText(/username/i)
    const submitButton = screen.getByRole('button', { name: /sign in/i })
    
    await user.type(usernameInput, 'ab')
    await user.click(submitButton)
    
    expect(await screen.findByText('Username must be at least 3 characters')).toBeInTheDocument()
  })

  it('validates password length', async () => {
    renderWithAuth(<LoginForm />)
    
    const passwordInput = screen.getByLabelText(/password/i)
    const submitButton = screen.getByRole('button', { name: /sign in/i })
    
    await user.type(passwordInput, '123')
    await user.click(submitButton)
    
    expect(await screen.findByText('Password must be at least 6 characters')).toBeInTheDocument()
  })

  it('toggles password visibility', async () => {
    renderWithAuth(<LoginForm />)
    
    const passwordInput = screen.getByLabelText(/password/i) as HTMLInputElement
    const toggleButton = screen.getByRole('button', { name: /toggle password visibility/i })
    
    expect(passwordInput.type).toBe('password')
    
    await user.click(toggleButton)
    expect(passwordInput.type).toBe('text')
    
    await user.click(toggleButton)
    expect(passwordInput.type).toBe('password')
  })

  it('submits form with valid data', async () => {
    mockLogin.mockResolvedValueOnce(undefined)
    renderWithAuth(<LoginForm />)
    
    const usernameInput = screen.getByLabelText(/username/i)
    const passwordInput = screen.getByLabelText(/password/i)
    const submitButton = screen.getByRole('button', { name: /sign in/i })
    
    await user.type(usernameInput, 'testuser')
    await user.type(passwordInput, 'password123')
    await user.click(submitButton)
    
    await waitFor(() => {
      expect(mockLogin).toHaveBeenCalledWith('testuser', 'password123')
    })
  })

  it('handles login errors', async () => {
    const mockError = new Error('Invalid credentials')
    mockLogin.mockRejectedValueOnce(mockError)
    
    renderWithAuth(<LoginForm />)
    
    const usernameInput = screen.getByLabelText(/username/i)
    const passwordInput = screen.getByLabelText(/password/i)
    const submitButton = screen.getByRole('button', { name: /sign in/i })
    
    await user.type(usernameInput, 'testuser')
    await user.type(passwordInput, 'wrongpassword')
    await user.click(submitButton)
    
    await waitFor(() => {
      expect(screen.getByText('Invalid credentials')).toBeInTheDocument()
    })
  })

  it('shows loading state during submission', async () => {
    // Mock a delayed login
    mockLogin.mockImplementation(() => new Promise(resolve => setTimeout(resolve, 100)))
    
    renderWithAuth(<LoginForm />)
    
    const usernameInput = screen.getByLabelText(/username/i)
    const passwordInput = screen.getByLabelText(/password/i)
    const submitButton = screen.getByRole('button', { name: /sign in/i })
    
    await user.type(usernameInput, 'testuser')
    await user.type(passwordInput, 'password123')
    await user.click(submitButton)
    
    expect(submitButton).toBeDisabled()
    expect(screen.getByText('Signing in...')).toBeInTheDocument()
    
    await waitFor(() => {
      expect(submitButton).not.toBeDisabled()
    })
  })

  it('remembers user preference', async () => {
    renderWithAuth(<LoginForm />)
    
    const rememberCheckbox = screen.getByRole('checkbox', { name: /remember me/i })
    
    expect(rememberCheckbox).not.toBeChecked()
    
    await user.click(rememberCheckbox)
    expect(rememberCheckbox).toBeChecked()
  })

  it('handles keyboard navigation', async () => {
    renderWithAuth(<LoginForm />)
    
    const usernameInput = screen.getByLabelText(/username/i)
    const passwordInput = screen.getByLabelText(/password/i)
    const submitButton = screen.getByRole('button', { name: /sign in/i })
    
    await user.click(usernameInput)
    await user.keyboard('{Tab}')
    expect(passwordInput).toHaveFocus()
    
    await user.keyboard('{Tab}')
    expect(screen.getByRole('checkbox', { name: /remember me/i })).toHaveFocus()
    
    await user.keyboard('{Tab}')
    expect(submitButton).toHaveFocus()
  })

  it('clears error message on input change', async () => {
    const mockError = new Error('Invalid credentials')
    mockLogin.mockRejectedValueOnce(mockError)
    
    renderWithAuth(<LoginForm />)
    
    const usernameInput = screen.getByLabelText(/username/i)
    const passwordInput = screen.getByLabelText(/password/i)
    const submitButton = screen.getByRole('button', { name: /sign in/i })
    
    // Trigger error
    await user.type(usernameInput, 'testuser')
    await user.type(passwordInput, 'wrongpassword')
    await user.click(submitButton)
    
    await waitFor(() => {
      expect(screen.getByText('Invalid credentials')).toBeInTheDocument()
    })
    
    // Change input to clear error
    await user.type(usernameInput, 'x')
    
    expect(screen.queryByText('Invalid credentials')).not.toBeInTheDocument()
  })

  it('handles special characters in credentials', async () => {
    mockLogin.mockResolvedValueOnce(undefined)
    renderWithAuth(<LoginForm />)
    
    const usernameInput = screen.getByLabelText(/username/i)
    const passwordInput = screen.getByLabelText(/password/i)
    const submitButton = screen.getByRole('button', { name: /sign in/i })
    
    await user.type(usernameInput, 'user@domain.com')
    await user.type(passwordInput, 'P@ssw0rd!')
    await user.click(submitButton)
    
    await waitFor(() => {
      expect(mockLogin).toHaveBeenCalledWith('user@domain.com', 'P@ssw0rd!')
    })
  })
})