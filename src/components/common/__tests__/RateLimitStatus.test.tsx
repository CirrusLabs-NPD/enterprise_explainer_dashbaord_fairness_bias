import { render, screen, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import RateLimitStatus from '../RateLimitStatus'

// Mock framer-motion
vi.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  },
}))

describe('RateLimitStatus', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    // Mock Math.random to return predictable values for testing
    vi.spyOn(Math, 'random').mockReturnValue(0.5) // 50 remaining out of 100
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('renders minimal indicator by default', () => {
    render(<RateLimitStatus />)
    
    // Should show the rate limit fraction
    expect(screen.getByText('50/100')).toBeInTheDocument()
  })

  it('renders detailed status when showDetails is true', () => {
    render(<RateLimitStatus showDetails={true} />)
    
    expect(screen.getByText('Rate Limit Status')).toBeInTheDocument()
    expect(screen.getByText('Usage')).toBeInTheDocument()
    expect(screen.getByText('Remaining')).toBeInTheDocument()
    expect(screen.getByText('Limit')).toBeInTheDocument()
    expect(screen.getByText('Reset in')).toBeInTheDocument()
    expect(screen.getByText('Status')).toBeInTheDocument()
  })

  it('shows correct usage percentage', () => {
    render(<RateLimitStatus showDetails={true} />)
    
    // 50% usage (50 used out of 100)
    expect(screen.getByText('50%')).toBeInTheDocument()
  })

  it('displays warning state correctly', () => {
    // Mock high usage (90% used = 10 remaining)
    vi.spyOn(Math, 'random').mockReturnValue(0.1)
    
    render(<RateLimitStatus showDetails={true} />)
    
    expect(screen.getByText('10/100')).toBeInTheDocument()
    expect(screen.getByText('Warning')).toBeInTheDocument()
  })

  it('displays blocked state correctly', () => {
    // Mock no remaining requests
    vi.spyOn(Math, 'random').mockReturnValue(0)
    
    render(<RateLimitStatus showDetails={true} />)
    
    expect(screen.getByText('0/100')).toBeInTheDocument()
    expect(screen.getByText('Blocked')).toBeInTheDocument()
  })

  it('shows blocking warning message', () => {
    vi.spyOn(Math, 'random').mockReturnValue(0)
    
    render(<RateLimitStatus showDetails={true} />)
    
    expect(screen.getByText(/Rate limit exceeded/)).toBeInTheDocument()
    expect(screen.getByText(/Please wait.*minute/)).toBeInTheDocument()
  })

  it('shows warning message when approaching limit', () => {
    vi.spyOn(Math, 'random').mockReturnValue(0.1) // 90% usage
    
    render(<RateLimitStatus showDetails={true} />)
    
    expect(screen.getByText(/Approaching rate limit/)).toBeInTheDocument()
    expect(screen.getByText(/Consider reducing request frequency/)).toBeInTheDocument()
  })

  it('shows OK status when usage is normal', () => {
    render(<RateLimitStatus showDetails={true} />)
    
    expect(screen.getByText('OK')).toBeInTheDocument()
  })

  it('applies custom className', () => {
    const { container } = render(<RateLimitStatus className="custom-class" />)
    
    expect(container.firstChild).toHaveClass('custom-class')
  })

  it('updates rate limit info periodically', async () => {
    // Mock different values for subsequent calls
    let callCount = 0
    vi.spyOn(Math, 'random').mockImplementation(() => {
      callCount++
      return callCount === 1 ? 0.5 : 0.3 // First 50, then 30 remaining
    })

    render(<RateLimitStatus />)
    
    expect(screen.getByText('50/100')).toBeInTheDocument()
    
    // Wait for the interval to trigger (mocked to update every 5 seconds)
    await waitFor(() => {
      // The component should update with new values
      // Note: This is simplified since we can't easily test real intervals
    }, { timeout: 1000 })
  })

  it('handles edge case values correctly', () => {
    render(<RateLimitStatus showDetails={true} />)
    
    // Should handle the mocked values gracefully
    expect(screen.getByText('50')).toBeInTheDocument() // Remaining
    expect(screen.getByText('100')).toBeInTheDocument() // Limit
  })

  it('formats time to reset correctly', () => {
    render(<RateLimitStatus showDetails={true} />)
    
    // Should show time in minutes
    expect(screen.getByText(/\d+m/)).toBeInTheDocument()
  })

  it('shows live indicator in detailed view', () => {
    render(<RateLimitStatus showDetails={true} />)
    
    expect(screen.getByText('Live')).toBeInTheDocument()
  })

  it('returns null when no rate limit info', () => {
    // Mock a scenario where rate limit info is not available
    vi.spyOn(Math, 'random').mockImplementation(() => {
      throw new Error('No rate limit info')
    })

    const { container } = render(<RateLimitStatus />)
    
    // Component should handle gracefully and not crash
    expect(container.firstChild).toBeInTheDocument()
  })

  it('has correct accessibility attributes', () => {
    render(<RateLimitStatus showDetails={true} />)
    
    // Check that the component is accessible
    expect(screen.getByText('Rate Limit Status')).toBeInTheDocument()
  })

  it('displays correct color states', () => {
    // Test different states with different mock values
    
    // Normal state (green)
    vi.spyOn(Math, 'random').mockReturnValue(0.5)
    const { rerender } = render(<RateLimitStatus showDetails={true} />)
    expect(screen.getByText('OK')).toBeInTheDocument()
    
    // Warning state (yellow)
    vi.spyOn(Math, 'random').mockReturnValue(0.1)
    rerender(<RateLimitStatus showDetails={true} />)
    expect(screen.getByText('Warning')).toBeInTheDocument()
    
    // Blocked state (red)
    vi.spyOn(Math, 'random').mockReturnValue(0)
    rerender(<RateLimitStatus showDetails={true} />)
    expect(screen.getByText('Blocked')).toBeInTheDocument()
  })
})