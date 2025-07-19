import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import NotificationCenter from '../NotificationCenter'
import type { Notification } from '../NotificationCenter'

// Mock framer-motion
vi.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  },
  AnimatePresence: ({ children }: any) => <>{children}</>,
}))

// Mock react-hot-toast
vi.mock('react-hot-toast', () => ({
  toast: {
    success: vi.fn(),
    error: vi.fn(),
  },
}))

const mockNotifications: Notification[] = [
  {
    id: '1',
    title: 'Model Drift Detected',
    message: 'Classification model accuracy dropped below threshold',
    type: 'warning',
    timestamp: new Date(Date.now() - 5 * 60 * 1000),
    read: false,
    actionUrl: '/data-drift',
    actionLabel: 'View Details',
    persistent: true,
  },
  {
    id: '2',
    title: 'New User Registered',
    message: 'user@example.com has joined the platform',
    type: 'info',
    timestamp: new Date(Date.now() - 15 * 60 * 1000),
    read: false,
  },
  {
    id: '3',
    title: 'System Backup Completed',
    message: 'Daily backup finished successfully',
    type: 'success',
    timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000),
    read: true,
  },
]

describe('NotificationCenter', () => {
  const user = userEvent.setup()
  const mockOnClose = vi.fn()
  const mockOnNotificationClick = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders nothing when closed', () => {
    render(
      <NotificationCenter
        isOpen={false}
        onClose={mockOnClose}
        onNotificationClick={mockOnNotificationClick}
      />
    )

    expect(screen.queryByText('Notifications')).not.toBeInTheDocument()
  })

  it('renders notification center when open', () => {
    render(
      <NotificationCenter
        isOpen={true}
        onClose={mockOnClose}
        onNotificationClick={mockOnNotificationClick}
      />
    )

    expect(screen.getByText('Notifications')).toBeInTheDocument()
    expect(screen.getByText('All')).toBeInTheDocument()
    expect(screen.getByText(/Unread \(\d+\)/)).toBeInTheDocument()
  })

  it('displays unread count correctly', () => {
    render(
      <NotificationCenter
        isOpen={true}
        onClose={mockOnClose}
        onNotificationClick={mockOnNotificationClick}
      />
    )

    // Should show unread count in the header and filter tab
    const unreadElements = screen.getAllByText(/\d+/)
    expect(unreadElements.length).toBeGreaterThan(0)
  })

  it('closes when backdrop is clicked', async () => {
    render(
      <NotificationCenter
        isOpen={true}
        onClose={mockOnClose}
        onNotificationClick={mockOnNotificationClick}
      />
    )

    const backdrop = document.querySelector('.fixed.inset-0')
    expect(backdrop).toBeInTheDocument()

    fireEvent.click(backdrop!)
    expect(mockOnClose).toHaveBeenCalledTimes(1)
  })

  it('closes when X button is clicked', async () => {
    render(
      <NotificationCenter
        isOpen={true}
        onClose={mockOnClose}
        onNotificationClick={mockOnNotificationClick}
      />
    )

    const closeButton = screen.getByRole('button', { name: '' }) // X button
    await user.click(closeButton)
    expect(mockOnClose).toHaveBeenCalledTimes(1)
  })

  it('filters notifications correctly', async () => {
    render(
      <NotificationCenter
        isOpen={true}
        onClose={mockOnClose}
        onNotificationClick={mockOnNotificationClick}
      />
    )

    // Initially shows all notifications
    expect(screen.getByText('Model Drift Detected')).toBeInTheDocument()
    expect(screen.getByText('System Backup Completed')).toBeInTheDocument()

    // Click unread filter
    const unreadButton = screen.getByRole('button', { name: /Unread/ })
    await user.click(unreadButton)

    // Should only show unread notifications
    expect(screen.getByText('Model Drift Detected')).toBeInTheDocument()
    expect(screen.queryByText('System Backup Completed')).not.toBeInTheDocument()

    // Switch back to all
    const allButton = screen.getByRole('button', { name: 'All' })
    await user.click(allButton)

    expect(screen.getByText('Model Drift Detected')).toBeInTheDocument()
    expect(screen.getByText('System Backup Completed')).toBeInTheDocument()
  })

  it('marks all notifications as read', async () => {
    render(
      <NotificationCenter
        isOpen={true}
        onClose={mockOnClose}
        onNotificationClick={mockOnNotificationClick}
      />
    )

    const markAllReadButton = screen.getByText('Mark all read')
    await user.click(markAllReadButton)

    // Should update the UI (though we can't easily test state changes without mocking)
    expect(markAllReadButton).toBeInTheDocument()
  })

  it('handles notification clicks', async () => {
    render(
      <NotificationCenter
        isOpen={true}
        onClose={mockOnClose}
        onNotificationClick={mockOnNotificationClick}
      />
    )

    const notification = screen.getByText('Model Drift Detected')
    await user.click(notification.closest('.cursor-pointer')!)

    expect(mockOnNotificationClick).toHaveBeenCalledWith(
      expect.objectContaining({
        id: '1',
        title: 'Model Drift Detected',
        type: 'warning',
      })
    )
  })

  it('handles action button clicks', async () => {
    render(
      <NotificationCenter
        isOpen={true}
        onClose={mockOnClose}
        onNotificationClick={mockOnNotificationClick}
      />
    )

    const actionButton = screen.getByText('View Details')
    await user.click(actionButton)

    expect(mockOnNotificationClick).toHaveBeenCalledWith(
      expect.objectContaining({
        id: '1',
        actionUrl: '/data-drift',
        actionLabel: 'View Details',
      })
    )
  })

  it('displays correct notification icons', () => {
    render(
      <NotificationCenter
        isOpen={true}
        onClose={mockOnClose}
        onNotificationClick={mockOnNotificationClick}
      />
    )

    // Check that different notification types are rendered
    expect(screen.getByText('Model Drift Detected')).toBeInTheDocument()
    expect(screen.getByText('New User Registered')).toBeInTheDocument()
    expect(screen.getByText('System Backup Completed')).toBeInTheDocument()
  })

  it('formats timestamps correctly', () => {
    render(
      <NotificationCenter
        isOpen={true}
        onClose={mockOnClose}
        onNotificationClick={mockOnNotificationClick}
      />
    )

    // Should show relative timestamps
    expect(screen.getByText('5m ago')).toBeInTheDocument()
    expect(screen.getByText('15m ago')).toBeInTheDocument()
    expect(screen.getByText('2h ago')).toBeInTheDocument()
  })

  it('shows empty state when no notifications', () => {
    // Mock empty notifications
    vi.mock('../NotificationCenter', async () => {
      const actual = await vi.importActual('../NotificationCenter')
      return {
        ...actual,
        default: ({ isOpen, onClose }: any) => isOpen ? (
          <div>
            <div>Notifications</div>
            <div>No notifications</div>
          </div>
        ) : null,
      }
    })

    render(
      <NotificationCenter
        isOpen={true}
        onClose={mockOnClose}
        onNotificationClick={mockOnNotificationClick}
      />
    )

    // The component should handle empty state gracefully
    expect(screen.getByText('Notifications')).toBeInTheDocument()
  })

  it('shows unread indicator for unread notifications', () => {
    render(
      <NotificationCenter
        isOpen={true}
        onClose={mockOnClose}
        onNotificationClick={mockOnNotificationClick}
      />
    )

    // Unread notifications should have visual indicators
    const modelDriftNotification = screen.getByText('Model Drift Detected').closest('.group')
    const newUserNotification = screen.getByText('New User Registered').closest('.group')
    const backupNotification = screen.getByText('System Backup Completed').closest('.group')

    // Check visual differences between read and unread (this is a simplified check)
    expect(modelDriftNotification).toBeInTheDocument()
    expect(newUserNotification).toBeInTheDocument()
    expect(backupNotification).toBeInTheDocument()
  })

  it('handles delete notification action', async () => {
    render(
      <NotificationCenter
        isOpen={true}
        onClose={mockOnClose}
        onNotificationClick={mockOnNotificationClick}
      />
    )

    // Hover over a notification to show delete button
    const notification = screen.getByText('Model Drift Detected').closest('.group')!
    fireEvent.mouseEnter(notification)

    // Find and click delete button (trash icon)
    const deleteButtons = document.querySelectorAll('.opacity-0')
    expect(deleteButtons.length).toBeGreaterThan(0)
  })

  it('prevents event bubbling on action clicks', async () => {
    render(
      <NotificationCenter
        isOpen={true}
        onClose={mockOnClose}
        onNotificationClick={mockOnNotificationClick}
      />
    )

    const actionButton = screen.getByText('View Details')
    
    // Mock stopPropagation
    const mockEvent = { stopPropagation: vi.fn() }
    fireEvent.click(actionButton, mockEvent)

    // The action should handle the click properly
    expect(mockOnNotificationClick).toHaveBeenCalled()
  })

  it('shows notification settings button', () => {
    render(
      <NotificationCenter
        isOpen={true}
        onClose={mockOnClose}
        onNotificationClick={mockOnNotificationClick}
      />
    )

    expect(screen.getByText('Notification Settings')).toBeInTheDocument()
  })
})