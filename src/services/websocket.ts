/**
 * WebSocket Service for Real-time Updates
 * Handles WebSocket connections and real-time communication
 */

// WebSocket Configuration
const WS_BASE_URL = 'ws://localhost:8000' || process.env.REACT_APP_WS_URL;
const WS_ENDPOINT = '/api/v1/ws/connect';

// Types
export interface WebSocketMessage {
  type: string;
  data?: any;
  model_id?: string;
  timestamp?: string;
  message?: string;
}

export interface ModelUpdate {
  type: 'model_update';
  model_id: string;
  update_type: 'prediction' | 'explanation' | 'drift_alert' | 'performance_alert';
  data: any;
  timestamp: string;
}

export interface SystemNotification {
  type: 'system_notification';
  message: string;
  severity: 'info' | 'warning' | 'error' | 'success';
  timestamp: string;
}

export interface ConnectionStats {
  active_connections: number;
  model_subscriptions: number;
  user_subscriptions: number;
  total_model_subscribers: number;
  connection_metadata: Record<string, any>;
}

// Event Types
export type WebSocketEventType = 
  | 'connection_established'
  | 'connection_lost'
  | 'model_update'
  | 'system_notification'
  | 'drift_alert'
  | 'performance_alert'
  | 'prediction_update'
  | 'explanation_update'
  | 'error'
  | 'pong'
  | 'stats';

export type WebSocketEventHandler = (data: any) => void;

// WebSocket Manager Class
class WebSocketManager {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private connectionId: string | null = null;
  private userId: string | null = null;
  private isConnecting = false;
  private isConnected = false;
  private eventHandlers: Map<WebSocketEventType, WebSocketEventHandler[]> = new Map();
  private subscribedModels: Set<string> = new Set();
  private heartbeatInterval: NodeJS.Timeout | null = null;
  private reconnectTimeout: NodeJS.Timeout | null = null;

  constructor() {
    this.setupEventHandlers();
  }

  private setupEventHandlers() {
    // Initialize event handler maps
    const eventTypes: WebSocketEventType[] = [
      'connection_established',
      'connection_lost',
      'model_update',
      'system_notification',
      'drift_alert',
      'performance_alert',
      'prediction_update',
      'explanation_update',
      'error',
      'pong',
      'stats'
    ];

    eventTypes.forEach(type => {
      this.eventHandlers.set(type, []);
    });
  }

  // Public API
  async connect(token?: string, userId?: string): Promise<boolean> {
    if (this.isConnecting || this.isConnected) {
      return this.isConnected;
    }

    this.isConnecting = true;
    this.userId = userId || null;

    try {
      const wsUrl = `${WS_BASE_URL}${WS_ENDPOINT}${token ? `?token=${token}` : ''}`;
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = this.handleOpen.bind(this);
      this.ws.onclose = this.handleClose.bind(this);
      this.ws.onerror = this.handleError.bind(this);
      this.ws.onmessage = this.handleMessage.bind(this);

      return new Promise((resolve) => {
        const timeout = setTimeout(() => {
          this.isConnecting = false;
          resolve(false);
        }, 5000);

        this.on('connection_established', () => {
          clearTimeout(timeout);
          this.isConnecting = false;
          resolve(true);
        });
      });
    } catch (error) {
      console.error('WebSocket connection error:', error);
      this.isConnecting = false;
      return false;
    }
  }

  disconnect(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }

    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    this.isConnected = false;
    this.connectionId = null;
    this.subscribedModels.clear();
    this.reconnectAttempts = 0;
  }

  // Event Management
  on(event: WebSocketEventType, handler: WebSocketEventHandler): void {
    const handlers = this.eventHandlers.get(event) || [];
    handlers.push(handler);
    this.eventHandlers.set(event, handlers);
  }

  off(event: WebSocketEventType, handler: WebSocketEventHandler): void {
    const handlers = this.eventHandlers.get(event) || [];
    const index = handlers.indexOf(handler);
    if (index > -1) {
      handlers.splice(index, 1);
      this.eventHandlers.set(event, handlers);
    }
  }

  private emit(event: WebSocketEventType, data?: any): void {
    const handlers = this.eventHandlers.get(event) || [];
    handlers.forEach(handler => {
      try {
        handler(data);
      } catch (error) {
        console.error(`Error in WebSocket event handler for ${event}:`, error);
      }
    });
  }

  // Model Subscriptions
  subscribeToModel(modelId: string): void {
    if (!this.isConnected) {
      console.warn('Cannot subscribe to model: WebSocket not connected');
      return;
    }

    if (this.subscribedModels.has(modelId)) {
      return;
    }

    this.send({
      type: 'subscribe_model',
      model_id: modelId
    });

    this.subscribedModels.add(modelId);
  }

  unsubscribeFromModel(modelId: string): void {
    if (!this.isConnected) {
      console.warn('Cannot unsubscribe from model: WebSocket not connected');
      return;
    }

    if (!this.subscribedModels.has(modelId)) {
      return;
    }

    this.send({
      type: 'unsubscribe_model',
      model_id: modelId
    });

    this.subscribedModels.delete(modelId);
  }

  // Messaging
  send(message: any): void {
    if (!this.isConnected || !this.ws) {
      console.warn('Cannot send message: WebSocket not connected');
      return;
    }

    try {
      this.ws.send(JSON.stringify(message));
    } catch (error) {
      console.error('Error sending WebSocket message:', error);
    }
  }

  ping(): void {
    this.send({
      type: 'ping',
      timestamp: new Date().toISOString()
    });
  }

  getStats(): void {
    this.send({
      type: 'get_stats'
    });
  }

  // Connection Status
  getConnectionStatus(): {
    connected: boolean;
    connectionId: string | null;
    userId: string | null;
    subscribedModels: string[];
  } {
    return {
      connected: this.isConnected,
      connectionId: this.connectionId,
      userId: this.userId,
      subscribedModels: Array.from(this.subscribedModels)
    };
  }

  // Private Event Handlers
  private handleOpen(): void {
    console.log('WebSocket connection opened');
    this.isConnected = true;
    this.reconnectAttempts = 0;
    
    // Start heartbeat
    this.startHeartbeat();
  }

  private handleClose(event: CloseEvent): void {
    console.log('WebSocket connection closed:', event.code, event.reason);
    this.isConnected = false;
    this.connectionId = null;
    
    // Stop heartbeat
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }

    this.emit('connection_lost', { code: event.code, reason: event.reason });

    // Attempt to reconnect
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.scheduleReconnect();
    }
  }

  private handleError(error: Event): void {
    console.error('WebSocket error:', error);
    this.emit('error', error);
  }

  private handleMessage(event: MessageEvent): void {
    try {
      const message: WebSocketMessage = JSON.parse(event.data);
      this.processMessage(message);
    } catch (error) {
      console.error('Error parsing WebSocket message:', error);
    }
  }

  private processMessage(message: WebSocketMessage): void {
    switch (message.type) {
      case 'connection_established':
        this.connectionId = message.data?.connection_id || null;
        this.emit('connection_established', message.data);
        break;

      case 'subscription_confirmed':
        console.log('Subscription confirmed:', message.data);
        break;

      case 'unsubscription_confirmed':
        console.log('Unsubscription confirmed:', message.data);
        break;

      case 'model_update':
        this.handleModelUpdate(message as ModelUpdate);
        break;

      case 'system_notification':
        this.emit('system_notification', message);
        break;

      case 'alert':
        this.handleAlert(message);
        break;

      case 'drift_report':
        this.emit('drift_alert', message.data);
        break;

      case 'model_drift_report':
        this.emit('performance_alert', message.data);
        break;

      case 'pong':
        this.emit('pong', message);
        break;

      case 'stats':
        this.emit('stats', message.data);
        break;

      case 'error':
        this.emit('error', message);
        break;

      default:
        console.warn('Unknown WebSocket message type:', message.type);
    }
  }

  private handleModelUpdate(update: ModelUpdate): void {
    this.emit('model_update', update);
    
    // Emit specific update types
    switch (update.update_type) {
      case 'prediction':
        this.emit('prediction_update', update);
        break;
      case 'explanation':
        this.emit('explanation_update', update);
        break;
      case 'drift_alert':
        this.emit('drift_alert', update);
        break;
      case 'performance_alert':
        this.emit('performance_alert', update);
        break;
    }
  }

  private handleAlert(message: WebSocketMessage): void {
    const alertData = message.data;
    
    switch (alertData.alert_type) {
      case 'drift':
        this.emit('drift_alert', alertData);
        break;
      case 'performance':
        this.emit('performance_alert', alertData);
        break;
      default:
        this.emit('system_notification', {
          type: 'system_notification',
          message: alertData.message,
          severity: alertData.severity,
          timestamp: alertData.timestamp
        });
    }
  }

  private startHeartbeat(): void {
    this.heartbeatInterval = setInterval(() => {
      this.ping();
    }, 30000); // Ping every 30 seconds
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
    }

    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts);
    console.log(`Scheduling reconnect in ${delay}ms (attempt ${this.reconnectAttempts + 1})`);

    this.reconnectTimeout = setTimeout(() => {
      this.reconnectAttempts++;
      this.connect(undefined, this.userId || undefined);
    }, delay);
  }
}

// Export singleton instance
export const webSocketManager = new WebSocketManager();
export default webSocketManager;