"""
WebSocket Manager for Real-time Updates
Handles WebSocket connections and broadcasting
"""

import asyncio
import json
import logging
from typing import Dict, List, Set, Optional, Any
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime
import structlog

logger = structlog.get_logger(__name__)


class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        # Active connections by connection_id
        self.active_connections: Dict[str, WebSocket] = {}
        
        # Model subscriptions: model_id -> set of connection_ids
        self.model_subscriptions: Dict[str, Set[str]] = {}
        
        # User subscriptions: user_id -> set of connection_ids
        self.user_subscriptions: Dict[str, Set[str]] = {}
        
        # Connection metadata
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        
        logger.info("ConnectionManager initialized")
    
    async def connect(self, websocket: WebSocket, connection_id: str, user_id: str = None) -> bool:
        """Accept a new WebSocket connection"""
        try:
            await websocket.accept()
            
            self.active_connections[connection_id] = websocket
            
            # Store connection metadata
            self.connection_metadata[connection_id] = {
                "user_id": user_id,
                "connected_at": datetime.utcnow(),
                "last_ping": datetime.utcnow()
            }
            
            # Add to user subscriptions
            if user_id:
                if user_id not in self.user_subscriptions:
                    self.user_subscriptions[user_id] = set()
                self.user_subscriptions[user_id].add(connection_id)
            
            logger.info(f"WebSocket connection established: {connection_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error accepting WebSocket connection: {e}")
            return False
    
    async def disconnect(self, connection_id: str):
        """Disconnect a WebSocket connection"""
        try:
            # Remove from active connections
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]
            
            # Remove from model subscriptions
            for model_id in list(self.model_subscriptions.keys()):
                if connection_id in self.model_subscriptions[model_id]:
                    self.model_subscriptions[model_id].discard(connection_id)
                    if not self.model_subscriptions[model_id]:
                        del self.model_subscriptions[model_id]
            
            # Remove from user subscriptions
            if connection_id in self.connection_metadata:
                user_id = self.connection_metadata[connection_id].get("user_id")
                if user_id and user_id in self.user_subscriptions:
                    self.user_subscriptions[user_id].discard(connection_id)
                    if not self.user_subscriptions[user_id]:
                        del self.user_subscriptions[user_id]
                
                del self.connection_metadata[connection_id]
            
            logger.info(f"WebSocket connection closed: {connection_id}")
            
        except Exception as e:
            logger.error(f"Error disconnecting WebSocket: {e}")
    
    async def send_personal_message(self, connection_id: str, message: str):
        """Send a message to a specific connection"""
        try:
            if connection_id in self.active_connections:
                websocket = self.active_connections[connection_id]
                await websocket.send_text(message)
                return True
            return False
            
        except WebSocketDisconnect:
            await self.disconnect(connection_id)
            return False
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            return False
    
    async def broadcast_to_user(self, user_id: str, message: str):
        """Broadcast a message to all connections of a user"""
        if user_id not in self.user_subscriptions:
            return
        
        disconnected_connections = []
        
        for connection_id in self.user_subscriptions[user_id]:
            try:
                if connection_id in self.active_connections:
                    websocket = self.active_connections[connection_id]
                    await websocket.send_text(message)
            except WebSocketDisconnect:
                disconnected_connections.append(connection_id)
            except Exception as e:
                logger.error(f"Error broadcasting to user {user_id}: {e}")
        
        # Clean up disconnected connections
        for connection_id in disconnected_connections:
            await self.disconnect(connection_id)
    
    async def subscribe_to_model(self, connection_id: str, model_id: str):
        """Subscribe a connection to model updates"""
        if connection_id not in self.active_connections:
            return False
        
        if model_id not in self.model_subscriptions:
            self.model_subscriptions[model_id] = set()
        
        self.model_subscriptions[model_id].add(connection_id)
        logger.info(f"Connection {connection_id} subscribed to model {model_id}")
        return True
    
    async def unsubscribe_from_model(self, connection_id: str, model_id: str):
        """Unsubscribe a connection from model updates"""
        if model_id in self.model_subscriptions:
            self.model_subscriptions[model_id].discard(connection_id)
            if not self.model_subscriptions[model_id]:
                del self.model_subscriptions[model_id]
            logger.info(f"Connection {connection_id} unsubscribed from model {model_id}")
    
    async def broadcast_to_model_subscribers(self, model_id: str, message: str):
        """Broadcast a message to all subscribers of a model"""
        if model_id not in self.model_subscriptions:
            return
        
        disconnected_connections = []
        
        for connection_id in self.model_subscriptions[model_id]:
            try:
                if connection_id in self.active_connections:
                    websocket = self.active_connections[connection_id]
                    await websocket.send_text(message)
            except WebSocketDisconnect:
                disconnected_connections.append(connection_id)
            except Exception as e:
                logger.error(f"Error broadcasting to model {model_id}: {e}")
        
        # Clean up disconnected connections
        for connection_id in disconnected_connections:
            await self.disconnect(connection_id)
    
    async def broadcast_to_all(self, message: str):
        """Broadcast a message to all active connections"""
        disconnected_connections = []
        
        for connection_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(message)
            except WebSocketDisconnect:
                disconnected_connections.append(connection_id)
            except Exception as e:
                logger.error(f"Error broadcasting to all: {e}")
        
        # Clean up disconnected connections
        for connection_id in disconnected_connections:
            await self.disconnect(connection_id)
    
    def get_active_connections_count(self) -> int:
        """Get the number of active connections"""
        return len(self.active_connections)
    
    def get_model_subscribers_count(self, model_id: str) -> int:
        """Get the number of subscribers for a model"""
        return len(self.model_subscriptions.get(model_id, set()))
    
    def get_user_connections_count(self, user_id: str) -> int:
        """Get the number of connections for a user"""
        return len(self.user_subscriptions.get(user_id, set()))
    
    async def ping_all_connections(self):
        """Send ping to all connections to check if they're alive"""
        disconnected_connections = []
        
        for connection_id, websocket in self.active_connections.items():
            try:
                await websocket.ping()
                # Update last ping time
                if connection_id in self.connection_metadata:
                    self.connection_metadata[connection_id]["last_ping"] = datetime.utcnow()
            except Exception:
                disconnected_connections.append(connection_id)
        
        # Clean up disconnected connections
        for connection_id in disconnected_connections:
            await self.disconnect(connection_id)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            "active_connections": len(self.active_connections),
            "model_subscriptions": len(self.model_subscriptions),
            "user_subscriptions": len(self.user_subscriptions),
            "total_model_subscribers": sum(len(subs) for subs in self.model_subscriptions.values()),
            "connection_metadata": {
                conn_id: {
                    "user_id": meta.get("user_id"),
                    "connected_at": meta.get("connected_at").isoformat() if meta.get("connected_at") else None,
                    "last_ping": meta.get("last_ping").isoformat() if meta.get("last_ping") else None
                }
                for conn_id, meta in self.connection_metadata.items()
            }
        }


class WebSocketManager:
    """
    High-level WebSocket manager for the ML Explainer Dashboard
    Handles different types of real-time updates
    """
    
    def __init__(self):
        self.connection_manager = ConnectionManager()
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.running = False
        self.message_processor_task: Optional[asyncio.Task] = None
        
        logger.info("WebSocketManager initialized")
    
    async def start(self):
        """Start the WebSocket manager"""
        if self.running:
            return
        
        self.running = True
        self.message_processor_task = asyncio.create_task(self._process_messages())
        
        # Start periodic ping task
        asyncio.create_task(self._periodic_ping())
        
        logger.info("WebSocketManager started")
    
    async def stop(self):
        """Stop the WebSocket manager"""
        self.running = False
        
        if self.message_processor_task:
            self.message_processor_task.cancel()
            try:
                await self.message_processor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("WebSocketManager stopped")
    
    async def connect(self, websocket: WebSocket, connection_id: str, user_id: str = None) -> bool:
        """Accept a new WebSocket connection"""
        return await self.connection_manager.connect(websocket, connection_id, user_id)
    
    async def disconnect(self, connection_id: str):
        """Disconnect a WebSocket connection"""
        await self.connection_manager.disconnect(connection_id)
    
    async def subscribe_to_model(self, connection_id: str, model_id: str):
        """Subscribe to model updates"""
        await self.connection_manager.subscribe_to_model(connection_id, model_id)
    
    async def unsubscribe_from_model(self, connection_id: str, model_id: str):
        """Unsubscribe from model updates"""
        await self.connection_manager.unsubscribe_from_model(connection_id, model_id)
    
    async def send_model_update(self, model_id: str, update_type: str, data: Dict[str, Any]):
        """Send a model update to all subscribers"""
        message = {
            "type": "model_update",
            "model_id": model_id,
            "update_type": update_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.message_queue.put(("model_broadcast", model_id, json.dumps(message)))
    
    async def send_prediction_update(self, model_id: str, prediction_data: Dict[str, Any]):
        """Send a prediction update"""
        await self.send_model_update(model_id, "prediction", prediction_data)
    
    async def send_explanation_update(self, model_id: str, explanation_data: Dict[str, Any]):
        """Send an explanation update"""
        await self.send_model_update(model_id, "explanation", explanation_data)
    
    async def send_drift_alert(self, model_id: str, alert_data: Dict[str, Any]):
        """Send a drift detection alert"""
        await self.send_model_update(model_id, "drift_alert", alert_data)
    
    async def send_performance_alert(self, model_id: str, alert_data: Dict[str, Any]):
        """Send a performance alert"""
        await self.send_model_update(model_id, "performance_alert", alert_data)
    
    async def send_system_notification(self, message: str, severity: str = "info"):
        """Send a system-wide notification"""
        notification = {
            "type": "system_notification",
            "message": message,
            "severity": severity,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.message_queue.put(("broadcast_all", json.dumps(notification)))
    
    async def send_user_notification(self, user_id: str, message: str, notification_type: str = "info"):
        """Send a notification to a specific user"""
        notification = {
            "type": "user_notification",
            "message": message,
            "notification_type": notification_type,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.message_queue.put(("user_broadcast", user_id, json.dumps(notification)))
    
    async def broadcast_to_model_subscribers(self, model_id: str, message: str):
        """Broadcast a message to model subscribers"""
        await self.connection_manager.broadcast_to_model_subscribers(model_id, message)
    
    async def _process_messages(self):
        """Process messages from the queue"""
        while self.running:
            try:
                message_type, *args = await asyncio.wait_for(
                    self.message_queue.get(), 
                    timeout=1.0
                )
                
                if message_type == "model_broadcast":
                    model_id, message = args
                    await self.connection_manager.broadcast_to_model_subscribers(model_id, message)
                elif message_type == "user_broadcast":
                    user_id, message = args
                    await self.connection_manager.broadcast_to_user(user_id, message)
                elif message_type == "broadcast_all":
                    message = args[0]
                    await self.connection_manager.broadcast_to_all(message)
                elif message_type == "personal_message":
                    connection_id, message = args
                    await self.connection_manager.send_personal_message(connection_id, message)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing message: {e}")
    
    async def _periodic_ping(self):
        """Periodically ping all connections"""
        while self.running:
            try:
                await asyncio.sleep(30)  # Ping every 30 seconds
                await self.connection_manager.ping_all_connections()
            except Exception as e:
                logger.error(f"Error in periodic ping: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get WebSocket statistics"""
        return self.connection_manager.get_connection_stats()


# Global WebSocket manager instance
websocket_manager = WebSocketManager()