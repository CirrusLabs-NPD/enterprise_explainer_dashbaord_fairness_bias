"""
WebSocket API endpoints for real-time updates
"""

import json
import uuid
from typing import Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from fastapi.responses import HTMLResponse
import structlog

from app.core.websocket_manager import WebSocketManager
from app.core.dependencies import get_websocket_manager
from app.core.security import get_current_user_optional

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/ws", tags=["websocket"])


@router.websocket("/connect")
async def websocket_endpoint(
    websocket: WebSocket,
    token: str = Query(None),
    websocket_manager: WebSocketManager = Depends(get_websocket_manager)
):
    """Main WebSocket connection endpoint"""
    connection_id = str(uuid.uuid4())
    user_id = None
    
    try:
        # Try to authenticate user if token provided
        if token:
            try:
                # Simple token validation for demo
                if token == "test_token":
                    user_id = "test_user"
                elif token.startswith("dev_"):
                    user_id = token.replace("dev_", "")
                else:
                    user_id = "anonymous"
            except Exception:
                user_id = "anonymous"
        
        # Connect to WebSocket manager
        success = await websocket_manager.connect(websocket, connection_id, user_id)
        
        if not success:
            await websocket.close(code=4000, reason="Connection failed")
            return
        
        logger.info(f"WebSocket connected: {connection_id} (user: {user_id})")
        
        # Send welcome message
        await websocket.send_text(json.dumps({
            "type": "connection_established",
            "connection_id": connection_id,
            "user_id": user_id,
            "message": "Connected to ML Explainer Dashboard"
        }))
        
        # Handle messages
        while True:
            try:
                # Receive message
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle different message types
                await handle_websocket_message(
                    websocket, connection_id, user_id, message, websocket_manager
                )
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format"
                }))
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Internal server error"
                }))
    
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    
    finally:
        # Clean up connection
        await websocket_manager.disconnect(connection_id)
        logger.info(f"WebSocket disconnected: {connection_id}")


async def handle_websocket_message(
    websocket: WebSocket,
    connection_id: str,
    user_id: str,
    message: Dict[str, Any],
    websocket_manager: WebSocketManager
):
    """Handle incoming WebSocket messages"""
    message_type = message.get("type")
    
    if message_type == "subscribe_model":
        # Subscribe to model updates
        model_id = message.get("model_id")
        if model_id:
            await websocket_manager.subscribe_to_model(connection_id, model_id)
            await websocket.send_text(json.dumps({
                "type": "subscription_confirmed",
                "model_id": model_id,
                "message": f"Subscribed to model {model_id}"
            }))
        else:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "model_id required for subscription"
            }))
    
    elif message_type == "unsubscribe_model":
        # Unsubscribe from model updates
        model_id = message.get("model_id")
        if model_id:
            await websocket_manager.unsubscribe_from_model(connection_id, model_id)
            await websocket.send_text(json.dumps({
                "type": "unsubscription_confirmed",
                "model_id": model_id,
                "message": f"Unsubscribed from model {model_id}"
            }))
        else:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "model_id required for unsubscription"
            }))
    
    elif message_type == "ping":
        # Respond to ping
        await websocket.send_text(json.dumps({
            "type": "pong",
            "timestamp": message.get("timestamp")
        }))
    
    elif message_type == "get_stats":
        # Send connection statistics
        stats = websocket_manager.get_stats()
        await websocket.send_text(json.dumps({
            "type": "stats",
            "data": stats
        }))
    
    elif message_type == "echo":
        # Echo message back
        await websocket.send_text(json.dumps({
            "type": "echo_response",
            "original_message": message.get("message", ""),
            "timestamp": message.get("timestamp")
        }))
    
    else:
        # Unknown message type
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Unknown message type: {message_type}"
        }))


@router.get("/test-page")
async def get_websocket_test_page():
    """Test page for WebSocket connections"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ML Explainer Dashboard - WebSocket Test</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 800px; margin: 0 auto; }
            .message-box { 
                border: 1px solid #ddd; 
                padding: 10px; 
                margin: 10px 0; 
                height: 300px; 
                overflow-y: auto; 
                background: #f9f9f9;
            }
            .controls { margin: 20px 0; }
            button { margin: 5px; padding: 10px 15px; }
            input[type="text"] { width: 200px; padding: 5px; margin: 5px; }
            .status { 
                padding: 10px; 
                margin: 10px 0; 
                border-radius: 5px; 
            }
            .connected { background: #d4edda; color: #155724; }
            .disconnected { background: #f8d7da; color: #721c24; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>ML Explainer Dashboard - WebSocket Test</h1>
            
            <div id="status" class="status disconnected">
                Status: Disconnected
            </div>
            
            <div class="controls">
                <button onclick="connect()">Connect</button>
                <button onclick="disconnect()">Disconnect</button>
                <input type="text" id="tokenInput" placeholder="Token (optional)" value="test_token">
            </div>
            
            <div class="controls">
                <input type="text" id="modelIdInput" placeholder="Model ID" value="test_model">
                <button onclick="subscribeToModel()">Subscribe to Model</button>
                <button onclick="unsubscribeFromModel()">Unsubscribe from Model</button>
            </div>
            
            <div class="controls">
                <button onclick="sendPing()">Send Ping</button>
                <button onclick="getStats()">Get Stats</button>
                <button onclick="clearMessages()">Clear Messages</button>
            </div>
            
            <div class="message-box" id="messages"></div>
        </div>
        
        <script>
            let ws = null;
            let messages = document.getElementById('messages');
            let status = document.getElementById('status');
            
            function addMessage(message) {
                const div = document.createElement('div');
                div.innerHTML = '<strong>' + new Date().toLocaleTimeString() + ':</strong> ' + 
                               JSON.stringify(message, null, 2);
                messages.appendChild(div);
                messages.scrollTop = messages.scrollHeight;
            }
            
            function updateStatus(connected) {
                if (connected) {
                    status.textContent = 'Status: Connected';
                    status.className = 'status connected';
                } else {
                    status.textContent = 'Status: Disconnected';
                    status.className = 'status disconnected';
                }
            }
            
            function connect() {
                const token = document.getElementById('tokenInput').value;
                const wsUrl = `ws://localhost:8000/api/v1/ws/connect${token ? '?token=' + token : ''}`;
                
                ws = new WebSocket(wsUrl);
                
                ws.onopen = function(event) {
                    addMessage({type: 'system', message: 'Connected to WebSocket'});
                    updateStatus(true);
                };
                
                ws.onmessage = function(event) {
                    const message = JSON.parse(event.data);
                    addMessage(message);
                };
                
                ws.onclose = function(event) {
                    addMessage({type: 'system', message: 'WebSocket connection closed'});
                    updateStatus(false);
                };
                
                ws.onerror = function(error) {
                    addMessage({type: 'error', message: 'WebSocket error: ' + error});
                    updateStatus(false);
                };
            }
            
            function disconnect() {
                if (ws) {
                    ws.close();
                    ws = null;
                }
            }
            
            function subscribeToModel() {
                const modelId = document.getElementById('modelIdInput').value;
                if (ws && modelId) {
                    ws.send(JSON.stringify({
                        type: 'subscribe_model',
                        model_id: modelId
                    }));
                }
            }
            
            function unsubscribeFromModel() {
                const modelId = document.getElementById('modelIdInput').value;
                if (ws && modelId) {
                    ws.send(JSON.stringify({
                        type: 'unsubscribe_model',
                        model_id: modelId
                    }));
                }
            }
            
            function sendPing() {
                if (ws) {
                    ws.send(JSON.stringify({
                        type: 'ping',
                        timestamp: new Date().toISOString()
                    }));
                }
            }
            
            function getStats() {
                if (ws) {
                    ws.send(JSON.stringify({
                        type: 'get_stats'
                    }));
                }
            }
            
            function clearMessages() {
                messages.innerHTML = '';
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@router.get("/stats")
async def get_websocket_stats(
    websocket_manager: WebSocketManager = Depends(get_websocket_manager)
):
    """Get WebSocket connection statistics"""
    try:
        stats = websocket_manager.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting WebSocket stats: {e}")
        return {"error": str(e)}


@router.post("/broadcast")
async def broadcast_message(
    message: str,
    message_type: str = "system_notification",
    websocket_manager: WebSocketManager = Depends(get_websocket_manager)
):
    """Broadcast a message to all connected clients"""
    try:
        await websocket_manager.send_system_notification(message, message_type)
        return {"message": "Broadcast sent successfully"}
    except Exception as e:
        logger.error(f"Error broadcasting message: {e}")
        return {"error": str(e)}


@router.post("/notify-model/{model_id}")
async def notify_model_subscribers(
    model_id: str,
    message: str,
    update_type: str = "general",
    websocket_manager: WebSocketManager = Depends(get_websocket_manager)
):
    """Send notification to model subscribers"""
    try:
        await websocket_manager.send_model_update(model_id, update_type, {
            "message": message,
            "update_type": update_type
        })
        return {"message": f"Notification sent to {model_id} subscribers"}
    except Exception as e:
        logger.error(f"Error notifying model subscribers: {e}")
        return {"error": str(e)}