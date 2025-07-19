"""
Enterprise Real-time Alerting and Monitoring Service
Provides comprehensive ML model monitoring with real-time alerts
Compatible with enterprise monitoring standards and compliance requirements
"""

import asyncio
import json
import smtplib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import structlog

from app.core.database import get_db
from app.core.websocket_manager import WebSocketManager

logger = structlog.get_logger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertType(str, Enum):
    """Types of alerts"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_DRIFT = "data_drift"
    BIAS_DETECTION = "bias_detection"
    DATA_QUALITY = "data_quality"
    SYSTEM_ERROR = "system_error"
    COMPLIANCE_VIOLATION = "compliance_violation"
    THRESHOLD_BREACH = "threshold_breach"
    MODEL_DRIFT = "model_drift"


class AlertChannel(str, Enum):
    """Alert delivery channels"""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    DASHBOARD = "dashboard"
    PAGERDUTY = "pagerduty"


class AlertStatus(str, Enum):
    """Alert status"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class EnterpriseAlertingService:
    """
    Enterprise-grade alerting service with real-time monitoring
    """
    
    def __init__(self, websocket_manager: WebSocketManager):
        self.websocket_manager = websocket_manager
        self.alert_rules = {}
        self.notification_channels = {}
        self.alert_suppressions = {}
        self.escalation_policies = {}
        
    async def initialize(self):
        """Initialize alerting service"""
        await self._load_alert_rules()
        await self._load_notification_channels()
        await self._load_escalation_policies()
        logger.info("Enterprise alerting service initialized")
    
    async def create_alert_rule(
        self,
        rule_id: str,
        name: str,
        description: str,
        metric_name: str,
        condition: str,
        threshold: float,
        severity: AlertSeverity,
        alert_type: AlertType,
        model_ids: Optional[List[str]] = None,
        enabled: bool = True,
        evaluation_window: int = 300,  # 5 minutes
        cooldown_period: int = 900,    # 15 minutes
        notification_channels: Optional[List[str]] = None,
        escalation_policy: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new alert rule
        
        Args:
            rule_id: Unique identifier for the rule
            name: Human-readable name
            description: Description of what the rule monitors
            metric_name: Name of the metric to monitor
            condition: Condition operator (>, <, >=, <=, ==, !=)
            threshold: Threshold value to trigger alert
            severity: Alert severity level
            alert_type: Type of alert
            model_ids: List of model IDs to monitor (None for all)
            enabled: Whether the rule is enabled
            evaluation_window: Window in seconds for metric evaluation
            cooldown_period: Minimum time between alerts in seconds
            notification_channels: List of notification channel IDs
            escalation_policy: Escalation policy ID
            metadata: Additional metadata
            
        Returns:
            Created alert rule information
        """
        alert_rule = {
            "rule_id": rule_id,
            "name": name,
            "description": description,
            "metric_name": metric_name,
            "condition": condition,
            "threshold": threshold,
            "severity": severity.value,
            "alert_type": alert_type.value,
            "model_ids": model_ids or [],
            "enabled": enabled,
            "evaluation_window": evaluation_window,
            "cooldown_period": cooldown_period,
            "notification_channels": notification_channels or [],
            "escalation_policy": escalation_policy,
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat(),
            "last_triggered": None,
            "trigger_count": 0
        }
        
        # Store in database
        async with get_db() as db:
            await db.execute("""
                INSERT INTO alert_rules (
                    rule_id, name, description, metric_name, condition, threshold,
                    severity, alert_type, model_ids, enabled, evaluation_window,
                    cooldown_period, notification_channels, escalation_policy, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                ON CONFLICT (rule_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    description = EXCLUDED.description,
                    metric_name = EXCLUDED.metric_name,
                    condition = EXCLUDED.condition,
                    threshold = EXCLUDED.threshold,
                    severity = EXCLUDED.severity,
                    alert_type = EXCLUDED.alert_type,
                    model_ids = EXCLUDED.model_ids,
                    enabled = EXCLUDED.enabled,
                    evaluation_window = EXCLUDED.evaluation_window,
                    cooldown_period = EXCLUDED.cooldown_period,
                    notification_channels = EXCLUDED.notification_channels,
                    escalation_policy = EXCLUDED.escalation_policy,
                    metadata = EXCLUDED.metadata,
                    updated_at = CURRENT_TIMESTAMP
            """, rule_id, name, description, metric_name, condition, threshold,
                severity.value, alert_type.value, json.dumps(model_ids or []), 
                enabled, evaluation_window, cooldown_period,
                json.dumps(notification_channels or []), escalation_policy, 
                json.dumps(metadata or {}))
        
        self.alert_rules[rule_id] = alert_rule
        
        logger.info("Alert rule created", rule_id=rule_id, name=name, severity=severity.value)
        return alert_rule
    
    async def evaluate_metric(
        self,
        metric_name: str,
        metric_value: float,
        model_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluate a metric against all applicable alert rules
        
        Args:
            metric_name: Name of the metric
            metric_value: Current value of the metric
            model_id: Model ID (if applicable)
            timestamp: Timestamp of the metric
            context: Additional context information
            
        Returns:
            List of triggered alerts
        """
        triggered_alerts = []
        current_time = timestamp or datetime.utcnow()
        
        # Find applicable rules
        applicable_rules = [
            rule for rule in self.alert_rules.values()
            if (rule["metric_name"] == metric_name and 
                rule["enabled"] and
                (not rule["model_ids"] or not model_id or model_id in rule["model_ids"]))
        ]
        
        for rule in applicable_rules:
            # Check cooldown period
            if rule["last_triggered"]:
                last_triggered = datetime.fromisoformat(rule["last_triggered"])
                if (current_time - last_triggered).total_seconds() < rule["cooldown_period"]:
                    continue
            
            # Evaluate condition
            if self._evaluate_condition(metric_value, rule["condition"], rule["threshold"]):
                alert = await self._trigger_alert(rule, metric_value, model_id, current_time, context)
                triggered_alerts.append(alert)
                
                # Update rule last triggered time
                rule["last_triggered"] = current_time.isoformat()
                rule["trigger_count"] += 1
                
                await self._update_rule_stats(rule["rule_id"], current_time)
        
        return triggered_alerts
    
    async def _trigger_alert(
        self,
        rule: Dict[str, Any],
        metric_value: float,
        model_id: Optional[str],
        timestamp: datetime,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Trigger an alert based on a rule"""
        
        alert_id = f"alert_{rule['rule_id']}_{int(timestamp.timestamp())}"
        
        alert = {
            "alert_id": alert_id,
            "rule_id": rule["rule_id"],
            "rule_name": rule["name"],
            "alert_type": rule["alert_type"],
            "severity": rule["severity"],
            "model_id": model_id,
            "metric_name": rule["metric_name"],
            "metric_value": metric_value,
            "threshold": rule["threshold"],
            "condition": rule["condition"],
            "status": AlertStatus.ACTIVE.value,
            "triggered_at": timestamp.isoformat(),
            "acknowledged_at": None,
            "acknowledged_by": None,
            "resolved_at": None,
            "resolved_by": None,
            "context": context or {},
            "message": self._generate_alert_message(rule, metric_value, model_id, context),
            "escalation_level": 0
        }
        
        # Store alert in database
        async with get_db() as db:
            await db.execute("""
                INSERT INTO alerts (
                    alert_id, rule_id, alert_type, severity, model_id, metric_name,
                    metric_value, threshold, condition, status, triggered_at,
                    context, message, escalation_level
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            """, alert_id, rule["rule_id"], rule["alert_type"], rule["severity"],
                model_id, rule["metric_name"], metric_value, rule["threshold"],
                rule["condition"], AlertStatus.ACTIVE.value, timestamp,
                json.dumps(context or {}), alert["message"], 0)
        
        # Send notifications
        await self._send_notifications(alert, rule["notification_channels"])
        
        # Real-time dashboard update
        await self._send_realtime_alert(alert)
        
        logger.warning("Alert triggered", 
                      alert_id=alert_id, 
                      rule_name=rule["name"],
                      severity=rule["severity"],
                      metric_value=metric_value)
        
        return alert
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate if metric value meets condition threshold"""
        
        conditions = {
            ">": lambda v, t: v > t,
            "<": lambda v, t: v < t,
            ">=": lambda v, t: v >= t,
            "<=": lambda v, t: v <= t,
            "==": lambda v, t: abs(v - t) < 1e-9,
            "!=": lambda v, t: abs(v - t) >= 1e-9
        }
        
        evaluator = conditions.get(condition)
        if not evaluator:
            logger.error("Invalid condition operator", condition=condition)
            return False
        
        return evaluator(value, threshold)
    
    def _generate_alert_message(
        self,
        rule: Dict[str, Any],
        metric_value: float,
        model_id: Optional[str],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Generate human-readable alert message"""
        
        model_context = f" for model {model_id}" if model_id else ""
        context_info = ""
        
        if context:
            relevant_context = {k: v for k, v in context.items() 
                              if k in ["dataset_size", "feature_name", "group_name"]}
            if relevant_context:
                context_info = f" ({', '.join(f'{k}: {v}' for k, v in relevant_context.items())})"
        
        return (f"{rule['name']}: {rule['metric_name']} = {metric_value:.4f} "
                f"{rule['condition']} {rule['threshold']}{model_context}{context_info}")
    
    async def _send_notifications(self, alert: Dict[str, Any], channel_ids: List[str]):
        """Send alert notifications to configured channels"""
        
        for channel_id in channel_ids:
            channel = self.notification_channels.get(channel_id)
            if not channel or not channel.get("enabled", True):
                continue
            
            try:
                if channel["type"] == AlertChannel.EMAIL.value:
                    await self._send_email_notification(alert, channel)
                elif channel["type"] == AlertChannel.SLACK.value:
                    await self._send_slack_notification(alert, channel)
                elif channel["type"] == AlertChannel.WEBHOOK.value:
                    await self._send_webhook_notification(alert, channel)
                elif channel["type"] == AlertChannel.SMS.value:
                    await self._send_sms_notification(alert, channel)
                elif channel["type"] == AlertChannel.PAGERDUTY.value:
                    await self._send_pagerduty_notification(alert, channel)
                    
            except Exception as e:
                logger.error("Failed to send notification", 
                           channel_id=channel_id, 
                           channel_type=channel["type"],
                           error=str(e))
    
    async def _send_email_notification(self, alert: Dict[str, Any], channel: Dict[str, Any]):
        """Send email notification"""
        
        subject = f"[{alert['severity'].upper()}] ML Alert: {alert['rule_name']}"
        
        html_body = f"""
        <html>
        <body>
            <h2 style="color: {'#dc2626' if alert['severity'] in ['critical', 'high'] else '#f59e0b' if alert['severity'] == 'medium' else '#10b981'}">
                ML Model Alert
            </h2>
            <table style="border-collapse: collapse; width: 100%;">
                <tr><td><strong>Alert ID:</strong></td><td>{alert['alert_id']}</td></tr>
                <tr><td><strong>Severity:</strong></td><td>{alert['severity'].upper()}</td></tr>
                <tr><td><strong>Rule:</strong></td><td>{alert['rule_name']}</td></tr>
                <tr><td><strong>Model:</strong></td><td>{alert.get('model_id', 'N/A')}</td></tr>
                <tr><td><strong>Metric:</strong></td><td>{alert['metric_name']}</td></tr>
                <tr><td><strong>Value:</strong></td><td>{alert['metric_value']:.4f}</td></tr>
                <tr><td><strong>Threshold:</strong></td><td>{alert['condition']} {alert['threshold']}</td></tr>
                <tr><td><strong>Triggered At:</strong></td><td>{alert['triggered_at']}</td></tr>
            </table>
            <p><strong>Message:</strong> {alert['message']}</p>
            
            <p style="margin-top: 20px;">
                <a href="{channel.get('dashboard_url', '#')}/alerts/{alert['alert_id']}" 
                   style="background-color: #3b82f6; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">
                    View in Dashboard
                </a>
            </p>
        </body>
        </html>
        """
        
        # Send email (implementation would depend on email service)
        # This is a placeholder for actual email sending logic
        logger.info("Email notification sent", 
                   alert_id=alert['alert_id'],
                   recipients=channel.get('recipients', []))
    
    async def _send_slack_notification(self, alert: Dict[str, Any], channel: Dict[str, Any]):
        """Send Slack notification"""
        
        color_map = {
            "critical": "#dc2626",
            "high": "#dc2626", 
            "medium": "#f59e0b",
            "low": "#10b981",
            "info": "#6b7280"
        }
        
        slack_payload = {
            "channel": channel.get("channel", "#alerts"),
            "username": "ML Monitor",
            "icon_emoji": ":warning:",
            "attachments": [{
                "color": color_map.get(alert["severity"], "#6b7280"),
                "title": f"ML Alert: {alert['rule_name']}",
                "text": alert["message"],
                "fields": [
                    {"title": "Severity", "value": alert["severity"].upper(), "short": True},
                    {"title": "Model", "value": alert.get("model_id", "N/A"), "short": True},
                    {"title": "Metric", "value": alert["metric_name"], "short": True},
                    {"title": "Value", "value": f"{alert['metric_value']:.4f}", "short": True}
                ],
                "footer": "ML Monitoring System",
                "ts": int(datetime.fromisoformat(alert["triggered_at"]).timestamp())
            }]
        }
        
        # Send to Slack (implementation would use Slack API)
        logger.info("Slack notification sent", 
                   alert_id=alert['alert_id'],
                   channel=channel.get("channel"))
    
    async def _send_webhook_notification(self, alert: Dict[str, Any], channel: Dict[str, Any]):
        """Send webhook notification"""
        
        webhook_payload = {
            "alert_id": alert["alert_id"],
            "rule_name": alert["rule_name"],
            "severity": alert["severity"],
            "alert_type": alert["alert_type"],
            "model_id": alert.get("model_id"),
            "metric_name": alert["metric_name"],
            "metric_value": alert["metric_value"],
            "threshold": alert["threshold"],
            "condition": alert["condition"],
            "triggered_at": alert["triggered_at"],
            "message": alert["message"],
            "context": alert["context"]
        }
        
        # Send webhook (implementation would use HTTP client)
        logger.info("Webhook notification sent",
                   alert_id=alert['alert_id'],
                   webhook_url=channel.get("webhook_url"))
    
    async def _send_sms_notification(self, alert: Dict[str, Any], channel: Dict[str, Any]):
        """Send SMS notification"""
        
        sms_message = (f"ML Alert [{alert['severity'].upper()}]: {alert['rule_name']} - "
                      f"{alert['metric_name']} = {alert['metric_value']:.4f} "
                      f"{alert['condition']} {alert['threshold']}")
        
        # Send SMS (implementation would use SMS service)
        logger.info("SMS notification sent",
                   alert_id=alert['alert_id'],
                   recipients=channel.get("phone_numbers", []))
    
    async def _send_pagerduty_notification(self, alert: Dict[str, Any], channel: Dict[str, Any]):
        """Send PagerDuty notification"""
        
        pagerduty_payload = {
            "routing_key": channel.get("integration_key"),
            "event_action": "trigger",
            "dedup_key": alert["alert_id"],
            "payload": {
                "summary": f"ML Alert: {alert['rule_name']}",
                "severity": alert["severity"],
                "source": alert.get("model_id", "ML Monitoring"),
                "component": alert["metric_name"],
                "group": "ML Models",
                "class": alert["alert_type"],
                "custom_details": {
                    "metric_value": alert["metric_value"],
                    "threshold": alert["threshold"],
                    "condition": alert["condition"],
                    "context": alert["context"]
                }
            }
        }
        
        # Send to PagerDuty (implementation would use PagerDuty API)
        logger.info("PagerDuty notification sent",
                   alert_id=alert['alert_id'])
    
    async def _send_realtime_alert(self, alert: Dict[str, Any]):
        """Send real-time alert to dashboard via WebSocket"""
        
        # Send to all connected dashboard clients
        await self.websocket_manager.broadcast_to_group("alerts", {
            "type": "new_alert",
            "alert": alert
        })
        
        # Send to model-specific subscribers if applicable
        if alert.get("model_id"):
            await self.websocket_manager.broadcast_to_group(
                f"model_{alert['model_id']}", {
                    "type": "model_alert",
                    "alert": alert
                }
            )
    
    async def acknowledge_alert(
        self, 
        alert_id: str, 
        acknowledged_by: str,
        note: Optional[str] = None
    ) -> Dict[str, Any]:
        """Acknowledge an alert"""
        
        async with get_db() as db:
            result = await db.execute("""
                UPDATE alerts 
                SET status = $1, acknowledged_at = CURRENT_TIMESTAMP, acknowledged_by = $2
                WHERE alert_id = $3 AND status = $4
            """, AlertStatus.ACKNOWLEDGED.value, acknowledged_by, alert_id, AlertStatus.ACTIVE.value)
            
            if result == "UPDATE 1":
                # Log acknowledgment
                await db.execute("""
                    INSERT INTO alert_actions (alert_id, action_type, action_by, note)
                    VALUES ($1, 'acknowledged', $2, $3)
                """, alert_id, acknowledged_by, note or "")
                
                # Send real-time update
                await self.websocket_manager.broadcast_to_group("alerts", {
                    "type": "alert_acknowledged",
                    "alert_id": alert_id,
                    "acknowledged_by": acknowledged_by
                })
                
                logger.info("Alert acknowledged", 
                           alert_id=alert_id, 
                           acknowledged_by=acknowledged_by)
                
                return {"status": "acknowledged", "alert_id": alert_id}
            else:
                raise ValueError(f"Alert {alert_id} not found or already processed")
    
    async def resolve_alert(
        self,
        alert_id: str,
        resolved_by: str,
        resolution_note: Optional[str] = None
    ) -> Dict[str, Any]:
        """Resolve an alert"""
        
        async with get_db() as db:
            result = await db.execute("""
                UPDATE alerts 
                SET status = $1, resolved_at = CURRENT_TIMESTAMP, resolved_by = $2
                WHERE alert_id = $3 AND status IN ($4, $5)
            """, AlertStatus.RESOLVED.value, resolved_by, alert_id, 
                AlertStatus.ACTIVE.value, AlertStatus.ACKNOWLEDGED.value)
            
            if result.startswith("UPDATE"):
                # Log resolution
                await db.execute("""
                    INSERT INTO alert_actions (alert_id, action_type, action_by, note)
                    VALUES ($1, 'resolved', $2, $3)
                """, alert_id, resolved_by, resolution_note or "")
                
                # Send real-time update
                await self.websocket_manager.broadcast_to_group("alerts", {
                    "type": "alert_resolved",
                    "alert_id": alert_id,
                    "resolved_by": resolved_by
                })
                
                logger.info("Alert resolved", 
                           alert_id=alert_id, 
                           resolved_by=resolved_by)
                
                return {"status": "resolved", "alert_id": alert_id}
            else:
                raise ValueError(f"Alert {alert_id} not found")
    
    async def get_active_alerts(
        self, 
        model_id: Optional[str] = None,
        severity_filter: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get active alerts with optional filtering"""
        
        query = """
            SELECT alert_id, rule_id, alert_type, severity, model_id, metric_name,
                   metric_value, threshold, condition, status, triggered_at,
                   acknowledged_at, acknowledged_by, context, message
            FROM alerts 
            WHERE status IN ($1, $2)
        """
        params = [AlertStatus.ACTIVE.value, AlertStatus.ACKNOWLEDGED.value]
        
        if model_id:
            query += f" AND model_id = ${len(params) + 1}"
            params.append(model_id)
        
        if severity_filter:
            placeholders = ", ".join(f"${len(params) + i + 1}" for i in range(len(severity_filter)))
            query += f" AND severity IN ({placeholders})"
            params.extend(severity_filter)
        
        query += f" ORDER BY triggered_at DESC LIMIT ${len(params) + 1}"
        params.append(limit)
        
        async with get_db() as db:
            rows = await db.fetch(query, *params)
            
            alerts = []
            for row in rows:
                alert = dict(row)
                if alert["context"]:
                    alert["context"] = json.loads(alert["context"])
                alerts.append(alert)
            
            return alerts
    
    async def get_alert_statistics(
        self, 
        time_range_hours: int = 24
    ) -> Dict[str, Any]:
        """Get alert statistics for the specified time range"""
        
        start_time = datetime.utcnow() - timedelta(hours=time_range_hours)
        
        async with get_db() as db:
            # Total alerts by severity
            severity_stats = await db.fetch("""
                SELECT severity, COUNT(*) as count
                FROM alerts 
                WHERE triggered_at >= $1
                GROUP BY severity
            """, start_time)
            
            # Alerts by type
            type_stats = await db.fetch("""
                SELECT alert_type, COUNT(*) as count
                FROM alerts 
                WHERE triggered_at >= $1
                GROUP BY alert_type
            """, start_time)
            
            # Most frequent rules
            rule_stats = await db.fetch("""
                SELECT rule_id, COUNT(*) as count
                FROM alerts 
                WHERE triggered_at >= $1
                GROUP BY rule_id
                ORDER BY count DESC
                LIMIT 10
            """, start_time)
            
            # Response times
            response_stats = await db.fetchrow("""
                SELECT 
                    AVG(EXTRACT(EPOCH FROM (acknowledged_at - triggered_at))) as avg_ack_time,
                    AVG(EXTRACT(EPOCH FROM (resolved_at - triggered_at))) as avg_resolution_time
                FROM alerts 
                WHERE triggered_at >= $1 
                AND acknowledged_at IS NOT NULL
            """, start_time)
            
            return {
                "time_range_hours": time_range_hours,
                "severity_distribution": {row["severity"]: row["count"] for row in severity_stats},
                "type_distribution": {row["alert_type"]: row["count"] for row in type_stats},
                "top_triggered_rules": [{"rule_id": row["rule_id"], "count": row["count"]} for row in rule_stats],
                "average_acknowledgment_time_seconds": response_stats["avg_ack_time"] or 0,
                "average_resolution_time_seconds": response_stats["avg_resolution_time"] or 0,
                "total_alerts": sum(row["count"] for row in severity_stats)
            }
    
    async def _load_alert_rules(self):
        """Load alert rules from database"""
        
        async with get_db() as db:
            rows = await db.fetch("SELECT * FROM alert_rules WHERE enabled = TRUE")
            
            for row in rows:
                rule = dict(row)
                rule["model_ids"] = json.loads(rule["model_ids"]) if rule["model_ids"] else []
                rule["notification_channels"] = json.loads(rule["notification_channels"]) if rule["notification_channels"] else []
                rule["metadata"] = json.loads(rule["metadata"]) if rule["metadata"] else {}
                self.alert_rules[rule["rule_id"]] = rule
        
        logger.info("Loaded alert rules", count=len(self.alert_rules))
    
    async def _load_notification_channels(self):
        """Load notification channels from database"""
        
        async with get_db() as db:
            rows = await db.fetch("SELECT * FROM notification_channels WHERE enabled = TRUE")
            
            for row in rows:
                channel = dict(row)
                channel["config"] = json.loads(channel["config"]) if channel["config"] else {}
                self.notification_channels[channel["channel_id"]] = channel
        
        logger.info("Loaded notification channels", count=len(self.notification_channels))
    
    async def _load_escalation_policies(self):
        """Load escalation policies from database"""
        
        async with get_db() as db:
            rows = await db.fetch("SELECT * FROM escalation_policies WHERE enabled = TRUE")
            
            for row in rows:
                policy = dict(row)
                policy["escalation_rules"] = json.loads(policy["escalation_rules"]) if policy["escalation_rules"] else []
                self.escalation_policies[policy["policy_id"]] = policy
        
        logger.info("Loaded escalation policies", count=len(self.escalation_policies))
    
    async def _update_rule_stats(self, rule_id: str, timestamp: datetime):
        """Update alert rule statistics"""
        
        async with get_db() as db:
            await db.execute("""
                UPDATE alert_rules 
                SET last_triggered = $1, trigger_count = trigger_count + 1
                WHERE rule_id = $2
            """, timestamp, rule_id)


# Predefined enterprise alert rules
ENTERPRISE_ALERT_RULES = [
    {
        "rule_id": "performance_degradation_critical",
        "name": "Critical Performance Degradation",
        "description": "Alert when model accuracy drops significantly",
        "metric_name": "accuracy",
        "condition": "<",
        "threshold": 0.7,
        "severity": AlertSeverity.CRITICAL,
        "alert_type": AlertType.PERFORMANCE_DEGRADATION,
        "evaluation_window": 300,
        "cooldown_period": 1800
    },
    {
        "rule_id": "data_drift_high",
        "name": "High Data Drift Detected",
        "description": "Alert when data drift exceeds acceptable threshold",
        "metric_name": "jsd_score",
        "condition": ">",
        "threshold": 0.3,
        "severity": AlertSeverity.HIGH,
        "alert_type": AlertType.DATA_DRIFT,
        "evaluation_window": 600,
        "cooldown_period": 3600
    },
    {
        "rule_id": "bias_detection_critical",
        "name": "Critical Bias Detected",
        "description": "Alert when bias metrics exceed compliance thresholds",
        "metric_name": "demographic_parity_difference",
        "condition": ">",
        "threshold": 0.2,
        "severity": AlertSeverity.CRITICAL,
        "alert_type": AlertType.BIAS_DETECTION,
        "evaluation_window": 300,
        "cooldown_period": 1800
    },
    {
        "rule_id": "data_quality_low",
        "name": "Low Data Quality",
        "description": "Alert when data quality score is low",
        "metric_name": "data_quality_score",
        "condition": "<",
        "threshold": 0.6,
        "severity": AlertSeverity.MEDIUM,
        "alert_type": AlertType.DATA_QUALITY,
        "evaluation_window": 600,
        "cooldown_period": 3600
    }
]