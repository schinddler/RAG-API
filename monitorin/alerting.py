import asyncio
import time
import logging
import json
import os
import smtplib
import threading
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib

# Import dependencies (assuming these exist in the project)
try:
    from .logs import log_alert_event
except ImportError:
    def log_alert_event(alert_type: str, message: str, metadata: dict = None, **kwargs):
        logging.info(f"Alert Event - Type: {alert_type}, Message: {message}")


class AlertType(Enum):
    """Alert types for consistent categorization"""
    FALLBACK_TRIGGERED = "fallback_triggered"
    TOKEN_OVERUSE = "token_overuse"
    LATENCY_EXCEEDED = "latency_exceeded"
    API_FAILURE = "api_failure"
    RATE_LIMIT_HIT = "rate_limit_hit"
    QUOTA_EXHAUSTED = "quota_exhausted"
    COST_OVERFLOW = "cost_overflow"
    SYSTEM_DEGRADED = "system_degraded"
    CRITICAL_ERROR = "critical_error"
    WARNING = "warning"


@dataclass
class Alert:
    """Structured alert data"""
    alert_type: AlertType
    message: str
    metadata: Dict[str, Any]
    timestamp: datetime
    severity: str = "info"
    deduplication_key: Optional[str] = None
    
    def __post_init__(self):
        if self.deduplication_key is None:
            # Generate deduplication key from content
            content = f"{self.alert_type.value}:{self.message}:{json.dumps(self.metadata, sort_keys=True)}"
            self.deduplication_key = hashlib.md5(content.encode()).hexdigest()


class AlertChannel:
    """Base class for alert channels"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", True)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert through this channel"""
        if not self.enabled:
            return True
        
        try:
            return await self._send(alert)
        except Exception as e:
            self.logger.error(f"Failed to send alert via {self.__class__.__name__}: {e}")
            return False
    
    async def _send(self, alert: Alert) -> bool:
        """Override in subclasses"""
        raise NotImplementedError


class LogChannel(AlertChannel):
    """Log-based alert channel"""
    
    async def _send(self, alert: Alert) -> bool:
        """Send alert to logs"""
        log_level = getattr(logging, alert.severity.upper(), logging.INFO)
        self.logger.log(log_level, f"ALERT [{alert.alert_type.value}]: {alert.message}")
        
        # Log to monitoring system
        log_alert_event(
            alert_type=alert.alert_type.value,
            message=alert.message,
            metadata=alert.metadata,
            severity=alert.severity,
            timestamp=alert.timestamp.isoformat()
        )
        return True


class EmailChannel(AlertChannel):
    """Email-based alert channel"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.smtp_server = config.get("smtp_server", "localhost")
        self.smtp_port = config.get("smtp_port", 587)
        self.username = config.get("username")
        self.password = config.get("password")
        self.from_email = config.get("from_email")
        self.to_emails = config.get("to_emails", [])
        self.use_tls = config.get("use_tls", True)
    
    async def _send(self, alert: Alert) -> bool:
        """Send alert via email"""
        if not self.from_email or not self.to_emails:
            self.logger.warning("Email channel not configured properly")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ", ".join(self.to_emails)
            msg['Subject'] = f"RAG Alert: {alert.alert_type.value.upper()}"
            
            # Create body
            body = f"""
Alert Type: {alert.alert_type.value}
Severity: {alert.severity.upper()}
Timestamp: {alert.timestamp.isoformat()}
Message: {alert.message}

Metadata:
{json.dumps(alert.metadata, indent=2)}
"""
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                if self.username and self.password:
                    server.login(self.username, self.password)
                server.send_message(msg)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
            return False


class SlackChannel(AlertChannel):
    """Slack webhook-based alert channel"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.webhook_url = config.get("webhook_url") or os.getenv("SLACK_WEBHOOK_URL")
        self.channel = config.get("channel", "#alerts")
        self.username = config.get("username", "RAG Alert Bot")
    
    async def _send(self, alert: Alert) -> bool:
        """Send alert via Slack webhook"""
        if not self.webhook_url:
            self.logger.warning("Slack webhook URL not configured")
            return False
        
        try:
            # Create Slack message
            color = {
                "critical": "#ff0000",
                "error": "#ff6b6b",
                "warning": "#ffa500",
                "info": "#4ecdc4"
            }.get(alert.severity, "#4ecdc4")
            
            payload = {
                "channel": self.channel,
                "username": self.username,
                "attachments": [{
                    "color": color,
                    "title": f"RAG Alert: {alert.alert_type.value.replace('_', ' ').title()}",
                    "text": alert.message,
                    "fields": [
                        {"title": "Severity", "value": alert.severity.upper(), "short": True},
                        {"title": "Timestamp", "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S"), "short": True}
                    ],
                    "footer": "RAG System Alert"
                }]
            }
            
            # Add metadata as additional fields
            if alert.metadata:
                for key, value in alert.metadata.items():
                    payload["attachments"][0]["fields"].append({
                        "title": key.replace("_", " ").title(),
                        "value": str(value),
                        "short": True
                    })
            
            # Send to Slack
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send Slack alert: {e}")
            return False


class PagerDutyChannel(AlertChannel):
    """PagerDuty-based alert channel"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key") or os.getenv("PAGERDUTY_API_KEY")
        self.service_id = config.get("service_id") or os.getenv("PAGERDUTY_SERVICE_ID")
        self.api_url = "https://api.pagerduty.com/v2/incidents"
    
    async def _send(self, alert: Alert) -> bool:
        """Send alert via PagerDuty API"""
        if not self.api_key or not self.service_id:
            self.logger.warning("PagerDuty not configured properly")
            return False
        
        try:
            # Determine urgency based on severity
            urgency = "high" if alert.severity in ["critical", "error"] else "low"
            
            payload = {
                "incident": {
                    "type": "incident",
                    "title": f"RAG Alert: {alert.alert_type.value.replace('_', ' ').title()}",
                    "service": {"id": self.service_id, "type": "service_reference"},
                    "urgency": urgency,
                    "body": {
                        "type": "incident_body",
                        "details": f"{alert.message}\n\nMetadata: {json.dumps(alert.metadata)}"
                    }
                }
            }
            
            headers = {
                "Authorization": f"Token token={self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/vnd.pagerduty+json;version=2"
            }
            
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send PagerDuty alert: {e}")
            return False


class AlertManager:
    """
    Central alerting system for RAG operations with multiple channels and deduplication.
    Provides non-blocking, batched alert delivery with retry logic.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the alert manager.
        
        Args:
            config: Configuration dictionary with channel settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize channels
        self.channels = self._init_channels()
        
        # Alert deduplication
        self.recent_alerts = {}
        self.deduplication_window = timedelta(minutes=5)
        
        # Batching
        self.alert_queue = []
        self.batch_size = config.get("batch_size", 10)
        self.batch_timeout = config.get("batch_timeout", 30)  # seconds
        
        # Start background processing
        self._start_background_processor()
    
    def _init_channels(self) -> List[AlertChannel]:
        """Initialize alert channels based on configuration"""
        channels = []
        
        # Log channel (always enabled)
        channels.append(LogChannel({"enabled": True}))
        
        # Email channel
        if self.config.get("email", {}).get("enabled", False):
            channels.append(EmailChannel(self.config.get("email", {})))
        
        # Slack channel
        if self.config.get("slack", {}).get("enabled", False):
            channels.append(SlackChannel(self.config.get("slack", {})))
        
        # PagerDuty channel
        if self.config.get("pagerduty", {}).get("enabled", False):
            channels.append(PagerDutyChannel(self.config.get("pagerduty", {})))
        
        return channels
    
    def _start_background_processor(self):
        """Start background thread for processing alerts"""
        def processor():
            while True:
                try:
                    if self.alert_queue:
                        self._process_batch()
                    time.sleep(1)
                except Exception as e:
                    self.logger.error(f"Background alert processor error: {e}")
        
        thread = threading.Thread(target=processor, daemon=True)
        thread.start()
    
    def _process_batch(self):
        """Process a batch of alerts"""
        if not self.alert_queue:
            return
        
        # Take up to batch_size alerts
        batch = self.alert_queue[:self.batch_size]
        self.alert_queue = self.alert_queue[self.batch_size:]
        
        # Send alerts in parallel
        asyncio.run(self._send_batch(batch))
    
    async def _send_batch(self, alerts: List[Alert]):
        """Send a batch of alerts through all channels"""
        tasks = []
        
        for alert in alerts:
            for channel in self.channels:
                task = channel.send_alert(alert)
                tasks.append(task)
        
        # Wait for all sends to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log results
        success_count = sum(1 for result in results if result is True)
        self.logger.info(f"Sent {len(alerts)} alerts through {len(self.channels)} channels: {success_count} successful")
    
    def send_alert(self, event_type: str, message: str, metadata: Dict[str, Any] = None) -> None:
        """
        Send an alert (non-blocking).
        
        Args:
            event_type: Type of alert event
            message: Alert message
            metadata: Additional metadata
        """
        try:
            # Create alert
            alert_type = AlertType(event_type) if event_type in [e.value for e in AlertType] else AlertType.WARNING
            alert = Alert(
                alert_type=alert_type,
                message=message,
                metadata=metadata or {},
                timestamp=datetime.now(),
                severity=self._determine_severity(alert_type)
            )
            
            # Check deduplication
            if self._is_duplicate(alert):
                self.logger.debug(f"Dropping duplicate alert: {alert.deduplication_key}")
                return
            
            # Add to queue
            self.alert_queue.append(alert)
            self.recent_alerts[alert.deduplication_key] = datetime.now()
            
            # Clean old deduplication entries
            self._cleanup_old_alerts()
            
        except Exception as e:
            self.logger.error(f"Failed to create alert: {e}")
    
    def alert_on_fallback(self, primary: str, fallback: str, reason: str) -> None:
        """Alert when LLM fallback is triggered"""
        metadata = {
            "primary_model": primary,
            "fallback_model": fallback,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }
        
        self.send_alert(
            event_type="fallback_triggered",
            message=f"LLM fallback triggered: {primary} -> {fallback} due to {reason}",
            metadata=metadata
        )
    
    def alert_on_token_overuse(self, model: str, token_count: int, threshold: int) -> None:
        """Alert when token usage exceeds threshold"""
        metadata = {
            "model": model,
            "token_count": token_count,
            "threshold": threshold,
            "usage_percentage": (token_count / threshold) * 100
        }
        
        self.send_alert(
            event_type="token_overuse",
            message=f"Token usage exceeded threshold: {model} used {token_count} tokens (threshold: {threshold})",
            metadata=metadata
        )
    
    def alert_on_latency(self, model: str, latency_ms: float, budget_ms: float) -> None:
        """Alert when latency exceeds budget"""
        metadata = {
            "model": model,
            "latency_ms": latency_ms,
            "budget_ms": budget_ms,
            "exceeded_by": latency_ms - budget_ms
        }
        
        self.send_alert(
            event_type="latency_exceeded",
            message=f"Latency exceeded budget: {model} took {latency_ms:.2f}ms (budget: {budget_ms:.2f}ms)",
            metadata=metadata
        )
    
    def alert_on_api_failure(self, model: str, error: Exception) -> None:
        """Alert when API call fails"""
        metadata = {
            "model": model,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.now().isoformat()
        }
        
        self.send_alert(
            event_type="api_failure",
            message=f"API failure for {model}: {type(error).__name__}: {str(error)}",
            metadata=metadata
        )
    
    def alert_on_rate_limit(self, model: str, retry_after: Optional[int] = None) -> None:
        """Alert when rate limit is hit"""
        metadata = {
            "model": model,
            "retry_after": retry_after,
            "timestamp": datetime.now().isoformat()
        }
        
        self.send_alert(
            event_type="rate_limit_hit",
            message=f"Rate limit hit for {model}" + (f" (retry after {retry_after}s)" if retry_after else ""),
            metadata=metadata
        )
    
    def alert_on_quota_exhausted(self, model: str, quota_type: str) -> None:
        """Alert when API quota is exhausted"""
        metadata = {
            "model": model,
            "quota_type": quota_type,
            "timestamp": datetime.now().isoformat()
        }
        
        self.send_alert(
            event_type="quota_exhausted",
            message=f"Quota exhausted for {model}: {quota_type}",
            metadata=metadata
        )
    
    def alert_on_cost_overflow(self, model: str, cost: float, budget: float) -> None:
        """Alert when cost exceeds budget"""
        metadata = {
            "model": model,
            "cost": cost,
            "budget": budget,
            "exceeded_by": cost - budget
        }
        
        self.send_alert(
            event_type="cost_overflow",
            message=f"Cost exceeded budget: {model} cost ${cost:.4f} (budget: ${budget:.4f})",
            metadata=metadata
        )
    
    def _determine_severity(self, alert_type: AlertType) -> str:
        """Determine alert severity based on type"""
        severity_map = {
            AlertType.CRITICAL_ERROR: "critical",
            AlertType.API_FAILURE: "error",
            AlertType.RATE_LIMIT_HIT: "warning",
            AlertType.QUOTA_EXHAUSTED: "error",
            AlertType.FALLBACK_TRIGGERED: "warning",
            AlertType.TOKEN_OVERUSE: "warning",
            AlertType.LATENCY_EXCEEDED: "warning",
            AlertType.COST_OVERFLOW: "warning",
            AlertType.SYSTEM_DEGRADED: "error",
            AlertType.WARNING: "warning"
        }
        return severity_map.get(alert_type, "info")
    
    def _is_duplicate(self, alert: Alert) -> bool:
        """Check if alert is a duplicate within the deduplication window"""
        if alert.deduplication_key not in self.recent_alerts:
            return False
        
        last_seen = self.recent_alerts[alert.deduplication_key]
        return datetime.now() - last_seen < self.deduplication_window
    
    def _cleanup_old_alerts(self):
        """Clean up old deduplication entries"""
        cutoff = datetime.now() - self.deduplication_window
        self.recent_alerts = {
            key: timestamp for key, timestamp in self.recent_alerts.items()
            if timestamp > cutoff
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get alert manager statistics"""
        return {
            "channels": len(self.channels),
            "queue_size": len(self.alert_queue),
            "recent_alerts": len(self.recent_alerts),
            "enabled_channels": [c.__class__.__name__ for c in self.channels if c.enabled]
        }


# Global instance for easy access
_alert_manager = None

def get_alert_manager(config: Optional[Dict[str, Any]] = None) -> AlertManager:
    """Get the global alert manager instance"""
    global _alert_manager
    if _alert_manager is None:
        if config is None:
            config = {
                "email": {"enabled": False},
                "slack": {"enabled": False},
                "pagerduty": {"enabled": False},
                "batch_size": 10,
                "batch_timeout": 30
            }
        _alert_manager = AlertManager(config)
    return _alert_manager


# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    EXAMPLE_CONFIG = {
        "email": {
            "enabled": False,
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "username": "alerts@example.com",
            "password": "password",
            "from_email": "alerts@example.com",
            "to_emails": ["admin@example.com"]
        },
        "slack": {
            "enabled": False,
            "webhook_url": "https://hooks.slack.com/services/...",
            "channel": "#alerts"
        },
        "pagerduty": {
            "enabled": False,
            "api_key": "pd_key_...",
            "service_id": "service_id"
        },
        "batch_size": 5,
        "batch_timeout": 10
    }
    
    # Test the alert manager
    manager = AlertManager(EXAMPLE_CONFIG)
    
    # Test various alerts
    manager.alert_on_fallback("gpt-4", "claude", "timeout")
    manager.alert_on_token_overuse("gpt-4", 120000, 128000)
    manager.alert_on_latency("claude", 3500, 2500)
    manager.alert_on_api_failure("mistral", Exception("Connection timeout"))
    manager.alert_on_rate_limit("gpt-4", 60)
    
    print(f"Alert manager stats: {manager.get_stats()}")
