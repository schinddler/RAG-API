import logging
import json
import os
import time
import threading
import queue
import contextvars
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Union
from pathlib import Path
import traceback
from logging.handlers import RotatingFileHandler
import uuid

# Context variable for correlation ID
correlation_id = contextvars.ContextVar('correlation_id', default=None)

# Global logger instance
_logger_instance = None
_logger_lock = threading.Lock()


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record):
        # Create base log entry
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
            "correlation_id": correlation_id.get(),
            "thread_id": record.thread,
            "process_id": record.process
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields from record
        if hasattr(record, 'metadata'):
            log_entry["metadata"] = record.metadata
        
        if hasattr(record, 'event_type'):
            log_entry["event_type"] = record.event_type
        
        if hasattr(record, 'severity'):
            log_entry["severity"] = record.severity
        
        if hasattr(record, 'duration_ms'):
            log_entry["duration_ms"] = record.duration_ms
        
        if hasattr(record, 'action'):
            log_entry["action"] = record.action
        
        if hasattr(record, 'details'):
            log_entry["details"] = record.details
        
        return json.dumps(log_entry, ensure_ascii=False)


class AsyncLogHandler(logging.Handler):
    """Asynchronous log handler with queue-based buffering"""
    
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
        self.queue = queue.Queue(maxsize=10000)
        self.thread = threading.Thread(target=self._process_logs, daemon=True)
        self.thread.start()
        self.running = True
    
    def emit(self, record):
        try:
            # Non-blocking put with timeout
            self.queue.put(record, timeout=0.1)
        except queue.Full:
            # If queue is full, log to stderr as fallback
            print(f"Log queue full, dropping log entry: {record.getMessage()}")
    
    def _process_logs(self):
        """Background thread to process log entries"""
        while self.running:
            try:
                # Get log entry with timeout
                record = self.queue.get(timeout=1)
                if record is None:
                    break
                
                # Format and write to handlers
                formatted = self.format(record)
                for handler in self.handlers:
                    try:
                        handler.write(formatted + '\n')
                        handler.flush()
                    except Exception as e:
                        print(f"Failed to write log: {e}")
                
                self.queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in log processor: {e}")
    
    def close(self):
        """Close the async handler"""
        self.running = False
        self.queue.put(None)  # Signal shutdown
        self.thread.join(timeout=5)


class Logger:
    """Centralized structured logging system for RAG operations"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the logger with configuration"""
        self.config = config or {}
        self.logger = logging.getLogger("rag_system")
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        # Get log level from config or environment
        log_level = self.config.get("log_level", os.getenv("LOG_LEVEL", "INFO"))
        level = getattr(logging, log_level.upper(), logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        self.logger.setLevel(level)
        
        # Create formatter
        formatter = JSONFormatter()
        
        # Create rotating file handler
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_dir / "system.log",
            maxBytes=100 * 1024 * 1024,  # 100MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        
        # Create console handler if enabled
        if self.config.get("console_output", True):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            console_handler.setLevel(level)
            self.logger.addHandler(console_handler)
        
        # Add file handler
        self.logger.addHandler(file_handler)
        
        # Prevent propagation to root logger
        self.logger.propagate = False
    
    @staticmethod
    def get_logger() -> 'Logger':
        """Get the global logger instance (singleton)"""
        global _logger_instance
        if _logger_instance is None:
            with _logger_lock:
                if _logger_instance is None:
                    _logger_instance = Logger()
        return _logger_instance
    
    @staticmethod
    def set_correlation_id(corr_id: str):
        """Set correlation ID for current context"""
        correlation_id.set(corr_id)
    
    @staticmethod
    def get_correlation_id() -> Optional[str]:
        """Get current correlation ID"""
        return correlation_id.get()
    
    @staticmethod
    def generate_correlation_id() -> str:
        """Generate a new correlation ID"""
        return str(uuid.uuid4())
    
    def log_event(self, event_type: str, message: str, metadata: Optional[Dict[str, Any]] = None, 
                  severity: str = "info", **kwargs):
        """Log a structured event"""
        extra = {
            'event_type': event_type,
            'severity': severity,
            'metadata': metadata or {},
            **kwargs
        }
        
        level = getattr(logging, severity.upper(), logging.INFO)
        self.logger.log(level, message, extra=extra)
    
    def log_success(self, module: str, action: str, details: Optional[Dict[str, Any]] = None, **kwargs):
        """Log a successful operation"""
        message = f"Success: {module}.{action}"
        extra = {
            'event_type': 'success',
            'severity': 'info',
            'action': action,
            'details': details or {},
            **kwargs
        }
        
        self.logger.info(message, extra=extra)
    
    def log_failure(self, module: str, action: str, error: Exception, 
                    details: Optional[Dict[str, Any]] = None, **kwargs):
        """Log a failed operation"""
        message = f"Failure: {module}.{action} - {type(error).__name__}: {str(error)}"
        extra = {
            'event_type': 'failure',
            'severity': 'error',
            'action': action,
            'details': details or {},
            **kwargs
        }
        
        self.logger.error(message, extra=extra, exc_info=True)
    
    def log_latency(self, module: str, operation: str, duration_ms: float, 
                    details: Optional[Dict[str, Any]] = None, **kwargs):
        """Log operation latency"""
        message = f"Latency: {module}.{operation} took {duration_ms:.2f}ms"
        extra = {
            'event_type': 'latency',
            'severity': 'info',
            'duration_ms': duration_ms,
            'details': details or {},
            **kwargs
        }
        
        self.logger.info(message, extra=extra)
    
    def log_llm_call(self, model_name: str, success: bool, latency_ms: float, **kwargs):
        """Log LLM API call"""
        event_type = "llm_success" if success else "llm_failure"
        severity = "info" if success else "warning"
        message = f"LLM Call: {model_name} {'succeeded' if success else 'failed'} in {latency_ms:.2f}ms"
        
        extra = {
            'event_type': event_type,
            'severity': severity,
            'model_name': model_name,
            'success': success,
            'latency_ms': latency_ms,
            **kwargs
        }
        
        level = logging.INFO if success else logging.WARNING
        self.logger.log(level, message, extra=extra)
    
    def log_token_estimation(self, model_name: str, prompt_tokens: int, context_tokens: int, 
                            total_tokens: int, **kwargs):
        """Log token estimation"""
        message = f"Token Estimation: {model_name} - Prompt: {prompt_tokens}, Context: {context_tokens}, Total: {total_tokens}"
        
        extra = {
            'event_type': 'token_estimation',
            'severity': 'info',
            'model_name': model_name,
            'prompt_tokens': prompt_tokens,
            'context_tokens': context_tokens,
            'total_tokens': total_tokens,
            **kwargs
        }
        
        self.logger.info(message, extra=extra)
    
    def log_alert_event(self, alert_type: str, message: str, metadata: Optional[Dict[str, Any]] = None, **kwargs):
        """Log alert event"""
        extra = {
            'event_type': 'alert',
            'severity': 'warning',
            'alert_type': alert_type,
            'metadata': metadata or {},
            **kwargs
        }
        
        self.logger.warning(message, extra=extra)
    
    def log_fallback(self, primary_model: str, fallback_model: str, reason: str, **kwargs):
        """Log LLM fallback event"""
        message = f"Fallback: {primary_model} -> {fallback_model} due to {reason}"
        
        extra = {
            'event_type': 'fallback',
            'severity': 'warning',
            'primary_model': primary_model,
            'fallback_model': fallback_model,
            'reason': reason,
            **kwargs
        }
        
        self.logger.warning(message, extra=extra)
    
    def log_retrieval(self, query: str, num_results: int, latency_ms: float, **kwargs):
        """Log retrieval operation"""
        message = f"Retrieval: Found {num_results} results in {latency_ms:.2f}ms"
        
        extra = {
            'event_type': 'retrieval',
            'severity': 'info',
            'query_length': len(query),
            'num_results': num_results,
            'latency_ms': latency_ms,
            **kwargs
        }
        
        self.logger.info(message, extra=extra)
    
    def log_api_request(self, method: str, path: str, status_code: int, latency_ms: float, **kwargs):
        """Log API request"""
        message = f"API Request: {method} {path} -> {status_code} in {latency_ms:.2f}ms"
        
        extra = {
            'event_type': 'api_request',
            'severity': 'info',
            'method': method,
            'path': path,
            'status_code': status_code,
            'latency_ms': latency_ms,
            **kwargs
        }
        
        self.logger.info(message, extra=extra)
    
    def log_cost(self, model_name: str, tokens_used: int, cost_usd: float, **kwargs):
        """Log cost information"""
        message = f"Cost: {model_name} used {tokens_used} tokens, cost ${cost_usd:.6f}"
        
        extra = {
            'event_type': 'cost',
            'severity': 'info',
            'model_name': model_name,
            'tokens_used': tokens_used,
            'cost_usd': cost_usd,
            **kwargs
        }
        
        self.logger.info(message, extra=extra)
    
    def log_rate_limit(self, model_name: str, retry_after: Optional[int] = None, **kwargs):
        """Log rate limit event"""
        message = f"Rate Limit: {model_name}" + (f" (retry after {retry_after}s)" if retry_after else "")
        
        extra = {
            'event_type': 'rate_limit',
            'severity': 'warning',
            'model_name': model_name,
            'retry_after': retry_after,
            **kwargs
        }
        
        self.logger.warning(message, extra=extra)
    
    def log_system_health(self, component: str, status: str, details: Optional[Dict[str, Any]] = None, **kwargs):
        """Log system health check"""
        message = f"Health Check: {component} - {status}"
        
        extra = {
            'event_type': 'health_check',
            'severity': 'info',
            'component': component,
            'status': status,
            'details': details or {},
            **kwargs
        }
        
        self.logger.info(message, extra=extra)
    
    def log_performance_metric(self, metric_name: str, value: float, unit: str = "", **kwargs):
        """Log performance metric"""
        message = f"Performance: {metric_name} = {value}{unit}"
        
        extra = {
            'event_type': 'performance_metric',
            'severity': 'info',
            'metric_name': metric_name,
            'value': value,
            'unit': unit,
            **kwargs
        }
        
        self.logger.info(message, extra=extra)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logger statistics"""
        return {
            "handlers_count": len(self.logger.handlers),
            "level": self.logger.level,
            "correlation_id": self.get_correlation_id(),
            "config": self.config
        }


# Convenience functions for easy access
def log_event(event_type: str, message: str, metadata: Optional[Dict[str, Any]] = None, **kwargs):
    """Log a structured event"""
    Logger.get_logger().log_event(event_type, message, metadata, **kwargs)


def log_success(module: str, action: str, details: Optional[Dict[str, Any]] = None, **kwargs):
    """Log a successful operation"""
    Logger.get_logger().log_success(module, action, details, **kwargs)


def log_failure(module: str, action: str, error: Exception, details: Optional[Dict[str, Any]] = None, **kwargs):
    """Log a failed operation"""
    Logger.get_logger().log_failure(module, action, error, details, **kwargs)


def log_latency(module: str, operation: str, duration_ms: float, details: Optional[Dict[str, Any]] = None, **kwargs):
    """Log operation latency"""
    Logger.get_logger().log_latency(module, operation, duration_ms, details, **kwargs)


def log_llm_call(model_name: str, success: bool, latency_ms: float, **kwargs):
    """Log LLM API call"""
    Logger.get_logger().log_llm_call(model_name, success, latency_ms, **kwargs)


def log_token_estimation(model_name: str, prompt_tokens: int, context_tokens: int, total_tokens: int, **kwargs):
    """Log token estimation"""
    Logger.get_logger().log_token_estimation(model_name, prompt_tokens, context_tokens, total_tokens, **kwargs)


def log_alert_event(alert_type: str, message: str, metadata: Optional[Dict[str, Any]] = None, **kwargs):
    """Log alert event"""
    Logger.get_logger().log_alert_event(alert_type, message, metadata, **kwargs)


def set_correlation_id(corr_id: str):
    """Set correlation ID for current context"""
    Logger.set_correlation_id(corr_id)


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID"""
    return Logger.get_correlation_id()


def generate_correlation_id() -> str:
    """Generate a new correlation ID"""
    return Logger.generate_correlation_id()


# Example usage and testing
if __name__ == "__main__":
    # Test the logger
    logger = Logger.get_logger()
    
    # Set correlation ID
    logger.set_correlation_id("test-123")
    
    # Test various log types
    logger.log_event("test", "This is a test event", {"test": True})
    logger.log_success("test_module", "test_action", {"result": "success"})
    logger.log_failure("test_module", "test_action", Exception("Test error"), {"error": "test"})
    logger.log_latency("test_module", "test_operation", 150.5, {"details": "test"})
    logger.log_llm_call("gpt-4", True, 2500.0, tokens_used=1000)
    logger.log_token_estimation("gpt-4", 50, 950, 1000)
    logger.log_alert_event("test_alert", "Test alert message", {"alert": "test"})
    logger.log_fallback("gpt-4", "claude", "timeout")
    logger.log_retrieval("test query", 5, 100.0)
    logger.log_api_request("POST", "/api/query", 200, 500.0)
    logger.log_cost("gpt-4", 1000, 0.03)
    logger.log_rate_limit("gpt-4", 60)
    logger.log_system_health("database", "healthy", {"connections": 10})
    logger.log_performance_metric("response_time", 150.5, "ms")
    
    print(f"Logger stats: {logger.get_stats()}")
