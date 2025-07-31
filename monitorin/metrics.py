import time
import threading
import statistics
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import json
import logging

# Import dependencies (assuming these exist in the project)
try:
    from .logs import log_event, log_alert_event
    from .alerting import get_alert_manager
except ImportError:
    def log_event(event_type: str, message: str, metadata: dict = None, **kwargs):
        logging.info(f"Metrics Event - {event_type}: {message}")
    
    def log_alert_event(alert_type: str, message: str, metadata: dict = None, **kwargs):
        logging.warning(f"Metrics Alert - {alert_type}: {message}")
    
    def get_alert_manager():
        return None


@dataclass
class ModelCosts:
    """Model-specific token costs per 1K tokens"""
    prompt_cost_per_1k: float
    completion_cost_per_1k: float


# Centralized model costs (USD per 1K tokens)
MODEL_COSTS = {
    "gpt-4": ModelCosts(prompt_cost_per_1k=0.03, completion_cost_per_1k=0.06),
    "gpt-4-turbo": ModelCosts(prompt_cost_per_1k=0.01, completion_cost_per_1k=0.03),
    "gpt-4-turbo-preview": ModelCosts(prompt_cost_per_1k=0.01, completion_cost_per_1k=0.03),
    "gpt-3.5-turbo": ModelCosts(prompt_cost_per_1k=0.0015, completion_cost_per_1k=0.002),
    "claude-3-opus": ModelCosts(prompt_cost_per_1k=0.015, completion_cost_per_1k=0.075),
    "claude-3-sonnet": ModelCosts(prompt_cost_per_1k=0.003, completion_cost_per_1k=0.015),
    "claude-3-haiku": ModelCosts(prompt_cost_per_1k=0.00025, completion_cost_per_1k=0.00125),
    "claude-3-sonnet-20240229": ModelCosts(prompt_cost_per_1k=0.003, completion_cost_per_1k=0.015),
    "mistral-large": ModelCosts(prompt_cost_per_1k=0.007, completion_cost_per_1k=0.024),
    "mistral-large-latest": ModelCosts(prompt_cost_per_1k=0.007, completion_cost_per_1k=0.024),
    "mistral-medium": ModelCosts(prompt_cost_per_1k=0.0027, completion_cost_per_1k=0.0081),
    "mistral-small": ModelCosts(prompt_cost_per_1k=0.002, completion_cost_per_1k=0.006)
}


@dataclass
class LatencyMetric:
    """Latency metric data structure"""
    module: str
    action: str
    duration_ms: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TokenMetric:
    """Token usage metric data structure"""
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    timestamp: datetime = field(default_factory=datetime.now)


class RollingWindow:
    """Thread-safe rolling window for calculating moving averages"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
        self._lock = threading.Lock()
    
    def add(self, value: float):
        """Add a value to the rolling window"""
        with self._lock:
            self.values.append(value)
    
    def get_average(self) -> float:
        """Get the average of values in the window"""
        with self._lock:
            if not self.values:
                return 0.0
            return sum(self.values) / len(self.values)
    
    def get_percentile(self, percentile: float) -> float:
        """Get percentile of values in the window"""
        with self._lock:
            if not self.values:
                return 0.0
            sorted_values = sorted(self.values)
            index = int(len(sorted_values) * percentile / 100)
            return sorted_values[index]
    
    def get_count(self) -> int:
        """Get number of values in the window"""
        with self._lock:
            return len(self.values)


class MetricsCollector:
    """
    High-performance metrics collector for RAG system with thread-safe operations,
    rolling averages, and Prometheus-compatible output.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the metrics collector.
        
        Args:
            config: Configuration dictionary with alerting thresholds and settings
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.alert_manager = get_alert_manager()
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Latency metrics
        self.latency_metrics: List[LatencyMetric] = []
        self.latency_windows: Dict[str, RollingWindow] = defaultdict(lambda: RollingWindow(100))
        
        # Token metrics
        self.token_metrics: List[TokenMetric] = []
        self.token_windows: Dict[str, RollingWindow] = defaultdict(lambda: RollingWindow(100))
        
        # Cumulative counters
        self.total_requests = 0
        self.total_tokens = 0
        self.total_cost_usd = 0.0
        self.total_latency_ms = 0.0
        
        # Per-model counters
        self.model_requests: Dict[str, int] = defaultdict(int)
        self.model_tokens: Dict[str, int] = defaultdict(int)
        self.model_costs: Dict[str, float] = defaultdict(float)
        self.model_latencies: Dict[str, List[float]] = defaultdict(list)
        
        # Per-module counters
        self.module_requests: Dict[str, int] = defaultdict(int)
        self.module_latencies: Dict[str, List[float]] = defaultdict(list)
        
        # Alerting thresholds
        self.latency_threshold_ms = self.config.get("latency_threshold_ms", 5000)
        self.token_threshold = self.config.get("token_threshold", 100000)
        self.cost_threshold_usd = self.config.get("cost_threshold_usd", 100.0)
        
        # Time-based filtering
        self.metrics_retention_hours = self.config.get("metrics_retention_hours", 24)
        
        self.logger.info("Metrics collector initialized")
    
    def log_latency(self, module: str, action: str, duration_ms: float) -> None:
        """
        Log latency metric for a module action.
        
        Args:
            module: Module name (e.g., 'llm', 'retrieval', 'indexer')
            action: Action name (e.g., 'generate', 'search', 'index')
            duration_ms: Duration in milliseconds
        """
        timestamp = datetime.now()
        
        with self._lock:
            # Create metric
            metric = LatencyMetric(module=module, action=action, duration_ms=duration_ms, timestamp=timestamp)
            self.latency_metrics.append(metric)
            
            # Update cumulative counters
            self.total_requests += 1
            self.total_latency_ms += duration_ms
            
            # Update per-module counters
            self.module_requests[module] += 1
            self.module_latencies[module].append(duration_ms)
            
            # Update rolling window
            window_key = f"{module}.{action}"
            self.latency_windows[window_key].add(duration_ms)
        
        # Log the metric
        log_event("latency_metric", f"{module}.{action} took {duration_ms:.2f}ms", {
            "module": module,
            "action": action,
            "duration_ms": duration_ms,
            "timestamp": timestamp.isoformat()
        })
        
        # Check for latency alerts
        if duration_ms > self.latency_threshold_ms:
            self._alert_high_latency(module, action, duration_ms)
    
    def log_tokens(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        """
        Log token usage for a model.
        
        Args:
            model: Model name
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
        """
        total_tokens = prompt_tokens + completion_tokens
        cost_usd = self._calculate_cost(model, prompt_tokens, completion_tokens)
        timestamp = datetime.now()
        
        with self._lock:
            # Create metric
            metric = TokenMetric(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_usd=cost_usd,
                timestamp=timestamp
            )
            self.token_metrics.append(metric)
            
            # Update cumulative counters
            self.total_tokens += total_tokens
            self.total_cost_usd += cost_usd
            
            # Update per-model counters
            self.model_requests[model] += 1
            self.model_tokens[model] += total_tokens
            self.model_costs[model] += cost_usd
            self.model_latencies[model].append(0)  # Placeholder for latency
        
        # Log the metric
        log_event("token_metric", f"{model} used {total_tokens} tokens, cost ${cost_usd:.6f}", {
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost_usd": cost_usd,
            "timestamp": timestamp.isoformat()
        })
        
        # Check for token alerts
        if total_tokens > self.token_threshold:
            self._alert_high_token_usage(model, total_tokens)
    
    def log_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        """
        Log cost for a model (alias for log_tokens with cost focus).
        
        Args:
            model: Model name
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
        """
        self.log_tokens(model, prompt_tokens, completion_tokens)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics summary.
        
        Returns:
            Dictionary with all metrics summary
        """
        with self._lock:
            # Clean old metrics
            self._cleanup_old_metrics()
            
            # Calculate averages
            avg_latency_ms = self.total_latency_ms / max(self.total_requests, 1)
            
            # Per-model summaries
            model_summaries = {}
            for model in self.model_requests:
                requests = self.model_requests[model]
                tokens = self.model_tokens[model]
                cost = self.model_costs[model]
                latencies = self.model_latencies[model]
                
                avg_latency = statistics.mean(latencies) if latencies else 0.0
                p95_latency = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else 0.0
                
                model_summaries[model] = {
                    "requests": requests,
                    "total_tokens": tokens,
                    "total_cost_usd": cost,
                    "avg_latency_ms": avg_latency,
                    "p95_latency_ms": p95_latency,
                    "avg_tokens_per_request": tokens / max(requests, 1),
                    "avg_cost_per_request": cost / max(requests, 1)
                }
            
            # Per-module summaries
            module_summaries = {}
            for module in self.module_requests:
                requests = self.module_requests[module]
                latencies = self.module_latencies[module]
                
                avg_latency = statistics.mean(latencies) if latencies else 0.0
                p95_latency = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else 0.0
                
                module_summaries[module] = {
                    "requests": requests,
                    "avg_latency_ms": avg_latency,
                    "p95_latency_ms": p95_latency
                }
            
            # Rolling window summaries
            window_summaries = {}
            for window_key, window in self.latency_windows.items():
                window_summaries[window_key] = {
                    "avg_latency_ms": window.get_average(),
                    "p95_latency_ms": window.get_percentile(95),
                    "count": window.get_count()
                }
            
            return {
                "total_requests": self.total_requests,
                "total_tokens": self.total_tokens,
                "total_cost_usd": self.total_cost_usd,
                "avg_latency_ms": avg_latency_ms,
                "model_summaries": model_summaries,
                "module_summaries": module_summaries,
                "window_summaries": window_summaries,
                "timestamp": datetime.now().isoformat()
            }
    
    def reset_metrics(self) -> None:
        """Reset all metrics counters and data."""
        with self._lock:
            self.latency_metrics.clear()
            self.token_metrics.clear()
            
            for window in self.latency_windows.values():
                window.values.clear()
            
            for window in self.token_windows.values():
                window.values.clear()
            
            self.total_requests = 0
            self.total_tokens = 0
            self.total_cost_usd = 0.0
            self.total_latency_ms = 0.0
            
            self.model_requests.clear()
            self.model_tokens.clear()
            self.model_costs.clear()
            self.model_latencies.clear()
            
            self.module_requests.clear()
            self.module_latencies.clear()
        
        self.logger.info("Metrics reset completed")
        log_event("metrics_reset", "All metrics have been reset")
    
    def get_prometheus_metrics(self) -> str:
        """
        Get metrics in Prometheus exposition format.
        
        Returns:
            String with Prometheus-compatible metrics
        """
        summary = self.get_summary()
        lines = []
        
        # Add timestamp
        lines.append(f"# HELP rag_metrics_timestamp Timestamp of metrics collection")
        lines.append(f"# TYPE rag_metrics_timestamp gauge")
        lines.append(f"rag_metrics_timestamp {int(time.time())}")
        
        # Total metrics
        lines.append(f"# HELP rag_total_requests Total number of requests")
        lines.append(f"# TYPE rag_total_requests counter")
        lines.append(f"rag_total_requests {summary['total_requests']}")
        
        lines.append(f"# HELP rag_total_tokens Total number of tokens used")
        lines.append(f"# TYPE rag_total_tokens counter")
        lines.append(f"rag_total_tokens {summary['total_tokens']}")
        
        lines.append(f"# HELP rag_total_cost_usd Total cost in USD")
        lines.append(f"# TYPE rag_total_cost_usd counter")
        lines.append(f"rag_total_cost_usd {summary['total_cost_usd']}")
        
        lines.append(f"# HELP rag_avg_latency_ms Average latency in milliseconds")
        lines.append(f"# TYPE rag_avg_latency_ms gauge")
        lines.append(f"rag_avg_latency_ms {summary['avg_latency_ms']}")
        
        # Per-model metrics
        for model, data in summary['model_summaries'].items():
            lines.append(f"# HELP rag_model_requests_total Total requests per model")
            lines.append(f"# TYPE rag_model_requests_total counter")
            lines.append(f'rag_model_requests_total{{model="{model}"}} {data["requests"]}')
            
            lines.append(f"# HELP rag_model_tokens_total Total tokens per model")
            lines.append(f"# TYPE rag_model_tokens_total counter")
            lines.append(f'rag_model_tokens_total{{model="{model}"}} {data["total_tokens"]}')
            
            lines.append(f"# HELP rag_model_cost_usd Total cost per model")
            lines.append(f"# TYPE rag_model_cost_usd counter")
            lines.append(f'rag_model_cost_usd{{model="{model}"}} {data["total_cost_usd"]}')
            
            lines.append(f"# HELP rag_model_avg_latency_ms Average latency per model")
            lines.append(f"# TYPE rag_model_avg_latency_ms gauge")
            lines.append(f'rag_model_avg_latency_ms{{model="{model}"}} {data["avg_latency_ms"]}')
            
            lines.append(f"# HELP rag_model_p95_latency_ms 95th percentile latency per model")
            lines.append(f"# TYPE rag_model_p95_latency_ms gauge")
            lines.append(f'rag_model_p95_latency_ms{{model="{model}"}} {data["p95_latency_ms"]}')
        
        # Per-module metrics
        for module, data in summary['module_summaries'].items():
            lines.append(f"# HELP rag_module_requests_total Total requests per module")
            lines.append(f"# TYPE rag_module_requests_total counter")
            lines.append(f'rag_module_requests_total{{module="{module}"}} {data["requests"]}')
            
            lines.append(f"# HELP rag_module_avg_latency_ms Average latency per module")
            lines.append(f"# TYPE rag_module_avg_latency_ms gauge")
            lines.append(f'rag_module_avg_latency_ms{{module="{module}"}} {data["avg_latency_ms"]}')
            
            lines.append(f"# HELP rag_module_p95_latency_ms 95th percentile latency per module")
            lines.append(f"# TYPE rag_module_p95_latency_ms gauge")
            lines.append(f'rag_module_p95_latency_ms{{module="{module}"}} {data["p95_latency_ms"]}')
        
        return "\n".join(lines)
    
    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost for token usage."""
        # Get model costs
        model_costs = MODEL_COSTS.get(model)
        if not model_costs:
            # Use default costs for unknown models
            model_costs = ModelCosts(prompt_cost_per_1k=0.01, completion_cost_per_1k=0.03)
        
        # Calculate costs
        prompt_cost = (prompt_tokens / 1000) * model_costs.prompt_cost_per_1k
        completion_cost = (completion_tokens / 1000) * model_costs.completion_cost_per_1k
        
        return prompt_cost + completion_cost
    
    def _alert_high_latency(self, module: str, action: str, duration_ms: float) -> None:
        """Alert on high latency."""
        if self.alert_manager:
            self.alert_manager.alert_on_latency(module, duration_ms, self.latency_threshold_ms)
        
        log_alert_event("high_latency", f"{module}.{action} exceeded threshold: {duration_ms:.2f}ms", {
            "module": module,
            "action": action,
            "duration_ms": duration_ms,
            "threshold_ms": self.latency_threshold_ms
        })
    
    def _alert_high_token_usage(self, model: str, total_tokens: int) -> None:
        """Alert on high token usage."""
        if self.alert_manager:
            self.alert_manager.alert_on_token_overuse(model, total_tokens, self.token_threshold)
        
        log_alert_event("high_token_usage", f"{model} exceeded token threshold: {total_tokens}", {
            "model": model,
            "total_tokens": total_tokens,
            "threshold": self.token_threshold
        })
    
    def _cleanup_old_metrics(self) -> None:
        """Remove metrics older than retention period."""
        cutoff_time = datetime.now() - timedelta(hours=self.metrics_retention_hours)
        
        # Clean latency metrics
        self.latency_metrics = [
            metric for metric in self.latency_metrics
            if metric.timestamp > cutoff_time
        ]
        
        # Clean token metrics
        self.token_metrics = [
            metric for metric in self.token_metrics
            if metric.timestamp > cutoff_time
        ]
    
    def get_model_costs(self) -> Dict[str, ModelCosts]:
        """Get current model costs."""
        return MODEL_COSTS.copy()
    
    def update_model_cost(self, model: str, prompt_cost: float, completion_cost: float) -> None:
        """Update cost for a specific model."""
        MODEL_COSTS[model] = ModelCosts(prompt_cost_per_1k=prompt_cost, completion_cost_per_1k=completion_cost)
        self.logger.info(f"Updated costs for {model}: prompt=${prompt_cost}/1K, completion=${completion_cost}/1K")


# Global instance for easy access
_metrics_collector = None
_metrics_lock = threading.Lock()


def get_metrics_collector(config: Optional[Dict[str, Any]] = None) -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        with _metrics_lock:
            if _metrics_collector is None:
                _metrics_collector = MetricsCollector(config)
    return _metrics_collector


# Convenience functions for easy access
def log_latency(module: str, action: str, duration_ms: float):
    """Log latency metric."""
    get_metrics_collector().log_latency(module, action, duration_ms)


def log_tokens(model: str, prompt_tokens: int, completion_tokens: int):
    """Log token usage."""
    get_metrics_collector().log_tokens(model, prompt_tokens, completion_tokens)


def log_cost(model: str, prompt_tokens: int, completion_tokens: int):
    """Log cost metric."""
    get_metrics_collector().log_cost(model, prompt_tokens, completion_tokens)


def get_summary() -> Dict[str, Any]:
    """Get metrics summary."""
    return get_metrics_collector().get_summary()


def reset_metrics():
    """Reset all metrics."""
    get_metrics_collector().reset_metrics()


def get_prometheus_metrics() -> str:
    """Get Prometheus-compatible metrics."""
    return get_metrics_collector().get_prometheus_metrics()


# FastAPI middleware integration
class MetricsMiddleware:
    """FastAPI middleware for automatic request metrics."""
    
    def __init__(self, app, metrics_collector: Optional[MetricsCollector] = None):
        self.app = app
        self.metrics_collector = metrics_collector or get_metrics_collector()
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        start_time = time.time()
        
        # Track request
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                # Calculate latency
                duration_ms = (time.time() - start_time) * 1000
                
                # Log metrics
                path = scope.get("path", "unknown")
                method = scope.get("method", "unknown")
                status = message.get("status", 0)
                
                self.metrics_collector.log_latency(
                    module="api",
                    action=f"{method}_{path}",
                    duration_ms=duration_ms
                )
                
                # Log additional API metrics
                log_event("api_request", f"{method} {path} -> {status}", {
                    "method": method,
                    "path": path,
                    "status": status,
                    "latency_ms": duration_ms
                })
            
            await send(message)
        
        await self.app(scope, receive, send_wrapper)


# Example usage and testing
if __name__ == "__main__":
    # Test the metrics collector
    collector = MetricsCollector({
        "latency_threshold_ms": 1000,
        "token_threshold": 50000,
        "cost_threshold_usd": 10.0
    })
    
    # Test latency metrics
    collector.log_latency("llm", "generate", 2500.0)
    collector.log_latency("retrieval", "search", 150.0)
    collector.log_latency("indexer", "index", 5000.0)
    
    # Test token metrics
    collector.log_tokens("gpt-4", 1000, 500)
    collector.log_tokens("claude-3-sonnet", 2000, 800)
    collector.log_tokens("mistral-large", 1500, 600)
    
    # Get summary
    summary = collector.get_summary()
    print(f"Summary: {json.dumps(summary, indent=2)}")
    
    # Get Prometheus metrics
    prometheus_metrics = collector.get_prometheus_metrics()
    print(f"Prometheus metrics:\n{prometheus_metrics}")
    
    # Test alerting thresholds
    collector.log_latency("llm", "generate", 6000.0)  # Should trigger alert
    collector.log_tokens("gpt-4", 60000, 30000)  # Should trigger alert
