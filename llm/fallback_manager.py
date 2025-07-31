import asyncio
import time
import logging
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
import traceback

# Import dependencies (assuming these exist in the project)
try:
    from .token_estimator import estimate_tokens
    from .config import LLM_CONFIG
    from .monitoring.logs import log_llm_event
except ImportError:
    # Fallback imports for development
    def estimate_tokens(query: str, context: str) -> int:
        return len(query.split()) + len(context.split())
    
    LLM_CONFIG = {
        "priority_order": ["gpt-4", "claude", "mistral"],
        "gpt-4": {"max_tokens": 128000, "cost_per_1k_tokens": 0.03, "latency_budget_ms": 3000},
        "claude": {"max_tokens": 200000, "cost_per_1k_tokens": 0.01, "latency_budget_ms": 2500},
        "mistral": {"max_tokens": 32000, "cost_per_1k_tokens": 0.002, "latency_budget_ms": 1200}
    }
    
    def log_llm_event(model_name: str, success: bool, reason: str = "", **kwargs):
        logging.info(f"LLM Event - Model: {model_name}, Success: {success}, Reason: {reason}")


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    @abstractmethod
    async def generate(self, query: str, context: str, **kwargs) -> str:
        """Generate response from LLM"""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model name"""
        pass


class FallbackManager:
    """
    Production-grade fallback manager for LLM calls with intelligent fallback logic,
    token estimation, and comprehensive error handling.
    """
    
    def __init__(self, llm_clients: Dict[str, BaseLLMClient], config: Dict[str, Any]):
        """
        Initialize the fallback manager.
        
        Args:
            llm_clients: Dictionary mapping model names to LLM client instances
            config: Configuration dictionary with model settings and priority order
        """
        self.llm_clients = llm_clients
        self.config = config
        self.priority_order = config.get("priority_order", [])
        self.logger = logging.getLogger(__name__)
        
        # Validate configuration
        self._validate_config()
        
        # Track fallback attempts for monitoring
        self.fallback_attempts = {}
        self.total_requests = 0
        self.successful_requests = 0
    
    def _validate_config(self) -> None:
        """Validate the configuration and raise errors for invalid configs."""
        if not self.priority_order:
            raise ValueError("Priority order must be specified in config")
        
        for model_name in self.priority_order:
            if model_name not in self.llm_clients:
                raise ValueError(f"Model {model_name} in priority order but not in llm_clients")
            if model_name not in self.config:
                raise ValueError(f"Model {model_name} missing from config")
            
            required_keys = ["max_tokens", "cost_per_1k_tokens", "latency_budget_ms"]
            for key in required_keys:
                if key not in self.config[model_name]:
                    raise ValueError(f"Model {model_name} missing required config key: {key}")
    
    async def call_with_fallback(self, query: str, context: str, **kwargs) -> str:
        """
        Call LLM with fallback logic, trying models in priority order.
        
        Args:
            query: User query
            context: Retrieved context
            **kwargs: Additional arguments to pass to LLM clients
            
        Returns:
            Generated response from the first successful LLM
            
        Raises:
            RuntimeError: If all LLMs fail
        """
        self.total_requests += 1
        start_time = time.time()
        
        # Estimate tokens for the request
        token_estimate = estimate_tokens(query, context)
        self.logger.info(f"Token estimate for request: {token_estimate}")
        
        # Try each model in priority order
        for model_name in self.priority_order:
            try:
                # Check if we should skip this model
                if self.should_skip_model(model_name, token_estimate):
                    self.log_event(model_name, False, f"Token limit exceeded: {token_estimate}")
                    continue
                
                # Check latency budget
                elapsed_time = (time.time() - start_time) * 1000
                if elapsed_time > self.config[model_name]["latency_budget_ms"]:
                    self.log_event(model_name, False, f"Latency budget exceeded: {elapsed_time:.2f}ms")
                    continue
                
                # Attempt to call the LLM
                self.logger.info(f"Attempting to call {model_name}")
                response = await self._call_llm_with_timeout(model_name, query, context, **kwargs)
                
                # Success - log and return
                self.successful_requests += 1
                self.log_event(model_name, True, "Success")
                self.logger.info(f"Successfully generated response using {model_name}")
                return response
                
            except Exception as e:
                # Handle failure and continue to next model
                fallback_reason = self.handle_failure(model_name, e)
                self.log_event(model_name, False, fallback_reason)
                
                # If this was the last model, raise the error
                if model_name == self.priority_order[-1]:
                    self.logger.error(f"All LLMs failed. Last error: {str(e)}")
                    raise RuntimeError(f"All LLMs failed. Last error: {str(e)}")
        
        # This should never be reached, but just in case
        raise RuntimeError("No LLMs available for fallback")
    
    async def _call_llm_with_timeout(self, model_name: str, query: str, context: str, **kwargs) -> str:
        """
        Call LLM with timeout handling.
        
        Args:
            model_name: Name of the model to call
            query: User query
            context: Retrieved context
            **kwargs: Additional arguments
            
        Returns:
            Generated response
            
        Raises:
            TimeoutError: If the call times out
            Exception: Any other error from the LLM client
        """
        client = self.llm_clients[model_name]
        timeout_seconds = self.config[model_name]["latency_budget_ms"] / 1000
        
        try:
            # Create task with timeout
            task = client.generate(query, context, **kwargs)
            response = await asyncio.wait_for(task, timeout=timeout_seconds)
            return response
            
        except asyncio.TimeoutError:
            raise TimeoutError(f"LLM call to {model_name} timed out after {timeout_seconds}s")
        except Exception as e:
            # Re-raise the original exception
            raise e
    
    def handle_failure(self, model_name: str, error: Exception) -> str:
        """
        Handle LLM failure and determine fallback reason.
        
        Args:
            model_name: Name of the failed model
            error: The exception that occurred
            
        Returns:
            Human-readable reason for the failure
        """
        error_str = str(error).lower()
        
        # Categorize the error
        if "timeout" in error_str or isinstance(error, TimeoutError):
            reason = "Timeout"
        elif "quota" in error_str or "rate limit" in error_str:
            reason = "Rate limit/quota exceeded"
        elif "token" in error_str or "length" in error_str:
            reason = "Token limit exceeded"
        elif "authentication" in error_str or "api key" in error_str:
            reason = "Authentication error"
        elif "network" in error_str or "connection" in error_str:
            reason = "Network error"
        else:
            reason = f"Unknown error: {type(error).__name__}"
        
        # Track fallback attempts
        if model_name not in self.fallback_attempts:
            self.fallback_attempts[model_name] = 0
        self.fallback_attempts[model_name] += 1
        
        self.logger.warning(f"LLM {model_name} failed: {reason} - {str(error)}")
        return reason
    
    def should_skip_model(self, model_name: str, token_estimate: int) -> bool:
        """
        Determine if a model should be skipped based on token limits.
        
        Args:
            model_name: Name of the model to check
            token_estimate: Estimated token count for the request
            
        Returns:
            True if the model should be skipped, False otherwise
        """
        max_tokens = self.config[model_name]["max_tokens"]
        
        # Add safety margin (80% of max tokens)
        safe_token_limit = int(max_tokens * 0.8)
        
        if token_estimate > safe_token_limit:
            self.logger.warning(
                f"Skipping {model_name}: token estimate {token_estimate} exceeds "
                f"safe limit {safe_token_limit} (max: {max_tokens})"
            )
            return True
        
        return False
    
    def log_event(self, model_name: str, success: bool, reason: str = "", **kwargs) -> None:
        """
        Log LLM event for monitoring and analytics.
        
        Args:
            model_name: Name of the model
            success: Whether the call was successful
            reason: Reason for failure (if applicable)
            **kwargs: Additional event data
        """
        event_data = {
            "model_name": model_name,
            "success": success,
            "reason": reason,
            "timestamp": time.time(),
            "fallback_attempts": self.fallback_attempts.get(model_name, 0),
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            **kwargs
        }
        
        # Log to monitoring system
        try:
            log_llm_event(**event_data)
        except Exception as e:
            self.logger.error(f"Failed to log LLM event: {e}")
        
        # Also log to standard logger
        if success:
            self.logger.info(f"LLM {model_name} succeeded")
        else:
            self.logger.warning(f"LLM {model_name} failed: {reason}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get fallback manager statistics for monitoring.
        
        Returns:
            Dictionary with statistics
        """
        success_rate = (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0
        
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "success_rate_percent": success_rate,
            "fallback_attempts": self.fallback_attempts.copy(),
            "priority_order": self.priority_order.copy()
        }
    
    def reset_statistics(self) -> None:
        """Reset all statistics counters."""
        self.total_requests = 0
        self.successful_requests = 0
        self.fallback_attempts = {}


# Example usage and testing
if __name__ == "__main__":
    # Example LLM client implementations
    class MockGPTClient(BaseLLMClient):
        async def generate(self, query: str, context: str, **kwargs) -> str:
            await asyncio.sleep(0.1)  # Simulate API call
            return f"GPT-4 response to: {query[:50]}..."
        
        def get_model_name(self) -> str:
            return "gpt-4"
    
    class MockClaudeClient(BaseLLMClient):
        async def generate(self, query: str, context: str, **kwargs) -> str:
            await asyncio.sleep(0.2)  # Simulate API call
            return f"Claude response to: {query[:50]}..."
        
        def get_model_name(self) -> str:
            return "claude"
    
    class MockMistralClient(BaseLLMClient):
        async def generate(self, query: str, context: str, **kwargs) -> str:
            await asyncio.sleep(0.05)  # Simulate API call
            return f"Mistral response to: {query[:50]}..."
        
        def get_model_name(self) -> str:
            return "mistral"
    
    # Example configuration
    EXAMPLE_CONFIG = {
        "priority_order": ["gpt-4", "claude", "mistral"],
        "gpt-4": {"max_tokens": 128000, "cost_per_1k_tokens": 0.03, "latency_budget_ms": 3000},
        "claude": {"max_tokens": 200000, "cost_per_1k_tokens": 0.01, "latency_budget_ms": 2500},
        "mistral": {"max_tokens": 32000, "cost_per_1k_tokens": 0.002, "latency_budget_ms": 1200}
    }
    
    # Example usage
    async def test_fallback_manager():
        llm_clients = {
            "gpt-4": MockGPTClient(),
            "claude": MockClaudeClient(),
            "mistral": MockMistralClient()
        }
        
        manager = FallbackManager(llm_clients, EXAMPLE_CONFIG)
        
        try:
            response = await manager.call_with_fallback(
                query="What is the capital of France?",
                context="France is a country in Europe. Paris is its capital city."
            )
            print(f"Response: {response}")
            print(f"Statistics: {manager.get_statistics()}")
        except Exception as e:
            print(f"Error: {e}")
    
    # Run test
    asyncio.run(test_fallback_manager())
