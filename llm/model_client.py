import asyncio
import time
import logging
import json
import aiohttp
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
import backoff
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Import dependencies (assuming these exist in the project)
try:
    from .config import LLM_CONFIG
    from .token_estimator import estimate_tokens, trim_to_token_limit
    from .monitoring.logs import log_llm_call
except ImportError:
    # Fallback imports for development
    LLM_CONFIG = {
        "gpt-4": {"api_key": "sk-...", "temperature": 0.7, "max_tokens": 128000},
        "claude": {"api_key": "sk-ant-...", "temperature": 0.7, "max_tokens": 200000},
        "mistral": {"api_key": "hf-...", "temperature": 0.7, "max_tokens": 32000}
    }
    
    def estimate_tokens(text: str) -> int:
        return len(text.split())
    
    def trim_to_token_limit(text: str, max_tokens: int) -> str:
        tokens = text.split()
        if len(tokens) > max_tokens:
            return " ".join(tokens[:max_tokens])
        return text
    
    def log_llm_call(model_name: str, success: bool, latency_ms: float, **kwargs):
        logging.info(f"LLM Call - Model: {model_name}, Success: {success}, Latency: {latency_ms:.2f}ms")


@dataclass
class LLMResponse:
    """Structured response from LLM calls"""
    text: str
    model_name: str
    latency_ms: float
    tokens_used: Optional[int] = None
    cost: Optional[float] = None


class LLMError(Exception):
    """Base exception for LLM-related errors"""
    pass


class RateLimitError(LLMError):
    """Raised when rate limit is exceeded"""
    pass


class TokenLimitError(LLMError):
    """Raised when token limit is exceeded"""
    pass


class AuthenticationError(LLMError):
    """Raised when authentication fails"""
    pass


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        """
        Initialize the LLM client.
        
        Args:
            model_name: Name of the model
            config: Configuration dictionary with API settings
        """
        self.model_name = model_name
        self.config = config
        self.api_key = config.get("api_key")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 4096)
        self.logger = logging.getLogger(f"{__name__}.{model_name}")
        
        if not self.api_key:
            raise ValueError(f"API key not found for model {model_name}")
    
    @abstractmethod
    async def _make_api_call(self, prompt: str, context: str, **kwargs) -> str:
        """Make the actual API call to the LLM service"""
        pass
    
    @abstractmethod
    def _format_prompt(self, prompt: str, context: str) -> str:
        """Format the prompt according to the model's requirements"""
        pass
    
    @abstractmethod
    def _parse_response(self, response_data: Any) -> str:
        """Parse the response from the API"""
        pass
    
    @abstractmethod
    def _handle_api_error(self, error: Exception) -> None:
        """Handle API-specific errors"""
        pass
    
    async def generate(self, prompt: str, context: str, **kwargs) -> str:
        """
        Generate response from LLM with retry logic and error handling.
        
        Args:
            prompt: User prompt
            context: Retrieved context
            **kwargs: Additional arguments
            
        Returns:
            Generated text response
            
        Raises:
            LLMError: If the API call fails after retries
        """
        start_time = time.time()
        
        try:
            # Format the prompt
            formatted_prompt = self._format_prompt(prompt, context)
            
            # Estimate tokens and trim if necessary
            token_estimate = estimate_tokens(formatted_prompt)
            if token_estimate > self.max_tokens:
                formatted_prompt = trim_to_token_limit(formatted_prompt, self.max_tokens)
                self.logger.warning(f"Prompt trimmed to {self.max_tokens} tokens")
            
            # Make the API call with retry logic
            response_text = await self._call_with_retry(formatted_prompt, **kwargs)
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Log successful call
            log_llm_call(
                model_name=self.model_name,
                success=True,
                latency_ms=latency_ms,
                tokens_used=token_estimate
            )
            
            self.logger.info(f"Successfully generated response in {latency_ms:.2f}ms")
            return response_text
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            
            # Log failed call
            log_llm_call(
                model_name=self.model_name,
                success=False,
                latency_ms=latency_ms,
                error=str(e)
            )
            
            self.logger.error(f"Failed to generate response: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((RateLimitError, asyncio.TimeoutError))
    )
    async def _call_with_retry(self, formatted_prompt: str, **kwargs) -> str:
        """Make API call with retry logic"""
        try:
            return await self._make_api_call(formatted_prompt, **kwargs)
        except Exception as e:
            self._handle_api_error(e)
            raise
    
    def get_model_name(self) -> str:
        """Get the model name"""
        return self.model_name


class GPTClient(BaseLLMClient):
    """OpenAI GPT client implementation"""
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        self.api_base = config.get("api_base", "https://api.openai.com/v1")
        self.model = config.get("model", "gpt-4-turbo-preview")
    
    def _format_prompt(self, prompt: str, context: str) -> str:
        """Format prompt for GPT models"""
        return f"""Context: {context}

Question: {prompt}

Answer:"""
    
    async def _make_api_call(self, formatted_prompt: str, **kwargs) -> str:
        """Make API call to OpenAI"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": formatted_prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **kwargs
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 429:
                    raise RateLimitError("OpenAI rate limit exceeded")
                elif response.status == 401:
                    raise AuthenticationError("Invalid OpenAI API key")
                elif response.status == 400:
                    error_data = await response.json()
                    if "context_length_exceeded" in str(error_data):
                        raise TokenLimitError("OpenAI token limit exceeded")
                    else:
                        raise LLMError(f"OpenAI API error: {error_data}")
                elif response.status != 200:
                    raise LLMError(f"OpenAI API error: {response.status}")
                
                response_data = await response.json()
                return self._parse_response(response_data)
    
    def _parse_response(self, response_data: Dict[str, Any]) -> str:
        """Parse OpenAI response"""
        try:
            return response_data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as e:
            raise LLMError(f"Invalid OpenAI response format: {e}")
    
    def _handle_api_error(self, error: Exception) -> None:
        """Handle OpenAI-specific errors"""
        if isinstance(error, RateLimitError):
            self.logger.warning("OpenAI rate limit hit, retrying...")
        elif isinstance(error, TokenLimitError):
            self.logger.error("OpenAI token limit exceeded")
        elif isinstance(error, AuthenticationError):
            self.logger.error("OpenAI authentication failed")
        else:
            self.logger.error(f"OpenAI API error: {error}")


class ClaudeClient(BaseLLMClient):
    """Anthropic Claude client implementation"""
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        self.api_base = config.get("api_base", "https://api.anthropic.com")
        self.model = config.get("model", "claude-3-sonnet-20240229")
    
    def _format_prompt(self, prompt: str, context: str) -> str:
        """Format prompt for Claude models"""
        return f"""\n\nHuman: Here is some context: {context}

Question: {prompt}

Assistant:"""
    
    async def _make_api_call(self, formatted_prompt: str, **kwargs) -> str:
        """Make API call to Anthropic"""
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": formatted_prompt}],
            **kwargs
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_base}/v1/messages",
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 429:
                    raise RateLimitError("Anthropic rate limit exceeded")
                elif response.status == 401:
                    raise AuthenticationError("Invalid Anthropic API key")
                elif response.status == 400:
                    error_data = await response.json()
                    if "context_length" in str(error_data):
                        raise TokenLimitError("Anthropic token limit exceeded")
                    else:
                        raise LLMError(f"Anthropic API error: {error_data}")
                elif response.status != 200:
                    raise LLMError(f"Anthropic API error: {response.status}")
                
                response_data = await response.json()
                return self._parse_response(response_data)
    
    def _parse_response(self, response_data: Dict[str, Any]) -> str:
        """Parse Anthropic response"""
        try:
            return response_data["content"][0]["text"].strip()
        except (KeyError, IndexError) as e:
            raise LLMError(f"Invalid Anthropic response format: {e}")
    
    def _handle_api_error(self, error: Exception) -> None:
        """Handle Anthropic-specific errors"""
        if isinstance(error, RateLimitError):
            self.logger.warning("Anthropic rate limit hit, retrying...")
        elif isinstance(error, TokenLimitError):
            self.logger.error("Anthropic token limit exceeded")
        elif isinstance(error, AuthenticationError):
            self.logger.error("Anthropic authentication failed")
        else:
            self.logger.error(f"Anthropic API error: {error}")


class MistralClient(BaseLLMClient):
    """Mistral client implementation (via HuggingFace or direct API)"""
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        self.api_base = config.get("api_base", "https://api.mistral.ai/v1")
        self.model = config.get("model", "mistral-large-latest")
    
    def _format_prompt(self, prompt: str, context: str) -> str:
        """Format prompt for Mistral models"""
        return f"""<s>[INST] Context: {context}

Question: {prompt} [/INST]"""
    
    async def _make_api_call(self, formatted_prompt: str, **kwargs) -> str:
        """Make API call to Mistral"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": formatted_prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **kwargs
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 429:
                    raise RateLimitError("Mistral rate limit exceeded")
                elif response.status == 401:
                    raise AuthenticationError("Invalid Mistral API key")
                elif response.status == 400:
                    error_data = await response.json()
                    if "context_length" in str(error_data):
                        raise TokenLimitError("Mistral token limit exceeded")
                    else:
                        raise LLMError(f"Mistral API error: {error_data}")
                elif response.status != 200:
                    raise LLMError(f"Mistral API error: {response.status}")
                
                response_data = await response.json()
                return self._parse_response(response_data)
    
    def _parse_response(self, response_data: Dict[str, Any]) -> str:
        """Parse Mistral response"""
        try:
            return response_data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as e:
            raise LLMError(f"Invalid Mistral response format: {e}")
    
    def _handle_api_error(self, error: Exception) -> None:
        """Handle Mistral-specific errors"""
        if isinstance(error, RateLimitError):
            self.logger.warning("Mistral rate limit hit, retrying...")
        elif isinstance(error, TokenLimitError):
            self.logger.error("Mistral token limit exceeded")
        elif isinstance(error, AuthenticationError):
            self.logger.error("Mistral authentication failed")
        else:
            self.logger.error(f"Mistral API error: {error}")


def create_llm_client(model_name: str, config: Dict[str, Any]) -> BaseLLMClient:
    """
    Factory function to create LLM client instances.
    
    Args:
        model_name: Name of the model
        config: Configuration dictionary
        
    Returns:
        Configured LLM client instance
        
    Raises:
        ValueError: If model is not supported
    """
    model_config = config.get(model_name, {})
    
    if model_name.startswith("gpt"):
        return GPTClient(model_name, model_config)
    elif model_name.startswith("claude"):
        return ClaudeClient(model_name, model_config)
    elif model_name.startswith("mistral"):
        return MistralClient(model_name, model_config)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    # Example configuration
    EXAMPLE_CONFIG = {
        "gpt-4": {
            "api_key": "sk-test-key",
            "temperature": 0.7,
            "max_tokens": 128000,
            "model": "gpt-4-turbo-preview"
        },
        "claude": {
            "api_key": "sk-ant-test-key",
            "temperature": 0.7,
            "max_tokens": 200000,
            "model": "claude-3-sonnet-20240229"
        },
        "mistral": {
            "api_key": "hf-test-key",
            "temperature": 0.7,
            "max_tokens": 32000,
            "model": "mistral-large-latest"
        }
    }
    
    async def test_clients():
        """Test the LLM clients"""
        clients = {}
        
        for model_name in ["gpt-4", "claude", "mistral"]:
            try:
                clients[model_name] = create_llm_client(model_name, EXAMPLE_CONFIG)
                print(f"Created client for {model_name}")
            except Exception as e:
                print(f"Failed to create client for {model_name}: {e}")
        
        # Test a client (this will fail due to invalid API keys, but shows the interface)
        if "gpt-4" in clients:
            try:
                response = await clients["gpt-4"].generate(
                    prompt="What is the capital of France?",
                    context="France is a country in Europe."
                )
                print(f"Response: {response}")
            except Exception as e:
                print(f"Expected error (invalid API key): {e}")
    
    asyncio.run(test_clients())
