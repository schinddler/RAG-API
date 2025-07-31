import time
import logging
import re
from typing import Dict, Tuple, Optional, Any
from functools import lru_cache
import asyncio

# Import tokenizer libraries with fallbacks
try:
    import tiktoken
except ImportError:
    tiktoken = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

# Import dependencies (assuming these exist in the project)
try:
    from .monitoring.logs import log_token_estimation
except ImportError:
    def log_token_estimation(model_name: str, prompt_tokens: int, context_tokens: int, total_tokens: int, **kwargs):
        logging.info(f"Token Estimation - Model: {model_name}, Prompt: {prompt_tokens}, Context: {context_tokens}, Total: {total_tokens}")


class TokenLimitExceededError(Exception):
    """Raised when token limit cannot be satisfied even with truncation"""
    pass


class TokenEstimator:
    """
    High-performance token estimator for multiple LLM models with fallback heuristics.
    Provides fast token estimation and intelligent truncation for RAG systems.
    """
    
    # Model token limits (conservative estimates)
    MODEL_TOKEN_LIMITS = {
        "gpt-4": 128000,
        "gpt-4-turbo": 128000,
        "gpt-4-turbo-preview": 128000,
        "gpt-3.5-turbo": 16385,
        "claude-3-opus": 200000,
        "claude-3-sonnet": 200000,
        "claude-3-haiku": 200000,
        "claude-3-sonnet-20240229": 200000,
        "mistral-large": 32000,
        "mistral-large-latest": 32000,
        "mistral-medium": 32000,
        "mistral-small": 32000
    }
    
    # Character-to-token ratios for fallback estimation
    CHAR_TO_TOKEN_RATIOS = {
        "gpt": 0.25,      # ~4 chars per token
        "claude": 0.3,    # ~3.3 chars per token  
        "mistral": 0.28   # ~3.6 chars per token
    }
    
    def __init__(self):
        """Initialize the token estimator with model-specific tokenizers"""
        self.logger = logging.getLogger(__name__)
        self._tokenizers = {}
        self._init_tokenizers()
    
    def _init_tokenizers(self) -> None:
        """Initialize tokenizers for different models"""
        try:
            # OpenAI tokenizer
            if tiktoken:
                self._tokenizers["gpt"] = tiktoken.get_encoding("cl100k_base")
                self.logger.info("Initialized tiktoken for GPT models")
        except Exception as e:
            self.logger.warning(f"Failed to initialize tiktoken: {e}")
        
        try:
            # Anthropic tokenizer
            if Anthropic:
                self._tokenizers["claude"] = Anthropic()
                self.logger.info("Initialized Anthropic tokenizer")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Anthropic tokenizer: {e}")
        
        try:
            # HuggingFace tokenizer for Mistral
            if AutoTokenizer:
                self._tokenizers["mistral"] = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
                self.logger.info("Initialized HuggingFace tokenizer for Mistral")
        except Exception as e:
            self.logger.warning(f"Failed to initialize HuggingFace tokenizer: {e}")
    
    def estimate_tokens(self, prompt: str, context: str, model_name: str) -> int:
        """
        Estimate total tokens for prompt + context pair.
        
        Args:
            prompt: User prompt
            context: Retrieved context
            model_name: Name of the model
            
        Returns:
            Estimated token count
            
        Raises:
            TokenLimitExceededError: If estimation fails
        """
        start_time = time.time()
        
        try:
            # Get model family for tokenizer selection
            model_family = self._get_model_family(model_name)
            
            # Estimate tokens using appropriate tokenizer
            if model_family in self._tokenizers:
                total_tokens = self._estimate_with_tokenizer(prompt, context, model_family)
            else:
                total_tokens = self._estimate_with_heuristics(prompt, context, model_family)
            
            # Log estimation
            latency_ms = (time.time() - start_time) * 1000
            log_token_estimation(
                model_name=model_name,
                prompt_tokens=len(prompt.split()),
                context_tokens=len(context.split()),
                total_tokens=total_tokens,
                latency_ms=latency_ms,
                method="tokenizer" if model_family in self._tokenizers else "heuristic"
            )
            
            self.logger.debug(f"Token estimation for {model_name}: {total_tokens} tokens in {latency_ms:.2f}ms")
            return total_tokens
            
        except Exception as e:
            self.logger.error(f"Token estimation failed for {model_name}: {e}")
            raise TokenLimitExceededError(f"Failed to estimate tokens for {model_name}: {e}")
    
    def _estimate_with_tokenizer(self, prompt: str, context: str, model_family: str) -> int:
        """Estimate tokens using model-specific tokenizer"""
        tokenizer = self._tokenizers[model_family]
        
        if model_family == "gpt":
            # OpenAI tiktoken
            prompt_tokens = len(tokenizer.encode(prompt))
            context_tokens = len(tokenizer.encode(context))
            return prompt_tokens + context_tokens
        
        elif model_family == "claude":
            # Anthropic tokenizer
            try:
                prompt_tokens = tokenizer.count_tokens(prompt)
                context_tokens = tokenizer.count_tokens(context)
                return prompt_tokens + context_tokens
            except Exception:
                # Fallback to character-based estimation
                return self._estimate_with_heuristics(prompt, context, model_family)
        
        elif model_family == "mistral":
            # HuggingFace tokenizer
            try:
                prompt_tokens = len(tokenizer.encode(prompt))
                context_tokens = len(tokenizer.encode(context))
                return prompt_tokens + context_tokens
            except Exception:
                # Fallback to character-based estimation
                return self._estimate_with_heuristics(prompt, context, model_family)
        
        else:
            return self._estimate_with_heuristics(prompt, context, model_family)
    
    def _estimate_with_heuristics(self, prompt: str, context: str, model_family: str) -> int:
        """Estimate tokens using character-based heuristics"""
        ratio = self.CHAR_TO_TOKEN_RATIOS.get(model_family, 0.25)
        
        # Count characters (excluding whitespace)
        prompt_chars = len(re.sub(r'\s+', '', prompt))
        context_chars = len(re.sub(r'\s+', '', context))
        
        # Apply model-specific ratio
        prompt_tokens = int(prompt_chars * ratio)
        context_tokens = int(context_chars * ratio)
        
        return prompt_tokens + context_tokens
    
    def get_token_limit(self, model_name: str) -> int:
        """
        Get the maximum token limit for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Maximum token limit
            
        Raises:
            ValueError: If model is not supported
        """
        # Try exact match first
        if model_name in self.MODEL_TOKEN_LIMITS:
            return self.MODEL_TOKEN_LIMITS[model_name]
        
        # Try prefix matching
        for prefix, limit in self.MODEL_TOKEN_LIMITS.items():
            if model_name.startswith(prefix):
                return limit
        
        # Default limits by model family
        model_family = self._get_model_family(model_name)
        if model_family == "gpt":
            return 128000
        elif model_family == "claude":
            return 200000
        elif model_family == "mistral":
            return 32000
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def truncate_to_fit(self, prompt: str, context: str, model_name: str) -> Tuple[str, str]:
        """
        Truncate context to fit within model's token limit while preserving prompt.
        
        Args:
            prompt: User prompt (preserved)
            context: Retrieved context (may be truncated)
            model_name: Name of the model
            
        Returns:
            Tuple of (truncated_prompt, truncated_context)
            
        Raises:
            TokenLimitExceededError: If even prompt alone exceeds limit
        """
        token_limit = self.get_token_limit(model_name)
        
        # Estimate prompt tokens
        prompt_tokens = self.estimate_tokens(prompt, "", model_name)
        
        # If prompt alone exceeds limit, raise error
        if prompt_tokens > token_limit:
            raise TokenLimitExceededError(f"Prompt alone ({prompt_tokens} tokens) exceeds limit ({token_limit} tokens)")
        
        # Calculate available tokens for context
        available_tokens = token_limit - prompt_tokens
        
        # Estimate context tokens
        context_tokens = self.estimate_tokens("", context, model_name)
        
        # If context fits, return as-is
        if context_tokens <= available_tokens:
            return prompt, context
        
        # Truncate context to fit
        truncated_context = self._truncate_context(context, available_tokens, model_name)
        
        self.logger.info(f"Truncated context from {context_tokens} to {self.estimate_tokens('', truncated_context, model_name)} tokens")
        return prompt, truncated_context
    
    def _truncate_context(self, context: str, max_tokens: int, model_name: str) -> str:
        """Intelligently truncate context to fit token limit"""
        model_family = self._get_model_family(model_name)
        
        # Start with conservative truncation
        words = context.split()
        truncated_words = words[:max_tokens // 2]  # Conservative estimate
        
        truncated_context = " ".join(truncated_words)
        
        # Iteratively adjust until we fit
        while self.estimate_tokens("", truncated_context, model_name) > max_tokens and len(truncated_words) > 0:
            truncated_words = truncated_words[:-1]
            truncated_context = " ".join(truncated_words)
        
        # If still too long, use character-based truncation
        if self.estimate_tokens("", truncated_context, model_name) > max_tokens:
            ratio = self.CHAR_TO_TOKEN_RATIOS.get(model_family, 0.25)
            max_chars = int(max_tokens / ratio)
            truncated_context = context[:max_chars]
        
        return truncated_context
    
    def _get_model_family(self, model_name: str) -> str:
        """Get the model family for tokenizer selection"""
        model_name_lower = model_name.lower()
        
        if model_name_lower.startswith("gpt"):
            return "gpt"
        elif model_name_lower.startswith("claude"):
            return "claude"
        elif model_name_lower.startswith("mistral"):
            return "mistral"
        else:
            # Default to GPT for unknown models
            return "gpt"
    
    @lru_cache(maxsize=1000)
    def _cached_estimate(self, text: str, model_family: str) -> int:
        """Cached token estimation for performance"""
        if model_family in self._tokenizers:
            return self._estimate_with_tokenizer(text, "", model_family)
        else:
            return self._estimate_with_heuristics(text, "", model_family)
    
    def get_estimation_stats(self) -> Dict[str, Any]:
        """Get statistics about token estimation performance"""
        return {
            "available_tokenizers": list(self._tokenizers.keys()),
            "model_limits": self.MODEL_TOKEN_LIMITS.copy(),
            "char_ratios": self.CHAR_TO_TOKEN_RATIOS.copy()
        }


# Global instance for easy access
_token_estimator = None

def get_token_estimator() -> TokenEstimator:
    """Get the global token estimator instance"""
    global _token_estimator
    if _token_estimator is None:
        _token_estimator = TokenEstimator()
    return _token_estimator

def estimate_tokens(prompt: str, context: str, model_name: str = "gpt-4") -> int:
    """Convenience function for token estimation"""
    return get_token_estimator().estimate_tokens(prompt, context, model_name)

def trim_to_token_limit(text: str, max_tokens: int, model_name: str = "gpt-4") -> str:
    """Convenience function for trimming text to token limit"""
    estimator = get_token_estimator()
    token_limit = estimator.get_token_limit(model_name)
    _, truncated_text = estimator.truncate_to_fit("", text, model_name)
    return truncated_text


# Example usage and testing
if __name__ == "__main__":
    # Test the token estimator
    estimator = TokenEstimator()
    
    # Test cases
    test_cases = [
        ("What is the capital of France?", "France is a country in Europe. Paris is its capital city.", "gpt-4"),
        ("Explain quantum computing", "Quantum computing is a field that uses quantum mechanical phenomena to process information.", "claude-3-sonnet"),
        ("What is machine learning?", "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.", "mistral-large")
    ]
    
    for prompt, context, model in test_cases:
        try:
            tokens = estimator.estimate_tokens(prompt, context, model)
            limit = estimator.get_token_limit(model)
            print(f"{model}: {tokens} tokens (limit: {limit})")
            
            # Test truncation
            truncated_prompt, truncated_context = estimator.truncate_to_fit(prompt, context, model)
            truncated_tokens = estimator.estimate_tokens(truncated_prompt, truncated_context, model)
            print(f"  Truncated: {truncated_tokens} tokens")
            
        except Exception as e:
            print(f"Error with {model}: {e}")
    
    # Print stats
    print(f"\nStats: {estimator.get_estimation_stats()}")
