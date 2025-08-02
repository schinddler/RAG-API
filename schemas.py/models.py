"""
Centralized model definitions for RAG pipeline.

This module provides clean wrappers for embedding, reranking, and reasoning models
with support for both local and API-based models, retry logic, and caching.
"""

import os
import time
import json
import asyncio
import logging
from typing import List, Dict, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from functools import wraps
import hashlib
import pickle

# Core ML libraries
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# API clients
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

# Optional local models
try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model instances."""
    # Embedding models
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = 384
    use_sparse_embeddings: bool = False
    
    # Reranker models
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"
    reranker_batch_size: int = 32
    
    # Reasoning models
    reasoning_model_name: str = "gpt-4-turbo-preview"
    reasoning_provider: str = "openai"  # openai, anthropic, deepseek, local
    max_tokens: int = 2048
    temperature: float = 0.1
    top_p: float = 0.9
    
    # API settings
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Caching
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying failed API calls."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                        time.sleep(delay * (2 ** attempt))  # Exponential backoff
                    else:
                        logger.error(f"All {max_retries} attempts failed. Last error: {e}")
            raise last_exception
        return wrapper
    return decorator


def cache_result(ttl: int = 3600):
    """Decorator for caching function results."""
    cache = {}
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key_data = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
            cache_key = hashlib.md5(key_data.encode()).hexdigest()
            
            # Check cache
            if cache_key in cache:
                timestamp, result = cache[cache_key]
                if time.time() - timestamp < ttl:
                    return result
                else:
                    del cache[cache_key]
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache[cache_key] = (time.time(), result)
            return result
        return wrapper
    return decorator


class EmbeddingModel:
    """Unified embedding model wrapper supporting dense and sparse embeddings."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize embedding model.
        
        Args:
            config: Model configuration. Uses defaults if None.
        """
        self.config = config or ModelConfig()
        self.dense_model = None
        self.sparse_model = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize dense and sparse embedding models."""
        try:
            # Initialize dense embedding model
            self.dense_model = SentenceTransformer(self.config.embedding_model_name)
            logger.info(f"Loaded dense embedding model: {self.config.embedding_model_name}")
        except Exception as e:
            logger.error(f"Failed to load dense embedding model: {e}")
            self.dense_model = None
        
        if self.config.use_sparse_embeddings:
            try:
                # Initialize sparse embedding model (TF-IDF)
                self.sparse_model = TfidfVectorizer(
                    max_features=10000,
                    ngram_range=(1, 2),
                    stop_words='english'
                )
                logger.info("Initialized sparse embedding model (TF-IDF)")
            except Exception as e:
                logger.error(f"Failed to load sparse embedding model: {e}")
                self.sparse_model = None
    
    @retry_on_failure(max_retries=3, delay=1.0)
    @cache_result(ttl=3600)
    def embed(self, texts: Union[str, List[str]], use_sparse: bool = False) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text string or list of texts
            use_sparse: Whether to use sparse embeddings (TF-IDF)
            
        Returns:
            Embedding array of shape (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([])
        
        if use_sparse and self.sparse_model is not None:
            return self._embed_sparse(texts)
        elif self.dense_model is not None:
            return self._embed_dense(texts)
        else:
            raise RuntimeError("No embedding model available")
    
    def _embed_dense(self, texts: List[str]) -> np.ndarray:
        """Generate dense embeddings using sentence-transformers."""
        try:
            embeddings = self.dense_model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embeddings
        except Exception as e:
            logger.error(f"Dense embedding failed: {e}")
            raise
    
    def _embed_sparse(self, texts: List[str]) -> np.ndarray:
        """Generate sparse embeddings using TF-IDF."""
        try:
            # Fit if not already fitted
            if not hasattr(self.sparse_model, 'vocabulary_'):
                self.sparse_model.fit(texts)
            
            # Transform to sparse matrix and convert to dense
            sparse_embeddings = self.sparse_model.transform(texts)
            return sparse_embeddings.toarray()
        except Exception as e:
            logger.error(f"Sparse embedding failed: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        if self.dense_model is not None:
            return self.dense_model.get_sentence_embedding_dimension()
        elif self.sparse_model is not None:
            return self.sparse_model.get_feature_names_out().shape[0]
        else:
            return self.config.embedding_dimension


class RerankerModel:
    """Unified reranker model wrapper supporting multiple providers."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize reranker model.
        
        Args:
            config: Model configuration. Uses defaults if None.
        """
        self.config = config or ModelConfig()
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize reranker model."""
        try:
            if TRANSFORMERS_AVAILABLE:
                # Use local transformer model
                self.model = SentenceTransformer(self.config.reranker_model_name)
                logger.info(f"Loaded local reranker model: {self.config.reranker_model_name}")
            else:
                logger.warning("Transformers not available, using API-based reranking")
                self.model = None
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            self.model = None
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def rerank(
        self, 
        query: str, 
        documents: List[str], 
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Rerank documents based on query relevance.
        
        Args:
            query: Search query
            documents: List of document texts to rerank
            top_k: Number of top results to return (None for all)
            
        Returns:
            List of (document_index, score) tuples sorted by score
        """
        if not documents:
            return []
        
        if self.model is not None:
            return self._rerank_local(query, documents, top_k)
        else:
            return self._rerank_api(query, documents, top_k)
    
    def _rerank_local(self, query: str, documents: List[str], top_k: Optional[int]) -> List[Tuple[int, float]]:
        """Rerank using local transformer model."""
        try:
            # Create query-document pairs
            pairs = [[query, doc] for doc in documents]
            
            # Get similarity scores
            scores = self.model.encode(
                pairs,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            # Calculate cosine similarities
            similarities = []
            for i, score in enumerate(scores):
                # For reranker models, the score is already a similarity measure
                similarities.append((i, float(score)))
            
            # Sort by score (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            if top_k is not None:
                similarities = similarities[:top_k]
            
            return similarities
            
        except Exception as e:
            logger.error(f"Local reranking failed: {e}")
            # Fallback: return documents in original order
            return [(i, 0.5) for i in range(len(documents))]
    
    def _rerank_api(self, query: str, documents: List[str], top_k: Optional[int]) -> List[Tuple[int, float]]:
        """Rerank using API-based service (placeholder for Cohere, etc.)."""
        # This would integrate with Cohere Rerank API or similar
        # For now, return documents in original order
        logger.warning("API reranking not implemented, returning original order")
        return [(i, 0.5) for i in range(len(documents))]


class ReasoningModel:
    """Unified reasoning model wrapper supporting multiple LLM providers."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize reasoning model.
        
        Args:
            config: Model configuration. Uses defaults if None.
        """
        self.config = config or ModelConfig()
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize API client based on provider."""
        try:
            if self.config.reasoning_provider == "openai" and OPENAI_AVAILABLE:
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable not set")
                
                self.client = openai.OpenAI(api_key=api_key)
                logger.info("Initialized OpenAI client")
                
            elif self.config.reasoning_provider == "anthropic" and ANTHROPIC_AVAILABLE:
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ValueError("ANTHROPIC_API_KEY environment variable not set")
                
                self.client = anthropic.Anthropic(api_key=api_key)
                logger.info("Initialized Anthropic client")
                
            else:
                raise ValueError(f"Unsupported reasoning provider: {self.config.reasoning_provider}")
                
        except Exception as e:
            logger.error(f"Failed to initialize reasoning client: {e}")
            self.client = None
    
    @retry_on_failure(max_retries=3, delay=2.0)
    def generate_answer(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate answer using LLM with context.
        
        Args:
            query: User query
            context_chunks: List of context chunks with 'text' and 'score' keys
            metadata: Optional metadata about the query/context
            system_prompt: Optional custom system prompt
            
        Returns:
            Dictionary with 'answer', 'sources', 'confidence', and 'raw_response'
        """
        if not self.client:
            raise RuntimeError("No reasoning client available")
        
        # Prepare context
        context_text = self._prepare_context(context_chunks)
        
        # Prepare system prompt
        if system_prompt is None:
            system_prompt = self._get_default_system_prompt()
        
        # Generate response
        if self.config.reasoning_provider == "openai":
            return self._generate_openai(query, context_text, system_prompt, metadata)
        elif self.config.reasoning_provider == "anthropic":
            return self._generate_anthropic(query, context_text, system_prompt, metadata)
        else:
            raise ValueError(f"Unsupported provider: {self.config.reasoning_provider}")
    
    def _prepare_context(self, context_chunks: List[Dict[str, Any]]) -> str:
        """Prepare context text from chunks."""
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            text = chunk.get('text', '')
            score = chunk.get('score', 0.0)
            doc_id = chunk.get('doc_id', 'unknown')
            
            context_parts.append(f"[Source {i} - Score: {score:.3f} - Doc: {doc_id}]\n{text}\n")
        
        return "\n".join(context_parts)
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for insurance/legal domain."""
        return """You are an expert assistant for insurance and legal document analysis. Your task is to:

1. Answer questions based on the provided context from insurance policies, legal documents, or compliance materials
2. Provide accurate, well-reasoned answers with specific references to source clauses
3. If the information is not available in the context, clearly state this
4. Use a professional, clear writing style appropriate for legal/insurance contexts
5. Include relevant clause numbers, section references, and specific details when available

Format your response as:
- Direct answer to the question
- Supporting evidence from the context (with source references)
- Any relevant limitations or conditions mentioned

Be precise, factual, and cite specific sources from the provided context."""

    def _generate_openai(
        self, 
        query: str, 
        context: str, 
        system_prompt: str, 
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate answer using OpenAI API."""
        try:
            user_prompt = f"Context:\n{context}\n\nQuestion: {query}"
            
            response = self.client.chat.completions.create(
                model=self.config.reasoning_model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p
            )
            
            answer = response.choices[0].message.content
            
            return {
                "answer": answer,
                "sources": self._extract_sources(context),
                "confidence": self._estimate_confidence(answer, context),
                "raw_response": response.model_dump(),
                "model": self.config.reasoning_model_name,
                "provider": "openai"
            }
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise
    
    def _generate_anthropic(
        self, 
        query: str, 
        context: str, 
        system_prompt: str, 
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate answer using Anthropic API."""
        try:
            user_prompt = f"Context:\n{context}\n\nQuestion: {query}"
            
            response = self.client.messages.create(
                model=self.config.reasoning_model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            
            answer = response.content[0].text
            
            return {
                "answer": answer,
                "sources": self._extract_sources(context),
                "confidence": self._estimate_confidence(answer, context),
                "raw_response": response.model_dump(),
                "model": self.config.reasoning_model_name,
                "provider": "anthropic"
            }
            
        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            raise
    
    def _extract_sources(self, context: str) -> List[Dict[str, Any]]:
        """Extract source information from context."""
        sources = []
        lines = context.split('\n')
        
        for line in lines:
            if line.startswith('[Source'):
                # Parse source line like "[Source 1 - Score: 0.95 - Doc: doc_123]"
                try:
                    parts = line.strip('[]').split(' - ')
                    source_num = int(parts[0].split()[1])
                    score = float(parts[1].split(':')[1])
                    doc_id = parts[2].split(':')[1]
                    
                    sources.append({
                        "source_id": source_num,
                        "score": score,
                        "doc_id": doc_id
                    })
                except (IndexError, ValueError):
                    continue
        
        return sources
    
    def _estimate_confidence(self, answer: str, context: str) -> float:
        """Estimate confidence based on answer characteristics."""
        # Simple heuristic: longer, more detailed answers tend to be more confident
        answer_length = len(answer.split())
        context_length = len(context.split())
        
        # Normalize by context length and cap at 1.0
        confidence = min(answer_length / max(context_length * 0.1, 1), 1.0)
        return round(confidence, 3)


# Default model instances
default_config = ModelConfig()

# Initialize default models
embedding_model = EmbeddingModel(default_config)
reranker_model = RerankerModel(default_config)
reasoning_model = ReasoningModel(default_config)


# Convenience functions
def get_embedding_model(config: Optional[ModelConfig] = None) -> EmbeddingModel:
    """Get embedding model instance."""
    if config is None:
        return embedding_model
    return EmbeddingModel(config)


def get_reranker_model(config: Optional[ModelConfig] = None) -> RerankerModel:
    """Get reranker model instance."""
    if config is None:
        return reranker_model
    return RerankerModel(config)


def get_reasoning_model(config: Optional[ModelConfig] = None) -> ReasoningModel:
    """Get reasoning model instance."""
    if config is None:
        return reasoning_model
    return ReasoningModel(config)


def test_models():
    """Test all model components."""
    print("Testing embedding model...")
    texts = ["Insurance policy covers medical expenses.", "Dental coverage is limited to $2000."]
    embeddings = embedding_model.embed(texts)
    print(f"Generated embeddings: {embeddings.shape}")
    
    print("\nTesting reranker model...")
    query = "What does the insurance policy cover?"
    documents = ["Medical expenses are covered up to $10000.", "Dental coverage is limited."]
    reranked = reranker_model.rerank(query, documents)
    print(f"Reranked results: {reranked}")
    
    print("\nTesting reasoning model...")
    context_chunks = [
        {"text": "Medical expenses are covered up to $10000.", "score": 0.95, "doc_id": "doc_123"},
        {"text": "Dental coverage is limited to $2000 per year.", "score": 0.87, "doc_id": "doc_123"}
    ]
    
    try:
        result = reasoning_model.generate_answer(
            "What medical expenses are covered?",
            context_chunks
        )
        print(f"Generated answer: {result['answer'][:100]}...")
    except Exception as e:
        print(f"Reasoning test failed (likely API key not set): {e}")


if __name__ == "__main__":
    test_models()
