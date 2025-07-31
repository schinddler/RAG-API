"""
High-Performance Cross-Encoder Reranker for RAG Systems

This module implements semantic reranking using cross-encoder models to enhance
retrieval quality by re-ranking candidate chunks after dense or hybrid retrieval.

Author: RAG System Team
License: MIT
"""

import logging
import time
import asyncio
import hashlib
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from collections import defaultdict
import re

# Optional imports with graceful fallbacks
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from sentence_transformers import CrossEncoder
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    torch = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    CrossEncoder = None

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class RerankedChunk:
    """Represents a reranked chunk with enhanced scoring."""
    text: str
    original_score: float
    rerank_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'text': self.text,
            'original_score': self.original_score,
            'rerank_score': self.rerank_score,
            'metadata': self.metadata,
            'chunk_id': self.chunk_id
        }


@dataclass
class RerankerConfig:
    """Configuration for cross-encoder reranker."""
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str = "auto"  # "auto", "cpu", "cuda"
    top_k: int = 5
    batch_size: int = 32
    max_length: int = 512
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour
    early_exit_threshold: float = 0.95  # Skip reranking if BM25 score is very high
    score_normalization: str = "minmax"  # "minmax", "zscore", "sigmoid"
    enable_gpu: bool = True
    quantized: bool = False
    debug_mode: bool = False


class BaseReranker(ABC):
    """Abstract base class for rerankers."""
    
    @abstractmethod
    def rerank(
        self, 
        query: str, 
        chunks: List[Dict[str, Any]], 
        top_k: Optional[int] = None
    ) -> List[RerankedChunk]:
        """Rerank chunks based on query relevance."""
        pass
    
    @abstractmethod
    def batch_rerank(
        self, 
        query: str, 
        chunks: List[Dict[str, Any]], 
        batch_size: Optional[int] = None
    ) -> List[RerankedChunk]:
        """Rerank chunks in batches for efficiency."""
        pass


class CrossEncoderReranker(BaseReranker):
    """
    High-performance cross-encoder reranker using HuggingFace models.
    
    Provides semantic reranking using cross-encoder models for optimal
    retrieval accuracy while maintaining low latency.
    """
    
    def __init__(
        self,
        config: Optional[RerankerConfig] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize cross-encoder reranker.
        
        Args:
            config: Configuration for reranker behavior
            model_name: Override model name from config
            device: Override device from config
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers and torch are required. Install with: "
                "pip install transformers torch sentence-transformers"
            )
        
        self.config = config or RerankerConfig()
        if model_name:
            self.config.model_name = model_name
        if device:
            self.config.device = device
            
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self._load_model()
        
        # Initialize cache
        self.cache = None
        if self.config.enable_caching and REDIS_AVAILABLE:
            try:
                self.cache = redis.Redis(host='localhost', port=6379, db=0)
                self.logger.info("Connected to Redis cache for reranking")
            except Exception as e:
                self.logger.warning(f"Failed to connect to Redis: {e}")
        
        # Performance tracking
        self.stats = {
            'rerank_count': 0,
            'total_rerank_time': 0.0,
            'batch_count': 0,
            'total_batch_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'early_exits': 0
        }
    
    def _load_model(self):
        """Load cross-encoder model and tokenizer."""
        try:
            self.logger.info(f"Loading cross-encoder model: {self.config.model_name}")
            
            # Determine device
            if self.config.device == "auto":
                device = "cuda" if torch.cuda.is_available() and self.config.enable_gpu else "cpu"
            else:
                device = self.config.device
            
            # Load model using sentence-transformers CrossEncoder
            self.model = CrossEncoder(
                self.config.model_name,
                device=device,
                max_length=self.config.max_length
            )
            
            self.logger.info(f"Cross-encoder model loaded successfully on {device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load cross-encoder model: {e}")
            raise
    
    def rerank(
        self, 
        query: str, 
        chunks: List[Dict[str, Any]], 
        top_k: Optional[int] = None
    ) -> List[RerankedChunk]:
        """
        Rerank chunks based on query relevance.
        
        Args:
            query: User query string
            chunks: List of chunk dictionaries with text and scores
            top_k: Number of top chunks to return (uses config if None)
            
        Returns:
            List[RerankedChunk]: Reranked chunks sorted by relevance
        """
        start_time = time.time()
        top_k = top_k or self.config.top_k
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(query, chunks, top_k)
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                self.stats['cache_hits'] += 1
                return cached_result
            
            self.stats['cache_misses'] += 1
            
            self.logger.info(f"Reranking {len(chunks)} chunks for query: '{query[:100]}...'")
            
            # Early exit check for high-scoring chunks
            high_score_chunks = self._check_early_exit(chunks)
            if high_score_chunks:
                self.stats['early_exits'] += 1
                self.logger.info("Early exit triggered - high-scoring chunks found")
                return high_score_chunks[:top_k]
            
            # Prepare text pairs for cross-encoder
            text_pairs = [(query, chunk.get('text', '')) for chunk in chunks]
            
            # Get rerank scores
            rerank_scores = self.model.predict(text_pairs, batch_size=self.config.batch_size)
            
            # Create reranked chunks
            reranked_chunks = []
            for i, chunk in enumerate(chunks):
                reranked_chunk = RerankedChunk(
                    text=chunk.get('text', ''),
                    original_score=chunk.get('score', 0.0),
                    rerank_score=float(rerank_scores[i]),
                    metadata=chunk.get('metadata', {}),
                    chunk_id=chunk.get('id', f"chunk_{i}")
                )
                reranked_chunks.append(reranked_chunk)
            
            # Normalize scores
            reranked_chunks = self._normalize_scores(reranked_chunks)
            
            # Sort by rerank score and return top_k
            reranked_chunks.sort(key=lambda x: x.rerank_score, reverse=True)
            final_chunks = reranked_chunks[:top_k]
            
            # Cache results
            self._cache_result(cache_key, final_chunks)
            
            # Update stats
            rerank_time = time.time() - start_time
            self.stats['rerank_count'] += 1
            self.stats['total_rerank_time'] += rerank_time
            
            self.logger.info(f"Reranking completed in {rerank_time:.3f}s, returned {len(final_chunks)} chunks")
            
            if self.config.debug_mode:
                self._log_debug_info(query, final_chunks)
            
            return final_chunks
            
        except Exception as e:
            self.logger.error(f"Failed to rerank chunks: {e}")
            # Return original chunks as fallback
            return self._create_fallback_chunks(chunks, top_k)
    
    def batch_rerank(
        self, 
        query: str, 
        chunks: List[Dict[str, Any]], 
        batch_size: Optional[int] = None
    ) -> List[RerankedChunk]:
        """
        Rerank chunks in batches for efficiency.
        
        Args:
            query: User query string
            chunks: List of chunk dictionaries
            batch_size: Batch size for processing (uses config if None)
            
        Returns:
            List[RerankedChunk]: Reranked chunks
        """
        batch_size = batch_size or self.config.batch_size
        start_time = time.time()
        
        try:
            self.logger.info(f"Batch reranking {len(chunks)} chunks with batch size {batch_size}")
            
            # Process in batches
            all_reranked_chunks = []
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                batch_result = self.rerank(query, batch_chunks)
                all_reranked_chunks.extend(batch_result)
            
            # Sort all results by rerank score
            all_reranked_chunks.sort(key=lambda x: x.rerank_score, reverse=True)
            
            batch_time = time.time() - start_time
            self.stats['batch_count'] += 1
            self.stats['total_batch_time'] += batch_time
            
            self.logger.info(f"Batch reranking completed in {batch_time:.3f}s")
            return all_reranked_chunks
            
        except Exception as e:
            self.logger.error(f"Failed to batch rerank: {e}")
            return self._create_fallback_chunks(chunks, len(chunks))
    
    def _check_early_exit(self, chunks: List[Dict[str, Any]]) -> Optional[List[RerankedChunk]]:
        """Check if early exit should be triggered based on high scores."""
        if not chunks:
            return None
        
        # Check for very high BM25 scores (indicates strong keyword match)
        high_score_chunks = []
        for chunk in chunks:
            score = chunk.get('score', 0.0)
            if score >= self.config.early_exit_threshold:
                reranked_chunk = RerankedChunk(
                    text=chunk.get('text', ''),
                    original_score=score,
                    rerank_score=score,  # Use original score
                    metadata=chunk.get('metadata', {}),
                    chunk_id=chunk.get('id', '')
                )
                high_score_chunks.append(reranked_chunk)
        
        if high_score_chunks:
            # Sort by original score and return
            high_score_chunks.sort(key=lambda x: x.original_score, reverse=True)
            return high_score_chunks
        
        return None
    
    def _normalize_scores(self, chunks: List[RerankedChunk]) -> List[RerankedChunk]:
        """Normalize rerank scores using configured method."""
        if not chunks:
            return chunks
        
        scores = [chunk.rerank_score for chunk in chunks]
        scores = np.array(scores)
        
        if self.config.score_normalization == "minmax":
            if scores.max() == scores.min():
                normalized_scores = np.ones_like(scores)
            else:
                normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
        
        elif self.config.score_normalization == "zscore":
            mean = scores.mean()
            std = scores.std()
            if std == 0:
                normalized_scores = np.ones_like(scores)
            else:
                normalized_scores = (scores - mean) / std
                # Convert to 0-1 range
                normalized_scores = (normalized_scores - normalized_scores.min()) / (normalized_scores.max() - normalized_scores.min())
        
        elif self.config.score_normalization == "sigmoid":
            normalized_scores = 1 / (1 + np.exp(-scores))
        
        else:
            normalized_scores = scores
        
        # Update chunks with normalized scores
        for i, chunk in enumerate(chunks):
            chunk.rerank_score = float(normalized_scores[i])
        
        return chunks
    
    def _create_fallback_chunks(self, chunks: List[Dict[str, Any]], top_k: int) -> List[RerankedChunk]:
        """Create fallback chunks when reranking fails."""
        fallback_chunks = []
        for i, chunk in enumerate(chunks[:top_k]):
            fallback_chunk = RerankedChunk(
                text=chunk.get('text', ''),
                original_score=chunk.get('score', 0.0),
                rerank_score=chunk.get('score', 0.0),  # Use original score
                metadata=chunk.get('metadata', {}),
                chunk_id=chunk.get('id', f"chunk_{i}")
            )
            fallback_chunks.append(fallback_chunk)
        
        return fallback_chunks
    
    def _generate_cache_key(self, query: str, chunks: List[Dict[str, Any]], top_k: int) -> str:
        """Generate cache key for reranking results."""
        # Create a hash of query and chunk texts
        chunk_texts = [chunk.get('text', '')[:100] for chunk in chunks]  # Limit text length
        key_data = {
            'query': query.lower().strip(),
            'chunk_texts': chunk_texts,
            'top_k': top_k,
            'model': self.config.model_name
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return f"rerank:{hashlib.md5(key_string.encode()).hexdigest()}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[List[RerankedChunk]]:
        """Get reranking result from cache."""
        if not self.cache:
            return None
        
        try:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                data = json.loads(cached_data)
                return [RerankedChunk(**item) for item in data]
        except Exception as e:
            self.logger.debug(f"Failed to get from cache: {e}")
        
        return None
    
    def _cache_result(self, cache_key: str, chunks: List[RerankedChunk]) -> None:
        """Cache reranking result."""
        if not self.cache:
            return
        
        try:
            data = [chunk.to_dict() for chunk in chunks]
            self.cache.setex(
                cache_key,
                self.config.cache_ttl,
                json.dumps(data)
            )
        except Exception as e:
            self.logger.debug(f"Failed to cache result: {e}")
    
    def _log_debug_info(self, query: str, chunks: List[RerankedChunk]):
        """Log debug information for fine-tuning."""
        self.logger.debug(f"Query: {query}")
        self.logger.debug(f"Reranked {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks[:3]):  # Log top 3
            self.logger.debug(f"  {i+1}. Score: {chunk.rerank_score:.3f}, Text: {chunk.text[:100]}...")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.stats.copy()
        if stats['rerank_count'] > 0:
            stats['avg_rerank_time'] = stats['total_rerank_time'] / stats['rerank_count']
            stats['cache_hit_rate'] = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])
        if stats['batch_count'] > 0:
            stats['avg_batch_time'] = stats['total_batch_time'] / stats['batch_count']
        return stats
    
    def clear_stats(self) -> None:
        """Clear performance statistics."""
        self.stats = {
            'rerank_count': 0,
            'total_rerank_time': 0.0,
            'batch_count': 0,
            'total_batch_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'early_exits': 0
        }


class MockReranker(BaseReranker):
    """Mock reranker for testing and development."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    def rerank(
        self, 
        query: str, 
        chunks: List[Dict[str, Any]], 
        top_k: Optional[int] = None
    ) -> List[RerankedChunk]:
        """Mock reranking that returns chunks with slightly modified scores."""
        top_k = top_k or 5
        
        reranked_chunks = []
        for i, chunk in enumerate(chunks[:top_k]):
            # Add some randomness to simulate reranking
            original_score = chunk.get('score', 0.0)
            rerank_score = original_score + np.random.uniform(-0.1, 0.1)
            rerank_score = max(0.0, min(1.0, rerank_score))  # Clamp to [0, 1]
            
            reranked_chunk = RerankedChunk(
                text=chunk.get('text', ''),
                original_score=original_score,
                rerank_score=rerank_score,
                metadata=chunk.get('metadata', {}),
                chunk_id=chunk.get('id', f"chunk_{i}")
            )
            reranked_chunks.append(reranked_chunk)
        
        # Sort by rerank score
        reranked_chunks.sort(key=lambda x: x.rerank_score, reverse=True)
        return reranked_chunks
    
    def batch_rerank(
        self, 
        query: str, 
        chunks: List[Dict[str, Any]], 
        batch_size: Optional[int] = None
    ) -> List[RerankedChunk]:
        """Mock batch reranking."""
        return self.rerank(query, chunks, len(chunks))


# Factory functions for easy usage
def create_reranker(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    top_k: int = 5,
    **kwargs
) -> BaseReranker:
    """
    Create a reranker with default configuration.
    
    Args:
        model_name: Cross-encoder model name
        device: Device for inference ("auto", "cpu", "cuda")
        top_k: Number of top chunks to return
        **kwargs: Additional configuration parameters
        
    Returns:
        BaseReranker: Configured reranker instance
    """
    if not TRANSFORMERS_AVAILABLE:
        logger.warning("Transformers not available, using mock reranker")
        return MockReranker()
    
    config = RerankerConfig(
        model_name=model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device=device or "auto",
        top_k=top_k,
        **kwargs
    )
    
    return CrossEncoderReranker(config)


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create reranker
    reranker = create_reranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k=3,
        debug_mode=True
    )
    
    # Test query and chunks
    query = "What is the liability coverage for auto insurance?"
    
    test_chunks = [
        {
            "text": "Auto insurance liability coverage protects you from financial loss if you cause an accident.",
            "score": 0.85,
            "metadata": {"source": "policy_doc_1", "page": 5},
            "id": "chunk_1"
        },
        {
            "text": "Liability coverage includes bodily injury and property damage protection.",
            "score": 0.78,
            "metadata": {"source": "policy_doc_1", "page": 6},
            "id": "chunk_2"
        },
        {
            "text": "Premium payments are due monthly and can be made online or by phone.",
            "score": 0.65,
            "metadata": {"source": "policy_doc_2", "page": 12},
            "id": "chunk_3"
        },
        {
            "text": "The policy covers medical expenses for injuries sustained in covered accidents.",
            "score": 0.72,
            "metadata": {"source": "policy_doc_1", "page": 8},
            "id": "chunk_4"
        },
        {
            "text": "Liability limits are clearly stated in your insurance policy document.",
            "score": 0.88,
            "metadata": {"source": "policy_doc_1", "page": 7},
            "id": "chunk_5"
        }
    ]
    
    # Perform reranking
    print(f"Reranking chunks for query: '{query}'")
    reranked_chunks = reranker.rerank(query, test_chunks, top_k=3)
    
    print(f"\nReranked {len(reranked_chunks)} chunks:")
    for i, chunk in enumerate(reranked_chunks):
        print(f"{i+1}. Rerank Score: {chunk.rerank_score:.3f}")
        print(f"   Original Score: {chunk.original_score:.3f}")
        print(f"   Text: {chunk.text}")
        print(f"   ID: {chunk.chunk_id}")
        print()
    
    # Get stats
    print(f"Reranker stats: {reranker.get_stats()}")
