"""
High-Performance Context Compression for RAG Systems

This module implements intelligent context compression to optimize retrieved chunks
for LLM consumption while preserving critical information and maintaining accuracy.

Author: RAG System Team
License: MIT
"""

import logging
import time
import asyncio
import hashlib
import json
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Any, Tuple, Set
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from collections import defaultdict, Counter

# Optional imports with graceful fallbacks
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    torch = None
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None
    pipeline = None
    SentenceTransformer = None

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class CompressedChunk:
    """Represents a compressed chunk with optimized content."""
    summary: str
    source_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance_score: float = 0.0
    compression_ratio: float = 0.0
    original_score: float = 0.0
    chunk_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'summary': self.summary,
            'source_text': self.source_text,
            'metadata': self.metadata,
            'importance_score': self.importance_score,
            'compression_ratio': self.compression_ratio,
            'original_score': self.original_score,
            'chunk_id': self.chunk_id
        }


@dataclass
class CompressionConfig:
    """Configuration for context compression."""
    max_context_tokens: int = 3000
    top_k_chunks: int = 10
    summarization_model: str = "facebook/bart-large-cnn"  # Fast and effective
    similarity_threshold: float = 0.85  # For redundancy removal
    min_importance_score: float = 0.3
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour
    batch_size: int = 8
    max_summary_length: int = 150
    enable_gpu: bool = True
    fallback_to_extractive: bool = True
    debug_mode: bool = False


class BaseCompressor(ABC):
    """Abstract base class for context compressors."""
    
    @abstractmethod
    def compress(
        self, 
        query: str, 
        chunks: List[Dict[str, Any]], 
        token_budget: Optional[int] = None
    ) -> List[CompressedChunk]:
        """Compress chunks into optimized context."""
        pass


class ContextCompressor(BaseCompressor):
    """
    High-performance context compressor using query-aware summarization.
    
    Provides intelligent compression of retrieved chunks while preserving
    critical information and maintaining relevance to the query.
    """
    
    def __init__(
        self,
        config: Optional[CompressionConfig] = None,
        model_name: Optional[str] = None
    ):
        """
        Initialize context compressor.
        
        Args:
            config: Configuration for compression behavior
            model_name: Override summarization model from config
        """
        self.config = config or CompressionConfig()
        if model_name:
            self.config.summarization_model = model_name
            
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Initialize models
        self.summarizer = None
        self.similarity_model = None
        self.tokenizer = None
        self._load_models()
        
        # Initialize cache
        self.cache = None
        if self.config.enable_caching and REDIS_AVAILABLE:
            try:
                self.cache = redis.Redis(host='localhost', port=6379, db=0)
                self.logger.info("Connected to Redis cache for compression")
            except Exception as e:
                self.logger.warning(f"Failed to connect to Redis: {e}")
        
        # Performance tracking
        self.stats = {
            'compression_count': 0,
            'total_compression_time': 0.0,
            'total_tokens_before': 0,
            'total_tokens_after': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'fallback_usage': 0
        }
    
    def _load_models(self):
        """Load summarization and similarity models."""
        if not TRANSFORMERS_AVAILABLE:
            self.logger.warning("Transformers not available, using extractive fallback")
            return
        
        try:
            # Load summarization model
            self.logger.info(f"Loading summarization model: {self.config.summarization_model}")
            
            device = "cuda" if torch.cuda.is_available() and self.config.enable_gpu else "cpu"
            
            self.summarizer = pipeline(
                "summarization",
                model=self.config.summarization_model,
                device=0 if device == "cuda" else -1,
                max_length=self.config.max_summary_length,
                min_length=30,
                do_sample=False
            )
            
            # Load similarity model for redundancy detection
            self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Load tokenizer for token counting
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.summarization_model)
            
            self.logger.info(f"Models loaded successfully on {device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            self.summarizer = None
            self.similarity_model = None
            self.tokenizer = None

    def compress(
        self, 
        query: str, 
        chunks: List[Dict[str, Any]], 
        token_budget: Optional[int] = None
    ) -> List[CompressedChunk]:
        """
        Compress chunks into optimized context.
        
        Args:
            query: User query string
            chunks: List of chunk dictionaries with text and scores
            token_budget: Maximum tokens allowed (uses config if None)
            
        Returns:
            List[CompressedChunk]: Compressed chunks optimized for LLM
        """
        start_time = time.time()
        token_budget = token_budget or self.config.max_context_tokens
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(query, chunks, token_budget)
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                self.stats['cache_hits'] += 1
                return cached_result
            
            self.stats['cache_misses'] += 1
            
            self.logger.info(f"Compressing {len(chunks)} chunks for query: '{query[:100]}...'")
            
            # Filter chunks by importance
            filtered_chunks = self._filter_by_importance(chunks)
            
            # Remove redundant chunks
            unique_chunks = self._remove_redundancy(filtered_chunks)
            
            # Summarize chunks with query focus
            summarized_chunks = self._summarize_chunks(query, unique_chunks)
            
            # Optimize for token budget
            optimized_chunks = self._optimize_for_budget(summarized_chunks, token_budget)
            
            # Cache results
            self._cache_result(cache_key, optimized_chunks)
            
            # Update stats
            compression_time = time.time() - start_time
            self.stats['compression_count'] += 1
            self.stats['total_compression_time'] += compression_time
            
            # Calculate token statistics
            tokens_before = sum(len(chunk.get('text', '').split()) for chunk in chunks)
            tokens_after = sum(len(chunk.summary.split()) for chunk in optimized_chunks)
            self.stats['total_tokens_before'] += tokens_before
            self.stats['total_tokens_after'] += tokens_after
            
            compression_ratio = tokens_after / max(tokens_before, 1)
            
            self.logger.info(
                f"Compression completed in {compression_time:.3f}s, "
                f"ratio: {compression_ratio:.2f}, "
                f"tokens: {tokens_before} → {tokens_after}"
            )
            
            if self.config.debug_mode:
                self._log_debug_info(query, optimized_chunks, compression_ratio)
            
            return optimized_chunks
            
        except Exception as e:
            self.logger.error(f"Failed to compress chunks: {e}")
            # Return fallback compression
            return self._fallback_compression(chunks, token_budget)
    
    def _filter_by_importance(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter chunks by importance score and query relevance."""
        if not chunks:
            return []
        
        # Calculate importance scores
        scored_chunks = []
        for chunk in chunks:
            importance_score = self._calculate_importance_score(chunk)
            if importance_score >= self.config.min_importance_score:
                chunk_copy = chunk.copy()
                chunk_copy['importance_score'] = importance_score
                scored_chunks.append(chunk_copy)
        
        # Sort by importance and take top_k
        scored_chunks.sort(key=lambda x: x['importance_score'], reverse=True)
        return scored_chunks[:self.config.top_k_chunks]
    
    def _calculate_importance_score(self, chunk: Dict[str, Any]) -> float:
        """Calculate importance score for a chunk."""
        # Base score from rerank score
        base_score = chunk.get('rerank_score', chunk.get('score', 0.0))
        
        # Boost for metadata importance (e.g., page number, source)
        metadata_boost = 0.0
        metadata = chunk.get('metadata', {})
        
        # Boost for policy-related sources
        if 'policy' in str(metadata.get('source', '')).lower():
            metadata_boost += 0.1
        
        # Boost for early pages (often contain key information)
        page_num = metadata.get('page_number', 0)
        if page_num <= 10:
            metadata_boost += 0.05
        
        return min(1.0, base_score + metadata_boost)

    def _remove_redundancy(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove redundant chunks using semantic similarity."""
        if not chunks or not self.similarity_model:
            return chunks
        
        unique_chunks = [chunks[0]]
        
        for chunk in chunks[1:]:
            is_redundant = False
            
            # Check similarity with existing chunks
            for existing_chunk in unique_chunks:
                similarity = self._calculate_similarity(
                    chunk.get('text', ''),
                    existing_chunk.get('text', '')
                )
                
                if similarity > self.config.similarity_threshold:
                    is_redundant = True
                    # Keep the one with higher importance score
                    if chunk.get('importance_score', 0) > existing_chunk.get('importance_score', 0):
                        unique_chunks.remove(existing_chunk)
                        unique_chunks.append(chunk)
                    break
            
            if not is_redundant:
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        try:
            embeddings = self.similarity_model.encode([text1, text2])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except Exception as e:
            self.logger.debug(f"Failed to calculate similarity: {e}")
            # Fallback to simple word overlap
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            overlap = len(words1.intersection(words2)) / len(words1.union(words2))
            return overlap
    
    def _summarize_chunks(self, query: str, chunks: List[Dict[str, Any]]) -> List[CompressedChunk]:
        """Summarize chunks with query focus."""
        if not chunks:
            return []
        
        compressed_chunks = []
        
        for chunk in chunks:
            try:
                # Create query-aware summary
                if self.summarizer:
                    summary = self._abstractive_summarize(query, chunk.get('text', ''))
                else:
                    summary = self._extractive_summarize(query, chunk.get('text', ''))
                
                # Calculate compression ratio
                original_length = len(chunk.get('text', '').split())
                summary_length = len(summary.split())
                compression_ratio = summary_length / max(original_length, 1)
                
                compressed_chunk = CompressedChunk(
                    summary=summary,
                    source_text=chunk.get('text', ''),
                    metadata=chunk.get('metadata', {}),
                    importance_score=chunk.get('importance_score', 0.0),
                    compression_ratio=compression_ratio,
                    original_score=chunk.get('rerank_score', chunk.get('score', 0.0)),
                    chunk_id=chunk.get('id', '')
                )
                compressed_chunks.append(compressed_chunk)
                
            except Exception as e:
                self.logger.error(f"Failed to summarize chunk: {e}")
                # Use extractive fallback
                summary = self._extractive_summarize(query, chunk.get('text', ''))
                compressed_chunk = CompressedChunk(
                    summary=summary,
                    source_text=chunk.get('text', ''),
                    metadata=chunk.get('metadata', {}),
                    importance_score=chunk.get('importance_score', 0.0),
                    compression_ratio=0.5,  # Estimate
                    original_score=chunk.get('rerank_score', chunk.get('score', 0.0)),
                    chunk_id=chunk.get('id', '')
                )
                compressed_chunks.append(compressed_chunk)
        
        return compressed_chunks
    
    def _abstractive_summarize(self, query: str, text: str) -> str:
        """Generate abstractive summary with query focus."""
        try:
            # Create query-aware prompt
            prompt = f"Query: {query}\n\nText: {text}\n\nSummary focusing on query relevance:"
            
            # Truncate text if too long for model
            max_input_length = 1024
            if len(prompt.split()) > max_input_length:
                # Keep query and truncate text
                query_part = f"Query: {query}\n\n"
                available_length = max_input_length - len(query_part.split()) - 10  # Buffer
                truncated_text = " ".join(text.split()[:available_length])
                prompt = f"{query_part}Text: {truncated_text}\n\nSummary focusing on query relevance:"
            
            # Generate summary
            result = self.summarizer(prompt, max_length=self.config.max_summary_length, min_length=30)
            summary = result[0]['summary_text']
            
            # Clean up summary
            summary = re.sub(r'^Summary focusing on query relevance:\s*', '', summary)
            summary = summary.strip()
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Abstractive summarization failed: {e}")
            self.stats['fallback_usage'] += 1
            return self._extractive_summarize(query, text)
    
    def _extractive_summarize(self, query: str, text: str) -> str:
        """Generate extractive summary by selecting key sentences."""
        try:
            # Split into sentences
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if not sentences:
                return text[:self.config.max_summary_length]
            
            # Score sentences by query relevance
            query_words = set(query.lower().split())
            sentence_scores = []
            
            for sentence in sentences:
                sentence_words = set(sentence.lower().split())
                overlap = len(query_words.intersection(sentence_words))
                score = overlap / max(len(query_words), 1)
                sentence_scores.append((sentence, score))
            
            # Sort by score and take top sentences
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Select sentences until we reach target length
            selected_sentences = []
            current_length = 0
            
            for sentence, score in sentence_scores:
                if current_length + len(sentence.split()) <= self.config.max_summary_length:
                    selected_sentences.append(sentence)
                    current_length += len(sentence.split())
                else:
                    break
            
            # If no sentences selected, take first few
            if not selected_sentences:
                selected_sentences = sentences[:3]
            
            summary = ". ".join(selected_sentences) + "."
            return summary
            
        except Exception as e:
            self.logger.error(f"Extractive summarization failed: {e}")
            # Last resort: truncate text
            return text[:self.config.max_summary_length]

    def _optimize_for_budget(self, chunks: List[CompressedChunk], token_budget: int) -> List[CompressedChunk]:
        """Optimize chunks to fit within token budget."""
        if not chunks:
            return []
        
        # Estimate tokens for each chunk
        total_tokens = 0
        optimized_chunks = []
        
        for chunk in chunks:
            estimated_tokens = len(chunk.summary.split())  # Rough estimation
            if total_tokens + estimated_tokens <= token_budget:
                optimized_chunks.append(chunk)
                total_tokens += estimated_tokens
            else:
                # Try to truncate this chunk
                remaining_tokens = token_budget - total_tokens
                if remaining_tokens > 50:  # Minimum meaningful length
                    truncated_summary = " ".join(chunk.summary.split()[:remaining_tokens])
                    chunk_copy = CompressedChunk(
                        summary=truncated_summary + "...",
                        source_text=chunk.source_text,
                        metadata=chunk.metadata,
                        importance_score=chunk.importance_score,
                        compression_ratio=chunk.compression_ratio,
                        original_score=chunk.original_score,
                        chunk_id=chunk.chunk_id
                    )
                    optimized_chunks.append(chunk_copy)
                break
        
        return optimized_chunks
    
    def _fallback_compression(self, chunks: List[Dict[str, Any]], token_budget: int) -> List[CompressedChunk]:
        """Fallback compression when main methods fail."""
        fallback_chunks = []
        
        for i, chunk in enumerate(chunks):
            text = chunk.get('text', '')
            # Simple truncation
            words = text.split()
            truncated_text = " ".join(words[:min(100, len(words))])
            
            fallback_chunk = CompressedChunk(
                summary=truncated_text,
                source_text=text,
                metadata=chunk.get('metadata', {}),
                importance_score=chunk.get('rerank_score', chunk.get('score', 0.0)),
                compression_ratio=0.5,
                original_score=chunk.get('rerank_score', chunk.get('score', 0.0)),
                chunk_id=chunk.get('id', f"chunk_{i}")
            )
            fallback_chunks.append(fallback_chunk)
        
        return self._optimize_for_budget(fallback_chunks, token_budget)
    
    def _generate_cache_key(self, query: str, chunks: List[Dict[str, Any]], token_budget: int) -> str:
        """Generate cache key for compression results."""
        # Create a hash of query and chunk texts
        chunk_texts = [chunk.get('text', '')[:100] for chunk in chunks]
        key_data = {
            'query': query.lower().strip(),
            'chunk_texts': chunk_texts,
            'token_budget': token_budget,
            'model': self.config.summarization_model
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return f"compress:{hashlib.md5(key_string.encode()).hexdigest()}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[List[CompressedChunk]]:
        """Get compression result from cache."""
        if not self.cache:
            return None
        
        try:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                data = json.loads(cached_data)
                return [CompressedChunk(**item) for item in data]
        except Exception as e:
            self.logger.debug(f"Failed to get from cache: {e}")
        
        return None
    
    def _cache_result(self, cache_key: str, chunks: List[CompressedChunk]) -> None:
        """Cache compression result."""
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
    
    def _log_debug_info(self, query: str, chunks: List[CompressedChunk], compression_ratio: float):
        """Log debug information for fine-tuning."""
        self.logger.debug(f"Query: {query}")
        self.logger.debug(f"Compression ratio: {compression_ratio:.2f}")
        self.logger.debug(f"Compressed {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks[:3]):  # Log top 3
            self.logger.debug(f"  {i+1}. Score: {chunk.importance_score:.3f}, Summary: {chunk.summary[:100]}...")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.stats.copy()
        if stats['compression_count'] > 0:
            stats['avg_compression_time'] = stats['total_compression_time'] / stats['compression_count']
            stats['avg_compression_ratio'] = stats['total_tokens_after'] / max(stats['total_tokens_before'], 1)
            stats['cache_hit_rate'] = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])
        return stats
    
    def clear_stats(self) -> None:
        """Clear performance statistics."""
        self.stats = {
            'compression_count': 0,
            'total_compression_time': 0.0,
            'total_tokens_before': 0,
            'total_tokens_after': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'fallback_usage': 0
        }


class MockCompressor(BaseCompressor):
    """Mock compressor for testing and development."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    def compress(
        self, 
        query: str, 
        chunks: List[Dict[str, Any]], 
        token_budget: Optional[int] = None
    ) -> List[CompressedChunk]:
        """Mock compression that truncates text."""
        token_budget = token_budget or 3000
        
        compressed_chunks = []
        for i, chunk in enumerate(chunks):
            text = chunk.get('text', '')
            # Simple truncation to simulate compression
            words = text.split()
            truncated_text = " ".join(words[:min(50, len(words))])
            
            compressed_chunk = CompressedChunk(
                summary=truncated_text,
                source_text=text,
                metadata=chunk.get('metadata', {}),
                importance_score=chunk.get('rerank_score', chunk.get('score', 0.0)),
                compression_ratio=0.5,
                original_score=chunk.get('rerank_score', chunk.get('score', 0.0)),
                chunk_id=chunk.get('id', f"chunk_{i}")
            )
            compressed_chunks.append(compressed_chunk)
        
        return compressed_chunks


# Factory functions for easy usage
def create_compressor(
    model_name: Optional[str] = None,
    max_context_tokens: int = 3000,
    **kwargs
) -> BaseCompressor:
    """
    Create a context compressor with default configuration.
    
    Args:
        model_name: Summarization model name
        max_context_tokens: Maximum tokens for compressed context
        **kwargs: Additional configuration parameters
        
    Returns:
        BaseCompressor: Configured compressor instance
    """
    if not TRANSFORMERS_AVAILABLE:
        logger.warning("Transformers not available, using mock compressor")
        return MockCompressor()
    
    config = CompressionConfig(
        summarization_model=model_name or "facebook/bart-large-cnn",
        max_context_tokens=max_context_tokens,
        **kwargs
    )
    
    return ContextCompressor(config)


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create compressor
    compressor = create_compressor(
        model_name="facebook/bart-large-cnn",
        max_context_tokens=2000,
        debug_mode=True
    )
    
    # Test query and chunks
    query = "What is the liability coverage for auto insurance?"
    
    test_chunks = [
        {
            "text": "Auto insurance liability coverage protects you from financial loss if you cause an accident. This coverage includes bodily injury and property damage protection. The policy covers medical expenses for injuries sustained in covered accidents. Liability limits are clearly stated in your insurance policy document.",
            "rerank_score": 0.95,
            "metadata": {"source": "policy_doc_1", "page": 5},
            "id": "chunk_1"
        },
        {
            "text": "Premium payments are due monthly and can be made online or by phone. Late payments may result in policy cancellation. Payment methods include credit card, bank transfer, and automatic deduction. Contact customer service for payment arrangements.",
            "rerank_score": 0.78,
            "metadata": {"source": "policy_doc_2", "page": 12},
            "id": "chunk_2"
        },
        {
            "text": "Liability coverage includes bodily injury and property damage protection. Bodily injury covers medical expenses, lost wages, and pain and suffering. Property damage covers repair or replacement of damaged property. Coverage limits vary by policy type.",
            "rerank_score": 0.88,
            "metadata": {"source": "policy_doc_1", "page": 6},
            "id": "chunk_3"
        }
    ]
    
    # Perform compression
    print(f"Compressing chunks for query: '{query}'")
    compressed_chunks = compressor.compress(query, test_chunks, token_budget=2000)
    
    print(f"\nCompressed {len(compressed_chunks)} chunks:")
    for i, chunk in enumerate(compressed_chunks):
        print(f"{i+1}. Importance Score: {chunk.importance_score:.3f}")
        print(f"   Compression Ratio: {chunk.compression_ratio:.2f}")
        print(f"   Summary: {chunk.summary}")
        print(f"   ID: {chunk.chunk_id}")
        print()
    
    # Get stats
    print(f"Compressor stats: {compressor.get_stats()}")
