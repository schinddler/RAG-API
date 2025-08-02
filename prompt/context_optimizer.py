import asyncio
import time
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import re
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

# Import dependencies (assuming these exist in the project)
try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError:
    SentenceTransformer = None
    faiss = None

try:
    from ..llm.token_estimator import estimate_tokens, get_token_limit
    from ..monitoring.logs import log_event, log_success, log_failure, log_latency
    from ..monitoring.alerting import get_alert_manager
    from ..config import MODEL_CONFIG
except ImportError:
    # Fallback imports for development
    def estimate_tokens(prompt: str, context: str, model_name: str = "gpt-4") -> int:
        return len(prompt.split()) + len(context.split())
    
    def get_token_limit(model_name: str) -> int:
        limits = {"gpt-4": 128000, "claude-3-sonnet": 200000, "mistral-large": 32000}
        return limits.get(model_name, 128000)
    
    def log_event(event_type: str, message: str, metadata: dict = None, **kwargs):
        logging.info(f"Context Optimizer - {event_type}: {message}")
    
    def log_success(module: str, action: str, details: dict = None, **kwargs):
        logging.info(f"Context Optimizer Success - {module}.{action}")
    
    def log_failure(module: str, action: str, error: Exception, details: dict = None, **kwargs):
        logging.error(f"Context Optimizer Failure - {module}.{action}: {error}")
    
    def log_latency(module: str, operation: str, duration_ms: float, details: dict = None, **kwargs):
        logging.info(f"Context Optimizer Latency - {module}.{operation}: {duration_ms:.2f}ms")
    
    def get_alert_manager():
        return None
    
    MODEL_CONFIG = {
        "gpt-4": {"max_tokens": 128000, "context_window": 128000},
        "claude-3-sonnet": {"max_tokens": 200000, "context_window": 200000},
        "mistral-large": {"max_tokens": 32000, "context_window": 32000}
    }


@dataclass
class ChunkInfo:
    """Information about a text chunk"""
    text: str
    relevance_score: float = 0.0
    token_count: int = 0
    embedding: Optional[np.ndarray] = None
    cluster_id: Optional[int] = None


class ContextOptimizer:
    """
    High-performance context optimizer for RAG systems with semantic ranking,
    token-aware compression, and redundancy removal.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the context optimizer.
        
        Args:
            config: Configuration dictionary with optimization settings
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.alert_manager = get_alert_manager()
        
        # Embedding model
        self.embedding_model = None
        self.embedding_model_name = self.config.get("embedding_model", "all-MiniLM-L6-v2")
        
        # Optimization settings
        self.min_context_tokens = self.config.get("min_context_tokens", 100)
        self.max_context_tokens = self.config.get("max_context_tokens", 32000)
        self.relevance_threshold = self.config.get("relevance_threshold", 0.3)
        self.redundancy_removal = self.config.get("redundancy_removal", False)
        self.cluster_threshold = self.config.get("cluster_threshold", 0.85)
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize embedding model and other components"""
        try:
            if SentenceTransformer and self.embedding_model_name:
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                self.logger.info(f"Initialized embedding model: {self.embedding_model_name}")
                log_success("context_optimizer", "initialize", {"embedding_model": self.embedding_model_name})
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {e}")
            log_failure("context_optimizer", "initialize", e, {"embedding_model": self.embedding_model_name})
            if self.alert_manager:
                self.alert_manager.alert_on_api_failure("context_optimizer", e)
    
    async def compress_context(self, chunks: List[str], model: str, max_tokens: int) -> str:
        """
        Compress and optimize context chunks for the target model.
        
        Args:
            chunks: List of text chunks to compress
            model: Target model name
            max_tokens: Maximum tokens allowed for context
            
        Returns:
            Compressed context string
        """
        start_time = time.time()
        
        try:
            if not chunks:
                return ""
            
            # Get model-specific limits
            model_config = MODEL_CONFIG.get(model, {})
            context_window = model_config.get("context_window", max_tokens)
            effective_max_tokens = min(max_tokens, context_window)
            
            # Create chunk info objects
            chunk_infos = [ChunkInfo(text=chunk.strip()) for chunk in chunks if chunk.strip()]
            
            # Estimate token counts
            for chunk_info in chunk_infos:
                chunk_info.token_count = estimate_tokens("", chunk_info.text, model)
            
            # Filter by minimum relevance if query is available
            if len(chunk_infos) > 1:
                # Use first chunk as pseudo-query for ranking
                pseudo_query = chunk_infos[0].text[:200]  # Use first 200 chars as query
                chunk_infos = await self.rank_chunks_by_relevance(
                    [chunk.text for chunk in chunk_infos], 
                    pseudo_query, 
                    model
                )
                chunk_infos = [ChunkInfo(text=chunk) for chunk in chunk_infos]
                for chunk_info in chunk_infos:
                    chunk_info.token_count = estimate_tokens("", chunk_info.text, model)
            
            # Remove redundancy if enabled
            if self.redundancy_removal and len(chunk_infos) > 1:
                chunk_infos = await self._remove_redundancy(chunk_infos)
            
            # Compress to fit token limit
            compressed_chunks = await self._compress_to_token_limit(chunk_infos, effective_max_tokens, model)
            
            # Combine chunks
            final_context = "\n\n".join(compressed_chunks)
            
            # Validate final token count
            final_tokens = estimate_tokens("", final_context, model)
            
            # Log optimization results
            latency_ms = (time.time() - start_time) * 1000
            log_success("context_optimizer", "compress_context", {
                "original_chunks": len(chunks),
                "final_chunks": len(compressed_chunks),
                "original_tokens": sum(chunk.token_count for chunk in chunk_infos),
                "final_tokens": final_tokens,
                "compression_ratio": final_tokens / max(sum(chunk.token_count for chunk in chunk_infos), 1),
                "model": model,
                "latency_ms": latency_ms
            })
            log_latency("context_optimizer", "compress_context", latency_ms, {"model": model})
            
            # Alert if context is too small
            if final_tokens < self.min_context_tokens:
                if self.alert_manager:
                    self.alert_manager.send_alert("context_too_small", f"Context compressed to {final_tokens} tokens (min: {self.min_context_tokens})")
                log_event("context_warning", f"Context too small: {final_tokens} tokens", {
                    "final_tokens": final_tokens,
                    "min_tokens": self.min_context_tokens,
                    "model": model
                })
            
            return final_context
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            log_failure("context_optimizer", "compress_context", e, {
                "chunks_count": len(chunks),
                "model": model,
                "latency_ms": latency_ms
            })
            
            if self.alert_manager:
                self.alert_manager.alert_on_api_failure("context_optimizer", e)
            
            # Return original chunks as fallback
            return "\n\n".join(chunks)
    
    async def rank_chunks_by_relevance(self, chunks: List[str], query: str, model: str) -> List[str]:
        """
        Rank chunks by relevance to the query using semantic similarity.
        
        Args:
            chunks: List of text chunks to rank
            query: Query string for relevance scoring
            model: Model name for token estimation
            
        Returns:
            List of chunks ranked by relevance (most relevant first)
        """
        start_time = time.time()
        
        try:
            if not chunks or not query:
                return chunks
            
            if not self.embedding_model:
                # Fallback to simple keyword matching
                return self._rank_by_keywords(chunks, query)
            
            # Generate embeddings
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
            chunk_embeddings = self.embedding_model.encode(chunks, convert_to_numpy=True)
            
            # Calculate cosine similarities
            similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
            
            # Create chunk info with relevance scores
            chunk_infos = []
            for i, (chunk, similarity) in enumerate(zip(chunks, similarities)):
                chunk_info = ChunkInfo(
                    text=chunk,
                    relevance_score=float(similarity),
                    token_count=estimate_tokens("", chunk, model),
                    embedding=chunk_embeddings[i]
                )
                chunk_infos.append(chunk_info)
            
            # Sort by relevance score (descending)
            chunk_infos.sort(key=lambda x: x.relevance_score, reverse=True)
            
            # Log ranking results
            latency_ms = (time.time() - start_time) * 1000
            log_success("context_optimizer", "rank_chunks", {
                "chunks_count": len(chunks),
                "avg_relevance": np.mean([c.relevance_score for c in chunk_infos]),
                "max_relevance": max(c.relevance_score for c in chunk_infos),
                "latency_ms": latency_ms
            })
            log_latency("context_optimizer", "rank_chunks", latency_ms, {"chunks_count": len(chunks)})
            
            return [chunk.text for chunk in chunk_infos]
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            log_failure("context_optimizer", "rank_chunks", e, {
                "chunks_count": len(chunks),
                "latency_ms": latency_ms
            })
            
            # Return original order as fallback
            return chunks
    
    async def filter_irrelevant_chunks(self, chunks: List[str], query: str, threshold: float) -> List[str]:
        """
        Filter out chunks with relevance below threshold.
        
        Args:
            chunks: List of text chunks to filter
            query: Query string for relevance scoring
            threshold: Minimum relevance score (0.0 to 1.0)
            
        Returns:
            List of relevant chunks
        """
        start_time = time.time()
        
        try:
            if not chunks or not query:
                return chunks
            
            if not self.embedding_model:
                # Fallback to keyword-based filtering
                return self._filter_by_keywords(chunks, query, threshold)
            
            # Generate embeddings
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
            chunk_embeddings = self.embedding_model.encode(chunks, convert_to_numpy=True)
            
            # Calculate similarities and filter
            similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
            relevant_chunks = [chunk for chunk, sim in zip(chunks, similarities) if sim >= threshold]
            
            # Log filtering results
            latency_ms = (time.time() - start_time) * 1000
            filtered_count = len(relevant_chunks)
            log_success("context_optimizer", "filter_chunks", {
                "original_count": len(chunks),
                "filtered_count": filtered_count,
                "threshold": threshold,
                "avg_similarity": float(np.mean(similarities)),
                "latency_ms": latency_ms
            })
            log_latency("context_optimizer", "filter_chunks", latency_ms, {
                "original_count": len(chunks),
                "filtered_count": filtered_count
            })
            
            return relevant_chunks
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            log_failure("context_optimizer", "filter_chunks", e, {
                "chunks_count": len(chunks),
                "threshold": threshold,
                "latency_ms": latency_ms
            })
            
            # Return all chunks as fallback
            return chunks
    
    async def _remove_redundancy(self, chunk_infos: List[ChunkInfo]) -> List[ChunkInfo]:
        """Remove redundant chunks using clustering."""
        try:
            if len(chunk_infos) < 2:
                return chunk_infos
            
            # Get embeddings for clustering
            embeddings = []
            valid_chunks = []
            
            for chunk_info in chunk_infos:
                if chunk_info.embedding is not None:
                    embeddings.append(chunk_info.embedding)
                    valid_chunks.append(chunk_info)
                else:
                    # Generate embedding if missing
                    if self.embedding_model:
                        embedding = self.embedding_model.encode([chunk_info.text], convert_to_numpy=True)[0]
                        chunk_info.embedding = embedding
                        embeddings.append(embedding)
                        valid_chunks.append(chunk_info)
            
            if len(embeddings) < 2:
                return chunk_infos
            
            # Perform clustering
            embeddings_array = np.array(embeddings)
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=1 - self.cluster_threshold,
                linkage='ward'
            )
            
            cluster_labels = clustering.fit_predict(embeddings_array)
            
            # Select representative chunk from each cluster
            cluster_chunks = defaultdict(list)
            for chunk_info, label in zip(valid_chunks, cluster_labels):
                cluster_chunks[label].append(chunk_info)
            
            # Select best chunk from each cluster (highest relevance or longest)
            selected_chunks = []
            for cluster_id, cluster_chunk_list in cluster_chunks.items():
                # Select chunk with highest relevance score, or longest if tied
                best_chunk = max(cluster_chunk_list, 
                               key=lambda x: (x.relevance_score, len(x.text)))
                best_chunk.cluster_id = cluster_id
                selected_chunks.append(best_chunk)
            
            # Add chunks without embeddings
            for chunk_info in chunk_infos:
                if chunk_info.embedding is None:
                    selected_chunks.append(chunk_info)
            
            self.logger.info(f"Removed redundancy: {len(chunk_infos)} -> {len(selected_chunks)} chunks")
            return selected_chunks
            
        except Exception as e:
            self.logger.error(f"Failed to remove redundancy: {e}")
            return chunk_infos
    
    async def _compress_to_token_limit(self, chunk_infos: List[ChunkInfo], max_tokens: int, model: str) -> List[str]:
        """Compress chunks to fit within token limit."""
        try:
            if not chunk_infos:
                return []
            
            # Sort by relevance score (descending)
            chunk_infos.sort(key=lambda x: x.relevance_score, reverse=True)
            
            selected_chunks = []
            total_tokens = 0
            
            for chunk_info in chunk_infos:
                if total_tokens + chunk_info.token_count <= max_tokens:
                    selected_chunks.append(chunk_info.text)
                    total_tokens += chunk_info.token_count
                else:
                    # Try to truncate chunk to fit
                    truncated_text = self._truncate_chunk(chunk_info.text, max_tokens - total_tokens, model)
                    if truncated_text:
                        selected_chunks.append(truncated_text)
                        break
            
            return selected_chunks
            
        except Exception as e:
            self.logger.error(f"Failed to compress to token limit: {e}")
            # Return first chunk as fallback
            return [chunk_infos[0].text] if chunk_infos else []
    
    def _truncate_chunk(self, text: str, max_tokens: int, model: str) -> str:
        """Truncate text to fit within token limit."""
        try:
            # Simple word-based truncation
            words = text.split()
            estimated_tokens = len(words)
            
            if estimated_tokens <= max_tokens:
                return text
            
            # Truncate to fit
            truncated_words = words[:max_tokens]
            return " ".join(truncated_words)
            
        except Exception as e:
            self.logger.error(f"Failed to truncate chunk: {e}")
            return text[:max_tokens * 4]  # Rough character-based fallback
    
    def _rank_by_keywords(self, chunks: List[str], query: str) -> List[str]:
        """Fallback ranking using keyword matching."""
        try:
            query_words = set(query.lower().split())
            
            def keyword_score(chunk):
                chunk_words = set(chunk.lower().split())
                intersection = len(query_words.intersection(chunk_words))
                return intersection / max(len(query_words), 1)
            
            # Sort by keyword score
            scored_chunks = [(chunk, keyword_score(chunk)) for chunk in chunks]
            scored_chunks.sort(key=lambda x: x[1], reverse=True)
            
            return [chunk for chunk, score in scored_chunks]
            
        except Exception as e:
            self.logger.error(f"Failed to rank by keywords: {e}")
            return chunks
    
    def _filter_by_keywords(self, chunks: List[str], query: str, threshold: float) -> List[str]:
        """Fallback filtering using keyword matching."""
        try:
            query_words = set(query.lower().split())
            
            def keyword_score(chunk):
                chunk_words = set(chunk.lower().split())
                intersection = len(query_words.intersection(chunk_words))
                return intersection / max(len(query_words), 1)
            
            # Filter by keyword score
            relevant_chunks = []
            for chunk in chunks:
                score = keyword_score(chunk)
                if score >= threshold:
                    relevant_chunks.append(chunk)
            
            return relevant_chunks
            
        except Exception as e:
            self.logger.error(f"Failed to filter by keywords: {e}")
            return chunks
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return {
            "embedding_model": self.embedding_model_name,
            "min_context_tokens": self.min_context_tokens,
            "max_context_tokens": self.max_context_tokens,
            "relevance_threshold": self.relevance_threshold,
            "redundancy_removal": self.redundancy_removal,
            "cluster_threshold": self.cluster_threshold
        }


# Global instance for easy access
_context_optimizer = None


def get_context_optimizer(config: Optional[Dict[str, Any]] = None) -> ContextOptimizer:
    """Get the global context optimizer instance."""
    global _context_optimizer
    if _context_optimizer is None:
        _context_optimizer = ContextOptimizer(config)
    return _context_optimizer


# Convenience functions for easy access
async def compress_context(chunks: List[str], model: str, max_tokens: int) -> str:
    """Compress context chunks."""
    return await get_context_optimizer().compress_context(chunks, model, max_tokens)


async def rank_chunks_by_relevance(chunks: List[str], query: str, model: str) -> List[str]:
    """Rank chunks by relevance."""
    return await get_context_optimizer().rank_chunks_by_relevance(chunks, query, model)


async def filter_irrelevant_chunks(chunks: List[str], query: str, threshold: float) -> List[str]:
    """Filter irrelevant chunks."""
    return await get_context_optimizer().filter_irrelevant_chunks(chunks, query, threshold)


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    # Test the context optimizer
    async def test_context_optimizer():
        optimizer = ContextOptimizer({
            "embedding_model": "all-MiniLM-L6-v2",
            "min_context_tokens": 100,
            "max_context_tokens": 8000,
            "relevance_threshold": 0.3,
            "redundancy_removal": True,
            "cluster_threshold": 0.85
        })
        
        # Test chunks
        test_chunks = [
            "The insurance policy covers medical expenses up to $50,000 per year.",
            "Legal documents must be signed by all parties involved in the transaction.",
            "Financial statements show revenue of $1.2 million for the fiscal year.",
            "The contract specifies payment terms of net 30 days from invoice date.",
            "Risk assessment procedures require quarterly reviews and updates."
        ]
        
        query = "What are the insurance coverage limits?"
        
        # Test ranking
        ranked_chunks = await optimizer.rank_chunks_by_relevance(test_chunks, query, "gpt-4")
        print(f"Ranked chunks: {ranked_chunks[:2]}")
        
        # Test filtering
        filtered_chunks = await optimizer.filter_irrelevant_chunks(test_chunks, query, 0.3)
        print(f"Filtered chunks: {len(filtered_chunks)}")
        
        # Test compression
        compressed = await optimizer.compress_context(test_chunks, "gpt-4", 1000)
        print(f"Compressed context length: {len(compressed)}")
        
        # Print stats
        print(f"Optimizer stats: {optimizer.get_optimization_stats()}")
    
    asyncio.run(test_context_optimizer())
