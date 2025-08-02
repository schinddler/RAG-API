"""
Context deduplication module for RAG system.

This module removes redundant or highly similar chunks after retrieval
but before final context assembly to minimize LLM token load.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class DeduplicationConfig:
    """Configuration for context deduplication."""
    similarity_threshold: float = 0.85
    max_chunks: int = 8
    preserve_high_score_chunks: int = 3
    use_clustering: bool = False
    cluster_threshold: float = 0.8
    token_budget: Optional[int] = None
    min_chunk_length: int = 50


class ContextDeduplicator:
    """Deduplicates retrieved chunks based on semantic similarity."""
    
    def __init__(self, config: Optional[DeduplicationConfig] = None):
        """
        Initialize deduplicator.
        
        Args:
            config: Deduplication configuration
        """
        self.config = config or DeduplicationConfig()
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if not 0.0 <= self.config.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        
        if self.config.max_chunks < 1:
            raise ValueError("max_chunks must be at least 1")
        
        if self.config.preserve_high_score_chunks > self.config.max_chunks:
            raise ValueError("preserve_high_score_chunks cannot exceed max_chunks")
    
    def deduplicate(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate chunks based on semantic similarity.
        
        Args:
            chunks: List of chunks with 'text', 'score', and optional 'embedding' keys
            
        Returns:
            Deduplicated list of chunks
        """
        if not chunks:
            return []
        
        # Validate input
        self._validate_chunks(chunks)
        
        # Sort by score (descending)
        sorted_chunks = sorted(chunks, key=lambda x: x.get('score', 0.0), reverse=True)
        
        # Keep top chunks regardless of similarity
        preserved_chunks = sorted_chunks[:self.config.preserve_high_score_chunks]
        remaining_chunks = sorted_chunks[self.config.preserve_high_score_chunks:]
        
        # Remove exact duplicates first
        unique_chunks = self._exact_duplicate_removal(remaining_chunks)
        
        # Apply similarity-based deduplication
        if len(unique_chunks) > 1:
            if self.config.use_clustering:
                deduplicated = self._cluster_based_deduplication(unique_chunks)
            else:
                deduplicated = self._similarity_based_deduplication(unique_chunks)
        else:
            deduplicated = unique_chunks
        
        # Combine preserved and deduplicated chunks
        result = preserved_chunks + deduplicated
        
        # Apply token budget if specified
        if self.config.token_budget:
            result = self._apply_token_budget(result)
        
        # Limit to max_chunks
        result = result[:self.config.max_chunks]
        
        logger.info(f"Deduplication: {len(chunks)} -> {len(result)} chunks")
        return result
    
    def _validate_chunks(self, chunks: List[Dict[str, Any]]):
        """Validate chunk format."""
        for chunk in chunks:
            if 'text' not in chunk:
                raise ValueError("Each chunk must have a 'text' field")
            if not isinstance(chunk.get('text', ''), str):
                raise ValueError("Chunk text must be a string")
    
    def _exact_duplicate_removal(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove exact text duplicates."""
        seen_texts = set()
        unique_chunks = []
        
        for chunk in chunks:
            # Normalize text for comparison
            normalized_text = chunk['text'].strip().lower()
            
            if normalized_text not in seen_texts and len(normalized_text) >= self.config.min_chunk_length:
                seen_texts.add(normalized_text)
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    def _similarity_based_deduplication(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove chunks based on semantic similarity."""
        if len(chunks) <= 1:
            return chunks
        
        # Get embeddings (use provided or compute)
        embeddings = self._get_embeddings(chunks)
        
        if embeddings is None:
            # Fallback: keep all chunks if embeddings unavailable
            return chunks
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Keep chunks that are not too similar to already kept chunks
        kept_chunks = []
        kept_indices = set()
        
        for i, chunk in enumerate(chunks):
            if i in kept_indices:
                continue
            
            # Check similarity with already kept chunks
            too_similar = False
            for kept_idx in kept_indices:
                if similarity_matrix[i, kept_idx] > self.config.similarity_threshold:
                    too_similar = True
                    break
            
            if not too_similar:
                kept_chunks.append(chunk)
                kept_indices.add(i)
        
        return kept_chunks
    
    def _cluster_based_deduplication(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Use clustering to group and select representative chunks."""
        if len(chunks) <= 1:
            return chunks
        
        embeddings = self._get_embeddings(chunks)
        
        if embeddings is None:
            return chunks
        
        # Perform clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1 - self.config.cluster_threshold,
            linkage='complete'
        )
        
        cluster_labels = clustering.fit_predict(embeddings)
        
        # Select representative chunk from each cluster (highest score)
        cluster_chunks = {}
        for i, label in enumerate(cluster_labels):
            if label not in cluster_chunks:
                cluster_chunks[label] = chunks[i]
            elif chunks[i].get('score', 0) > cluster_chunks[label].get('score', 0):
                cluster_chunks[label] = chunks[i]
        
        return list(cluster_chunks.values())
    
    def _get_embeddings(self, chunks: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        """Get embeddings for chunks."""
        # Check if embeddings are already provided
        if all('embedding' in chunk for chunk in chunks):
            return np.array([chunk['embedding'] for chunk in chunks])
        
        # Try to compute embeddings
        try:
            return self._compute_embeddings([chunk['text'] for chunk in chunks])
        except Exception as e:
            logger.warning(f"Could not compute embeddings: {e}")
            return None
    
    def _compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """Compute embeddings for texts."""
        # Try to import and use the embedder
        try:
            from ..embedder.embedder import create_embedder
            embedder = create_embedder()
            return embedder.embed_chunks(texts).vectors
        except ImportError:
            # Fallback: use simple hash-based "embeddings"
            import hashlib
            embeddings = []
            for text in texts:
                # Create a simple hash-based vector
                hash_obj = hashlib.md5(text.encode())
                hash_bytes = hash_obj.digest()
                # Convert to float array
                embedding = [float(b) / 255.0 for b in hash_bytes]
                # Pad or truncate to consistent length
                embedding = embedding[:384] + [0.0] * max(0, 384 - len(embedding))
                embeddings.append(embedding)
            return np.array(embeddings)
    
    def _apply_token_budget(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply token budget constraint."""
        if not self.config.token_budget:
            return chunks
        
        # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
        total_tokens = 0
        filtered_chunks = []
        
        for chunk in chunks:
            estimated_tokens = len(chunk['text']) // 4
            if total_tokens + estimated_tokens <= self.config.token_budget:
                filtered_chunks.append(chunk)
                total_tokens += estimated_tokens
            else:
                break
        
        return filtered_chunks
    
    def get_deduplication_stats(self, original_count: int, final_count: int) -> Dict[str, Any]:
        """Get deduplication statistics."""
        return {
            'original_chunks': original_count,
            'final_chunks': final_count,
            'removed_chunks': original_count - final_count,
            'reduction_percentage': round((original_count - final_count) / original_count * 100, 2) if original_count > 0 else 0
        }


def deduplicate_chunks(chunks: List[Dict[str, Any]], config: Optional[DeduplicationConfig] = None) -> List[Dict[str, Any]]:
    """Convenience function for chunk deduplication."""
    deduplicator = ContextDeduplicator(config)
    return deduplicator.deduplicate(chunks)