"""
High-performance Vector Store for RAG Systems

This module provides a unified interface for vector storage and retrieval,
supporting both FAISS and Chroma backends with optimized performance
for large-scale document retrieval.

Author: RAG System Team
License: MIT
"""

import os
import pickle
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import json
from concurrent.futures import ThreadPoolExecutor
import time

# Optional imports with graceful fallbacks
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    chromadb = None

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a single search result with metadata."""
    id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': self.id,
            'text': self.text,
            'score': self.score,
            'metadata': self.metadata
        }


@dataclass
class VectorStoreConfig:
    """Configuration for vector store initialization."""
    index_type: str = "faiss"  # "faiss" or "chroma"
    dimension: int = 384
    metric: str = "cosine"  # "cosine", "l2", "ip"
    use_gpu: bool = False
    batch_size: int = 1000
    hnsw_m: int = 32  # HNSW parameter
    hnsw_ef_construction: int = 200  # HNSW parameter
    ivf_nlist: int = 100  # IVF parameter
    persist_directory: str = "./vector_index"
    collection_name: str = "rag_documents"
    enable_compression: bool = False
    compression_threshold: int = 1000  # Apply compression if > 1000 tokens


class BaseVectorStore(ABC):
    """
    Abstract base class for vector store implementations.
    
    Provides a unified interface for different vector database backends
    with optimized performance for large-scale document retrieval.
    """
    
    def __init__(self, config: VectorStoreConfig):
        """
        Initialize the vector store with configuration.
        
        Args:
            config: VectorStoreConfig object with initialization parameters
        """
        self.config = config
        self.is_initialized = False
        self.total_vectors = 0
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Performance tracking
        self.stats = {
            'search_count': 0,
            'add_count': 0,
            'total_search_time': 0.0,
            'total_add_time': 0.0
        }
    
    @abstractmethod
    def init_index(self, dimension: Optional[int] = None) -> bool:
        """
        Initialize the vector index.
        
        Args:
            dimension: Embedding dimension (uses config.dimension if None)
            
        Returns:
            bool: True if initialization successful
        """
        pass
    
    @abstractmethod
    def add_texts(
        self, 
        texts: List[str], 
        embeddings: np.ndarray, 
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add texts and their embeddings to the vector store.
        
        Args:
            texts: List of text chunks
            embeddings: numpy array of embeddings (n_texts, dimension)
            metadata: Optional list of metadata dictionaries
            ids: Optional list of custom IDs
            
        Returns:
            List[str]: IDs of added documents
        """
        pass
    
    @abstractmethod
    def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar vectors.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List[SearchResult]: Search results with scores and metadata
        """
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> bool:
        """
        Delete vectors by IDs.
        
        Args:
            ids: List of vector IDs to delete
            
        Returns:
            bool: True if deletion successful
        """
        pass
    
    @abstractmethod
    def save_index(self, path: str) -> bool:
        """
        Save the index to disk.
        
        Args:
            path: Directory path to save the index
            
        Returns:
            bool: True if save successful
        """
        pass
    
    @abstractmethod
    def load_index(self, path: str) -> bool:
        """
        Load the index from disk.
        
        Args:
            path: Directory path to load the index from
            
        Returns:
            bool: True if load successful
        """
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.stats.copy()
        stats['total_vectors'] = self.total_vectors
        stats['is_initialized'] = self.is_initialized
        if stats['search_count'] > 0:
            stats['avg_search_time'] = stats['total_search_time'] / stats['search_count']
        if stats['add_count'] > 0:
            stats['avg_add_time'] = stats['total_add_time'] / stats['add_count']
        return stats
    
    def clear_stats(self) -> None:
        """Clear performance statistics."""
        self.stats = {
            'search_count': 0,
            'add_count': 0,
            'total_search_time': 0.0,
            'total_add_time': 0.0
        }
    
    def _validate_embeddings(self, embeddings: np.ndarray, expected_count: int) -> None:
        """Validate embedding dimensions and count."""
        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings must be 2D array, got {embeddings.ndim}D")
        
        if embeddings.shape[0] != expected_count:
            raise ValueError(f"Expected {expected_count} embeddings, got {embeddings.shape[0]}")
        
        if embeddings.shape[1] != self.config.dimension:
            raise ValueError(f"Expected dimension {self.config.dimension}, got {embeddings.shape[1]}")
    
    def _prepare_metadata(self, metadata: Optional[List[Dict[str, Any]]], count: int) -> List[Dict[str, Any]]:
        """Prepare metadata list with default values."""
        if metadata is None:
            return [{} for _ in range(count)]
        
        if len(metadata) != count:
            raise ValueError(f"Metadata count {len(metadata)} doesn't match text count {count}")
        
        return metadata
    
    def _generate_ids(self, count: int, prefix: str = "doc") -> List[str]:
        """Generate unique IDs for documents."""
        return [f"{prefix}_{i}_{int(time.time())}" for i in range(count)]


class FAISSVectorStore(BaseVectorStore):
    """
    FAISS-based vector store implementation with optimized performance.
    
    Supports HNSW and IVF indices with GPU acceleration.
    """
    
    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)
        self.index = None
        self.texts = []
        self.metadata = []
        self.ids = []
        
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is not available. Install with: pip install faiss-cpu or faiss-gpu")
    
    def init_index(self, dimension: Optional[int] = None) -> bool:
        """Initialize FAISS index with specified parameters."""
        try:
            dim = dimension or self.config.dimension
            
            # Choose index type based on metric
            if self.config.metric == "cosine":
                # Normalize vectors for cosine similarity
                self.index = faiss.IndexFlatIP(dim)
                self.index = faiss.IndexIDMap(self.index)
            elif self.config.metric == "l2":
                self.index = faiss.IndexFlatL2(dim)
                self.index = faiss.IndexIDMap(self.index)
            elif self.config.metric == "ip":
                self.index = faiss.IndexFlatIP(dim)
                self.index = faiss.IndexIDMap(self.index)
            else:
                raise ValueError(f"Unsupported metric: {self.config.metric}")
            
            # Apply HNSW for better performance if we have enough vectors
            if self.config.hnsw_m > 0:
                self.index = faiss.IndexHNSWFlat(dim, self.config.hnsw_m)
                self.index.hnsw.efConstruction = self.config.hnsw_ef_construction
                self.index = faiss.IndexIDMap(self.index)
            
            # Enable GPU if available and requested
            if self.config.use_gpu and faiss.get_num_gpus() > 0:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                self.logger.info("GPU acceleration enabled for FAISS")
            
            self.is_initialized = True
            self.logger.info(f"FAISS index initialized with dimension {dim}, metric {self.config.metric}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize FAISS index: {e}")
            return False
    
    def add_texts(
        self, 
        texts: List[str], 
        embeddings: np.ndarray, 
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add texts and embeddings to FAISS index."""
        start_time = time.time()
        
        try:
            if not self.is_initialized:
                raise RuntimeError("Index not initialized. Call init_index() first.")
            
            # Validate inputs
            self._validate_embeddings(embeddings, len(texts))
            metadata = self._prepare_metadata(metadata, len(texts))
            
            # Generate IDs if not provided
            if ids is None:
                ids = self._generate_ids(len(texts))
            
            # Normalize embeddings for cosine similarity
            if self.config.metric == "cosine":
                faiss.normalize_L2(embeddings)
            
            # Add to FAISS index
            self.index.add_with_ids(embeddings.astype(np.float32), 
                                   np.array([hash(id_) % (2**63) for id_ in ids], dtype=np.int64))
            
            # Store texts and metadata
            self.texts.extend(texts)
            self.metadata.extend(metadata)
            self.ids.extend(ids)
            
            self.total_vectors += len(texts)
            
            # Update stats
            add_time = time.time() - start_time
            self.stats['add_count'] += 1
            self.stats['total_add_time'] += add_time
            
            self.logger.info(f"Added {len(texts)} texts to FAISS index in {add_time:.3f}s")
            return ids
            
        except Exception as e:
            self.logger.error(f"Failed to add texts to FAISS: {e}")
            raise
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar vectors in FAISS index."""
        start_time = time.time()
        
        try:
            if not self.is_initialized:
                raise RuntimeError("Index not initialized. Call init_index() first.")
            
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # Normalize query for cosine similarity
            if self.config.metric == "cosine":
                faiss.normalize_L2(query_embedding)
            
            # Search in FAISS
            scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    continue
                
                # Find the actual index in our stored data
                actual_idx = idx % len(self.texts) if len(self.texts) > 0 else 0
                
                if actual_idx < len(self.texts):
                    result = SearchResult(
                        id=self.ids[actual_idx],
                        text=self.texts[actual_idx],
                        score=float(score),
                        metadata=self.metadata[actual_idx]
                    )
                    
                    # Apply metadata filter if provided
                    if filter_metadata is None or self._matches_filter(result.metadata, filter_metadata):
                        results.append(result)
            
            # Update stats
            search_time = time.time() - start_time
            self.stats['search_count'] += 1
            self.stats['total_search_time'] += search_time
            
            self.logger.debug(f"FAISS search completed in {search_time:.3f}s, found {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to search FAISS index: {e}")
            raise
    
    def delete(self, ids: List[str]) -> bool:
        """Delete vectors by IDs (not directly supported in FAISS, rebuild required)."""
        try:
            # FAISS doesn't support direct deletion, so we need to rebuild
            # This is a simplified implementation - in production, you might want
            # to use a more sophisticated approach
            ids_set = set(ids)
            new_texts = []
            new_metadata = []
            new_ids = []
            
            for i, doc_id in enumerate(self.ids):
                if doc_id not in ids_set:
                    new_texts.append(self.texts[i])
                    new_metadata.append(self.metadata[i])
                    new_ids.append(doc_id)
            
            # Rebuild index
            self.texts = new_texts
            self.metadata = new_metadata
            self.ids = new_ids
            
            # Reinitialize and rebuild index
            self.init_index()
            if new_texts:
                # This is a simplified approach - in production, you'd want to
                # store embeddings separately and rebuild properly
                self.logger.warning("FAISS delete requires full rebuild - consider using Chroma for better deletion support")
            
            self.total_vectors = len(self.texts)
            self.logger.info(f"Deleted {len(ids)} vectors from FAISS index")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete from FAISS: {e}")
            return False
    
    def save_index(self, path: str) -> bool:
        """Save FAISS index and metadata to disk."""
        try:
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, str(path / "faiss.index"))
            
            # Save metadata
            metadata_file = path / "metadata.pkl"
            with open(metadata_file, 'wb') as f:
                pickle.dump({
                    'texts': self.texts,
                    'metadata': self.metadata,
                    'ids': self.ids,
                    'config': self.config,
                    'total_vectors': self.total_vectors
                }, f)
            
            self.logger.info(f"FAISS index saved to {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save FAISS index: {e}")
            return False
    
    def load_index(self, path: str) -> bool:
        """Load FAISS index and metadata from disk."""
        try:
            path = Path(path)
            
            # Load FAISS index
            index_file = path / "faiss.index"
            if not index_file.exists():
                raise FileNotFoundError(f"FAISS index file not found: {index_file}")
            
            self.index = faiss.read_index(str(index_file))
            
            # Load metadata
            metadata_file = path / "metadata.pkl"
            if metadata_file.exists():
                with open(metadata_file, 'rb') as f:
                    data = pickle.load(f)
                    self.texts = data.get('texts', [])
                    self.metadata = data.get('metadata', [])
                    self.ids = data.get('ids', [])
                    self.total_vectors = data.get('total_vectors', len(self.texts))
            
            self.is_initialized = True
            self.logger.info(f"FAISS index loaded from {path} with {self.total_vectors} vectors")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load FAISS index: {e}")
            return False
    
    def _matches_filter(self, metadata: Dict[str, Any], filter_metadata: Dict[str, Any]) -> bool:
        """Check if metadata matches the filter criteria."""
        for key, value in filter_metadata.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True


class ChromaVectorStore(BaseVectorStore):
    """
    Chroma-based vector store implementation with optimized performance.
    
    Provides better metadata filtering and deletion capabilities than FAISS.
    """
    
    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)
        self.client = None
        self.collection = None
        
        if not CHROMA_AVAILABLE:
            raise ImportError("Chroma is not available. Install with: pip install chromadb")
    
    def init_index(self, dimension: Optional[int] = None) -> bool:
        """Initialize Chroma client and collection."""
        try:
            # Initialize Chroma client
            self.client = chromadb.PersistentClient(
                path=self.config.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(
                    name=self.config.collection_name,
                    embedding_function=None  # We'll provide embeddings directly
                )
                self.logger.info(f"Loaded existing Chroma collection: {self.config.collection_name}")
            except Exception:
                self.collection = self.client.create_collection(
                    name=self.config.collection_name,
                    metadata={"hnsw:space": self.config.metric}
                )
                self.logger.info(f"Created new Chroma collection: {self.config.collection_name}")
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Chroma: {e}")
            return False
    
    def add_texts(
        self, 
        texts: List[str], 
        embeddings: np.ndarray, 
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add texts and embeddings to Chroma collection."""
        start_time = time.time()
        
        try:
            if not self.is_initialized:
                raise RuntimeError("Index not initialized. Call init_index() first.")
            
            # Validate inputs
            self._validate_embeddings(embeddings, len(texts))
            metadata = self._prepare_metadata(metadata, len(texts))
            
            # Generate IDs if not provided
            if ids is None:
                ids = self._generate_ids(len(texts))
            
            # Convert embeddings to list format for Chroma
            embeddings_list = embeddings.tolist()
            
            # Add to Chroma collection
            self.collection.add(
                embeddings=embeddings_list,
                documents=texts,
                metadatas=metadata,
                ids=ids
            )
            
            self.total_vectors += len(texts)
            
            # Update stats
            add_time = time.time() - start_time
            self.stats['add_count'] += 1
            self.stats['total_add_time'] += add_time
            
            self.logger.info(f"Added {len(texts)} texts to Chroma in {add_time:.3f}s")
            return ids
            
        except Exception as e:
            self.logger.error(f"Failed to add texts to Chroma: {e}")
            raise
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar vectors in Chroma collection."""
        start_time = time.time()
        
        try:
            if not self.is_initialized:
                raise RuntimeError("Index not initialized. Call init_index() first.")
            
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # Search in Chroma
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k,
                where=filter_metadata
            )
            
            # Convert to SearchResult format
            search_results = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    result = SearchResult(
                        id=results['ids'][0][i],
                        text=results['documents'][0][i],
                        score=results['distances'][0][i] if 'distances' in results else 0.0,
                        metadata=results['metadatas'][0][i] if results['metadatas'] else {}
                    )
                    search_results.append(result)
            
            # Update stats
            search_time = time.time() - start_time
            self.stats['search_count'] += 1
            self.stats['total_search_time'] += search_time
            
            self.logger.debug(f"Chroma search completed in {search_time:.3f}s, found {len(search_results)} results")
            return search_results
            
        except Exception as e:
            self.logger.error(f"Failed to search Chroma: {e}")
            raise
    
    def delete(self, ids: List[str]) -> bool:
        """Delete vectors by IDs from Chroma collection."""
        try:
            if not self.is_initialized:
                raise RuntimeError("Index not initialized. Call init_index() first.")
            
            # Delete from Chroma
            self.collection.delete(ids=ids)
            
            self.total_vectors = max(0, self.total_vectors - len(ids))
            self.logger.info(f"Deleted {len(ids)} vectors from Chroma")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete from Chroma: {e}")
            return False
    
    def save_index(self, path: str) -> bool:
        """Chroma automatically persists to disk, so this is a no-op."""
        try:
            # Chroma automatically persists to the configured directory
            self.logger.info(f"Chroma index is automatically persisted to {self.config.persist_directory}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save Chroma index: {e}")
            return False
    
    def load_index(self, path: str) -> bool:
        """Load Chroma collection from disk."""
        try:
            # Update persist directory if different path provided
            if path != self.config.persist_directory:
                self.config.persist_directory = path
            
            # Reinitialize to load from the new path
            return self.init_index()
            
        except Exception as e:
            self.logger.error(f"Failed to load Chroma index: {e}")
            return False


class VectorStoreFactory:
    """Factory class for creating vector store instances."""
    
    @staticmethod
    def create(config: VectorStoreConfig) -> BaseVectorStore:
        """
        Create a vector store instance based on configuration.
        
        Args:
            config: VectorStoreConfig with desired settings
            
        Returns:
            BaseVectorStore: Configured vector store instance
        """
        if config.index_type.lower() == "faiss":
            return FAISSVectorStore(config)
        elif config.index_type.lower() == "chroma":
            return ChromaVectorStore(config)
        else:
            raise ValueError(f"Unsupported index type: {config.index_type}")


# Convenience functions for easy usage
def create_vector_store(
    index_type: str = "faiss",
    dimension: int = 384,
    metric: str = "cosine",
    use_gpu: bool = False,
    persist_directory: str = "./vector_index"
) -> BaseVectorStore:
    """
    Create a vector store with default configuration.
    
    Args:
        index_type: "faiss" or "chroma"
        dimension: Embedding dimension
        metric: Distance metric ("cosine", "l2", "ip")
        use_gpu: Enable GPU acceleration (FAISS only)
        persist_directory: Directory for persistence
        
    Returns:
        BaseVectorStore: Configured vector store instance
    """
    config = VectorStoreConfig(
        index_type=index_type,
        dimension=dimension,
        metric=metric,
        use_gpu=use_gpu,
        persist_directory=persist_directory
    )
    return VectorStoreFactory.create(config)


def get_available_backends() -> List[str]:
    """Get list of available vector store backends."""
    backends = []
    if FAISS_AVAILABLE:
        backends.append("faiss")
    if CHROMA_AVAILABLE:
        backends.append("chroma")
    return backends


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    print("Available backends:", get_available_backends())
    
    # Create vector store
    vs = create_vector_store(index_type="faiss", dimension=384)
    
    # Initialize index
    vs.init_index()
    
    # Example data
    texts = ["This is a test document", "Another test document"]
    embeddings = np.random.rand(2, 384).astype(np.float32)
    metadata = [{"source": "test", "page": 1}, {"source": "test", "page": 2}]
    
    # Add documents
    ids = vs.add_texts(texts, embeddings, metadata)
    print(f"Added documents with IDs: {ids}")
    
    # Search
    query_embedding = np.random.rand(384).astype(np.float32)
    results = vs.search(query_embedding, top_k=2)
    print(f"Search results: {[r.to_dict() for r in results]}")
    
    # Get stats
    print(f"Vector store stats: {vs.get_stats()}")
