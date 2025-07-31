"""
High-Performance Hybrid Retriever for RAG Systems

This module implements a hybrid retrieval strategy that combines dense vector
retrieval (semantic similarity) with sparse BM25 retrieval (keyword matching)
for optimal recall and precision in large-scale document retrieval.

Author: RAG System Team
License: MIT
"""

import logging
import time
import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from collections import defaultdict
import json

# Optional imports with graceful fallbacks
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    BM25Okapi = None

try:
    from elasticsearch import Elasticsearch
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False
    Elasticsearch = None

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class HybridSearchResult:
    """Represents a hybrid search result with combined scores."""
    id: str
    text: str
    dense_score: float
    sparse_score: float
    combined_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    retrieval_sources: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': self.id,
            'text': self.text,
            'dense_score': self.dense_score,
            'sparse_score': self.sparse_score,
            'combined_score': self.combined_score,
            'metadata': self.metadata,
            'retrieval_sources': self.retrieval_sources
        }


@dataclass
class HybridRetrieverConfig:
    """Configuration for hybrid retriever."""
    weight_dense: float = 0.6
    weight_sparse: float = 0.4
    top_k_dense: int = 20
    top_k_sparse: int = 20
    top_k_final: int = 10
    enable_parallel: bool = True
    enable_caching: bool = False
    cache_ttl: int = 3600  # 1 hour
    score_normalization: str = "minmax"  # "minmax", "zscore", "rank"
    fusion_method: str = "weighted_sum"  # "weighted_sum", "reciprocal_rank", "rrf"
    rrf_k: float = 60.0  # Reciprocal rank fusion parameter
    debug_mode: bool = False


class BaseBM25Retriever(ABC):
    """Abstract base class for BM25 retrievers."""
    
    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for documents using BM25."""
        pass
    
    @abstractmethod
    def add_documents(self, documents: List[str], doc_ids: Optional[List[str]] = None) -> bool:
        """Add documents to the BM25 index."""
        pass


class BM25Retriever(BaseBM25Retriever):
    """BM25 retriever using rank-bm25 library."""
    
    def __init__(self):
        if not BM25_AVAILABLE:
            raise ImportError("rank-bm25 is not available. Install with: pip install rank-bm25")
        
        self.bm25 = None
        self.documents = []
        self.doc_ids = []
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    def add_documents(self, documents: List[str], doc_ids: Optional[List[str]] = None) -> bool:
        """Add documents to BM25 index."""
        try:
            if doc_ids is None:
                doc_ids = [f"doc_{i}" for i in range(len(documents))]
            
            # Tokenize documents
            tokenized_docs = [doc.lower().split() for doc in documents]
            
            # Initialize BM25
            self.bm25 = BM25Okapi(tokenized_docs)
            self.documents = documents
            self.doc_ids = doc_ids
            
            self.logger.info(f"Added {len(documents)} documents to BM25 index")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add documents to BM25: {e}")
            return False
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for documents using BM25."""
        try:
            if self.bm25 is None:
                raise RuntimeError("BM25 index not initialized. Call add_documents() first.")
            
            # Tokenize query
            tokenized_query = query.lower().split()
            
            # Get BM25 scores
            scores = self.bm25.get_scores(tokenized_query)
            
            # Get top-k results
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if scores[idx] > 0:  # Only include documents with positive scores
                    results.append({
                        'id': self.doc_ids[idx],
                        'text': self.documents[idx],
                        'score': float(scores[idx]),
                        'metadata': {'retrieval_type': 'bm25'}
                    })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to search BM25: {e}")
            return []


class ElasticsearchRetriever(BaseBM25Retriever):
    """Elasticsearch retriever for BM25 search."""
    
    def __init__(self, host: str = "localhost", port: int = 9200, index_name: str = "documents"):
        if not ELASTICSEARCH_AVAILABLE:
            raise ImportError("elasticsearch is not available. Install with: pip install elasticsearch")
        
        self.client = Elasticsearch([{'host': host, 'port': port}])
        self.index_name = index_name
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    def add_documents(self, documents: List[str], doc_ids: Optional[List[str]] = None) -> bool:
        """Add documents to Elasticsearch index."""
        try:
            if doc_ids is None:
                doc_ids = [f"doc_{i}" for i in range(len(documents))]
            
            # Create index if it doesn't exist
            if not self.client.indices.exists(index=self.index_name):
                self.client.indices.create(
                    index=self.index_name,
                    body={
                        "mappings": {
                            "properties": {
                                "text": {"type": "text"},
                                "doc_id": {"type": "keyword"}
                            }
                        }
                    }
                )
            
            # Bulk index documents
            actions = []
            for doc_id, text in zip(doc_ids, documents):
                actions.append({
                    "_index": self.index_name,
                    "_id": doc_id,
                    "_source": {"text": text, "doc_id": doc_id}
                })
            
            from elasticsearch.helpers import bulk
            bulk(self.client, actions)
            
            self.logger.info(f"Added {len(documents)} documents to Elasticsearch index")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add documents to Elasticsearch: {e}")
            return False
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for documents using Elasticsearch BM25."""
        try:
            response = self.client.search(
                index=self.index_name,
                body={
                    "query": {
                        "match": {
                            "text": query
                        }
                    },
                    "size": top_k
                }
            )
            
            results = []
            for hit in response['hits']['hits']:
                results.append({
                    'id': hit['_id'],
                    'text': hit['_source']['text'],
                    'score': float(hit['_score']),
                    'metadata': {'retrieval_type': 'elasticsearch_bm25'}
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to search Elasticsearch: {e}")
            return []


class HybridRetriever:
    """
    Hybrid retriever that combines dense vector retrieval with sparse BM25 retrieval.
    
    Provides optimal recall and precision by leveraging both semantic similarity
    and keyword matching approaches.
    """
    
    def __init__(
        self,
        vector_store,
        bm25_retriever: BaseBM25Retriever,
        config: Optional[HybridRetrieverConfig] = None
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_store: Vector store instance (FAISS/Chroma)
            bm25_retriever: BM25 retriever instance
            config: Configuration for hybrid retrieval
        """
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.config = config or HybridRetrieverConfig()
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Validate weights
        if abs(self.config.weight_dense + self.config.weight_sparse - 1.0) > 1e-6:
            self.logger.warning("Weights should sum to 1.0 for optimal performance")
        
        # Performance tracking
        self.stats = {
            'search_count': 0,
            'total_search_time': 0.0,
            'dense_search_time': 0.0,
            'sparse_search_time': 0.0,
            'fusion_time': 0.0
        }
    
    def search(
        self, 
        query: str, 
        query_embedding: Optional[np.ndarray] = None,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[HybridSearchResult]:
        """
        Perform hybrid search combining dense and sparse retrieval.
        
        Args:
            query: Text query
            query_embedding: Pre-computed query embedding (optional)
            top_k: Number of results to return
            filter_metadata: Metadata filters for dense retrieval
            
        Returns:
            List[HybridSearchResult]: Combined and ranked search results
        """
        start_time = time.time()
        top_k = top_k or self.config.top_k_final
        
        try:
            self.logger.info(f"Starting hybrid search for query: '{query[:100]}...'")
            
            # Perform parallel retrieval if enabled
            if self.config.enable_parallel:
                dense_results, sparse_results = self._parallel_retrieval(
                    query, query_embedding, filter_metadata
                )
            else:
                dense_results, sparse_results = self._sequential_retrieval(
                    query, query_embedding, filter_metadata
                )
            
            # Combine and rank results
            combined_results = self._combine_results(
                dense_results, sparse_results, top_k
            )
            
            # Update stats
            search_time = time.time() - start_time
            self.stats['search_count'] += 1
            self.stats['total_search_time'] += search_time
            
            self.logger.info(f"Hybrid search completed in {search_time:.3f}s, found {len(combined_results)} results")
            
            if self.config.debug_mode:
                self._log_debug_info(query, dense_results, sparse_results, combined_results)
            
            return combined_results
            
        except Exception as e:
            self.logger.error(f"Failed to perform hybrid search: {e}")
            raise
    
    def _parallel_retrieval(
        self, 
        query: str, 
        query_embedding: Optional[np.ndarray],
        filter_metadata: Optional[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Perform dense and sparse retrieval in parallel."""
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both retrieval tasks
            dense_future = executor.submit(
                self._dense_retrieval, query, query_embedding, filter_metadata
            )
            sparse_future = executor.submit(
                self._sparse_retrieval, query
            )
            
            # Wait for both to complete
            dense_results = dense_future.result()
            sparse_results = sparse_future.result()
            
            return dense_results, sparse_results
    
    def _sequential_retrieval(
        self, 
        query: str, 
        query_embedding: Optional[np.ndarray],
        filter_metadata: Optional[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Perform dense and sparse retrieval sequentially."""
        dense_results = self._dense_retrieval(query, query_embedding, filter_metadata)
        sparse_results = self._sparse_retrieval(query)
        return dense_results, sparse_results
    
    def _dense_retrieval(
        self, 
        query: str, 
        query_embedding: Optional[np.ndarray],
        filter_metadata: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Perform dense vector retrieval."""
        dense_start = time.time()
        
        try:
            # Get query embedding if not provided
            if query_embedding is None:
                # This would typically come from an embedding model
                # For now, we'll use a placeholder - in production, you'd use sentence-transformers
                query_embedding = np.random.rand(self.vector_store.config.dimension).astype(np.float32)
            
            # Search vector store
            search_results = self.vector_store.search(
                query_embedding, 
                top_k=self.config.top_k_dense,
                filter_metadata=filter_metadata
            )
            
            # Convert to standard format
            dense_results = []
            for result in search_results:
                dense_results.append({
                    'id': result.id,
                    'text': result.text,
                    'score': result.score,
                    'metadata': result.metadata
                })
            
            dense_time = time.time() - dense_start
            self.stats['dense_search_time'] += dense_time
            
            self.logger.debug(f"Dense retrieval completed in {dense_time:.3f}s, found {len(dense_results)} results")
            return dense_results
            
        except Exception as e:
            self.logger.error(f"Failed to perform dense retrieval: {e}")
            return []
    
    def _sparse_retrieval(self, query: str) -> List[Dict[str, Any]]:
        """Perform sparse BM25 retrieval."""
        sparse_start = time.time()
        
        try:
            sparse_results = self.bm25_retriever.search(query, top_k=self.config.top_k_sparse)
            
            sparse_time = time.time() - sparse_start
            self.stats['sparse_search_time'] += sparse_time
            
            self.logger.debug(f"Sparse retrieval completed in {sparse_time:.3f}s, found {len(sparse_results)} results")
            return sparse_results
            
        except Exception as e:
            self.logger.error(f"Failed to perform sparse retrieval: {e}")
            return []
    
    def _combine_results(
        self, 
        dense_results: List[Dict[str, Any]], 
        sparse_results: List[Dict[str, Any]],
        top_k: int
    ) -> List[HybridSearchResult]:
        """Combine and rank results from both retrievers."""
        fusion_start = time.time()
        
        try:
            # Normalize scores
            dense_scores = self._normalize_scores([r['score'] for r in dense_results])
            sparse_scores = self._normalize_scores([r['score'] for r in sparse_results])
            
            # Create result mapping
            result_map = {}
            
            # Add dense results
            for i, result in enumerate(dense_results):
                doc_id = result['id']
                result_map[doc_id] = {
                    'id': doc_id,
                    'text': result['text'],
                    'dense_score': dense_scores[i],
                    'sparse_score': 0.0,
                    'metadata': result['metadata'],
                    'retrieval_sources': ['dense']
                }
            
            # Add sparse results and merge
            for i, result in enumerate(sparse_results):
                doc_id = result['id']
                if doc_id in result_map:
                    # Document found in both retrievers
                    result_map[doc_id]['sparse_score'] = sparse_scores[i]
                    result_map[doc_id]['retrieval_sources'].append('sparse')
                else:
                    # Document only in sparse retrieval
                    result_map[doc_id] = {
                        'id': doc_id,
                        'text': result['text'],
                        'dense_score': 0.0,
                        'sparse_score': sparse_scores[i],
                        'metadata': result['metadata'],
                        'retrieval_sources': ['sparse']
                    }
            
            # Calculate combined scores
            hybrid_results = []
            for doc_id, result in result_map.items():
                combined_score = self._calculate_combined_score(
                    result['dense_score'], 
                    result['sparse_score']
                )
                
                hybrid_result = HybridSearchResult(
                    id=result['id'],
                    text=result['text'],
                    dense_score=result['dense_score'],
                    sparse_score=result['sparse_score'],
                    combined_score=combined_score,
                    metadata=result['metadata'],
                    retrieval_sources=result['retrieval_sources']
                )
                hybrid_results.append(hybrid_result)
            
            # Sort by combined score and return top_k
            hybrid_results.sort(key=lambda x: x.combined_score, reverse=True)
            
            fusion_time = time.time() - fusion_start
            self.stats['fusion_time'] += fusion_time
            
            self.logger.debug(f"Result fusion completed in {fusion_time:.3f}s")
            return hybrid_results[:top_k]
            
        except Exception as e:
            self.logger.error(f"Failed to combine results: {e}")
            return []
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores using the configured method."""
        if not scores:
            return []
        
        scores = np.array(scores)
        
        if self.config.score_normalization == "minmax":
            if scores.max() == scores.min():
                return [1.0] * len(scores)
            return ((scores - scores.min()) / (scores.max() - scores.min())).tolist()
        
        elif self.config.score_normalization == "zscore":
            mean = scores.mean()
            std = scores.std()
            if std == 0:
                return [1.0] * len(scores)
            return ((scores - mean) / std).tolist()
        
        elif self.config.score_normalization == "rank":
            # Convert to rank-based scores (1/rank)
            ranks = np.argsort(np.argsort(-scores)) + 1
            return (1.0 / ranks).tolist()
        
        else:
            return scores.tolist()
    
    def _calculate_combined_score(self, dense_score: float, sparse_score: float) -> float:
        """Calculate combined score using the configured fusion method."""
        if self.config.fusion_method == "weighted_sum":
            return (self.config.weight_dense * dense_score + 
                   self.config.weight_sparse * sparse_score)
        
        elif self.config.fusion_method == "reciprocal_rank":
            # Reciprocal rank fusion
            dense_rank = 1.0 / (dense_score + self.config.rrf_k)
            sparse_rank = 1.0 / (sparse_score + self.config.rrf_k)
            return dense_rank + sparse_rank
        
        elif self.config.fusion_method == "rrf":
            # Reciprocal rank fusion with weights
            dense_rank = self.config.weight_dense / (dense_score + self.config.rrf_k)
            sparse_rank = self.config.weight_sparse / (sparse_score + self.config.rrf_k)
            return dense_rank + sparse_rank
        
        else:
            # Default to weighted sum
            return (self.config.weight_dense * dense_score + 
                   self.config.weight_sparse * sparse_score)
    
    def _log_debug_info(
        self, 
        query: str, 
        dense_results: List[Dict[str, Any]], 
        sparse_results: List[Dict[str, Any]], 
        combined_results: List[HybridSearchResult]
    ):
        """Log debug information for fine-tuning."""
        self.logger.debug(f"Query: {query}")
        self.logger.debug(f"Dense results (top 3): {dense_results[:3]}")
        self.logger.debug(f"Sparse results (top 3): {sparse_results[:3]}")
        self.logger.debug(f"Combined results (top 3): {[r.to_dict() for r in combined_results[:3]]}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.stats.copy()
        if stats['search_count'] > 0:
            stats['avg_search_time'] = stats['total_search_time'] / stats['search_count']
            stats['avg_dense_time'] = stats['dense_search_time'] / stats['search_count']
            stats['avg_sparse_time'] = stats['sparse_search_time'] / stats['search_count']
            stats['avg_fusion_time'] = stats['fusion_time'] / stats['search_count']
        return stats
    
    def clear_stats(self) -> None:
        """Clear performance statistics."""
        self.stats = {
            'search_count': 0,
            'total_search_time': 0.0,
            'dense_search_time': 0.0,
            'sparse_search_time': 0.0,
            'fusion_time': 0.0
        }


# Factory functions for easy usage
def create_bm25_retriever(retriever_type: str = "rank_bm25", **kwargs) -> BaseBM25Retriever:
    """
    Create a BM25 retriever instance.
    
    Args:
        retriever_type: "rank_bm25" or "elasticsearch"
        **kwargs: Additional arguments for the retriever
        
    Returns:
        BaseBM25Retriever: Configured BM25 retriever instance
    """
    if retriever_type == "rank_bm25":
        return BM25Retriever()
    elif retriever_type == "elasticsearch":
        return ElasticsearchRetriever(**kwargs)
    else:
        raise ValueError(f"Unsupported retriever type: {retriever_type}")


def create_hybrid_retriever(
    vector_store,
    bm25_retriever: Optional[BaseBM25Retriever] = None,
    weight_dense: float = 0.6,
    weight_sparse: float = 0.4,
    **kwargs
) -> HybridRetriever:
    """
    Create a hybrid retriever with default configuration.
    
    Args:
        vector_store: Vector store instance
        bm25_retriever: BM25 retriever instance (creates default if None)
        weight_dense: Weight for dense retrieval scores
        weight_sparse: Weight for sparse retrieval scores
        **kwargs: Additional configuration parameters
        
    Returns:
        HybridRetriever: Configured hybrid retriever instance
    """
    if bm25_retriever is None:
        bm25_retriever = create_bm25_retriever()
    
    config = HybridRetrieverConfig(
        weight_dense=weight_dense,
        weight_sparse=weight_sparse,
        **kwargs
    )
    
    return HybridRetriever(vector_store, bm25_retriever, config)


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    from retrieval.vector_store import create_vector_store
    
    # Create vector store
    vs = create_vector_store(index_type="faiss", dimension=384)
    vs.init_index()
    
    # Create BM25 retriever
    bm25 = create_bm25_retriever()
    
    # Add sample documents
    documents = [
        "Insurance policy covers medical expenses",
        "Claim processing requires documentation",
        "Premium calculation based on risk factors",
        "Policy renewal terms and conditions",
        "Coverage limits and exclusions apply"
    ]
    
    # Add to both retrievers
    embeddings = np.random.rand(len(documents), 384).astype(np.float32)
    metadata = [{"source": f"doc_{i}"} for i in range(len(documents))]
    
    vs.add_texts(documents, embeddings, metadata)
    bm25.add_documents(documents)
    
    # Create hybrid retriever
    hybrid = create_hybrid_retriever(
        vs, bm25, 
        weight_dense=0.6, 
        weight_sparse=0.4,
        debug_mode=True
    )
    
    # Perform hybrid search
    query = "insurance coverage medical"
    results = hybrid.search(query, top_k=5)
    
    print(f"Hybrid search results for '{query}':")
    for i, result in enumerate(results):
        print(f"{i+1}. {result.text}")
        print(f"   Combined Score: {result.combined_score:.3f}")
        print(f"   Dense Score: {result.dense_score:.3f}")
        print(f"   Sparse Score: {result.sparse_score:.3f}")
        print(f"   Sources: {result.retrieval_sources}")
        print()
    
    # Get stats
    print(f"Hybrid retriever stats: {hybrid.get_stats()}")
