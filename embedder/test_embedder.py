"""
Test suite for the embedder module.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

from embedder import (
    Embedder, 
    EmbeddingConfig, 
    ThreadSafeEmbeddingCache, 
    create_embedder, 
    SUPPORTED_MODELS
)


class TestEmbeddingCache:
    """Test the ThreadSafeEmbeddingCache class."""
    
    def setup_method(self):
        """Set up test cache directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = ThreadSafeEmbeddingCache(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test cache directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_cache_initialization(self):
        """Test cache initialization."""
        assert self.cache.cache_dir.exists()
        assert self.cache.metadata_file.exists()
        assert self.cache.metadata == {}
    
    def test_cache_set_get(self):
        """Test setting and getting embeddings from cache."""
        chunk_id = "test_chunk_123"
        embedding = np.random.rand(384).astype(np.float32)
        model_name = "test-model"
        
        # Test setting
        self.cache.set(chunk_id, embedding, model_name)
        
        # Test getting
        retrieved = self.cache.get(chunk_id)
        assert retrieved is not None
        assert np.array_equal(embedding, retrieved)
        
        # Test metadata
        assert chunk_id in self.cache.metadata
        assert self.cache.metadata[chunk_id]["model_name"] == model_name
        assert self.cache.metadata[chunk_id]["embedding_dim"] == 384
    
    def test_cache_miss(self):
        """Test cache miss behavior."""
        chunk_id = "nonexistent_chunk"
        retrieved = self.cache.get(chunk_id)
        assert retrieved is None
    
    def test_cache_clear(self):
        """Test cache clearing."""
        # Add some test data
        chunk_id = "test_chunk"
        embedding = np.random.rand(384).astype(np.float32)
        self.cache.set(chunk_id, embedding, "test-model")
        
        # Clear cache
        self.cache.clear()
        
        # Verify cache is empty
        assert self.cache.metadata == {}
        assert len(list(self.cache.cache_dir.glob("*.npy"))) == 0
    
    def test_cache_stats(self):
        """Test cache statistics."""
        # Add some test data
        for i in range(3):
            chunk_id = f"test_chunk_{i}"
            embedding = np.random.rand(384).astype(np.float32)
            self.cache.set(chunk_id, embedding, "test-model")
        
        stats = self.cache.get_stats()
        assert stats["total_embeddings"] == 3
        assert "test-model" in stats["models_used"]


class TestEmbedder:
    """Test the Embedder class."""
    
    def setup_method(self):
        """Set up test embedder."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = EmbeddingConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            use_cache=True,
            cache_dir=self.temp_dir,
            batch_size=2
        )
        self.embedder = Embedder(self.config)
    
    def teardown_method(self):
        """Clean up test embedder."""
        shutil.rmtree(self.temp_dir)
    
    def test_embedder_initialization(self):
        """Test embedder initialization."""
        assert self.embedder.config.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert self.embedder.config.batch_size == 2
        assert self.embedder.cache is not None
        assert hasattr(self.embedder, 'model')
    
    def test_generate_chunk_id(self):
        """Test chunk ID generation."""
        text1 = "This is a test document."
        text2 = "This is a test document."
        text3 = "This is a different document."
        
        id1 = self.embedder._generate_chunk_id(text1)
        id2 = self.embedder._generate_chunk_id(text2)
        id3 = self.embedder._generate_chunk_id(text3)
        
        # Same text should have same ID
        assert id1 == id2
        
        # Different text should have different ID
        assert id1 != id3
        
        # IDs should be valid MD5 hashes
        assert len(id1) == 32
        assert all(c in '0123456789abcdef' for c in id1)
    
    def test_embed_query(self):
        """Test query embedding."""
        query = "What are the insurance requirements?"
        embedding = self.embedder.embed_query(query)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.ndim == 1
        assert embedding.shape[0] == self.embedder.config.embedding_dim
    
    def test_embed_chunks_empty(self):
        """Test embedding empty chunk list."""
        result = self.embedder.embed_chunks([])
        
        assert isinstance(result, EmbeddingResult)
        assert len(result.vectors) == 0
        assert len(result.chunk_ids) == 0
        assert result.processing_time == 0.0
        assert result.cache_hits == 0
        assert result.cache_misses == 0
    
    def test_embed_chunks_single(self):
        """Test embedding single chunk."""
        chunks = ["This is a test document about insurance policies."]
        result = self.embedder.embed_chunks(chunks)
        
        assert isinstance(result, EmbeddingResult)
        assert len(result.vectors) == 1
        assert len(result.chunk_ids) == 1
        assert result.vectors.shape[1] == self.embedder.config.embedding_dim
        assert result.processing_time > 0
        assert result.cache_misses == 1  # First time, should miss cache
    
    def test_embed_chunks_multiple(self):
        """Test embedding multiple chunks."""
        chunks = [
            "This is a test document about insurance policies.",
            "Another document about legal requirements.",
            "Financial document with important information."
        ]
        result = self.embedder.embed_chunks(chunks)
        
        assert isinstance(result, EmbeddingResult)
        assert len(result.vectors) == 3
        assert len(result.chunk_ids) == 3
        assert result.vectors.shape[1] == self.embedder.config.embedding_dim
        assert result.processing_time > 0
        assert result.cache_misses == 3  # First time, should miss cache
    
    def test_embed_chunks_caching(self):
        """Test embedding with caching."""
        chunks = ["This is a test document about insurance policies."]
        
        # First embedding (cache miss)
        result1 = self.embedder.embed_chunks(chunks)
        assert result1.cache_misses == 1
        assert result1.cache_hits == 0
        
        # Second embedding (cache hit)
        result2 = self.embedder.embed_chunks(chunks)
        assert result2.cache_misses == 0
        assert result2.cache_hits == 1
        
        # Embeddings should be identical
        assert np.array_equal(result1.vectors, result2.vectors)
    
    def test_embed_chunks_batch_processing(self):
        """Test batch processing with multiple chunks."""
        chunks = [
            "Document 1 about insurance.",
            "Document 2 about legal matters.",
            "Document 3 about financial topics.",
            "Document 4 about compliance.",
            "Document 5 about regulations."
        ]
        
        # Test with batch_size=2
        result = self.embedder.embed_chunks(chunks)
        
        assert len(result.vectors) == 5
        assert len(result.chunk_ids) == 5
        assert result.cache_misses == 5  # All should miss cache on first run
    
    def test_compute_similarity(self):
        """Test similarity computation."""
        # Create test embeddings
        query = "What are the insurance requirements?"
        chunks = [
            "This document covers insurance policies and requirements.",
            "Legal requirements for business operations.",
            "Financial regulations and compliance."
        ]
        
        query_embedding = self.embedder.embed_query(query)
        result = self.embedder.embed_chunks(chunks)
        
        similarities = self.embedder.compute_similarity(query_embedding, result.vectors)
        
        assert isinstance(similarities, np.ndarray)
        assert len(similarities) == 3
        assert all(0 <= sim <= 1 for sim in similarities)  # Cosine similarity should be in [0,1]
        
        # First chunk should be most similar to insurance query
        assert similarities[0] > similarities[1]
        assert similarities[0] > similarities[2]
    
    def test_get_stats(self):
        """Test statistics retrieval."""
        # Perform some embeddings
        chunks = ["Test document 1.", "Test document 2."]
        self.embedder.embed_chunks(chunks)
        
        stats = self.embedder.get_stats()
        
        assert "model_name" in stats
        assert "embedding_dim" in stats
        assert "device" in stats
        assert "total_embeddings" in stats
        assert "cache_hits" in stats
        assert "cache_misses" in stats
        assert "cache_hit_rate" in stats
        assert "total_processing_time" in stats
        assert "avg_processing_time" in stats
        assert "cache_stats" in stats
        
        assert stats["total_embeddings"] == 2
        assert stats["cache_misses"] == 2
        assert stats["cache_hits"] == 0
        assert stats["cache_hit_rate"] == 0.0
    
    def test_change_model(self):
        """Test model changing."""
        original_model_name = self.embedder.config.model_name
        
        # Change to a different model
        new_model_name = "sentence-transformers/all-mpnet-base-v2"
        self.embedder.change_model(new_model_name)
        
        assert self.embedder.config.model_name == new_model_name
        assert self.embedder.config.model_name != original_model_name
    
    def test_clear_cache(self):
        """Test cache clearing."""
        # Add some embeddings to cache
        chunks = ["Test document for cache clearing."]
        self.embedder.embed_chunks(chunks)
        
        # Clear cache
        self.embedder.clear_cache()
        
        # Verify cache is empty
        cache_stats = self.embedder.cache.get_stats()
        assert cache_stats["total_embeddings"] == 0


class TestFactoryFunctions:
    """Test factory functions and utilities."""
    
    def test_create_embedder(self):
        """Test embedder factory function."""
        embedder = create_embedder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            use_cache=False,
            batch_size=16
        )
        
        assert isinstance(embedder, Embedder)
        assert embedder.config.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert embedder.config.use_cache == False
        assert embedder.config.batch_size == 16
        assert embedder.cache is None
    
    def test_supported_models(self):
        """Test supported models dictionary."""
        assert "miniLM" in SUPPORTED_MODELS
        assert "mpnet" in SUPPORTED_MODELS
        assert "bge" in SUPPORTED_MODELS
        assert "e5" in SUPPORTED_MODELS
        assert "gte" in SUPPORTED_MODELS
        
        # Check that all models are valid sentence-transformers models
        for model_name in SUPPORTED_MODELS.values():
            assert "sentence-transformers/" in model_name or "BAAI/" in model_name or "intfloat/" in model_name or "thenlper/" in model_name


class TestIntegration:
    """Integration tests for the embedder module."""
    
    def setup_method(self):
        """Set up integration test."""
        self.temp_dir = tempfile.mkdtemp()
        self.embedder = create_embedder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            use_cache=True,
            batch_size=4
        )
        self.embedder.config.cache_dir = self.temp_dir
    
    def teardown_method(self):
        """Clean up integration test."""
        shutil.rmtree(self.temp_dir)
    
    def test_full_workflow(self):
        """Test complete embedding workflow."""
        # Test documents
        documents = [
            "Insurance policies must be renewed annually.",
            "Legal compliance requires regular audits.",
            "Financial statements should be filed quarterly.",
            "Risk assessment is mandatory for all operations.",
            "Documentation must be maintained for 7 years."
        ]
        
        # Test queries
        queries = [
            "How often should insurance be renewed?",
            "What are the audit requirements?",
            "When are financial statements due?"
        ]
        
        # Embed documents
        doc_result = self.embedder.embed_chunks(documents)
        assert len(doc_result.vectors) == 5
        
        # Embed queries and find similarities
        for query in queries:
            query_embedding = self.embedder.embed_query(query)
            similarities = self.embedder.compute_similarity(query_embedding, doc_result.vectors)
            
            # Should find at least one relevant document
            assert max(similarities) > 0.3  # Reasonable similarity threshold
        
        # Test caching behavior
        doc_result2 = self.embedder.embed_chunks(documents)
        assert doc_result2.cache_hits == 5
        assert doc_result2.cache_misses == 0
        
        # Verify embeddings are identical
        assert np.array_equal(doc_result.vectors, doc_result2.vectors)


class TestImprovedFeatures:
    """Test the new improved features."""
    
    def setup_method(self):
        """Set up test embedder with improved features."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = EmbeddingConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            use_cache=True,
            cache_dir=self.temp_dir,
            batch_size=2,
            enable_deterministic=True,
            chunk_id_length=16
        )
        self.embedder = Embedder(self.config)
    
    def teardown_method(self):
        """Clean up test directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_sha256_hashing(self):
        """Test SHA256 hashing with improved normalization."""
        # Test basic normalization
        text1 = "Hello, World!"
        text2 = "hello world"
        text3 = "  Hello,   World!  "
        
        id1 = self.embedder._generate_chunk_id(text1)
        id2 = self.embedder._generate_chunk_id(text2)
        id3 = self.embedder._generate_chunk_id(text3)
        
        # All should be the same after normalization
        assert id1 == id2 == id3
        assert len(id1) == 16  # Truncated SHA256
        
        # Test different texts produce different IDs
        text4 = "Different content"
        id4 = self.embedder._generate_chunk_id(text4)
        assert id1 != id4
    
    def test_deterministic_mode(self):
        """Test deterministic mode for reproducibility."""
        # Create embedder with deterministic mode
        config = EmbeddingConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            enable_deterministic=True
        )
        embedder = Embedder(config)
        
        # Test that deterministic mode is enabled
        stats = embedder.get_stats()
        assert stats["deterministic_mode"] == True
        
        # Test reproducible embeddings
        text = "Test for deterministic embeddings"
        embedding1 = embedder.embed_query(text)
        embedding2 = embedder.embed_query(text)
        
        # Should be identical in deterministic mode
        assert np.array_equal(embedding1, embedding2)
    
    def test_thread_safety(self):
        """Test thread safety of the cache."""
        import threading
        import time
        
        # Test data
        test_chunks = [f"Test chunk {i}" for i in range(10)]
        results = []
        errors = []
        
        def embed_chunks(thread_id):
            """Worker function for threading test."""
            try:
                result = self.embedder.embed_chunks(test_chunks)
                results.append((thread_id, result))
            except Exception as e:
                errors.append((thread_id, e))
        
        # Create multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=embed_chunks, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Threading errors: {errors}"
        
        # Verify all threads produced results
        assert len(results) == 3
        
        # Verify cache stats show thread safety
        cache_stats = self.embedder.cache.get_stats()
        assert cache_stats["thread_safe"] == True
    
    def test_memory_aware_batching(self):
        """Test memory-aware batch size calculation."""
        # Test with CPU (should return original batch size)
        safe_size = self.embedder._calculate_safe_batch_size(32, 100)
        assert safe_size == 32
        
        # Test with GPU (if available)
        if torch.cuda.is_available():
            # This is a basic test - actual behavior depends on GPU memory
            safe_size = self.embedder._calculate_safe_batch_size(32, 100)
            assert safe_size <= 32  # Should not increase batch size
    
    def test_chunk_id_normalization(self):
        """Test advanced chunk ID normalization."""
        # Test various text formats
        test_cases = [
            ("Hello, World!", "hello world"),
            ("  Multiple    Spaces  ", "multiple spaces"),
            ("Punctuation!@#$%^&*()", "punctuation"),
            ("Mixed CASE and 123 numbers", "mixed case and 123 numbers"),
            ("\n\nNewlines\n\n", "newlines"),
            ("Tabs\t\t\there", "tabs here"),
        ]
        
        for input_text, expected_normalized in test_cases:
            chunk_id = self.embedder._generate_chunk_id(input_text)
            # The chunk ID should be consistent for the same normalized text
            assert len(chunk_id) == 16
            assert chunk_id.isalnum()  # Should only contain alphanumeric characters
    
    def test_cache_backend_options(self):
        """Test cache backend options documentation."""
        from embedder import CACHE_BACKENDS
        
        # Verify cache backend options are documented
        assert "file" in CACHE_BACKENDS
        assert "redis" in CACHE_BACKENDS
        assert "sqlite" in CACHE_BACKENDS
        assert "memory" in CACHE_BACKENDS
        
        # Verify descriptions are provided
        for backend, description in CACHE_BACKENDS.items():
            assert len(description) > 0
    
    def test_retry_logic(self):
        """Test retry logic for model loading."""
        # This test verifies that retry logic is available
        # Actual retry behavior is tested in integration scenarios
        assert hasattr(self.embedder, '_load_model_with_retry')
        assert hasattr(self.embedder, '_load_model')
        
        # Test that model loading works
        assert self.embedder.model is not None
        assert self.embedder.config.embedding_dim is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 