"""
Production-grade document indexing module for RAG system.
Integrates FAISS vector indexing with PostgreSQL metadata storage.
Supports document-level hashing, versioning, and hybrid retrieval capabilities.
"""

import os
import json
import logging
import hashlib
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

import numpy as np
import faiss
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from psycopg2.pool import SimpleConnectionPool

# Import shared utilities
try:
    from ..utils.hashing import generate_chunk_id
    from ..config import get_model_name
except ImportError:
    # Handle case when running as script
    from utils.hashing import generate_chunk_id
    from config import get_model_name

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IndexType(Enum):
    """Supported index types for different use cases."""
    FLAT_IP = "flat_ip"  # Inner product (cosine similarity)
    FLAT_L2 = "flat_l2"  # L2 distance
    IVF_FLAT = "ivf_flat"  # Inverted file with flat quantizer
    HNSW = "hnsw"  # Hierarchical navigable small world


class CompressionType(Enum):
    """Supported compression types for large-scale deployments."""
    NONE = "none"  # No compression
    FLOAT16 = "float16"  # Half precision
    PQ = "pq"  # Product quantization


@dataclass
class IndexingConfig:
    """Configuration for document indexing."""
    # FAISS settings
    index_type: IndexType = IndexType.FLAT_IP
    compression_type: CompressionType = CompressionType.NONE
    normalize_vectors: bool = True
    
    # PostgreSQL settings
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "rag_index"
    db_user: str = "postgres"
    db_password: str = ""
    db_pool_size: int = 5
    
    # File storage
    index_dir: str = "./faiss_indexes"
    index_filename: str = "faiss.index"
    
    # Performance settings
    batch_size: int = 1000
    enable_document_hashing: bool = True
    enable_metadata_validation: bool = True
    
    # Hybrid retrieval support
    enable_sparse_vectors: bool = False
    enable_keyword_tags: bool = False


@dataclass
class IndexingResult:
    """Result container for indexing operations."""
    doc_id: str
    total_chunks: int
    total_vectors: int
    processing_time: float
    index_size_mb: float
    metadata_rows: int
    document_hash: Optional[str] = None
    is_reindex: bool = False
    skipped_reindex: bool = False
    errors: List[str] = field(default_factory=list)


@dataclass
class ChunkMetadata:
    """Metadata for a document chunk."""
    doc_id: str
    chunk_id: str
    chunk_index: int
    char_start: int
    char_end: int
    section_title: Optional[str] = None
    source_filename: Optional[str] = None
    token_count: Optional[int] = None
    quality_score: Optional[float] = None
    is_suspicious: bool = False
    # Hybrid retrieval fields
    sparse_vector: Optional[np.ndarray] = None
    keyword_tags: Optional[List[str]] = None
    # Additional metadata
    extra_metadata: Dict[str, Any] = field(default_factory=dict)


class FAISSIndexManager:
    """Manages FAISS index operations with versioning and compression."""
    
    def __init__(self, config: IndexingConfig):
        self.config = config
        self.index_dir = Path(config.index_dir)
        self.index_dir.mkdir(exist_ok=True)
        self.index_path = self.index_dir / config.index_filename
        
        self.index = None
        self.vector_dimension = None
        self.total_vectors = 0
        
    def create_index(self, dimension: int) -> None:
        """Create a new FAISS index based on configuration."""
        self.vector_dimension = dimension
        
        if self.config.index_type == IndexType.FLAT_IP:
            self.index = faiss.IndexFlatIP(dimension)
        elif self.config.index_type == IndexType.FLAT_L2:
            self.index = faiss.IndexFlatL2(dimension)
        elif self.config.index_type == IndexType.IVF_FLAT:
            # Use 100 clusters for IVF
            nlist = min(100, max(1, self.total_vectors // 100))
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        elif self.config.index_type == IndexType.HNSW:
            self.index = faiss.IndexHNSWFlat(dimension, 32)  # 32 neighbors
        
        # Apply compression if specified
        if self.config.compression_type == CompressionType.FLOAT16:
            self.index = faiss.IndexScalarQuantizer(
                self.index, faiss.ScalarQuantizer.QT_fp16
            )
        elif self.config.compression_type == CompressionType.PQ:
            # Product quantization with 8 bits per sub-vector
            bits = 8
            nsub = dimension // 8
            self.index = faiss.IndexPQ(dimension, nsub, bits)
        
        logger.info(f"Created FAISS index: {self.config.index_type.value} "
                   f"with compression: {self.config.compression_type.value}")
    
    def add_vectors(self, vectors: np.ndarray, ids: Optional[np.ndarray] = None) -> None:
        """Add vectors to the index."""
        if self.index is None:
            raise ValueError("Index not initialized. Call create_index() first.")
        
        # Normalize vectors if required
        if self.config.normalize_vectors:
            faiss.normalize_L2(vectors)
        
        # Add vectors to index
        if ids is not None:
            self.index.add_with_ids(vectors, ids)
        else:
            self.index.add(vectors)
        
        self.total_vectors += len(vectors)
        logger.info(f"Added {len(vectors)} vectors to index. Total: {self.total_vectors}")
    
    def remove_vectors(self, doc_id: str, metadata_db) -> int:
        """Remove all vectors for a specific document."""
        if self.index is None:
            return 0
        
        # Get vector IDs for this document from metadata DB
        vector_ids = metadata_db.get_vector_ids_for_document(doc_id)
        if not vector_ids:
            return 0
        
        # Remove from FAISS index
        vector_ids_array = np.array(vector_ids, dtype=np.int64)
        self.index.remove_ids(vector_ids_array)
        
        removed_count = len(vector_ids)
        self.total_vectors -= removed_count
        
        logger.info(f"Removed {removed_count} vectors for document {doc_id}")
        return removed_count
    
    def search(self, query_vector: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar vectors."""
        if self.index is None:
            raise ValueError("Index not initialized.")
        
        # Normalize query vector if required
        if self.config.normalize_vectors:
            query_vector = query_vector.reshape(1, -1)
            faiss.normalize_L2(query_vector)
        
        # Search
        distances, indices = self.index.search(query_vector, k)
        return distances[0], indices[0]
    
    def save_index(self) -> None:
        """Save the index to disk."""
        if self.index is None:
            raise ValueError("No index to save.")
        
        faiss.write_index(self.index, str(self.index_path))
        logger.info(f"Saved FAISS index to {self.index_path}")
    
    def load_index(self) -> bool:
        """Load the index from disk."""
        if not self.index_path.exists():
            logger.warning(f"Index file not found: {self.index_path}")
            return False
        
        try:
            self.index = faiss.read_index(str(self.index_path))
            self.total_vectors = self.index.ntotal
            self.vector_dimension = self.index.d
            
            logger.info(f"Loaded FAISS index with {self.total_vectors} vectors "
                       f"of dimension {self.vector_dimension}")
            return True
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        if self.index is None:
            return {"error": "Index not initialized"}
        
        stats = {
            "total_vectors": self.total_vectors,
            "vector_dimension": self.vector_dimension,
            "index_type": self.config.index_type.value,
            "compression_type": self.config.compression_type.value,
            "is_trained": self.index.is_trained if hasattr(self.index, 'is_trained') else True
        }
        
        # Calculate index size
        if self.index_path.exists():
            stats["index_size_mb"] = self.index_path.stat().st_size / (1024 * 1024)
        
        return stats


class PostgreSQLManager:
    """Manages PostgreSQL metadata storage."""
    
    def __init__(self, config: IndexingConfig):
        self.config = config
        self.pool = None
        self._initialize_connection_pool()
        self._create_tables()
    
    def _initialize_connection_pool(self) -> None:
        """Initialize connection pool."""
        try:
            self.pool = SimpleConnectionPool(
                minconn=1,
                maxconn=self.config.db_pool_size,
                host=self.config.db_host,
                port=self.config.db_port,
                database=self.config.db_name,
                user=self.config.db_user,
                password=self.config.db_password
            )
            logger.info("PostgreSQL connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL pool: {e}")
            raise
    
    def _create_tables(self) -> None:
        """Create necessary tables if they don't exist."""
        create_tables_sql = """
        CREATE TABLE IF NOT EXISTS document_metadata (
            doc_id VARCHAR(255) PRIMARY KEY,
            document_hash VARCHAR(64) NOT NULL,
            source_filename VARCHAR(500),
            total_chunks INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS chunk_metadata (
            embedding_id SERIAL PRIMARY KEY,
            doc_id VARCHAR(255) NOT NULL,
            chunk_id VARCHAR(64) NOT NULL,
            chunk_index INTEGER NOT NULL,
            char_start INTEGER,
            char_end INTEGER,
            section_title TEXT,
            source_filename VARCHAR(500),
            token_count INTEGER,
            quality_score FLOAT,
            is_suspicious BOOLEAN DEFAULT FALSE,
            sparse_vector BYTEA,
            keyword_tags JSONB,
            extra_metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (doc_id) REFERENCES document_metadata(doc_id) ON DELETE CASCADE
        );
        
        CREATE INDEX IF NOT EXISTS idx_chunk_metadata_doc_id ON chunk_metadata(doc_id);
        CREATE INDEX IF NOT EXISTS idx_chunk_metadata_chunk_id ON chunk_metadata(chunk_id);
        CREATE INDEX IF NOT EXISTS idx_chunk_metadata_embedding_id ON chunk_metadata(embedding_id);
        """
        
        with self.pool.getconn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(create_tables_sql)
                conn.commit()
        
        self.pool.putconn(conn)
        logger.info("PostgreSQL tables created/verified")
    
    def insert_document_metadata(self, doc_id: str, document_hash: str, 
                                source_filename: str, total_chunks: int) -> None:
        """Insert or update document metadata."""
        sql = """
        INSERT INTO document_metadata (doc_id, document_hash, source_filename, total_chunks, updated_at)
        VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT (doc_id) DO UPDATE SET
            document_hash = EXCLUDED.document_hash,
            source_filename = EXCLUDED.source_filename,
            total_chunks = EXCLUDED.total_chunks,
            updated_at = CURRENT_TIMESTAMP
        """
        
        with self.pool.getconn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, (doc_id, document_hash, source_filename, total_chunks))
                conn.commit()
        
        self.pool.putconn(conn)
    
    def insert_chunk_metadata_batch(self, metadata_list: List[ChunkMetadata]) -> List[int]:
        """Insert chunk metadata in batch and return embedding IDs."""
        sql = """
        INSERT INTO chunk_metadata 
        (doc_id, chunk_id, chunk_index, char_start, char_end, section_title, 
         source_filename, token_count, quality_score, is_suspicious, 
         sparse_vector, keyword_tags, extra_metadata)
        VALUES %s
        RETURNING embedding_id
        """
        
        # Prepare data for batch insert
        data = []
        for meta in metadata_list:
            # Serialize sparse vector if present
            sparse_vector_bytes = None
            if meta.sparse_vector is not None:
                sparse_vector_bytes = meta.sparse_vector.tobytes()
            
            # Serialize keyword tags if present
            keyword_tags_json = None
            if meta.keyword_tags is not None:
                keyword_tags_json = json.dumps(meta.keyword_tags)
            
            data.append((
                meta.doc_id, meta.chunk_id, meta.chunk_index,
                meta.char_start, meta.char_end, meta.section_title,
                meta.source_filename, meta.token_count, meta.quality_score,
                meta.is_suspicious, sparse_vector_bytes, keyword_tags_json,
                json.dumps(meta.extra_metadata)
            ))
        
        embedding_ids = []
        with self.pool.getconn() as conn:
            with conn.cursor() as cursor:
                execute_values(cursor, sql, data, fetch=True)
                embedding_ids = [row[0] for row in cursor.fetchall()]
                conn.commit()
        
        self.pool.putconn(conn)
        logger.info(f"Inserted {len(embedding_ids)} chunk metadata records")
        return embedding_ids
    
    def get_vector_ids_for_document(self, doc_id: str) -> List[int]:
        """Get all embedding IDs for a specific document."""
        sql = "SELECT embedding_id FROM chunk_metadata WHERE doc_id = %s ORDER BY chunk_index"
        
        with self.pool.getconn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, (doc_id,))
                embedding_ids = [row[0] for row in cursor.fetchall()]
        
        self.pool.putconn(conn)
        return embedding_ids
    
    def delete_document_metadata(self, doc_id: str) -> int:
        """Delete all metadata for a specific document."""
        sql = "DELETE FROM chunk_metadata WHERE doc_id = %s"
        
        with self.pool.getconn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, (doc_id,))
                deleted_count = cursor.rowcount
                conn.commit()
        
        self.pool.putconn(conn)
        logger.info(f"Deleted {deleted_count} chunk metadata records for document {doc_id}")
        return deleted_count
    
    def get_document_hash(self, doc_id: str) -> Optional[str]:
        """Get the document hash for a specific document."""
        sql = "SELECT document_hash FROM document_metadata WHERE doc_id = %s"
        
        with self.pool.getconn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, (doc_id,))
                result = cursor.fetchone()
        
        self.pool.putconn(conn)
        return result[0] if result else None
    
    def get_chunk_metadata(self, embedding_id: int) -> Optional[Dict[str, Any]]:
        """Get chunk metadata by embedding ID."""
        sql = """
        SELECT doc_id, chunk_id, chunk_index, char_start, char_end, section_title,
               source_filename, token_count, quality_score, is_suspicious,
               sparse_vector, keyword_tags, extra_metadata
        FROM chunk_metadata WHERE embedding_id = %s
        """
        
        with self.pool.getconn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(sql, (embedding_id,))
                result = cursor.fetchone()
        
        self.pool.putconn(conn)
        
        if result:
            metadata = dict(result)
            # Deserialize sparse vector if present
            if metadata['sparse_vector']:
                metadata['sparse_vector'] = np.frombuffer(metadata['sparse_vector'])
            # Deserialize JSON fields
            if metadata['keyword_tags']:
                metadata['keyword_tags'] = json.loads(metadata['keyword_tags'])
            if metadata['extra_metadata']:
                metadata['extra_metadata'] = json.loads(metadata['extra_metadata'])
            return metadata
        return None


class DocumentIndexer:
    """Main document indexing class that orchestrates FAISS and PostgreSQL operations."""
    
    def __init__(self, config: Optional[IndexingConfig] = None):
        self.config = config or IndexingConfig()
        self.faiss_manager = FAISSIndexManager(self.config)
        self.postgres_manager = PostgreSQLManager(self.config)
        
        # Load existing index if available
        self.faiss_manager.load_index()
        
        logger.info("DocumentIndexer initialized")
    
    def _calculate_document_hash(self, chunks: List[str]) -> str:
        """Calculate SHA256 hash of document content for change detection."""
        # Concatenate all chunks and hash
        content = "".join(chunks)
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _validate_inputs(self, doc_id: str, chunks: List[str], 
                        embeddings: np.ndarray, metadata_list: List[ChunkMetadata]) -> None:
        """Validate input data integrity."""
        if not chunks:
            raise ValueError("Chunks list cannot be empty")
        
        if len(chunks) != len(embeddings):
            raise ValueError(f"Number of chunks ({len(chunks)}) must match "
                           f"number of embeddings ({len(embeddings)})")
        
        if len(chunks) != len(metadata_list):
            raise ValueError(f"Number of chunks ({len(chunks)}) must match "
                           f"number of metadata records ({len(metadata_list)})")
        
        # Validate metadata consistency
        for i, meta in enumerate(metadata_list):
            if meta.doc_id != doc_id:
                raise ValueError(f"Metadata doc_id mismatch at index {i}")
            
            # Validate chunk ID consistency
            expected_chunk_id = generate_chunk_id(chunks[i])
            if meta.chunk_id != expected_chunk_id:
                raise ValueError(f"Chunk ID mismatch at index {i}")
        
        logger.info(f"Input validation passed for document {doc_id}")
    
    def _prepare_metadata(self, doc_id: str, chunks: List[str], 
                         metadata_list: List[ChunkMetadata]) -> List[ChunkMetadata]:
        """Prepare and validate metadata for indexing."""
        prepared_metadata = []
        
        for i, (chunk, meta) in enumerate(zip(chunks, metadata_list)):
            # Ensure chunk_id is consistent
            chunk_id = generate_chunk_id(chunk)
            
            # Create prepared metadata
            prepared_meta = ChunkMetadata(
                doc_id=doc_id,
                chunk_id=chunk_id,
                chunk_index=i,
                char_start=meta.char_start,
                char_end=meta.char_end,
                section_title=meta.section_title,
                source_filename=meta.source_filename,
                token_count=meta.token_count,
                quality_score=meta.quality_score,
                is_suspicious=meta.is_suspicious,
                sparse_vector=meta.sparse_vector,
                keyword_tags=meta.keyword_tags,
                extra_metadata=meta.extra_metadata
            )
            prepared_metadata.append(prepared_meta)
        
        return prepared_metadata
    
    def index_document(self, doc_id: str, chunks: List[str], 
                      embeddings: np.ndarray, metadata_list: List[ChunkMetadata],
                      overwrite: bool = False, source_filename: str = "unknown") -> IndexingResult:
        """
        Index a document with its chunks, embeddings, and metadata.
        
        Args:
            doc_id: Unique document identifier
            chunks: List of text chunks
            embeddings: NumPy array of embeddings (n_chunks x embedding_dim)
            metadata_list: List of chunk metadata
            overwrite: Whether to overwrite existing document
            source_filename: Original filename
            
        Returns:
            IndexingResult with processing statistics
        """
        start_time = time.time()
        result = IndexingResult(doc_id=doc_id, total_chunks=len(chunks))
        
        try:
            # Validate inputs
            self._validate_inputs(doc_id, chunks, embeddings, metadata_list)
            
            # Calculate document hash for change detection
            document_hash = self._calculate_document_hash(chunks)
            result.document_hash = document_hash
            
            # Check if document needs re-indexing
            if not overwrite and self.config.enable_document_hashing:
                existing_hash = self.postgres_manager.get_document_hash(doc_id)
                if existing_hash == document_hash:
                    logger.info(f"Document {doc_id} unchanged, skipping re-indexing")
                    result.skipped_reindex = True
                    result.processing_time = time.time() - start_time
                    return result
            
            # Remove existing document if overwriting
            if overwrite:
                logger.info(f"Overwriting existing document {doc_id}")
                self.faiss_manager.remove_vectors(doc_id, self.postgres_manager)
                self.postgres_manager.delete_document_metadata(doc_id)
                result.is_reindex = True
            
            # Prepare metadata
            prepared_metadata = self._prepare_metadata(doc_id, chunks, metadata_list)
            
            # Initialize FAISS index if needed
            if self.faiss_manager.index is None:
                embedding_dim = embeddings.shape[1]
                self.faiss_manager.create_index(embedding_dim)
            
            # Insert metadata into PostgreSQL and get embedding IDs
            embedding_ids = self.postgres_manager.insert_chunk_metadata_batch(prepared_metadata)
            result.metadata_rows = len(embedding_ids)
            
            # Add vectors to FAISS with embedding IDs
            embedding_ids_array = np.array(embedding_ids, dtype=np.int64)
            self.faiss_manager.add_vectors(embeddings, embedding_ids_array)
            
            # Update document metadata
            self.postgres_manager.insert_document_metadata(
                doc_id, document_hash, source_filename, len(chunks)
            )
            
            # Save FAISS index
            self.faiss_manager.save_index()
            
            # Calculate result statistics
            result.total_vectors = len(embeddings)
            result.processing_time = time.time() - start_time
            
            # Get index size
            index_stats = self.faiss_manager.get_index_stats()
            result.index_size_mb = index_stats.get("index_size_mb", 0.0)
            
            logger.info(f"Successfully indexed document {doc_id}: "
                       f"{result.total_chunks} chunks, {result.total_vectors} vectors, "
                       f"{result.processing_time:.2f}s")
            
        except Exception as e:
            error_msg = f"Failed to index document {doc_id}: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)
            raise
        
        return result
    
    def search_similar(self, query_embedding: np.ndarray, k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar chunks and return with metadata."""
        distances, indices = self.faiss_manager.search(query_embedding, k)
        
        results = []
        for distance, embedding_id in zip(distances, indices):
            if embedding_id == -1:  # FAISS returns -1 for invalid indices
                continue
            
            metadata = self.postgres_manager.get_chunk_metadata(embedding_id)
            if metadata:
                result = {
                    "embedding_id": embedding_id,
                    "similarity_score": float(distance),
                    "metadata": metadata
                }
                results.append(result)
        
        return results
    
    def get_document_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document."""
        vector_ids = self.postgres_manager.get_vector_ids_for_document(doc_id)
        
        chunks = []
        for embedding_id in vector_ids:
            metadata = self.postgres_manager.get_chunk_metadata(embedding_id)
            if metadata:
                chunks.append(metadata)
        
        return sorted(chunks, key=lambda x: x['chunk_index'])
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get comprehensive index statistics."""
        faiss_stats = self.faiss_manager.get_index_stats()
        
        # Get PostgreSQL stats
        with self.postgres_manager.pool.getconn() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM document_metadata")
                total_documents = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM chunk_metadata")
                total_chunks = cursor.fetchone()[0]
                
                cursor.execute("SELECT AVG(total_chunks) FROM document_metadata")
                avg_chunks_per_doc = cursor.fetchone()[0] or 0
        
        self.postgres_manager.pool.putconn(conn)
        
        stats = {
            "faiss": faiss_stats,
            "postgresql": {
                "total_documents": total_documents,
                "total_chunks": total_chunks,
                "avg_chunks_per_document": float(avg_chunks_per_doc)
            },
            "index_dir": str(self.config.index_dir),
            "index_filename": self.config.index_filename
        }
        
        return stats
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and all its chunks from the index."""
        try:
            # Remove from FAISS
            removed_vectors = self.faiss_manager.remove_vectors(doc_id, self.postgres_manager)
            
            # Remove from PostgreSQL
            removed_metadata = self.postgres_manager.delete_document_metadata(doc_id)
            
            # Save updated index
            self.faiss_manager.save_index()
            
            logger.info(f"Deleted document {doc_id}: {removed_vectors} vectors, {removed_metadata} metadata rows")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False
    
    def close(self) -> None:
        """Clean up resources."""
        if self.postgres_manager.pool:
            self.postgres_manager.pool.closeall()
        logger.info("DocumentIndexer resources cleaned up")


# Factory function for easy indexer creation
def create_indexer(
    index_type: str = "flat_ip",
    compression_type: str = "none",
    db_host: str = "localhost",
    db_name: str = "rag_index",
    db_user: str = "postgres",
    db_password: str = "",
    index_dir: str = "./faiss_indexes"
) -> DocumentIndexer:
    """
    Factory function to create an indexer with common configurations.
    
    Args:
        index_type: FAISS index type ("flat_ip", "flat_l2", "ivf_flat", "hnsw")
        compression_type: Compression type ("none", "float16", "pq")
        db_host: PostgreSQL host
        db_name: PostgreSQL database name
        db_user: PostgreSQL username
        db_password: PostgreSQL password
        index_dir: Directory for FAISS index files
        
    Returns:
        Configured DocumentIndexer instance
    """
    try:
        index_type_enum = IndexType(index_type)
    except ValueError:
        logger.warning(f"Invalid index_type '{index_type}', using 'flat_ip'")
        index_type_enum = IndexType.FLAT_IP
    
    try:
        compression_type_enum = CompressionType(compression_type)
    except ValueError:
        logger.warning(f"Invalid compression_type '{compression_type}', using 'none'")
        compression_type_enum = CompressionType.NONE
    
    config = IndexingConfig(
        index_type=index_type_enum,
        compression_type=compression_type_enum,
        db_host=db_host,
        db_name=db_name,
        db_user=db_user,
        db_password=db_password,
        index_dir=index_dir
    )
    
    return DocumentIndexer(config)


# Utility functions for index management
def load_faiss_index(index_path: str) -> Optional[faiss.Index]:
    """Load a FAISS index from disk."""
    try:
        index = faiss.read_index(index_path)
        logger.info(f"Loaded FAISS index from {index_path}")
        return index
    except Exception as e:
        logger.error(f"Failed to load FAISS index from {index_path}: {e}")
        return None


def save_faiss_index(index: faiss.Index, index_path: str) -> bool:
    """Save a FAISS index to disk."""
    try:
        faiss.write_index(index, index_path)
        logger.info(f"Saved FAISS index to {index_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save FAISS index to {index_path}: {e}")
        return False


# CLI interface for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Document indexer CLI")
    parser.add_argument("--doc-id", required=True, help="Document ID")
    parser.add_argument("--chunks", nargs="+", required=True, help="Text chunks")
    parser.add_argument("--embeddings", required=True, help="Embeddings file (.npy)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing document")
    parser.add_argument("--index-type", default="flat_ip", help="FAISS index type")
    parser.add_argument("--db-host", default="localhost", help="PostgreSQL host")
    parser.add_argument("--db-name", default="rag_index", help="PostgreSQL database")
    parser.add_argument("--db-user", default="postgres", help="PostgreSQL user")
    parser.add_argument("--db-password", default="", help="PostgreSQL password")
    
    args = parser.parse_args()
    
    # Load embeddings
    embeddings = np.load(args.embeddings)
    
    # Create sample metadata
    metadata_list = []
    for i, chunk in enumerate(args.chunks):
        metadata = ChunkMetadata(
            doc_id=args.doc_id,
            chunk_id=generate_chunk_id(chunk),
            chunk_index=i,
            char_start=i * 100,  # Placeholder
            char_end=(i + 1) * 100,  # Placeholder
            source_filename="cli_test.txt"
        )
        metadata_list.append(metadata)
    
    # Create indexer and index document
    indexer = create_indexer(
        index_type=args.index_type,
        db_host=args.db_host,
        db_name=args.db_name,
        db_user=args.db_user,
        db_password=args.db_password
    )
    
    try:
        result = indexer.index_document(
            doc_id=args.doc_id,
            chunks=args.chunks,
            embeddings=embeddings,
            metadata_list=metadata_list,
            overwrite=args.overwrite
        )
        
        print(f"Indexing completed successfully!")
        print(f"  Document ID: {result.doc_id}")
        print(f"  Total chunks: {result.total_chunks}")
        print(f"  Total vectors: {result.total_vectors}")
        print(f"  Processing time: {result.processing_time:.2f}s")
        print(f"  Index size: {result.index_size_mb:.2f} MB")
        print(f"  Metadata rows: {result.metadata_rows}")
        
        # Show index stats
        stats = indexer.get_index_stats()
        print(f"\nIndex Statistics:")
        print(f"  FAISS vectors: {stats['faiss']['total_vectors']}")
        print(f"  PostgreSQL documents: {stats['postgresql']['total_documents']}")
        print(f"  PostgreSQL chunks: {stats['postgresql']['total_chunks']}")
        
    finally:
        indexer.close()
