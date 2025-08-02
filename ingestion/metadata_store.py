"""
Metadata Store Module for RAG Backend

This module provides a centralized metadata storage system that tracks every document,
chunk, and transformation in the RAG pipeline. It supports PostgreSQL with SQLite fallback,
ACID compliance, audit logging, and comprehensive querying capabilities.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from uuid import UUID, uuid4
import json
from pathlib import Path

# Database imports
import asyncpg
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, Boolean, DateTime, Text, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
import sqlite3

# Custom exceptions
class MetadataStoreError(Exception):
    """Base exception for metadata store errors"""
    pass

class DocumentNotFoundError(MetadataStoreError):
    """Exception raised when document is not found"""
    pass

class DuplicateDocumentError(MetadataStoreError):
    """Exception raised when document already exists"""
    pass

# Enums
class DocumentStatus(str, Enum):
    """Document processing status"""
    UPLOADED = "uploaded"
    PARSED = "parsed"
    CHUNKED = "chunked"
    EMBEDDED = "embedded"
    INDEXED = "indexed"
    ERROR = "error"

class DocumentType(str, Enum):
    """Document type classification"""
    LEGAL = "legal"
    INSURANCE = "insurance"
    FINANCIAL = "financial"
    CONTRACT = "contract"
    POLICY = "policy"
    REPORT = "report"
    MANUAL = "manual"
    OTHER = "other"

# SQLAlchemy Base
Base = declarative_base()

class Document(Base):
    """Document metadata table"""
    __tablename__ = "documents"
    
    doc_id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    filename = Column(String(255), nullable=False)
    source_url = Column(Text, nullable=True)
    doc_type = Column(String(50), nullable=False)
    extension = Column(String(20), nullable=False)
    file_size = Column(Integer, nullable=False)
    doc_hash = Column(String(64), nullable=False, unique=True, index=True)
    requires_ocr = Column(Boolean, default=False)
    upload_time = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    status = Column(SQLEnum(DocumentStatus), nullable=False, default=DocumentStatus.UPLOADED)
    error_msg = Column(Text, nullable=True)
    language = Column(String(10), nullable=True)
    source_type = Column(String(50), nullable=True)  # 'upload', 'url', 's3', etc.
    user_id = Column(String(100), nullable=True)
    version = Column(Integer, default=1)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

class Chunk(Base):
    """Document chunk metadata table"""
    __tablename__ = "chunks"
    
    chunk_id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    doc_id = Column(PGUUID(as_uuid=True), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    embedding_id = Column(String(100), nullable=True, index=True)
    token_count = Column(Integer, nullable=True)
    page_number = Column(Integer, nullable=True)
    section_title = Column(String(255), nullable=True)
    quality_score = Column(Integer, nullable=True)  # 0-100
    is_suspicious = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

class IngestionLog(Base):
    """Audit log for ingestion pipeline"""
    __tablename__ = "ingestion_log"
    
    log_id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    doc_id = Column(PGUUID(as_uuid=True), nullable=True, index=True)
    step = Column(String(50), nullable=False)  # 'load', 'parse', 'chunk', 'embed', 'index'
    status = Column(String(20), nullable=False)  # 'started', 'success', 'error'
    message = Column(Text, nullable=True)
    metadata_json = Column(Text, nullable=True)  # JSON string for additional data
    timestamp = Column(DateTime(timezone=True), default=datetime.utcnow)
    processing_time_ms = Column(Integer, nullable=True)

class MetadataStore:
    """
    Centralized metadata storage for RAG backend.
    
    Provides ACID-compliant storage for documents, chunks, and audit logs
    with PostgreSQL support and SQLite fallback for development.
    """
    
    def __init__(
        self,
        db_url: str,
        enable_audit_logging: bool = True,
        max_retries: int = 3,
        pool_size: int = 10
    ):
        """
        Initialize metadata store.
        
        Args:
            db_url: Database connection URL (postgresql:// or sqlite://)
            enable_audit_logging: Enable detailed audit logging
            max_retries: Maximum retry attempts for database operations
            pool_size: Connection pool size
        """
        self.db_url = db_url
        self.enable_audit_logging = enable_audit_logging
        self.max_retries = max_retries
        self.pool_size = pool_size
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize database connection
        self._setup_database()
    
    def _setup_database(self):
        """Setup database connection and create tables."""
        try:
            if self.db_url.startswith('postgresql'):
                # PostgreSQL with async support
                self.engine = create_async_engine(
                    self.db_url,
                    pool_size=self.pool_size,
                    max_overflow=20,
                    echo=False
                )
                self.is_postgres = True
            else:
                # SQLite fallback
                self.engine = create_engine(
                    self.db_url,
                    echo=False,
                    connect_args={"check_same_thread": False} if "sqlite" in self.db_url else {}
                )
                self.is_postgres = False
            
            # Create tables
            Base.metadata.create_all(self.engine)
            
            # Setup session factory
            if self.is_postgres:
                self.async_session = sessionmaker(
                    self.engine, class_=AsyncSession, expire_on_commit=False
                )
            else:
                self.session = sessionmaker(bind=self.engine)
            
            self.logger.info(f"Database initialized: {self.db_url}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise MetadataStoreError(f"Database initialization failed: {e}")
    
    async def _get_session(self):
        """Get database session."""
        if self.is_postgres:
            return self.async_session()
        else:
            return self.session()
    
    async def save_document_metadata(self, metadata: Dict[str, Any]) -> UUID:
        """
        Save document metadata to database.
        
        Args:
            metadata: Document metadata from loader
            
        Returns:
            UUID of the created document record
            
        Raises:
            DuplicateDocumentError: If document with same hash already exists
            MetadataStoreError: For other database errors
        """
        for attempt in range(self.max_retries):
            try:
                # Check for existing document with same hash
                existing_doc = await self.get_doc_by_hash(metadata['doc_hash'])
                if existing_doc:
                    raise DuplicateDocumentError(f"Document with hash {metadata['doc_hash']} already exists")
                
                # Create new document record
                doc = Document(
                    filename=metadata['filename'],
                    source_url=metadata.get('source_url'),
                    doc_type=metadata.get('doc_type', DocumentType.OTHER.value),
                    extension=metadata['extension'],
                    file_size=metadata['file_size'],
                    doc_hash=metadata['doc_hash'],
                    requires_ocr=metadata.get('requires_ocr', False),
                    upload_time=datetime.fromisoformat(metadata['upload_time'].replace('Z', '+00:00')),
                    status=DocumentStatus.UPLOADED,
                    language=metadata.get('language'),
                    source_type=metadata.get('source_type', 'upload'),
                    user_id=metadata.get('user_id')
                )
                
                if self.is_postgres:
                    async with self.async_session() as session:
                        session.add(doc)
                        await session.commit()
                        await session.refresh(doc)
                else:
                    with self.session() as session:
                        session.add(doc)
                        session.commit()
                        session.refresh(doc)
                
                doc_id = doc.doc_id
                self.logger.info(f"Saved document metadata: {doc_id}")
                
                # Log audit event
                if self.enable_audit_logging:
                    await self._log_audit_event(
                        doc_id=doc_id,
                        step="document_save",
                        status="success",
                        message=f"Document {metadata['filename']} saved successfully"
                    )
                
                return doc_id
                
            except DuplicateDocumentError:
                raise
            except Exception as e:
                if attempt == self.max_retries - 1:
                    self.logger.error(f"Failed to save document metadata after {self.max_retries} attempts: {e}")
                    raise MetadataStoreError(f"Failed to save document metadata: {e}")
                self.logger.warning(f"Attempt {attempt + 1} failed, retrying: {e}")
                await asyncio.sleep(0.1 * (attempt + 1))  # Exponential backoff
    
    async def get_doc_by_hash(self, doc_hash: str) -> Optional[Dict[str, Any]]:
        """
        Get document by SHA-256 hash.
        
        Args:
            doc_hash: SHA-256 hash of document content
            
        Returns:
            Document metadata dict or None if not found
        """
        try:
            if self.is_postgres:
                async with self.async_session() as session:
                    from sqlalchemy import select
                    stmt = select(Document).where(Document.doc_hash == doc_hash)
                    result = await session.execute(stmt)
                    doc = result.scalar_one_or_none()
            else:
                with self.session() as session:
                    from sqlalchemy import select
                    stmt = select(Document).where(Document.doc_hash == doc_hash)
                    result = session.execute(stmt)
                    doc = result.scalar_one_or_none()
            
            if doc:
                return {
                    'doc_id': doc.doc_id,
                    'filename': doc.filename,
                    'source_url': doc.source_url,
                    'doc_type': doc.doc_type,
                    'extension': doc.extension,
                    'file_size': doc.file_size,
                    'doc_hash': doc.doc_hash,
                    'requires_ocr': doc.requires_ocr,
                    'upload_time': doc.upload_time.isoformat(),
                    'status': doc.status.value,
                    'error_msg': doc.error_msg,
                    'language': doc.language,
                    'source_type': doc.source_type,
                    'user_id': doc.user_id,
                    'version': doc.version,
                    'created_at': doc.created_at.isoformat(),
                    'updated_at': doc.updated_at.isoformat()
                }
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting document by hash: {e}")
            return None
    
    async def get_document_by_id(self, doc_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Get document by UUID.
        
        Args:
            doc_id: Document UUID
            
        Returns:
            Document metadata dict or None if not found
        """
        try:
            if self.is_postgres:
                async with self.async_session() as session:
                    from sqlalchemy import select
                    stmt = select(Document).where(Document.doc_id == doc_id)
                    result = await session.execute(stmt)
                    doc = result.scalar_one_or_none()
            else:
                with self.session() as session:
                    from sqlalchemy import select
                    stmt = select(Document).where(Document.doc_id == doc_id)
                    result = session.execute(stmt)
                    doc = result.scalar_one_or_none()
            
            if doc:
                return {
                    'doc_id': doc.doc_id,
                    'filename': doc.filename,
                    'source_url': doc.source_url,
                    'doc_type': doc.doc_type,
                    'extension': doc.extension,
                    'file_size': doc.file_size,
                    'doc_hash': doc.doc_hash,
                    'requires_ocr': doc.requires_ocr,
                    'upload_time': doc.upload_time.isoformat(),
                    'status': doc.status.value,
                    'error_msg': doc.error_msg,
                    'language': doc.language,
                    'source_type': doc.source_type,
                    'user_id': doc.user_id,
                    'version': doc.version,
                    'created_at': doc.created_at.isoformat(),
                    'updated_at': doc.updated_at.isoformat()
                }
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting document by ID: {e}")
            return None
    
    async def save_chunks(self, doc_id: UUID, chunks: List[Dict[str, Any]]) -> List[UUID]:
        """
        Save chunk metadata to database.
        
        Args:
            doc_id: Document UUID
            chunks: List of chunk metadata dictionaries
            
        Returns:
            List of chunk UUIDs
        """
        chunk_ids = []
        
        try:
            if self.is_postgres:
                async with self.async_session() as session:
                    for chunk_data in chunks:
                        chunk = Chunk(
                            doc_id=doc_id,
                            chunk_index=chunk_data['chunk_index'],
                            chunk_text=chunk_data['content'],
                            embedding_id=chunk_data.get('embedding_id'),
                            token_count=chunk_data.get('token_count'),
                            page_number=chunk_data.get('page_number'),
                            section_title=chunk_data.get('section_title'),
                            quality_score=chunk_data.get('quality_score'),
                            is_suspicious=chunk_data.get('is_suspicious', False)
                        )
                        session.add(chunk)
                        chunk_ids.append(chunk.chunk_id)
                    
                    await session.commit()
            else:
                with self.session() as session:
                    for chunk_data in chunks:
                        chunk = Chunk(
                            doc_id=doc_id,
                            chunk_index=chunk_data['chunk_index'],
                            chunk_text=chunk_data['content'],
                            embedding_id=chunk_data.get('embedding_id'),
                            token_count=chunk_data.get('token_count'),
                            page_number=chunk_data.get('page_number'),
                            section_title=chunk_data.get('section_title'),
                            quality_score=chunk_data.get('quality_score'),
                            is_suspicious=chunk_data.get('is_suspicious', False)
                        )
                        session.add(chunk)
                        chunk_ids.append(chunk.chunk_id)
                    
                    session.commit()
            
            self.logger.info(f"Saved {len(chunks)} chunks for document {doc_id}")
            
            # Log audit event
            if self.enable_audit_logging:
                await self._log_audit_event(
                    doc_id=doc_id,
                    step="chunk_save",
                    status="success",
                    message=f"Saved {len(chunks)} chunks",
                    metadata={"chunk_count": len(chunks)}
                )
            
            return chunk_ids
            
        except Exception as e:
            self.logger.error(f"Failed to save chunks for document {doc_id}: {e}")
            raise MetadataStoreError(f"Failed to save chunks: {e}")
    
    async def get_chunks_by_doc(self, doc_id: UUID) -> List[Dict[str, Any]]:
        """
        Get all chunks for a document.
        
        Args:
            doc_id: Document UUID
            
        Returns:
            List of chunk metadata dictionaries
        """
        try:
            if self.is_postgres:
                async with self.async_session() as session:
                    from sqlalchemy import select
                    stmt = select(Chunk).where(Chunk.doc_id == doc_id).order_by(Chunk.chunk_index)
                    result = await session.execute(stmt)
                    chunks = result.scalars().all()
            else:
                with self.session() as session:
                    from sqlalchemy import select
                    stmt = select(Chunk).where(Chunk.doc_id == doc_id).order_by(Chunk.chunk_index)
                    result = session.execute(stmt)
                    chunks = result.scalars().all()
            
            return [{
                'chunk_id': chunk.chunk_id,
                'doc_id': chunk.doc_id,
                'chunk_index': chunk.chunk_index,
                'chunk_text': chunk.chunk_text,
                'embedding_id': chunk.embedding_id,
                'token_count': chunk.token_count,
                'page_number': chunk.page_number,
                'section_title': chunk.section_title,
                'quality_score': chunk.quality_score,
                'is_suspicious': chunk.is_suspicious,
                'created_at': chunk.created_at.isoformat()
            } for chunk in chunks]
            
        except Exception as e:
            self.logger.error(f"Error getting chunks for document {doc_id}: {e}")
            return []
    
    async def mark_doc_status(self, doc_id: UUID, status: DocumentStatus, error_msg: Optional[str] = None):
        """
        Update document processing status.
        
        Args:
            doc_id: Document UUID
            status: New status
            error_msg: Optional error message
        """
        try:
            if self.is_postgres:
                async with self.async_session() as session:
                    from sqlalchemy import update
                    stmt = update(Document).where(Document.doc_id == doc_id).values(
                        status=status,
                        error_msg=error_msg,
                        updated_at=datetime.utcnow()
                    )
                    await session.execute(stmt)
                    await session.commit()
            else:
                with self.session() as session:
                    from sqlalchemy import update
                    stmt = update(Document).where(Document.doc_id == doc_id).values(
                        status=status,
                        error_msg=error_msg,
                        updated_at=datetime.utcnow()
                    )
                    session.execute(stmt)
                    session.commit()
            
            self.logger.info(f"Updated document {doc_id} status to {status.value}")
            
            # Log audit event
            if self.enable_audit_logging:
                await self._log_audit_event(
                    doc_id=doc_id,
                    step="status_update",
                    status="success",
                    message=f"Status updated to {status.value}",
                    metadata={"new_status": status.value, "error_msg": error_msg}
                )
                
        except Exception as e:
            self.logger.error(f"Failed to update document status: {e}")
            raise MetadataStoreError(f"Failed to update document status: {e}")
    
    async def update_chunk_embedding(self, chunk_id: UUID, embedding_id: str):
        """
        Update chunk with embedding ID.
        
        Args:
            chunk_id: Chunk UUID
            embedding_id: Embedding identifier
        """
        try:
            if self.is_postgres:
                async with self.async_session() as session:
                    from sqlalchemy import update
                    stmt = update(Chunk).where(Chunk.chunk_id == chunk_id).values(embedding_id=embedding_id)
                    await session.execute(stmt)
                    await session.commit()
            else:
                with self.session() as session:
                    from sqlalchemy import update
                    stmt = update(Chunk).where(Chunk.chunk_id == chunk_id).values(embedding_id=embedding_id)
                    session.execute(stmt)
                    session.commit()
            
            self.logger.debug(f"Updated chunk {chunk_id} with embedding {embedding_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to update chunk embedding: {e}")
            raise MetadataStoreError(f"Failed to update chunk embedding: {e}")
    
    async def get_documents_by_status(self, status: DocumentStatus, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get documents by processing status.
        
        Args:
            status: Document status to filter by
            limit: Maximum number of documents to return
            
        Returns:
            List of document metadata dictionaries
        """
        try:
            if self.is_postgres:
                async with self.async_session() as session:
                    from sqlalchemy import select
                    stmt = select(Document).where(Document.status == status).order_by(Document.upload_time.desc()).limit(limit)
                    result = await session.execute(stmt)
                    docs = result.scalars().all()
            else:
                with self.session() as session:
                    from sqlalchemy import select
                    stmt = select(Document).where(Document.status == status).order_by(Document.upload_time.desc()).limit(limit)
                    result = session.execute(stmt)
                    docs = result.scalars().all()
            
            return [{
                'doc_id': doc.doc_id,
                'filename': doc.filename,
                'source_url': doc.source_url,
                'doc_type': doc.doc_type,
                'extension': doc.extension,
                'file_size': doc.file_size,
                'doc_hash': doc.doc_hash,
                'requires_ocr': doc.requires_ocr,
                'upload_time': doc.upload_time.isoformat(),
                'status': doc.status.value,
                'error_msg': doc.error_msg,
                'language': doc.language,
                'source_type': doc.source_type,
                'user_id': doc.user_id,
                'version': doc.version,
                'created_at': doc.created_at.isoformat(),
                'updated_at': doc.updated_at.isoformat()
            } for doc in docs]
            
        except Exception as e:
            self.logger.error(f"Error getting documents by status: {e}")
            return []
    
    async def get_document_stats(self) -> Dict[str, Any]:
        """
        Get document processing statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        try:
            if self.is_postgres:
                async with self.async_session() as session:
                    from sqlalchemy import func, select
                    
                    # Total documents
                    result = await session.execute(select(func.count(Document.doc_id)))
                    total_docs = result.scalar()
                    
                    # Documents by status
                    result = await session.execute(
                        select(Document.status, func.count(Document.doc_id)).group_by(Document.status)
                    )
                    status_counts = dict(result.fetchall())
                    
                    # Total chunks
                    result = await session.execute(select(func.count(Chunk.chunk_id)))
                    total_chunks = result.scalar()
                    
                    # Average chunks per document
                    from sqlalchemy import text
                    result = await session.execute(
                        text("SELECT AVG(chunk_count) FROM (SELECT doc_id, COUNT(*) as chunk_count FROM chunks GROUP BY doc_id) as chunk_stats")
                    )
                    avg_chunks = result.scalar() or 0
                    
            else:
                with self.session() as session:
                    from sqlalchemy import func, select
                    
                    # Total documents
                    result = session.execute(select(func.count(Document.doc_id)))
                    total_docs = result.scalar()
                    
                    # Documents by status
                    result = session.execute(
                        select(Document.status, func.count(Document.doc_id)).group_by(Document.status)
                    )
                    status_counts = dict(result.fetchall())
                    
                    # Total chunks
                    result = session.execute(select(func.count(Chunk.chunk_id)))
                    total_chunks = result.scalar()
                    
                    # Average chunks per document
                    from sqlalchemy import text
                    result = session.execute(
                        text("SELECT AVG(chunk_count) FROM (SELECT doc_id, COUNT(*) as chunk_count FROM chunks GROUP BY doc_id) as chunk_stats")
                    )
                    avg_chunks = result.scalar() or 0
            
            return {
                "total_documents": total_docs,
                "total_chunks": total_chunks,
                "avg_chunks_per_doc": round(avg_chunks, 2),
                "status_breakdown": status_counts
            }
            
        except Exception as e:
            self.logger.error(f"Error getting document stats: {e}")
            return {}
    
    async def _log_audit_event(
        self,
        doc_id: Optional[UUID],
        step: str,
        status: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
        processing_time_ms: Optional[int] = None
    ):
        """
        Log audit event to ingestion_log table.
        
        Args:
            doc_id: Document UUID (optional)
            step: Processing step
            status: Event status
            message: Event message
            metadata: Additional metadata (JSON)
            processing_time_ms: Processing time in milliseconds
        """
        try:
            log_entry = IngestionLog(
                doc_id=doc_id,
                step=step,
                status=status,
                message=message,
                metadata_json=json.dumps(metadata) if metadata else None,
                processing_time_ms=processing_time_ms,
                timestamp=datetime.utcnow()
            )
            
            if self.is_postgres:
                async with self.async_session() as session:
                    session.add(log_entry)
                    await session.commit()
            else:
                with self.session() as session:
                    session.add(log_entry)
                    session.commit()
                    
        except Exception as e:
            self.logger.warning(f"Failed to log audit event: {e}")
    
    async def get_audit_log(
        self,
        doc_id: Optional[UUID] = None,
        step: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get audit log entries.
        
        Args:
            doc_id: Filter by document ID
            step: Filter by processing step
            limit: Maximum number of entries
            
        Returns:
            List of audit log entries
        """
        try:
            if self.is_postgres:
                async with self.async_session() as session:
                    from sqlalchemy import select
                    stmt = select(IngestionLog)
                    
                    if doc_id:
                        stmt = stmt.where(IngestionLog.doc_id == doc_id)
                    if step:
                        stmt = stmt.where(IngestionLog.step == step)
                    
                    stmt = stmt.order_by(IngestionLog.timestamp.desc()).limit(limit)
                    result = await session.execute(stmt)
                    logs = result.scalars().all()
            else:
                with self.session() as session:
                    from sqlalchemy import select
                    stmt = select(IngestionLog)
                    
                    if doc_id:
                        stmt = stmt.where(IngestionLog.doc_id == doc_id)
                    if step:
                        stmt = stmt.where(IngestionLog.step == step)
                    
                    stmt = stmt.order_by(IngestionLog.timestamp.desc()).limit(limit)
                    result = session.execute(stmt)
                    logs = result.scalars().all()
            
            return [{
                'log_id': log.log_id,
                'doc_id': log.doc_id,
                'step': log.step,
                'status': log.status,
                'message': log.message,
                'metadata_json': log.metadata_json,
                'timestamp': log.timestamp.isoformat(),
                'processing_time_ms': log.processing_time_ms
            } for log in logs]
            
        except Exception as e:
            self.logger.error(f"Error getting audit log: {e}")
            return []
    
    async def cleanup_old_logs(self, days_to_keep: int = 30):
        """
        Clean up old audit log entries.
        
        Args:
            days_to_keep: Number of days to keep logs
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            
            if self.is_postgres:
                async with self.async_session() as session:
                    from sqlalchemy import delete
                    stmt = delete(IngestionLog).where(IngestionLog.timestamp < cutoff_date)
                    await session.execute(stmt)
                    await session.commit()
            else:
                with self.session() as session:
                    from sqlalchemy import delete
                    stmt = delete(IngestionLog).where(IngestionLog.timestamp < cutoff_date)
                    session.execute(stmt)
                    session.commit()
            
            self.logger.info(f"Cleaned up audit logs older than {days_to_keep} days")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old logs: {e}")
    
    async def close(self):
        """Close database connections."""
        try:
            if hasattr(self, 'engine'):
                if self.is_postgres:
                    await self.engine.dispose()
                else:
                    self.engine.dispose()
            self.logger.info("Database connections closed")
        except Exception as e:
            self.logger.error(f"Error closing database connections: {e}")


# Convenience functions for backward compatibility
async def create_metadata_store(db_url: str, **kwargs) -> MetadataStore:
    """
    Create and initialize metadata store.
    
    Args:
        db_url: Database connection URL
        **kwargs: Additional arguments for MetadataStore
        
    Returns:
        Initialized MetadataStore instance
    """
    return MetadataStore(db_url, **kwargs)


# Example usage
async def example_usage():
    """Example usage of the metadata store."""
    
    # Initialize metadata store
    store = await create_metadata_store("postgresql://user:pass@localhost/rag_db")
    
    # Example metadata from loader
    doc_metadata = {
        "filename": "Policy_TnC.pdf",
        "source_url": "https://example.com/policies/tnc.pdf",
        "extension": ".pdf",
        "file_size": 233812,
        "doc_type": "insurance",
        "upload_time": "2025-07-31T14:55:00Z",
        "doc_hash": "a3e81bcf1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8",
        "requires_ocr": False,
        "status": "uploaded"
    }
    
    # Save document metadata
    doc_id = await store.save_document_metadata(doc_metadata)
    print(f"Saved document with ID: {doc_id}")
    
    # Update status
    await store.mark_doc_status(doc_id, DocumentStatus.PARSED)
    
    # Save chunks
    chunks = [
        {
            "chunk_index": 0,
            "content": "This is the first chunk of the document...",
            "token_count": 150,
            "page_number": 1,
            "section_title": "Introduction"
        },
        {
            "chunk_index": 1,
            "content": "This is the second chunk of the document...",
            "token_count": 200,
            "page_number": 1,
            "section_title": "Terms and Conditions"
        }
    ]
    
    chunk_ids = await store.save_chunks(doc_id, chunks)
    print(f"Saved {len(chunk_ids)} chunks")
    
    # Get document stats
    stats = await store.get_document_stats()
    print(f"Document stats: {stats}")
    
    # Close connections
    await store.close()


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
