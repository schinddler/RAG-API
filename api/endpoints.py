"""
FastAPI endpoints for RAG system.

This module provides the API layer for the intelligent query-retrieval system,
handling document processing, question answering, and structured JSON responses.
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import asyncio
from datetime import datetime

# Import RAG pipeline components
from ..ingestion.parser import parse_url
from ..ingestion.preprocessor import preprocess_text
from ..ingestion.chunkers import chunk_text
from ..schemas.models import get_embedding_model, get_reasoning_model
from ..retrieval.context_deduplicator import ContextDeduplicator, DeduplicationConfig
from ..prompt.prompt_builder import PromptBuilder, TaskType, ModelType, PromptConfig, ContextChunk, create_context_chunks_from_dicts
from ..services.cache_service import get_cache_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="HackRx6 RAG API",
    description="Intelligent Query-Retrieval System for Insurance, Legal, HR, and Compliance",
    version="1.0.0"
)

# Security
security = HTTPBearer()

# Configuration
API_TOKEN = "c742772b47bb55597517747abafcc3d472fa1c4403a1574461aa3f70ea2d9301"
MAX_QUESTIONS_PER_REQUEST = 10
MAX_DOCUMENT_SIZE_MB = 100
DEFAULT_TOP_K = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.7


class QuestionRequest(BaseModel):
    """Request model for question answering."""
    documents: str = Field(..., description="Blob URL of the document to process")
    questions: List[str] = Field(..., description="List of questions to answer", max_items=MAX_QUESTIONS_PER_REQUEST)
    
    @validator('questions')
    def validate_questions(cls, v):
        if not v:
            raise ValueError("At least one question is required")
        if len(v) > MAX_QUESTIONS_PER_REQUEST:
            raise ValueError(f"Maximum {MAX_QUESTIONS_PER_REQUEST} questions allowed per request")
        return v


class AnswerResponse(BaseModel):
    """Response model for individual answers."""
    question: str
    answer: str
    source_clauses: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    processing_time_ms: int
    tokens_used: Optional[int] = None


class HackRxResponse(BaseModel):
    """Response model for /hackrx/run endpoint."""
    answers: List[str]  # Changed to match actual response format
    performance: Dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> bool:
    """Verify API token."""
    if credentials.credentials != API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API token"
        )
    return True


class RAGService:
    """Service layer for RAG pipeline orchestration."""
    
    def __init__(self):
        """Initialize RAG service with all components."""
        self.embedding_model = get_embedding_model()
        self.reasoning_model = get_reasoning_model()
        self.cache_service = get_cache_service()
        self.prompt_builder = PromptBuilder(
            PromptConfig(
                task_type=TaskType.DOCUMENT_QA,
                model_type=ModelType.CLAUDE,
                max_tokens=6000
            )
        )
        self.deduplicator = ContextDeduplicator(
            DeduplicationConfig(
                similarity_threshold=0.85,
                max_chunks=8,
                preserve_high_score_chunks=3
            )
        )
    
    async def process_document(self, document_url: str) -> Dict[str, Any]:
        """
        Process document from blob URL.
        
        Args:
            document_url: Blob URL of the document
            
        Returns:
            Dictionary with parsed content and metadata
        """
        try:
            logger.info(f"Processing document from URL: {document_url}")
            
            # Parse document
            parsed_result = await parse_url(document_url)
            
            if not parsed_result or not parsed_result.get('text'):
                raise ValueError("No text content extracted from document")
            
            # Preprocess text
            raw_text = parsed_result['text']
            preprocessed_text = preprocess_text(raw_text)
            
            # Chunk text (keeping existing chunking logic)
            chunks = chunk_text(preprocessed_text)
            logger.info(f"Created {len(chunks)} chunks from document")
            
            if not chunks:
                logger.warning("No chunks created from document. Creating fallback chunks.")
                # Create fallback chunks from sentences
                sentences = preprocessed_text.split('. ')
                chunks = []
                for i, sentence in enumerate(sentences[:10]):  # Limit to first 10 sentences
                    if len(sentence.strip()) > 20:  # Only meaningful sentences
                        chunks.append({
                            'id': f'chunk_{i}',
                            'text': sentence.strip(),
                            'metadata': {'source': 'fallback', 'chunk_index': i}
                        })
                logger.info(f"Created {len(chunks)} fallback chunks")
            
            # Create embeddings for chunks
            chunk_texts = [chunk['text'] for chunk in chunks]
            if chunk_texts:
                embeddings = self.embedding_model.embed(chunk_texts)
            else:
                embeddings = []
                logger.error("No chunk texts available for embedding")
            
            # Store embeddings in FAISS (simplified - in production, use proper vector store)
            # For now, we'll use a simple in-memory approach
            chunk_embeddings = {}
            for i, chunk in enumerate(chunks):
                chunk_embeddings[chunk['id']] = {
                    'text': chunk['text'],
                    'embedding': embeddings[i],
                    'metadata': chunk.get('metadata', {})
                }
            
            return {
                'chunks': chunks,
                'embeddings': chunk_embeddings,
                'metadata': parsed_result.get('metadata', {}),
                'total_chunks': len(chunks)
            }
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to process document: {str(e)}"
            )
    
    async def retrieve_relevant_chunks(
        self, 
        question: str, 
        document_data: Dict[str, Any],
        top_k: int = DEFAULT_TOP_K
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a question.
        
        Args:
            question: User question
            document_data: Processed document data
            top_k: Number of top chunks to retrieve
            
        Returns:
            List of relevant chunks with scores
        """
        try:
            chunks = document_data['chunks']
            embeddings = document_data['embeddings']
            
            if not chunks or not embeddings:
                logger.warning("No chunks or embeddings available for retrieval")
                return []
            
            # Get question embedding
            question_embedding = self.embedding_model.embed([question])[0]
            
            # Calculate similarities with all chunks
            similarities = []
            
            for chunk in chunks:
                chunk_id = chunk['id']
                if chunk_id in embeddings and 'embedding' in embeddings[chunk_id]:
                    chunk_embedding = embeddings[chunk_id]['embedding']
                    similarity = self._cosine_similarity(question_embedding, chunk_embedding)
                    
                    similarities.append({
                        'chunk': chunk,
                        'similarity': similarity,
                        'text': chunk['text'],
                        'metadata': chunk.get('metadata', {})
                    })
                else:
                    logger.warning(f"Missing embedding for chunk {chunk_id}")
            
            # Sort by similarity and get top_k
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            top_chunks = similarities[:top_k]
            
            # Filter by similarity threshold
            relevant_chunks = [
                chunk for chunk in top_chunks 
                if chunk['similarity'] >= DEFAULT_SIMILARITY_THRESHOLD
            ]
            
            if not relevant_chunks:
                logger.warning(f"No relevant chunks found for question: {question}")
                return []
            
            return relevant_chunks
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            return []
    
    def _cosine_similarity(self, vec1, vec2) -> float:
        """Calculate cosine similarity between two vectors."""
        import numpy as np
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2)
    
    async def generate_answer(
        self, 
        question: str, 
        relevant_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate answer using LLM.
        
        Args:
            question: User question
            relevant_chunks: Retrieved relevant chunks
            
        Returns:
            Dictionary with answer and metadata
        """
        try:
            # Convert chunks to ContextChunk format
            context_chunks = []
            for chunk in relevant_chunks:
                context_chunk = ContextChunk(
                    text=chunk['text'],
                    source_id=chunk['metadata'].get('source_id', 'unknown'),
                    chunk_id=chunk['chunk']['id'],
                    score=chunk['similarity'],
                    metadata=chunk['metadata']
                )
                context_chunks.append(context_chunk)
            
            # Deduplicate chunks
            if len(context_chunks) > 1:
                context_chunks = self.deduplicator.deduplicate(context_chunks)
            
            # Build prompt
            prompt = self.prompt_builder.build_prompt(
                query=question,
                context_chunks=context_chunks,
                task_type=TaskType.DOCUMENT_QA
            )
            
            # Generate answer using LLM
            llm_response = self.reasoning_model.generate_answer(
                query=question,
                context_chunks=context_chunks
            )
            
            # Extract source clauses
            source_clauses = [chunk.text for chunk in context_chunks[:3]]  # Top 3 sources
            
            return {
                'answer': llm_response['answer'],
                'confidence': llm_response.get('confidence', 0.8),
                'source_clauses': source_clauses,
                'raw_response': llm_response
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                'answer': f"Unable to generate answer due to error: {str(e)}",
                'confidence': 0.0,
                'source_clauses': [],
                'raw_response': {}
            }


# Initialize RAG service
rag_service = RAGService()


@app.post(
    "/api/v1/hackrx/run",
    response_model=HackRxResponse,
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def hackrx_run(
    request: QuestionRequest,
    token_valid: bool = Depends(verify_token)
) -> HackRxResponse:
    """
    Process documents and answer questions using RAG pipeline.
    
    Args:
        request: Question request with document URL and questions
        token_valid: Verified API token
        
    Returns:
        Structured response with answers and metadata
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing request with {len(request.questions)} questions")
        
        # Process document
        document_data = await rag_service.process_document(request.documents)
        logger.info(f"Document processed: {document_data['total_chunks']} chunks extracted")
        
        # Process each question
        answers = []
        step_times = {}
        
        # Step 1: Document processing
        step1_start = time.time()
        logger.info(f"Processing document: {document_data['total_chunks']} chunks extracted")
        step_times["step1_processing"] = f"{time.time() - step1_start:.2f}s"
        
        for i, question in enumerate(request.questions):
            question_start_time = time.time()
            
            logger.info(f"Processing question {i+1}/{len(request.questions)}: {question[:50]}...")
            
            # Step 2: Retrieve relevant chunks
            step2_start = time.time()
            relevant_chunks = await rag_service.retrieve_relevant_chunks(
                question, document_data
            )
            step_times["step2_retrieval"] = f"{time.time() - step2_start:.2f}s"
            
            # Step 3: Generate answer
            step3_start = time.time()
            if not relevant_chunks:
                answer = "Not mentioned in the policy. The context provided does not contain information about this topic."
            else:
                answer_data = await rag_service.generate_answer(question, relevant_chunks)
                answer = answer_data.get('answer', "Unable to generate answer.")
            
            step_times["step3_generation"] = f"{time.time() - step3_start:.2f}s"
            answers.append(answer)
        
        total_time = time.time() - start_time
        total_tracked_time = sum(float(v[:-1]) for v in step_times.values())
        
        performance = {
            "total_time": f"{total_time:.2f}s",
            "total_tracked_time": f"{total_tracked_time:.2f}s",
            "untracked_time": f"{total_time - total_tracked_time:.2f}s",
            "step_times": step_times
        }
        
        logger.info(f"Request completed in {total_time:.2f}s")
        
        return HackRxResponse(
            answers=answers,
            performance=performance
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in hackrx_run: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": "1.0.0"
    }


@app.get("/api/v1/info")
async def api_info():
    """API information endpoint."""
    return {
        "name": "HackRx6 RAG API",
        "description": "Intelligent Query-Retrieval System",
        "version": "1.0.0",
        "endpoints": {
            "POST /api/v1/hackrx/run": "Process documents and answer questions",
            "GET /api/v1/health": "Health check",
            "GET /api/v1/info": "API information"
        },
        "max_questions_per_request": MAX_QUESTIONS_PER_REQUEST,
        "max_document_size_mb": MAX_DOCUMENT_SIZE_MB,
        "default_top_k": DEFAULT_TOP_K,
        "default_similarity_threshold": DEFAULT_SIMILARITY_THRESHOLD
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
