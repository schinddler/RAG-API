"""
Simple chunking interface for RAG API.
This module provides a simple chunk_text function that wraps the more complex chunker functionality.
"""

from typing import List, Dict, Any
from .chunker import create_chunker, ChunkConfig


def chunk_text(text: str, config: ChunkConfig = None) -> List[Dict[str, Any]]:
    """
    Simple interface to chunk text into semantic chunks.
    
    Args:
        text: Text to chunk
        config: Optional chunking configuration
        
    Returns:
        List of chunk dictionaries with 'id', 'text', and 'metadata' keys
    """
    if config is None:
        config = ChunkConfig()
    
    # Create chunker
    chunker = create_chunker(
        split_strategy=config.split_strategy.value,
        max_tokens=config.max_tokens,
        min_tokens=config.min_tokens,
        overlap_tokens=config.overlap_tokens,
        enable_dedup=config.enable_dedup,
        preclean_ocr=config.preclean_ocr
    )
    
    # Chunk the text
    result = chunker.chunk_document(text, source="document")
    
    # Convert to expected format
    chunks = []
    for chunk in result.chunks:
        chunk_dict = {
            'id': chunk.metadata.chunk_id,
            'text': chunk.content,
            'metadata': {
                'source': chunk.metadata.source,
                'chunk_index': chunk.metadata.chunk_index,
                'char_start': chunk.metadata.char_start,
                'char_end': chunk.metadata.char_end,
                'section_title': chunk.metadata.section_title,
                'is_suspicious': chunk.metadata.is_suspicious,
                'quality_score': chunk.metadata.quality_score,
                'token_count': chunk.metadata.token_count,
                'chunk_id': chunk.metadata.chunk_id
            }
        }
        chunks.append(chunk_dict)
    
    return chunks 