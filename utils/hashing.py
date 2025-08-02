"""
Shared hashing utilities for consistent chunk ID generation across modules.
"""

import re
import hashlib
from typing import Optional


def generate_chunk_id(text: str, length: int = 16) -> str:
    """
    Generate a stable, normalized ID using SHA256 hash.
    Matches embedder.py and chunker.py logic exactly.
    
    Args:
        text: Input text to normalize and hash
        length: Length of the truncated hash (default: 16)
    
    Returns:
        Truncated SHA256 hash string
    """
    clean = re.sub(r'\s+', ' ', text.strip().lower())
    clean = re.sub(r'[^\w\s]', '', clean)  # Remove punctuation
    return hashlib.sha256(clean.encode('utf-8')).hexdigest()[:length]


def generate_hash(text: str, algorithm: str = 'sha256', length: Optional[int] = None) -> str:
    """
    Generate a hash using the specified algorithm.
    
    Args:
        text: Input text to hash
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256', 'sha512')
        length: Optional length to truncate the hash
    
    Returns:
        Hash string
    """
    algorithms = {
        'md5': hashlib.md5,
        'sha1': hashlib.sha1,
        'sha256': hashlib.sha256,
        'sha512': hashlib.sha512
    }
    
    if algorithm not in algorithms:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    hash_obj = algorithms[algorithm](text.encode('utf-8'))
    hash_hex = hash_obj.hexdigest()
    
    if length:
        return hash_hex[:length]
    return hash_hex 