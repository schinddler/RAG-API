#!/usr/bin/env python3
"""
Simple test script to verify the fixes work.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_imports():
    """Test basic imports work."""
    try:
        # Test basic imports
        from ingestion.parser import parse_url
        from ingestion.preprocessor import preprocess_text
        from ingestion.chunkers import chunk_text
        from schemas import get_embedding_model, get_reasoning_model
        print("✅ Basic imports successful")
        return True
    except Exception as e:
        print(f"❌ Basic imports failed: {e}")
        return False

def test_embedding_model():
    """Test embedding model initialization."""
    try:
        from schemas import get_embedding_model
        model = get_embedding_model()
        print("✅ Embedding model initialized")
        return True
    except Exception as e:
        print(f"❌ Embedding model failed: {e}")
        return False

def test_reasoning_model():
    """Test reasoning model initialization."""
    try:
        from schemas import get_reasoning_model
        model = get_reasoning_model()
        print("✅ Reasoning model initialized")
        return True
    except Exception as e:
        print(f"❌ Reasoning model failed: {e}")
        return False

def test_chunking():
    """Test chunking functionality."""
    try:
        from ingestion.chunkers import chunk_text
        test_text = "This is a test document. It contains multiple sentences. Each sentence should be processed correctly."
        chunks = chunk_text(test_text)
        print(f"✅ Chunking successful: {len(chunks)} chunks created")
        return True
    except Exception as e:
        print(f"❌ Chunking failed: {e}")
        return False

def test_preprocessing():
    """Test preprocessing functionality."""
    try:
        from ingestion.preprocessor import preprocess_text
        test_text = "This is a test document with some formatting issues.   Multiple spaces.  Bullet points: • Item 1 • Item 2"
        processed = preprocess_text(test_text)
        print("✅ Preprocessing successful")
        return True
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Running simple tests...")
    
    tests = [
        test_basic_imports,
        test_embedding_model,
        test_reasoning_model,
        test_chunking,
        test_preprocessing
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The fixes are working.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main() 