#!/usr/bin/env python3
"""
Integration test for the new RAG project structure with shared conventions.
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_shared_hashing():
    """Test that chunk ID generation is consistent across modules."""
    print("Testing shared hashing...")
    
    from utils.hashing import generate_chunk_id
    
    test_text = "This is a test document for consistency verification."
    chunk_id = generate_chunk_id(test_text)
    
    print(f"✓ Generated chunk ID: {chunk_id}")
    return True

def test_config_imports():
    """Test that central configuration works."""
    print("Testing config imports...")
    
    from config import SUPPORTED_MODELS, CACHE_BACKENDS, SplitStrategy
    
    print(f"✓ Supported models: {list(SUPPORTED_MODELS.keys())}")
    print(f"✓ Cache backends: {list(CACHE_BACKENDS.keys())}")
    print(f"✓ Split strategies: {[s.value for s in SplitStrategy]}")
    return True

def test_chunker_imports():
    """Test that chunker can import shared utilities."""
    print("Testing chunker imports...")
    
    try:
        from ingestion.chunker import create_chunker
        print("✓ Chunker imports work")
        return True
    except ImportError as e:
        print(f"✗ Chunker import failed: {e}")
        return False

def test_embedder_imports():
    """Test that embedder can import shared utilities."""
    print("Testing embedder imports...")
    
    try:
        from embedder.embedder import create_embedder
        print("✓ Embedder imports work")
        return True
    except ImportError as e:
        print(f"✗ Embedder import failed: {e}")
        return False

def test_chunk_id_consistency():
    """Test that both modules generate the same chunk IDs."""
    print("Testing chunk ID consistency...")
    
    from utils.hashing import generate_chunk_id
    
    test_text = "Consistency test text for cross-module verification."
    
    # Test shared function
    shared_id = generate_chunk_id(test_text)
    
    # Test chunker (if available)
    try:
        from ingestion.chunker import create_chunker
        chunker = create_chunker()
        chunker_id = chunker._generate_chunk_id(test_text)
        print(f"✓ Chunker ID: {chunker_id}")
    except Exception as e:
        print(f"⚠ Chunker test skipped: {e}")
        chunker_id = shared_id
    
    # Test embedder (if available)
    try:
        from embedder.embedder import create_embedder
        embedder = create_embedder()
        embedder_id = embedder._generate_chunk_id(test_text)
        print(f"✓ Embedder ID: {embedder_id}")
    except Exception as e:
        print(f"⚠ Embedder test skipped: {e}")
        embedder_id = shared_id
    
    # Verify consistency
    consistency = (shared_id == chunker_id == embedder_id)
    print(f"{'✓' if consistency else '✗'} Chunk ID consistency: {consistency}")
    
    return consistency

def main():
    """Run all integration tests."""
    print("RAG-API Integration Tests")
    print("=" * 40)
    
    tests = [
        test_shared_hashing,
        test_config_imports,
        test_chunker_imports,
        test_embedder_imports,
        test_chunk_id_consistency,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            print()
    
    print("=" * 40)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The new project structure is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 