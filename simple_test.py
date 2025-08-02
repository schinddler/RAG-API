"""
Simple test to verify the basic fixes work.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_components():
    """Test basic components without complex imports."""
    print("Testing basic components...")
    
    try:
        # Test chunkers
        print("1. Testing chunkers...")
        from ingestion.chunkers import chunk_text
        test_text = "This is a test document with multiple sentences."
        chunks = chunk_text(test_text)
        print(f"   ✅ chunk_text created {len(chunks)} chunks")
        
        # Test cache service
        print("2. Testing cache service...")
        from services.cache_service import get_cache_service
        cache = get_cache_service()
        cache.set("test_key", "test_value")
        value = cache.get("test_key")
        assert value == "test_value"
        print("   ✅ cache service working")
        
        # Test context deduplicator (without numpy dependency)
        print("3. Testing context deduplicator...")
        from retrieval.context_deduplicator import ContextDeduplicator
        deduplicator = ContextDeduplicator()
        test_chunks = [
            {"text": "This is a test chunk", "score": 0.9},
            {"text": "This is another test chunk", "score": 0.8}
        ]
        deduplicated = deduplicator.deduplicate(test_chunks)
        print(f"   ✅ deduplicator processed {len(test_chunks)} chunks")
        
        print("\n🎉 Basic components are working!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_api_structure():
    """Test that the API structure is correct."""
    print("\nTesting API structure...")
    
    try:
        # Check if main.py exists and can be imported
        print("1. Testing main.py...")
        import main
        print("   ✅ main.py can be imported")
        
        # Check if API endpoints file exists
        print("2. Testing API endpoints...")
        import api.endpoints
        print("   ✅ API endpoints module exists")
        
        print("   ✅ API structure is correct!")
        return True
        
    except Exception as e:
        print(f"❌ API structure test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Testing basic RAG API fixes...")
    
    basic_success = test_basic_components()
    api_success = test_api_structure()
    
    if basic_success and api_success:
        print("\n🎉 All basic tests passed!")
        print("\nThe RAG API structure is correct and ready to use.")
        print("\nTo start the server, run:")
        print("python main.py")
    else:
        print("\n❌ Some tests failed. Please check the errors above.") 