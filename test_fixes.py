"""
Test script to verify that all import issues have been fixed.
"""

import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all the imports that were causing issues."""
    print("Testing imports...")
    
    try:
        # Test chunkers import
        print("1. Testing chunkers import...")
        from ingestion.chunkers import chunk_text
        print("   ✅ chunkers import successful")
        
        # Test models import
        print("2. Testing models import...")
        from schemas.models import get_embedding_model, get_reasoning_model
        print("   ✅ models import successful")
        
        # Test context_deduplicator import
        print("3. Testing context_deduplicator import...")
        from retrieval.context_deduplicator import ContextDeduplicator, DeduplicationConfig
        print("   ✅ context_deduplicator import successful")
        
        # Test cache_service import
        print("4. Testing cache_service import...")
        from services.cache_service import get_cache_service
        print("   ✅ cache_service import successful")
        
        # Test prompt_builder import
        print("5. Testing prompt_builder import...")
        from prompt.prompt_builder import PromptBuilder, TaskType, ModelType, PromptConfig, ContextChunk
        print("   ✅ prompt_builder import successful")
        
        # Test API endpoints import
        print("6. Testing API endpoints import...")
        from api.endpoints import app, RAGService
        print("   ✅ API endpoints import successful")
        
        print("\n🎉 All imports successful! The fixes are working.")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of the fixed components."""
    print("\nTesting basic functionality...")
    
    try:
        # Test chunk_text function
        print("1. Testing chunk_text function...")
        from ingestion.chunkers import chunk_text
        test_text = "This is a test document. It has multiple sentences. We want to chunk it properly."
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
        
        # Test context deduplicator
        print("3. Testing context deduplicator...")
        from retrieval.context_deduplicator import ContextDeduplicator, DeduplicationConfig
        deduplicator = ContextDeduplicator()
        test_chunks = [
            {"text": "This is a test chunk", "score": 0.9},
            {"text": "This is another test chunk", "score": 0.8},
            {"text": "This is a test chunk", "score": 0.7}  # Duplicate
        ]
        deduplicated = deduplicator.deduplicate(test_chunks)
        print(f"   ✅ deduplicator reduced {len(test_chunks)} to {len(deduplicated)} chunks")
        
        # Test models
        print("4. Testing models...")
        from schemas.models import get_embedding_model
        embedding_model = get_embedding_model()
        print("   ✅ embedding model created")
        
        print("\n🎉 All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

def test_api_endpoint():
    """Test that the API endpoint can be imported and initialized."""
    print("\nTesting API endpoint...")
    
    try:
        from api.endpoints import app, RAGService
        
        # Test that RAGService can be instantiated
        rag_service = RAGService()
        print("   ✅ RAGService instantiated successfully")
        
        # Test that FastAPI app exists
        assert app is not None
        print("   ✅ FastAPI app exists")
        
        print("   ✅ API endpoint test passed!")
        return True
        
    except Exception as e:
        print(f"❌ API endpoint test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Testing RAG API fixes...")
    
    # Run all tests
    import_success = test_imports()
    functionality_success = test_basic_functionality()
    api_success = test_api_endpoint()
    
    if import_success and functionality_success and api_success:
        print("\n🎉 All tests passed! The RAG API is ready to use.")
        print("\nTo start the server, run:")
        print("python main.py")
    else:
        print("\n❌ Some tests failed. Please check the errors above.") 