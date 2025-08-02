"""
Test script for the Document Loader

This script demonstrates the various capabilities of the DocumentLoader class
including URL downloads, file processing, ZIP extraction, and metadata capture.
"""

import asyncio
import logging
from pathlib import Path
from loader import DocumentLoader, load_document, load_document_async

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_local_file_loading():
    """Test loading a local file."""
    print("\n=== Testing Local File Loading ===")
    
    # Create a test text file
    test_file = Path("test_document.txt")
    test_content = "This is a test document for the RAG loader. It contains some sample text."
    
    with open(test_file, "w") as f:
        f.write(test_content)
    
    try:
        loader = DocumentLoader(cache_dir="./test_cache")
        file_path, metadata = loader.load_from_file(test_file)
        
        print(f"✅ Successfully loaded local file")
        print(f"   File path: {file_path}")
        print(f"   Filename: {metadata['filename']}")
        print(f"   File size: {metadata['file_size']} bytes")
        print(f"   Document hash: {metadata['doc_hash']}")
        print(f"   Requires OCR: {metadata['requires_ocr']}")
        
        # Test duplicate detection
        is_duplicate = loader.check_duplicate(metadata['doc_hash'])
        print(f"   Is duplicate: {is_duplicate}")
        
    except Exception as e:
        print(f"❌ Error loading local file: {e}")
    
    finally:
        # Cleanup
        test_file.unlink(missing_ok=True)

def test_url_download():
    """Test downloading a file from URL."""
    print("\n=== Testing URL Download ===")
    
    # Test with a simple text file URL (using a public sample)
    test_url = "https://raw.githubusercontent.com/microsoft/vscode/main/README.md"
    
    try:
        loader = DocumentLoader(cache_dir="./test_cache")
        file_path, metadata = loader.load_from_url(test_url)
        
        print(f"✅ Successfully downloaded file from URL")
        print(f"   File path: {file_path}")
        print(f"   Filename: {metadata['filename']}")
        print(f"   Source URL: {metadata['source_url']}")
        print(f"   File size: {metadata['file_size']} bytes")
        print(f"   Document hash: {metadata['doc_hash']}")
        print(f"   Content type: {metadata.get('content_type', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Error downloading from URL: {e}")

async def test_async_loading():
    """Test async document loading."""
    print("\n=== Testing Async Loading ===")
    
    test_url = "https://raw.githubusercontent.com/microsoft/vscode/main/README.md"
    
    try:
        loader = DocumentLoader(cache_dir="./test_cache", enable_async=True)
        file_path, metadata = await loader.load_from_url_async(test_url)
        
        print(f"✅ Successfully downloaded file asynchronously")
        print(f"   File path: {file_path}")
        print(f"   Filename: {metadata['filename']}")
        print(f"   File size: {metadata['file_size']} bytes")
        
    except Exception as e:
        print(f"❌ Error in async loading: {e}")

def test_zip_extraction():
    """Test ZIP file extraction."""
    print("\n=== Testing ZIP Extraction ===")
    
    import zipfile
    import tempfile
    
    # Create a test ZIP file with multiple documents
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_zip:
        zip_path = tmp_zip.name
    
    try:
        # Create a ZIP file with test documents
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.writestr("document1.txt", "This is the first document in the ZIP.")
            zipf.writestr("document2.txt", "This is the second document in the ZIP.")
            zipf.writestr("nested/document3.txt", "This is a nested document.")
        
        # Test loading the ZIP file
        loader = DocumentLoader(cache_dir="./test_cache")
        extracted_files = loader.load_from_file(zip_path)
        
        print(f"✅ Successfully extracted ZIP file")
        print(f"   Extracted {len(extracted_files)} files:")
        
        for file_path, metadata in extracted_files:
            print(f"     - {metadata['filename']} ({metadata['file_size']} bytes)")
            print(f"       Hash: {metadata['doc_hash'][:16]}...")
            print(f"       Path: {metadata.get('extracted_path', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Error extracting ZIP file: {e}")
    
    finally:
        # Cleanup
        Path(zip_path).unlink(missing_ok=True)

def test_security_validation():
    """Test security validation features."""
    print("\n=== Testing Security Validation ===")
    
    loader = DocumentLoader(cache_dir="./test_cache")
    
    # Test dangerous file extension
    try:
        loader._validate_file_extension("malicious.exe")
        print("❌ Should have rejected .exe file")
    except Exception as e:
        print(f"✅ Correctly rejected dangerous file: {e}")
    
    # Test unsupported file extension
    try:
        loader._validate_file_extension("unknown.xyz")
        print("❌ Should have rejected unknown file type")
    except Exception as e:
        print(f"✅ Correctly rejected unknown file type: {e}")
    
    # Test filename sanitization
    dangerous_filename = "file<with>dangerous:chars.txt"
    sanitized = loader._sanitize_filename(dangerous_filename)
    print(f"✅ Filename sanitization: '{dangerous_filename}' -> '{sanitized}'")

def test_convenience_functions():
    """Test convenience functions."""
    print("\n=== Testing Convenience Functions ===")
    
    # Create a test file
    test_file = Path("convenience_test.txt")
    with open(test_file, "w") as f:
        f.write("Test content for convenience function.")
    
    try:
        # Test sync convenience function
        file_path, metadata = load_document(str(test_file))
        print(f"✅ Convenience function worked: {metadata['filename']}")
        
    except Exception as e:
        print(f"❌ Error with convenience function: {e}")
    
    finally:
        test_file.unlink(missing_ok=True)

async def test_async_convenience():
    """Test async convenience function."""
    print("\n=== Testing Async Convenience Function ===")
    
    test_url = "https://raw.githubusercontent.com/microsoft/vscode/main/CONTRIBUTING.md"
    
    try:
        file_path, metadata = await load_document_async(test_url)
        print(f"✅ Async convenience function worked: {metadata['filename']}")
        
    except Exception as e:
        print(f"❌ Error with async convenience function: {e}")

def main():
    """Run all tests."""
    print("🚀 Starting Document Loader Tests")
    print("=" * 50)
    
    # Run synchronous tests
    test_local_file_loading()
    test_url_download()
    test_zip_extraction()
    test_security_validation()
    test_convenience_functions()
    
    # Run async tests
    async def run_async_tests():
        await test_async_loading()
        await test_async_convenience()
    
    asyncio.run(run_async_tests())
    
    print("\n" + "=" * 50)
    print("✅ All tests completed!")
    
    # Cleanup test cache
    import shutil
    test_cache = Path("./test_cache")
    if test_cache.exists():
        shutil.rmtree(test_cache)
        print("🧹 Cleaned up test cache directory")

if __name__ == "__main__":
    main() 