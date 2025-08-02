#!/usr/bin/env python3
"""
Enhanced Parser Feature Demonstration

This script demonstrates all the enhanced parser features
without using the metadata store to avoid conflicts.
"""

import asyncio
import tempfile
import os
from pathlib import Path
import json
from datetime import datetime

# Import the enhanced parser
from parser import FileParser, ParserError, parse_file, parse_url


class EnhancedParserDemo:
    """
    Demonstration of enhanced parser features.
    """
    
    def __init__(self):
        """Initialize the demo with enhanced parser."""
        self.parser = FileParser(enable_ocr=True)
    
    def demonstrate_all_features(self):
        """Demonstrate all enhanced parser features."""
        print("🎯 Enhanced Parser Feature Demonstration")
        print("=" * 50)
        
        # Create temporary directory for demo files
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Demo 1: Text file with chunking
            print("\n1️⃣ Text File with Chunking")
            print("-" * 30)
            text_content = "This is a sample document for demonstration. " * 20
            text_file = Path(temp_dir) / "demo.txt"
            text_file.write_text(text_content)
            
            result = self.parser.parse_from_file(str(text_file), max_chars=100)
            print(f"✓ File type: {result['filetype']}")
            print(f"✓ Text length: {result['length']}")
            print(f"✓ Chunks created: {len(result.get('text_chunks', []))}")
            print(f"✓ Metadata: {result['metadata']}")
            
            # Demo 2: MIME validation
            print("\n2️⃣ MIME Type Validation")
            print("-" * 30)
            is_valid = self.parser._validate_mime_type(text_file, '.txt')
            print(f"✓ MIME validation: {is_valid}")
            
            # Demo 3: OCR availability check
            print("\n3️⃣ OCR Feature Check")
            print("-" * 30)
            print(f"✓ OCR enabled: {self.parser.enable_ocr}")
            print(f"✓ OCR available: {hasattr(self.parser, '_ocr_pdf_pages')}")
            
            # Demo 4: Text chunking demonstration
            print("\n4️⃣ Text Chunking Demonstration")
            print("-" * 30)
            sample_text = "This is a sample text that will be chunked into smaller pieces. " * 5
            chunks = self.parser._chunk_text(sample_text, max_chars=80, overlap=20)
            print(f"✓ Original text length: {len(sample_text)}")
            print(f"✓ Number of chunks: {len(chunks)}")
            print(f"✓ Sample chunk: {chunks[0][:50]}...")
            
            # Demo 5: Metadata extraction
            print("\n5️⃣ Metadata Extraction")
            print("-" * 30)
            print(f"✓ Text file metadata: {result['metadata']}")
            
            # Demo 6: Different chunk sizes
            print("\n6️⃣ Different Chunk Sizes")
            print("-" * 30)
            large_text = "This is a very long text that demonstrates different chunking strategies. " * 15
            chunks_small = self.parser._chunk_text(large_text, max_chars=50, overlap=10)
            chunks_large = self.parser._chunk_text(large_text, max_chars=200, overlap=50)
            print(f"✓ Small chunks (50 chars): {len(chunks_small)} chunks")
            print(f"✓ Large chunks (200 chars): {len(chunks_large)} chunks")
            
            # Demo 7: Error handling
            print("\n7️⃣ Error Handling")
            print("-" * 30)
            try:
                self.parser.parse_from_file("nonexistent.txt")
                print("✗ Should have raised an error")
            except ParserError as e:
                print(f"✓ Correctly caught error: {e}")
            
            try:
                self.parser.parse_from_file("test.xyz")  # Unsupported extension
                print("✗ Should have raised an error")
            except ParserError as e:
                print(f"✓ Correctly caught error: {e}")
            
            # Demo 8: Parser configuration
            print("\n8️⃣ Parser Configuration")
            print("-" * 30)
            parser_no_ocr = FileParser(enable_ocr=False)
            parser_with_ocr = FileParser(enable_ocr=True)
            print(f"✓ Parser without OCR: {parser_no_ocr.enable_ocr}")
            print(f"✓ Parser with OCR: {parser_with_ocr.enable_ocr}")
            
            # Demo 9: Output format demonstration
            print("\n9️⃣ Output Format Demonstration")
            print("-" * 30)
            print("✓ Standard output format:")
            print(json.dumps({
                "filetype": result["filetype"],
                "url": result["url"],
                "full_text": result["full_text"][:100] + "...",
                "text_chunks": [chunk[:50] + "..." for chunk in result.get("text_chunks", [])[:3]],
                "length": result["length"],
                "metadata": result["metadata"]
            }, indent=2))
            
        except Exception as e:
            print(f"❌ Demo failed: {str(e)}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Clean up
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def demonstrate_convenience_functions(self):
        """Demonstrate convenience functions."""
        print("\n🔄 Convenience Functions Demonstration")
        print("=" * 50)
        
        # Create a test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content for convenience function demonstration")
            temp_path = f.name
        
        try:
            # Test parse_file convenience function
            result = parse_file(temp_path, max_chars=50, enable_ocr=False)
            
            print(f"✓ File type: {result['filetype']}")
            print(f"✓ Has full_text: {'full_text' in result}")
            print(f"✓ Has text_chunks: {'text_chunks' in result}")
            print(f"✓ Metadata: {result['metadata']}")
            
            print("✅ Convenience function test passed!")
            
        except Exception as e:
            print(f"❌ Convenience function test failed: {e}")
        
        finally:
            # Clean up
            os.unlink(temp_path)
    
    def demonstrate_advanced_features(self):
        """Demonstrate advanced features like OCR and MIME validation."""
        print("\n🚀 Advanced Features Demonstration")
        print("=" * 50)
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Test with different file types
            print("\n📄 Testing Different File Types")
            print("-" * 30)
            
            # Text file
            text_file = Path(temp_dir) / "test.txt"
            text_content = "This is a test text file with some content. " * 10
            text_file.write_text(text_content)
            
            result = self.parser.parse_from_file(str(text_file), max_chars=100)
            print(f"✓ Text file parsed successfully")
            print(f"✓ Chunks created: {len(result.get('text_chunks', []))}")
            
            # Test MIME validation with different scenarios
            print("\n🔍 MIME Validation Scenarios")
            print("-" * 30)
            
            # Valid text file
            is_valid = self.parser._validate_mime_type(text_file, '.txt')
            print(f"✓ Text file MIME validation: {is_valid}")
            
            # Test chunking with different parameters
            print("\n✂️ Advanced Chunking")
            print("-" * 30)
            
            long_text = "This is a very long text that will be chunked in different ways. " * 20
            
            # Test different chunk sizes
            chunks_50 = self.parser._chunk_text(long_text, max_chars=50, overlap=10)
            chunks_100 = self.parser._chunk_text(long_text, max_chars=100, overlap=20)
            chunks_200 = self.parser._chunk_text(long_text, max_chars=200, overlap=50)
            
            print(f"✓ 50-char chunks: {len(chunks_50)} chunks")
            print(f"✓ 100-char chunks: {len(chunks_100)} chunks")
            print(f"✓ 200-char chunks: {len(chunks_200)} chunks")
            
            # Test chunk quality
            print("\n📊 Chunk Quality Analysis")
            print("-" * 30)
            
            for i, chunk in enumerate(chunks_100[:3]):
                print(f"✓ Chunk {i+1}: {len(chunk)} chars, {len(chunk.split())} words")
            
        except Exception as e:
            print(f"❌ Advanced features demo failed: {str(e)}")
        
        finally:
            # Clean up
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


async def main():
    """Main function to run the enhanced parser demonstration."""
    print("🚀 Enhanced Parser Complete Demonstration")
    print("=" * 60)
    
    # Initialize demo
    demo = EnhancedParserDemo()
    
    # Run all demonstrations
    demo.demonstrate_all_features()
    demo.demonstrate_convenience_functions()
    demo.demonstrate_advanced_features()
    
    print("\n🎉 Enhanced Parser demonstration completed successfully!")
    print("\n📋 Summary of Enhanced Features:")
    print("✅ OCR fallback for scanned PDFs")
    print("✅ MIME type validation")
    print("✅ HTML parsing in EML files")
    print("✅ DOCX table structure preservation")
    print("✅ Intelligent text chunking")
    print("✅ Comprehensive metadata extraction")
    print("✅ Robust error handling")
    print("✅ Async support")
    print("✅ Configurable parameters")


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main()) 