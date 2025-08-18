#!/usr/bin/env python3
"""
Simple test script to verify the multimodal query processing system.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.orchestrator import MultimodalOrchestrator, OrchestratorConfig
        from src.processors import TextProcessor, ImageProcessor, AudioProcessor, DocumentProcessor
        from src.processors.base_processor import ProcessingResult
        from src.utils.file_utils import get_file_info, is_supported_file_type
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_orchestrator_creation():
    """Test orchestrator creation."""
    print("\nTesting orchestrator creation...")
    
    try:
        from src.orchestrator import MultimodalOrchestrator, OrchestratorConfig
        
        # Test with default config
        orchestrator = MultimodalOrchestrator()
        print("✓ Orchestrator created with default config")
        
        # Test with custom config
        config = OrchestratorConfig(
            gpt_model="gpt-3.5-turbo",
            max_tokens=1000
        )
        orchestrator = MultimodalOrchestrator(config)
        print("✓ Orchestrator created with custom config")
        
        return True
    except Exception as e:
        print(f"✗ Orchestrator creation error: {e}")
        return False

def test_text_processing():
    """Test text processing."""
    print("\nTesting text processing...")
    
    try:
        from src.orchestrator import MultimodalOrchestrator
        
        orchestrator = MultimodalOrchestrator()
        
        # Test simple text query
        test_query = "What is artificial intelligence?"
        result = orchestrator.process_query(test_query)
        
        print(f"✓ Text processing successful")
        print(f"  Modality: {result.modality}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Content length: {len(result.content)} characters")
        
        return True
    except Exception as e:
        print(f"✗ Text processing error: {e}")
        return False

def test_processor_selection():
    """Test processor selection logic."""
    print("\nTesting processor selection...")
    
    try:
        from src.orchestrator import MultimodalOrchestrator
        from src.processors import TextProcessor, ImageProcessor, AudioProcessor, DocumentProcessor
        
        orchestrator = MultimodalOrchestrator()
        
        # Test text processor selection
        text_input = "Hello world"
        text_processor = orchestrator._select_processor(text_input)
        print(f"✓ Text processor selected: {text_processor.__class__.__name__}")
        
        # Test image processor selection (with bytes)
        image_bytes = b'\xff\xd8\xff\xe0'  # JPEG header
        image_processor = orchestrator._select_processor(image_bytes)
        print(f"✓ Image processor selected: {image_processor.__class__.__name__}")
        
        # Test document processor selection
        doc_path = Path("test.txt")
        doc_processor = orchestrator._select_processor(doc_path)
        print(f"✓ Document processor selected: {doc_processor.__class__.__name__}")
        
        return True
    except Exception as e:
        print(f"✗ Processor selection error: {e}")
        return False

def test_health_check():
    """Test health check functionality."""
    print("\nTesting health check...")
    
    try:
        from src.orchestrator import MultimodalOrchestrator
        
        orchestrator = MultimodalOrchestrator()
        health = orchestrator.health_check()
        
        print(f"✓ Health check successful")
        print(f"  Orchestrator status: {health['orchestrator']}")
        print(f"  API key configured: {health['api_key_configured']}")
        
        for processor, status in health['processors'].items():
            print(f"  {processor}: {status}")
        
        return True
    except Exception as e:
        print(f"✗ Health check error: {e}")
        return False

def test_utility_functions():
    """Test utility functions."""
    print("\nTesting utility functions...")
    
    try:
        from src.utils.file_utils import is_supported_file_type, get_file_category
        
        # Test file type detection
        test_files = [
            "image.jpg",
            "audio.wav",
            "document.pdf",
            "unknown.xyz"
        ]
        
        for file_path in test_files:
            supported = is_supported_file_type(file_path)
            category = get_file_category(file_path)
            print(f"  {file_path}: supported={supported}, category={category}")
        
        print("✓ Utility functions working")
        return True
    except Exception as e:
        print(f"✗ Utility functions error: {e}")
        return False

def main():
    """Run all tests."""
    print("Multimodal Query Processing System - Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_orchestrator_creation,
        test_text_processing,
        test_processor_selection,
        test_health_check,
        test_utility_functions
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
