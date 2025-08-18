#!/usr/bin/env python3
"""
Basic test script that doesn't require API keys.
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
    """Test orchestrator creation without API key."""
    print("\nTesting orchestrator creation...")
    
    try:
        from src.orchestrator import MultimodalOrchestrator, OrchestratorConfig
        
        # Test with default config (no API key)
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

def test_processor_selection():
    """Test processor selection logic without API calls."""
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

def test_processor_capabilities():
    """Test processor capabilities without API calls."""
    print("\nTesting processor capabilities...")
    
    try:
        from src.processors import TextProcessor, ImageProcessor, AudioProcessor, DocumentProcessor
        
        # Test text processor
        text_processor = TextProcessor()
        print(f"✓ TextProcessor created")
        print(f"  Can process text: {text_processor.can_process('Hello world')}")
        print(f"  Can process bytes: {text_processor.can_process(b'Hello world')}")
        
        # Test image processor
        image_processor = ImageProcessor()
        print(f"✓ ImageProcessor created")
        image_bytes = b'\xff\xd8\xff\xe0'
        print(f"  Can process image bytes: {image_processor.can_process(image_bytes)}")
        print(f"  Can process image path: {image_processor.can_process(Path('image.jpg'))}")
        
        # Test audio processor
        audio_processor = AudioProcessor()
        print(f"✓ AudioProcessor created")
        print(f"  Can process audio path: {audio_processor.can_process(Path('audio.wav'))}")
        
        # Test document processor
        doc_processor = DocumentProcessor()
        print(f"✓ DocumentProcessor created")
        print(f"  Can process PDF: {doc_processor.can_process(Path('document.pdf'))}")
        print(f"  Can process DOCX: {doc_processor.can_process(Path('document.docx'))}")
        
        return True
    except Exception as e:
        print(f"✗ Processor capabilities error: {e}")
        return False

def main():
    """Run all basic tests."""
    print("Multimodal Query Processing System - Basic Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_orchestrator_creation,
        test_processor_selection,
        test_health_check,
        test_utility_functions,
        test_processor_capabilities
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
    print(f"Basic Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic tests passed! The system is working correctly.")
        print("\nTo use the full functionality:")
        print("1. Get an OpenAI API key from: https://platform.openai.com/api-keys")
        print("2. Create a .env file with: OPENAI_API_KEY=your_api_key_here")
        print("3. Run: python test_system.py")
        return 0
    else:
        print("❌ Some basic tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
