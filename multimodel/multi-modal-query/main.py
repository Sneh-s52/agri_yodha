#!/usr/bin/env python3
"""
Main entry point for multimodal query processing.
Provides a simple interface for processing various types of queries.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.orchestrator import MultimodalOrchestrator, OrchestratorConfig

def setup_environment():
    """Setup environment variables and configuration."""
    # Load environment variables from .env file if it exists
    env_file = Path(__file__).parent / ".env"
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file)  # This will work even if file doesn't exist
        print(f"Looking for .env file at: {env_file}")
        if env_file.exists():
            print("✓ .env file found and loaded")
        else:
            print("⚠️  .env file not found, using environment variables")
    except ImportError:
        print("⚠️  python-dotenv not available, using environment variables only")

def create_orchestrator(api_key: Optional[str] = None, config: Optional[dict] = None) -> MultimodalOrchestrator:
    """Create and configure the orchestrator."""
    # Use provided API key or get from environment
    api_key = api_key or os.getenv('OPENAI_API_KEY')
    
    # Create configuration with modern defaults
    orchestrator_config = OrchestratorConfig(
        openai_api_key=api_key,
        gpt_model=config.get('gpt_model', 'gpt-4o') if config else 'gpt-4o',
        vision_model=config.get('vision_model', 'gpt-4o') if config else 'gpt-4o', 
        max_tokens=config.get('max_tokens', 2000) if config else 2000,
        enable_fallback=config.get('enable_fallback', True) if config else True,
        log_level=config.get('log_level', 'INFO') if config else 'INFO'
    )
    
    return MultimodalOrchestrator(orchestrator_config)

def process_single_query(input_data, orchestrator: MultimodalOrchestrator, **kwargs):
    """Process a single query and return results."""
    try:
        result = orchestrator.process_query(input_data, **kwargs)
        
        # Format output
        output = {
            "success": True,
            "content": result.content,
            "modality": result.modality,
            "confidence": result.confidence,
            "metadata": result.metadata,
            "processed_data": result.processed_data
        }
        
        return output
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "content": f"Error processing query: {str(e)}",
            "modality": "error",
            "confidence": 0.0
        }

def process_file_input(file_path: str, orchestrator: MultimodalOrchestrator, **kwargs):
    """Process input from a file."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Read file content
    if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
        # Binary image file
        with open(file_path, 'rb') as f:
            input_data = f.read()
    elif file_path.suffix.lower() in ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac']:
        # Audio file - pass as path (pydub works better with file paths)
        input_data = file_path
    elif file_path.suffix.lower() in ['.pdf', '.docx', '.doc', '.txt', '.csv', '.json', '.xml']:
        # Document file - pass as path
        input_data = file_path
    else:
        # Try as text file
        try:
            input_data = file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            # If text fails, try as binary
            with open(file_path, 'rb') as f:
                input_data = f.read()
    
    return process_single_query(input_data, orchestrator, **kwargs)

def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="Multimodal Query Processor")
    parser.add_argument("input", help="Input text, file path, or '-' for stdin")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--model", default="gpt-4o", help="GPT model to use")
    parser.add_argument("--vision-model", default="gpt-4o", help="Vision model to use")
    parser.add_argument("--max-tokens", type=int, default=2000, help="Maximum tokens for response")
    parser.add_argument("--task", default="general", help="Processing task type")
    parser.add_argument("--output", help="Output file for results (JSON)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--health-check", action="store_true", help="Run health check")
    
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    # Create orchestrator
    config = {
        'gpt_model': args.model,
        'vision_model': args.vision_model,
        'max_tokens': args.max_tokens,
        'log_level': 'DEBUG' if args.verbose else 'INFO'
    }
    
    orchestrator = create_orchestrator(args.api_key, config)
    
    # Health check
    if args.health_check:
        health = orchestrator.health_check()
        print(json.dumps(health, indent=2))
        return
    
    # Process input
    try:
        if args.input == '-':
            # Read from stdin
            input_data = sys.stdin.read().strip()
            result = process_single_query(input_data, orchestrator, task=args.task)
        elif Path(args.input).exists():
            # Process file
            result = process_file_input(args.input, orchestrator, task=args.task)
        else:
            # Process as text
            result = process_single_query(args.input, orchestrator, task=args.task)
        
        # Output results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
        else:
            print(json.dumps(result, indent=2))
            
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "content": f"Error: {str(e)}",
            "modality": "error",
            "confidence": 0.0
        }
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(error_result, f, indent=2)
        else:
            print(json.dumps(error_result, indent=2))
        
        sys.exit(1)

def interactive_mode():
    """Interactive mode for processing multiple queries."""
    setup_environment()
    
    print("=== Multimodal Query Processor ===")
    print("Enter queries to process. Type 'quit' to exit.")
    print("For files, use: file:/path/to/file")
    print("For images, use: image:/path/to/image")
    print("For audio, use: audio:/path/to/audio")
    print()
    
    orchestrator = create_orchestrator()
    
    while True:
        try:
            query = input("Query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            # Process query
            if query.startswith('file:') or query.startswith('image:') or query.startswith('audio:'):
                # File input
                prefix, file_path = query.split(':', 1)
                result = process_file_input(file_path.strip(), orchestrator)
            else:
                # Text input
                result = process_single_query(query, orchestrator)
            
            # Display result
            print(f"\nModality: {result['modality']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Content: {result['content']}")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments, run interactive mode
        interactive_mode()
    else:
        # Run with command line arguments
        main()
