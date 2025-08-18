#!/usr/bin/env python3
"""
Script to help set up the .env file with your OpenAI API key.
"""

import os
from pathlib import Path

def setup_env_file():
    """Create .env file with user's OpenAI API key."""
    env_file = Path(".env")
    
    print("🔧 Setting up environment file")
    print("=" * 40)
    
    # Check if .env already exists
    if env_file.exists():
        print("✓ .env file already exists")
        
        # Read existing content
        with open(env_file, 'r') as f:
            content = f.read()
        
        if 'OPENAI_API_KEY=your_openai_api_key_here' in content:
            print("⚠️  Default API key found in .env file")
            print("\nPlease update your .env file with your actual OpenAI API key:")
            print(f"  File location: {env_file.absolute()}")
            print("  Replace 'your_openai_api_key_here' with your actual API key")
        elif 'OPENAI_API_KEY=' in content:
            print("✓ OpenAI API key is configured in .env file")
        else:
            print("⚠️  No OpenAI API key found in .env file")
            
        return
    
    # Get API key from user
    print("Please enter your OpenAI API key:")
    print("(You can get one from: https://platform.openai.com/api-keys)")
    
    api_key = input("API Key: ").strip()
    
    if not api_key:
        print("❌ No API key provided")
        return
    
    if api_key == "your_openai_api_key_here":
        print("❌ Please enter your actual API key, not the placeholder")
        return
    
    # Create .env file content
    env_content = f"""# OpenAI API Configuration
# Get your API key from: https://platform.openai.com/api-keys
OPENAI_API_KEY={api_key}

# Optional: Logging Configuration
LOG_LEVEL=INFO

# Optional: Model Configuration (these are defaults, you can override)
GPT_MODEL=gpt-3.5-turbo
VISION_MODEL=gpt-4-vision-preview
MAX_TOKENS=1000

# Optional: Processing Configuration
ENABLE_FALLBACK=true
"""
    
    # Write .env file
    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        print(f"✓ .env file created successfully at: {env_file.absolute()}")
        print("\n🎉 Setup complete! You can now use the multimodal query processing system.")
        
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")

def check_env_status():
    """Check current environment status."""
    print("\n📊 Environment Status")
    print("=" * 40)
    
    # Check .env file
    env_file = Path(".env")
    if env_file.exists():
        print("✓ .env file exists")
        
        with open(env_file, 'r') as f:
            content = f.read()
        
        if 'OPENAI_API_KEY=' in content and 'your_openai_api_key_here' not in content:
            print("✓ OpenAI API key is configured")
        else:
            print("⚠️  OpenAI API key needs to be configured")
    else:
        print("❌ .env file not found")
    
    # Check environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        if api_key == 'your_openai_api_key_here':
            print("⚠️  Default API key found in environment")
        else:
            print("✓ OpenAI API key found in environment variables")
    else:
        print("❌ OPENAI_API_KEY not found in environment variables")

def main():
    """Main function."""
    print("🔧 Multimodal Query Processing System - Environment Setup")
    print("=" * 60)
    
    # Check current status
    check_env_status()
    
    # Setup .env file
    print()
    setup_env_file()
    
    # Final status check
    check_env_status()
    
    print("\n💡 Usage:")
    print("  python main.py 'What is artificial intelligence?'")
    print("  python main.py --health-check")
    print("  python test_system.py")

if __name__ == "__main__":
    main()
