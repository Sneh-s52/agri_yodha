#!/usr/bin/env python3
"""
Setup script for the Multimodal Query Processing System.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def create_virtual_environment():
    """Create a virtual environment."""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("✓ Virtual environment already exists")
        return True
    
    print("Creating virtual environment...")
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✓ Virtual environment created")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to create virtual environment")
        return False

def install_dependencies():
    """Install required dependencies."""
    print("Installing dependencies...")
    
    # Determine the pip command based on the platform
    if os.name == 'nt':  # Windows
        pip_cmd = "venv\\Scripts\\pip"
    else:  # Unix/Linux/macOS
        pip_cmd = "venv/bin/pip"
    
    try:
        subprocess.run([pip_cmd, "install", "-r", "requirements.txt"], check=True)
        print("✓ Dependencies installed")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def setup_environment_file():
    """Create .env file from template."""
    env_file = Path(".env")
    env_example = Path("env.example")
    
    if env_file.exists():
        print("✓ .env file already exists")
        return True
    
    if env_example.exists():
        print("Creating .env file from template...")
        try:
            with open(env_example, 'r') as f:
                content = f.read()
            
            with open(env_file, 'w') as f:
                f.write(content)
            
            print("✓ .env file created")
            print("⚠️  Please edit .env file and add your OpenAI API key")
            return True
        except Exception as e:
            print(f"❌ Failed to create .env file: {e}")
            return False
    else:
        print("⚠️  env.example not found, creating basic .env file...")
        try:
            with open(env_file, 'w') as f:
                f.write("# OpenAI API Configuration\n")
                f.write("OPENAI_API_KEY=your_openai_api_key_here\n")
                f.write("\n# Optional: Logging Configuration\n")
                f.write("LOG_LEVEL=INFO\n")
            
            print("✓ Basic .env file created")
            print("⚠️  Please edit .env file and add your OpenAI API key")
            return True
        except Exception as e:
            print(f"❌ Failed to create .env file: {e}")
            return False

def run_tests():
    """Run system tests."""
    print("Running system tests...")
    
    # Determine the python command based on the platform
    if os.name == 'nt':  # Windows
        python_cmd = "venv\\Scripts\\python"
    else:  # Unix/Linux/macOS
        python_cmd = "venv/bin/python"
    
    try:
        result = subprocess.run([python_cmd, "test_system.py"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ System tests passed")
            return True
        else:
            print("❌ System tests failed")
            print(result.stdout)
            print(result.stderr)
            return False
    except subprocess.CalledProcessError:
        print("❌ Failed to run system tests")
        return False

def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit the .env file and add your OpenAI API key")
    print("2. Activate the virtual environment:")
    
    if os.name == 'nt':  # Windows
        print("   venv\\Scripts\\activate")
    else:  # Unix/Linux/macOS
        print("   source venv/bin/activate")
    
    print("\n3. Test the system:")
    print("   python test_system.py")
    
    print("\n4. Run the main application:")
    print("   python main.py")
    
    print("\n5. Try interactive mode:")
    print("   python main.py")
    
    print("\n6. Run examples:")
    print("   python examples/example_usage.py")
    
    print("\nFor more information, see the README.md file")

def main():
    """Main setup function."""
    print("Multimodal Query Processing System - Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    if not create_virtual_environment():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Setup environment file
    if not setup_environment_file():
        sys.exit(1)
    
    # Run tests
    if not run_tests():
        print("⚠️  Tests failed, but setup completed. You may need to configure your API key first.")
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()
