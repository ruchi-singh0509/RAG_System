
import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} is compatible")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    try:
        # Install from requirements.txt
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_tesseract():
    """Check if Tesseract OCR is installed"""
    print("\n🔍 Checking Tesseract OCR...")
    
    try:
        # Try to run tesseract
        result = subprocess.run(["tesseract", "--version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Tesseract OCR is installed")
            return True
        else:
            print("❌ Tesseract OCR is not working properly")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ Tesseract OCR is not installed")
        print("\n📋 To install Tesseract OCR:")
        
        system = platform.system().lower()
        if system == "windows":
            print("1. Download from: https://github.com/UB-Mannheim/tesseract/wiki")
            print("2. Install and add to PATH")
        elif system == "darwin":  # macOS
            print("Run: brew install tesseract")
        else:  # Linux
            print("Run: sudo apt-get install tesseract-ocr")
        
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = [
        "./faiss_db",
        "./uploads", 
        "./temp",
        "./logs",
        "./backups"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")
    
    return True

def check_environment_variables():
    """Check and setup environment variables"""
    print("\n🔧 Checking environment variables...")
    
    # Check if .env file exists
    env_file = Path(".env")
    env_example = Path("env_example.txt")
    
    if not env_file.exists():
        if env_example.exists():
            print("📝 Creating .env file from template...")
            with open(env_example, 'r') as f:
                env_content = f.read()
            
            with open(env_file, 'w') as f:
                f.write(env_content)
            
            print("✅ .env file created")
            print("⚠️ Please edit .env file with your OpenAI API key")
        else:
            print("⚠️ No .env file found. Please create one with your configuration.")
    
    # Check OpenAI API key
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key or openai_key == 'your_openai_api_key_here':
        print("⚠️ OpenAI API key not configured")
        print("Please set your OpenAI API key in the .env file")
        return False
    else:
        print("✅ OpenAI API key is configured")
        return True

def run_tests():
    """Run system tests"""
    print("\n🧪 Running system tests...")
    
    try:
        result = subprocess.run([sys.executable, "test_system.py"], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ All tests passed")
            return True
        else:
            print("❌ Some tests failed")
            print("Test output:")
            print(result.stdout)
            print("Test errors:")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("❌ Tests timed out")
        return False
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")
        return False

def show_next_steps():
    """Show next steps for the user"""
    print("\n🎉 Setup completed!")
    print("\n📋 Next steps:")
    print("1. Edit .env file with your OpenAI API key")
    print("2. Run the web application: streamlit run app.py")
    print("3. Or run the demo: python demo.py")
    print("4. Or run tests: python test_system.py")
    
    print("\n🚀 Quick start commands:")
    print("  streamlit run app.py          # Start web interface")
    print("  python demo.py               # Run demo")
    print("  python test_system.py        # Run tests")

def main():
    """Main setup function"""
    print("🚀 Visual Document Analysis RAG System Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed at dependency installation")
        sys.exit(1)
    
    # Check Tesseract
    tesseract_ok = check_tesseract()
    if not tesseract_ok:
        print("⚠️ Tesseract OCR is required for full functionality")
        print("You can continue setup, but OCR features won't work")
    
    # Create directories
    if not create_directories():
        print("❌ Setup failed at directory creation")
        sys.exit(1)
    
    # Check environment variables
    env_ok = check_environment_variables()
    if not env_ok:
        print("⚠️ Environment variables not fully configured")
        print("You can continue setup, but some features may not work")
    
    # Run tests (optional)
    print("\n🧪 Would you like to run system tests? (y/n): ", end="")
    choice = input().strip().lower()
    
    if choice in ['y', 'yes']:
        if not run_tests():
            print("⚠️ Tests failed, but setup can continue")
    
    # Show next steps
    show_next_steps()

if __name__ == "__main__":
    main()
