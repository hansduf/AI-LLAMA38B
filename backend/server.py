python -m uvicorn#!/usr/bin/env python3
import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import uvicorn
    from main import app
    
    print("🚀 Starting Dokai Chat Backend Server...")
    print("📡 Server will be available at: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔄 Auto-reload enabled for development")
    
    if __name__ == "__main__":
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
        
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("📦 Installing missing packages...")
    
    # Try to install missing packages
    import subprocess
    packages = [
        "fastapi", "uvicorn", "httpx", "python-multipart", 
        "docx2txt", "PyPDF4", "python-docx", "PyMuPDF", 
        "python-magic-bin", "pillow", "lxml"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError:
            print(f"⚠️ Failed to install {package}")
    
    print("🔄 Please run the server again...")
    
except Exception as e:
    print(f"❌ Error starting server: {e}")
    import traceback
    traceback.print_exc()
