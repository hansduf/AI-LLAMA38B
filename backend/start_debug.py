#!/usr/bin/env python3
"""
Startup script untuk Dokai Chat Backend
Debugging dan diagnostics untuk masalah startup
"""

import sys
import os
import traceback
from pathlib import Path

print("🔧 DOKAI CHAT BACKEND - STARTUP DIAGNOSTICS")
print("=" * 60)

# Check Python version
print(f"🐍 Python Version: {sys.version}")
print(f"📁 Working Directory: {os.getcwd()}")
print(f"📍 Script Location: {Path(__file__).absolute()}")

# Check required packages
required_packages = [
    ('fastapi', 'FastAPI'),
    ('uvicorn', 'Uvicorn'),
    ('httpx', 'HTTPX'),
    ('pydantic', 'Pydantic'),
    ('docx', 'python-docx'),
    ('docx2txt', 'docx2txt'),
    ('PyPDF4', 'PyPDF4'),
    ('fitz', 'PyMuPDF'),
    ('magic', 'python-magic'),
    ('PIL', 'Pillow')
]

print("\n📦 CHECKING REQUIRED PACKAGES:")
print("-" * 40)

missing_packages = []
for module_name, package_name in required_packages:
    try:
        __import__(module_name)
        print(f"✅ {package_name:<15} - OK")
    except ImportError as e:
        print(f"❌ {package_name:<15} - MISSING ({e})")
        missing_packages.append(package_name)

if missing_packages:
    print(f"\n⚠️  MISSING PACKAGES: {', '.join(missing_packages)}")
    print("💡 Run: pip install " + " ".join(missing_packages))
    sys.exit(1)

print("\n🔍 TESTING MAIN MODULE IMPORT:")
print("-" * 40)

try:
    # Add current directory to path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    print("📂 Added to Python path:", current_dir)
    
    # Import main module
    import main
    print("✅ main.py imported successfully")
    
    # Check FastAPI app
    if hasattr(main, 'app'):
        print("✅ FastAPI app found")
        print(f"📋 App type: {type(main.app)}")
    else:
        print("❌ FastAPI app not found in main module")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Failed to import main.py: {e}")
    print("\n🔍 DETAILED ERROR:")
    traceback.print_exc()
    sys.exit(1)

print("\n🚀 STARTING SERVER:")
print("-" * 40)

try:
    import uvicorn
    
    print("✅ Uvicorn imported successfully")
    print("🌐 Starting server on http://localhost:8000")
    print("📚 API docs will be available at: http://localhost:8000/docs")
    print("🔄 Auto-reload enabled")
    print("\n" + "=" * 60)
    print("🎯 SERVER STARTING...")
    print("=" * 60)
    
    # Start the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )
    
except KeyboardInterrupt:
    print("\n\n🛑 Server stopped by user")
except Exception as e:
    print(f"\n❌ Server startup failed: {e}")
    print("\n🔍 DETAILED ERROR:")
    traceback.print_exc()
    sys.exit(1)
