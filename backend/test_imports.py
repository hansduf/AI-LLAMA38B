import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing imports...")

try:
    from fastapi import FastAPI
    print("✅ FastAPI imported successfully")
except ImportError as e:
    print(f"❌ FastAPI import error: {e}")

try:
    import httpx
    print("✅ httpx imported successfully")
except ImportError as e:
    print(f"❌ httpx import error: {e}")

try:
    import fitz
    print("✅ PyMuPDF (fitz) imported successfully")
except ImportError as e:
    print(f"❌ PyMuPDF (fitz) import error: {e}")

try:
    import magic
    print("✅ python-magic imported successfully")
except ImportError as e:
    print(f"❌ python-magic import error: {e}")

try:
    from docx import Document
    print("✅ python-docx imported successfully")
except ImportError as e:
    print(f"❌ python-docx import error: {e}")

try:
    import PyPDF4
    print("✅ PyPDF4 imported successfully")
except ImportError as e:
    print(f"❌ PyPDF4 import error: {e}")

try:
    import docx2txt
    print("✅ docx2txt imported successfully")
except ImportError as e:
    print(f"❌ docx2txt import error: {e}")

print("\n" + "="*50)
print("Testing main.py import...")

try:
    from main import app
    print("✅ main.py imported successfully!")
    print(f"✅ FastAPI app created: {type(app)}")
except Exception as e:
    print(f"❌ main.py import error: {e}")
    import traceback
    traceback.print_exc()
