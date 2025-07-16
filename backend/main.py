from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import httpx
import logging
import os
import shutil
import uuid
import time
import hashlib
import docx2txt  # For basic DOCX processing
import PyPDF4   # For PDF processing
import time
import base64
import io
from docx import Document  # For advanced DOCX processing
from PIL import Image
import fitz  # PyMuPDF for better PDF processing
import magic  # For file type detection
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Folder untuk menyimpan file yang diunggah
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class ChatRequest(BaseModel):
    message: str
    context: Optional[str] = None  # Untuk menyimpan konteks dari dokumen
    model_type: Optional[str] = "llama3:latest"  # Model selection
    conversation_history: Optional[List[Dict[str, str]]] = []  # Chat history untuk konteks

class DocumentResponse(BaseModel):
    document_id: str
    content: str
    filename: str

# Document Library System
import json
from pathlib import Path

# Document Metadata Storage
METADATA_FILE = os.path.join(UPLOAD_FOLDER, "documents_metadata.json")

class DocumentMetadata(BaseModel):
    document_id: str
    filename: str
    original_filename: str
    upload_date: str
    file_size: int
    file_type: str  # .pdf or .docx
    content_preview: str  # First 200 chars
    analysis_summary: Dict[str, Any]
    is_active: bool = False

class DocumentLibrary:
    """Manage document library with persistent storage"""
    
    def __init__(self):
        self.metadata_file = METADATA_FILE
        self.ensure_metadata_file()
    
    def ensure_metadata_file(self):
        """Ensure metadata file exists"""
        if not os.path.exists(self.metadata_file):
            self.save_metadata([])
    
    def load_metadata(self) -> List[DocumentMetadata]:
        """Load all document metadata"""
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return [DocumentMetadata(**item) for item in data]
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            return []
    
    def save_metadata(self, documents: List[DocumentMetadata]):
        """Save document metadata"""
        try:
            data = [doc.dict() for doc in documents]
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def add_document(self, metadata: DocumentMetadata):
        """Add new document to library"""
        documents = self.load_metadata()
        
        # Set all other documents as inactive
        for doc in documents:
            doc.is_active = False
        
        # Add new document as active
        metadata.is_active = True
        documents.append(metadata)
        
        self.save_metadata(documents)
        logger.info(f"Added document to library: {metadata.filename}")
    
    def get_all_documents(self) -> List[DocumentMetadata]:
        """Get all documents in library"""
        return self.load_metadata()
    
    def get_document(self, document_id: str) -> Optional[DocumentMetadata]:
        """Get specific document by ID"""
        documents = self.load_metadata()
        for doc in documents:
            if doc.document_id == document_id:
                return doc
        return None
    
    def set_active_document(self, document_id: str) -> bool:
        """Set document as active"""
        documents = self.load_metadata()
        found = False
        
        for doc in documents:
            if doc.document_id == document_id:
                doc.is_active = True
                found = True
            else:
                doc.is_active = False
        
        if found:
            self.save_metadata(documents)
            logger.info(f"Set active document: {document_id}")
        
        return found
    
    def get_active_document(self) -> Optional[DocumentMetadata]:
        """Get currently active document"""
        documents = self.load_metadata()
        for doc in documents:
            if doc.is_active:
                return doc
        return None
    
    def delete_document(self, document_id: str) -> bool:
        """Delete document from library and filesystem"""
        documents = self.load_metadata()
        doc_to_delete = None
        
        # Find document to delete
        for i, doc in enumerate(documents):
            if doc.document_id == document_id:
                doc_to_delete = doc
                documents.pop(i)
                break
        
        if doc_to_delete:
            # Delete physical file
            file_path = os.path.join(UPLOAD_FOLDER, doc_to_delete.filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Deleted file: {file_path}")
            
            # Save updated metadata
            self.save_metadata(documents)
            logger.info(f"Deleted document from library: {doc_to_delete.original_filename}")
            return True
        
        return False

# Simplified Document Processing for Library System
class SimpleDocumentProcessor:
    """Simplified document processing for active document selection"""
    
    def enhance_context(self, query: str, full_context: str) -> str:
        """Simple context enhancement - just truncate if too long"""
        max_length = 2000  # 2KB context limit
        
        if not full_context:
            return ""
        
        if len(full_context) <= max_length:
            return full_context.strip()
        
        # Simple truncation
        return full_context[:max_length] + "..."

class SimplePromptEngineer:
    """Simplified prompt engineering for document and general chat"""
    
    def create_document_prompt(self, query: str, context: str, conversation_history: List[Dict] = None) -> str:
        """Create simple document analysis prompt"""
        
        conversation_context = ""
        if conversation_history:
            last_exchanges = conversation_history[-2:] if conversation_history else []
            for msg in last_exchanges:
                conversation_context += f"{msg.get('sender', 'User')}: {msg.get('content', '')[:200]}...\n"
        
        prompt = f"""DOKUMEN:
{context}

RIWAYAT:
{conversation_context}

PERTANYAAN: {query}

Jawab berdasarkan dokumen di atas dengan informatif dan detail.

JAWABAN:"""
        
        return prompt
    
    def create_general_prompt(self, query: str, conversation_history: List[Dict] = None) -> str:
        """Create simple general conversation prompt"""
        
        conversation_context = ""
        if conversation_history:
            last_exchanges = conversation_history[-2:] if conversation_history else []
            for msg in last_exchanges:
                conversation_context += f"{msg.get('sender', 'User')}: {msg.get('content', '')[:150]}...\n"
        
        prompt = f"""{conversation_context}

Pertanyaan: {query}
Jawaban:"""
        
        return prompt

# Simplified Performance Monitoring
class SimplePerformanceMonitor:
    """Simple performance monitoring for basic stats"""
    
    def __init__(self):
        self.last_response_time = 0
        
    def add_response_time(self, response_time_ms: float):
        """Add a response time measurement"""
        self.last_response_time = response_time_ms
    
    def get_simple_status(self) -> str:
        """Get simple performance status"""
        if self.last_response_time < 5000:
            return "excellent"
        elif self.last_response_time < 15000:
            return "good"
        elif self.last_response_time < 45000:
            return "slow"
        else:
            return "very_slow"

# Simple Document Analysis Functions
async def analyze_docx_structure(file_path: str) -> dict:
    """Simple DOCX structure analysis"""
    try:
        doc = Document(file_path)
        
        return {
            "document_type": "DOCX",
            "paragraphs": len([p for p in doc.paragraphs if p.text.strip()]),
            "tables": len(doc.tables),
            "text_length": sum(len(p.text) for p in doc.paragraphs),
        }
    except Exception as e:
        logger.error(f"Error analyzing DOCX: {e}")
        return {"error": str(e)}

async def analyze_pdf_structure(file_path: str) -> dict:
    """Simple PDF structure analysis"""
    try:
        pdf_document = fitz.open(file_path)
        
        text_length = 0
        images = 0
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            text_length += len(page.get_text())
            images += len(page.get_images())
        
        pdf_document.close()
        
        return {
            "document_type": "PDF",
            "pages": len(pdf_document),
            "images": images,
            "text_length": text_length,
        }
    except Exception as e:
        logger.error(f"Error analyzing PDF: {e}")
        return {"error": str(e)}

# Simplified Response Processing Classes
class ResponseCache:
    """Simple in-memory cache for responses"""
    
    def __init__(self, max_size: int = 100, ttl_minutes: int = 30):
        self.cache: Dict[str, Dict] = {}
        self.max_size = max_size
        self.ttl = timedelta(minutes=ttl_minutes)
    
    def _generate_key(self, prompt: str, context: str = None) -> str:
        """Generate cache key"""
        content = f"{prompt}:{context or ''}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, prompt: str, context: str = None) -> Optional[str]:
        """Get cached response"""
        key = self._generate_key(prompt, context)
        
        if key in self.cache:
            entry = self.cache[key]
            if datetime.now() - entry['timestamp'] < self.ttl:
                return entry['response']
            else:
                del self.cache[key]
        
        return None
    
    def set(self, prompt: str, response: str, context: str = None):
        """Cache a response"""
        key = self._generate_key(prompt, context)
        
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]
        
        self.cache[key] = {
            'response': response,
            'timestamp': datetime.now()
        }
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()

class ResponseMonitor:
    """Simple response quality monitor"""
    
    def clean_response(self, text: str) -> str:
        """Basic response cleaning"""
        if not text or text.strip() == "":
            return "Maaf, tidak ada jawaban yang dapat saya berikan."
        
        cleaned_text = text.strip()
        
        # Limit length if extremely long
        if len(cleaned_text) > 2000:
            cleaned_text = cleaned_text[:2000] + "..."
        
        return cleaned_text

# AI Model Configuration System
@dataclass
class ModelConfig:
    model_name: str
    temperature: float
    top_p: float
    top_k: int
    num_ctx: int
    num_predict: int
    repeat_penalty: float
    num_thread: int
    stop_tokens: list

class AIModelOptimizer:
    """Ultra-speed AI Model optimizer for gaming laptop - prioritizing speed"""
    
    def __init__(self):
        # ULTRA-SPEED GAMING LAPTOP configuration (22GB RAM!)
        self.config = ModelConfig(
            model_name="llama3:8b",
            temperature=0.3,        # Lower for faster, more deterministic responses
            top_p=0.7,             # Reduced for speed
            top_k=20,              # Much lower for speed
            num_ctx=2048,          # Reduced context for speed (still plenty)
            num_predict=200,       # Shorter responses for speed
            repeat_penalty=1.1,    # Standard penalty
            num_thread=-1,         # Use all CPU threads
            stop_tokens=["\n\n\n", "Human:", "Assistant:", "PERTANYAAN:", "JAWABAN:", "User:"]
        )
    
    def get_config(self) -> ModelConfig:
        return self.config
    
    def get_optimized_payload(self, prompt: str) -> Dict[str, Any]:
        config = self.get_config()
        
        return {
            "model": config.model_name,
            "prompt": prompt,
            "stream": False,  # Non-streaming for consistency
            "options": {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "num_ctx": config.num_ctx,
                "num_predict": config.num_predict,
                "num_thread": config.num_thread,
                "repeat_penalty": config.repeat_penalty,
                "repeat_last_n": 50,
                "stop": config.stop_tokens,
                
                # ULTRA-SPEED GAMING LAPTOP optimizations (22GB RAM!)
                "num_gpu": -1,       # Auto-detect GPU (Nitro might have dGPU)
                "low_vram": False,   # Disable - you have plenty RAM
                "f16_kv": False,     # Disable for speed (slight quality loss)
                "num_batch": 1024,   # Even larger batch for 22GB RAM
                "numa": False,       # Still disable for single socket
                "mlock": True,       # Lock model in memory (you have RAM!)
                "use_mmap": True,    # Enable memory mapping for speed
                "seed": -1,          # Random seed
                "num_gqa": -1,       # Default
                "rope_freq_base": 10000,
                "rope_freq_scale": 1.0,
                
                # Additional speed optimizations
                "penalize_newline": False,  # Don't penalize newlines for speed
                "presence_penalty": 0.0,    # Disable for speed
                "frequency_penalty": 0.0,   # Disable for speed
                "tfs_z": 1.0,              # Disable tail free sampling for speed
                "typical_p": 1.0,          # Disable typical sampling for speed
                "min_p": 0.0               # Disable min probability for speed
            }
        }

# Initialize optimizer
ai_optimizer = AIModelOptimizer()

# Initialize document library
document_library = DocumentLibrary()

# Initialize simple processors
doc_processor = SimpleDocumentProcessor()
prompt_engineer = SimplePromptEngineer()

# Initialize performance monitoring and caching
performance_monitor = SimplePerformanceMonitor()
response_cache = ResponseCache(max_size=200, ttl_minutes=60)
response_monitor = ResponseMonitor()

@app.post("/api/chat")
async def chat_with_llama(request: ChatRequest):
    # Start timing
    start_time = time.time()
    request_start = datetime.now()
    
    try:
        logger.info(f"🚀 [TIMING] Chat request started at {request_start.strftime('%H:%M:%S.%f')[:-3]}")
        logger.info(f"Received chat request: {request.message[:100]}...")
        
        # Check for very long context that might cause timeouts
        if request.context and len(request.context) > 5000:
            logger.warning(f"⚠️ Large document context detected: {len(request.context)} characters")
            logger.warning("This may cause slower response times. Consider asking more specific questions.")
        
        # Check for very long messages
        if len(request.message) > 500:
            logger.warning(f"⚠️ Long message detected: {len(request.message)} characters")
            logger.warning("Shorter, more specific questions typically get faster responses.")

        # Enhanced prompt engineering with conversation history and active document
        prompt_start = time.time()
        
        # Check for active document if no context provided
        final_context = request.context
        active_doc_info = ""
        
        if not final_context:
            # Check for active document in library
            active_doc = document_library.get_active_document()
            if active_doc:
                # Load content from active document
                file_path = os.path.join(UPLOAD_FOLDER, active_doc.filename)
                if os.path.exists(file_path):
                    try:
                        final_context = await extract_text_from_document(file_path)
                        active_doc_info = f"📄 Using active document: {active_doc.original_filename}"
                        logger.info(f"Using active document: {active_doc.original_filename}")
                    except Exception as e:
                        logger.warning(f"Failed to load active document: {e}")
        
        if final_context:
            # SAFETY: Limit document context size to prevent timeout
            if len(final_context) > 10000:  # Max 10KB context
                logger.warning(f"⚠️ Large document truncated: {len(final_context)} → 10000 chars")
                final_context = final_context[:10000] + "..."
            
            # Process document with enhanced intelligence and conversation history
            try:
                enhanced_context = doc_processor.enhance_context(request.message, final_context)
                optimized_prompt = prompt_engineer.create_document_prompt(
                    request.message, 
                    enhanced_context,
                    request.conversation_history or []
                )
                logger.info(f"Using enhanced document context with {len(enhanced_context)} characters")
            except Exception as e:
                logger.error(f"Document processing failed: {e}")
                # Fallback to simple context
                enhanced_context = final_context[:1500] + "..." if len(final_context) > 1500 else final_context
                optimized_prompt = prompt_engineer.create_document_prompt(
                    request.message, 
                    enhanced_context,
                    request.conversation_history or []
                )
                logger.info(f"Using fallback document context with {len(enhanced_context)} characters")
        else:
            # Enhanced general conversation with history
            optimized_prompt = prompt_engineer.create_general_prompt(
                request.message,
                request.conversation_history or []
            )
            logger.info("Using enhanced general conversation prompt with history")
        
        prompt_time = (time.time() - prompt_start) * 1000
        logger.info(f"⚡ [TIMING] Prompt engineering: {prompt_time:.2f}ms")
        
        # Check cache first
        cache_start = time.time()
        cached_response = response_cache.get(request.message, context=final_context)
        cache_time = (time.time() - cache_start) * 1000
        logger.info(f"⚡ [TIMING] Cache check: {cache_time:.2f}ms")
        
        if cached_response:
            total_time = (time.time() - start_time) * 1000
            logger.info(f"🎯 [TIMING] CACHE HIT! Total response time: {total_time:.2f}ms")
            
            # Add active document info to cached response if available
            response_data = {
                "response": cached_response,
                "status": "complete",
                "cached": True,
                "timing": {
                    "total_ms": total_time,
                    "source": "cache"
                }
            }
            
            if active_doc_info:
                response_data["document_info"] = active_doc_info
            
            return JSONResponse(response_data)
        
        # Prepare request payload
        payload_start = time.time()
        
        # Set timeout for gaming laptop (10 minutes for complex document analysis)
        timeout_seconds = 600.0  # 10 minutes for gaming laptop
        
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            # Get optimized payload
            request_payload = ai_optimizer.get_optimized_payload(optimized_prompt)
            
            payload_time = (time.time() - payload_start) * 1000
            logger.info(f"⚡ [TIMING] Payload preparation: {payload_time:.2f}ms")
            
            logger.info(f"Using optimized llama3:8b configuration (non-streaming)")
            logger.info(f"Context length: {request_payload['options']['num_ctx']}")
            logger.info(f"Max tokens: {request_payload['options']['num_predict']}")
            logger.info(f"Temperature: {request_payload['options']['temperature']}")
            logger.info(f"Batch size: {request_payload['options'].get('num_batch', 'default')}")
            logger.info(f"Timeout: {timeout_seconds}s")
            logger.info(f"Prompt length: {len(optimized_prompt)} characters")
            
            # Make request to Ollama with detailed timing
            ollama_start = time.time()
            logger.info(f"🤖 [TIMING] Sending request to Ollama at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
            
            response = await client.post(
                "http://localhost:11434/api/generate",
                json=request_payload,
                timeout=timeout_seconds
            )
            
            ollama_end = time.time()
            ollama_time = (ollama_end - ollama_start) * 1000
            logger.info(f"🤖 [TIMING] Ollama API response received: {ollama_time:.2f}ms")
            logger.info(f"Ollama response status: {response.status_code}")
            
            if response.status_code == 200:
                # Handle response processing with timing
                processing_start = time.time()
                result = response.json()
                
                logger.info(f"Raw Ollama response keys: {list(result.keys())}")
                
                ai_response = result.get("response", "")
                
                # Debug logging
                logger.info(f"AI response length: {len(ai_response)}")
                logger.info(f"AI response preview: {ai_response[:100]}...")
                
                # VALIDASI: Jika response terlalu pendek, ini kemungkinan error
                if len(ai_response.strip()) < 5:  # Kurang dari 5 karakter
                    logger.warning(f"⚠️ Response terlalu pendek ({len(ai_response)} chars): '{ai_response}'")
                    logger.warning("Kemungkinan stop tokens terlalu agresif atau model error")
                    ai_response = f"Maaf, sistem menghasilkan respons yang tidak lengkap ('{ai_response.strip()}'). Silakan coba lagi dengan pertanyaan yang berbeda."
                
                # If response is empty, provide a fallback
                if not ai_response or ai_response.strip() == "":
                    logger.warning("Empty response from Ollama, using fallback")
                    ai_response = "Maaf, saya tidak dapat memberikan jawaban saat ini. Silakan coba lagi dengan pertanyaan yang berbeda."
                
                # Clean the response
                cleaned_response = response_monitor.clean_response(ai_response)
                
                # Cache the response
                response_cache.set(request.message, cleaned_response, final_context)
                
                # Monitor performance
                performance_monitor.add_response_time(ollama_time)
                
                processing_time = (time.time() - processing_start) * 1000
                total_time = (time.time() - start_time) * 1000
                
                # Add simple performance monitoring
                perf_status = performance_monitor.get_simple_status()
                
                logger.info(f"⚡ [TIMING] Response processing: {processing_time:.2f}ms")
                logger.info(f"🎯 [TIMING] TOTAL REQUEST TIME: {total_time:.2f}ms ({total_time/1000:.2f}s)")
                logger.info(f"📊 [PERFORMANCE] Status: {perf_status}")
                
                if perf_status in ['slow', 'very_slow']:
                    logger.warning("🐌 [PERFORMANCE] Slow response detected!")
                
                logger.info(f"Final cleaned response length: {len(cleaned_response)} characters")
                
                # Prepare response data
                response_data = {
                    "response": cleaned_response,
                    "status": "complete",
                    "timing": {
                        "total_ms": total_time,
                        "ollama_ms": ollama_time,
                        "processing_ms": processing_time,
                        "prompt_ms": prompt_time,
                        "cache_ms": cache_time,
                        "breakdown": {
                            "prompt_engineering": prompt_time,
                            "cache_check": cache_time,
                            "payload_prep": payload_time,
                            "ollama_request": ollama_time,
                            "response_processing": processing_time
                        }
                    }
                }
                
                # Add document info if active document was used
                if active_doc_info:
                    response_data["document_info"] = active_doc_info
                
                return JSONResponse(response_data)
            else:
                error_msg = f"Ollama API returned status code {response.status_code}"
                total_time = (time.time() - start_time) * 1000
                logger.error(f"❌ [TIMING] Error after {total_time:.2f}ms: {error_msg}")
                logger.error(f"Response content: {response.text}")
                raise HTTPException(status_code=500, detail=error_msg)
                
    except httpx.ConnectError:
        total_time = (time.time() - start_time) * 1000
        error_msg = "Failed to connect to Ollama. Please ensure Ollama is running on http://localhost:11434"
        logger.error(f"❌ [TIMING] Connection error after {total_time:.2f}ms: {error_msg}")
        raise HTTPException(status_code=503, detail=error_msg)
    except httpx.TimeoutException:
        total_time = (time.time() - start_time) * 1000
        
        # Simple timeout handling
        timeout_msg = f"Request to Ollama timed out after {total_time:.2f}ms. Try asking a shorter, more specific question."
        
        logger.error(f"⏱️ [TIMING] Timeout after {total_time:.2f}ms")
        
        # Add specific suggestions based on context
        if request.context:
            timeout_msg += " For document analysis, try asking about just 1-2 specific words or concepts."
        else:
            timeout_msg += " Try questions like 'What is X?' or 'Define Y'."
            
        raise HTTPException(status_code=504, detail=timeout_msg)
    except Exception as e:
        total_time = (time.time() - start_time) * 1000
        error_msg = f"Unexpected error after {total_time:.2f}ms: {str(e)}"
        logger.error(f"❌ [TIMING] {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

async def extract_text_from_document(file_path: str) -> str:
    """Ekstrak teks dan deskripsi konten dari file dokumen (PDF atau DOCX) dengan support gambar."""
    try:
        # Log untuk debugging
        logger.info(f"Extracting text from document: {file_path}")
        
        # Deteksi tipe file dengan python-magic untuk akurasi lebih baik
        try:
            file_type = magic.from_file(file_path, mime=True)
            logger.info(f"Detected MIME type: {file_type}")
        except:
            # Fallback ke ekstensi file
            file_extension = os.path.splitext(file_path)[1].lower()
            logger.info(f"Fallback to file extension: {file_extension}")
            file_type = None
        
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf' or (file_type and 'pdf' in file_type):
            return await extract_text_from_pdf_advanced(file_path)
        elif file_extension == '.docx' or (file_type and 'wordprocessingml' in file_type):
            return await extract_text_from_docx_advanced(file_path)
        else:
            logger.error(f"Format file tidak didukung: {file_extension}")
            return f"Format file tidak didukung: {file_extension}"
    except Exception as e:
        logger.error(f"Error saat mengekstrak teks dari dokumen: {e}", exc_info=True)
        return f"Error saat mengekstrak teks: {str(e)}"

async def extract_text_from_docx_advanced(file_path: str) -> str:
    """Ekstrak teks dari file DOCX dengan support lengkap untuk semua elemen."""
    try:
        logger.info("Starting advanced DOCX extraction...")
        
        # Gunakan python-docx untuk ekstraksi yang lebih lengkap
        doc = Document(file_path)
        
        extracted_content = []
        image_count = 0
        table_count = 0
        
        # Header
        extracted_content.append("=== DOKUMEN WORD ===\n")
        
        # Ekstrak paragraf dengan formatting info
        for i, paragraph in enumerate(doc.paragraphs):
            text = paragraph.text.strip()
            if text:
                # Deteksi style/heading
                if paragraph.style.name.startswith('Heading'):
                    extracted_content.append(f"\n## {text}")
                elif paragraph.style.name == 'Title':
                    extracted_content.append(f"\n# {text}")
                else:
                    extracted_content.append(text)
        
        # Ekstrak tabel
        for table_idx, table in enumerate(doc.tables):
            table_count += 1
            extracted_content.append(f"\n[TABEL {table_count}]")
            
            for row_idx, row in enumerate(table.rows):
                row_data = []
                for cell in row.cells:
                    cell_text = cell.text.strip().replace('\n', ' ')
                    row_data.append(cell_text)
                
                if any(row_data):  # Hanya tambahkan jika ada data
                    extracted_content.append(" | ".join(row_data))
        
        # Ekstrak gambar dan beri deskripsi
        for rel in doc.part.rels.values():
            if "image" in rel.target_ref:
                image_count += 1
                extracted_content.append(f"\n[GAMBAR {image_count}] - Gambar ditemukan dalam dokumen")
        
        # Ekstrak properties dokumen
        if doc.core_properties.title:
            extracted_content.insert(1, f"Judul: {doc.core_properties.title}")
        if doc.core_properties.author:
            extracted_content.insert(2, f"Penulis: {doc.core_properties.author}")
        if doc.core_properties.subject:
            extracted_content.insert(3, f"Subjek: {doc.core_properties.subject}")
        
        # Tambahkan ringkasan konten
        summary = f"\n=== RINGKASAN DOKUMEN ===\n"
        summary += f"- Total paragraf: {len([p for p in doc.paragraphs if p.text.strip()])}\n"
        summary += f"- Total tabel: {table_count}\n"
        summary += f"- Total gambar: {image_count}\n"
        
        extracted_content.append(summary)
        
        result = "\n".join(extracted_content)
        logger.info(f"Advanced DOCX extraction completed: {len(result)} characters")
        return result
        
    except Exception as e:
        logger.error(f"Error dalam ekstraksi DOCX advanced: {e}")
        # Fallback ke docx2txt
        try:
            logger.info("Falling back to basic docx2txt extraction...")
            text = docx2txt.process(file_path)
            return f"=== DOKUMEN WORD (MODE SEDERHANA) ===\n{text}"
        except Exception as fallback_error:
            logger.error(f"Fallback juga gagal: {fallback_error}")
            return f"Error saat membaca DOCX: {str(e)}"

async def extract_text_from_pdf_advanced(file_path: str) -> str:
    """Ekstrak teks dari file PDF dengan PyMuPDF untuk hasil yang lebih baik."""
    try:
        logger.info("Starting advanced PDF extraction with PyMuPDF...")
        
        # Buka PDF dengan PyMuPDF
        pdf_document = fitz.open(file_path)
        extracted_content = []
        total_images = 0
        
        # Header
        extracted_content.append("=== DOKUMEN PDF ===\n")
        
        # Ekstrak metadata
        metadata = pdf_document.metadata
        if metadata.get('title'):
            extracted_content.append(f"Judul: {metadata['title']}")
        if metadata.get('author'):
            extracted_content.append(f"Penulis: {metadata['author']}")
        if metadata.get('subject'):
            extracted_content.append(f"Subjek: {metadata['subject']}")
        
        extracted_content.append(f"Total halaman: {len(pdf_document)}\n")
        
        # Ekstrak teks per halaman
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            
            # Ekstrak teks
            text = page.get_text()
            if text.strip():
                extracted_content.append(f"\n--- HALAMAN {page_num + 1} ---")
                extracted_content.append(text.strip())
            
            # Hitung gambar di halaman ini
            image_list = page.get_images()
            if image_list:
                page_images = len(image_list)
                total_images += page_images
                extracted_content.append(f"[{page_images} gambar ditemukan di halaman {page_num + 1}]")
        
        # Tambahkan ringkasan
        summary = f"\n=== RINGKASAN PDF ===\n"
        summary += f"- Total halaman: {len(pdf_document)}\n"
        summary += f"- Total gambar: {total_images}\n"
        
        extracted_content.append(summary)
        
        pdf_document.close()
        
        result = "\n".join(extracted_content)
        logger.info(f"Advanced PDF extraction completed: {len(result)} characters")
        return result
        
    except Exception as e:
        logger.error(f"Error dalam ekstraksi PDF advanced: {e}")
        # Fallback ke PyPDF4
        try:
            logger.info("Falling back to PyPDF4 extraction...")
            return extract_text_from_pdf(file_path)
        except Exception as fallback_error:
            logger.error(f"Fallback PDF juga gagal: {fallback_error}")
            return f"Error saat membaca PDF: {str(e)}"

def extract_text_from_pdf(file_path: str) -> str:
    """Ekstrak teks dari file PDF."""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            # Menggunakan PyPDF4 yang pure Python
            pdf_reader = PyPDF4.PdfFileReader(file)
            for page_num in range(pdf_reader.getNumPages()):
                page = pdf_reader.getPage(page_num)
                text += page.extractText() + "\n"
        return text
    except Exception as e:
        logger.error(f"Error saat mengekstrak teks dari PDF: {e}")
        return f"Error saat membaca PDF: {str(e)}"

def extract_text_from_docx(file_path: str) -> str:
    """Ekstrak teks dari file DOCX (fungsi fallback sederhana)."""
    try:
        text = docx2txt.process(file_path)
        return f"=== DOKUMEN WORD (MODE SEDERHANA) ===\n{text}" if text else "Dokumen kosong atau tidak dapat dibaca."
    except Exception as e:
        logger.error(f"Error saat mengekstrak teks dari DOCX: {e}")
        return f"Error saat membaca DOCX: {str(e)}"

# Endpoint untuk mengunggah dan memproses dokumen
@app.post("/api/upload_document", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload dan proses dokumen (PDF atau DOCX)."""
    logger.info(f"Menerima permintaan upload file: {file.filename}")
    
    try:
        # Validasi tipe file dengan deteksi yang lebih akurat
        if not file.filename:
            logger.error("Filename is empty")
            raise HTTPException(
                status_code=400, 
                detail="Filename tidak boleh kosong"
            )
            
        file_extension = os.path.splitext(file.filename)[1].lower()
        logger.info(f"File extension: {file_extension}")
        
        # Daftar ekstensi yang didukung dengan kemampuan baru
        supported_extensions = ['.pdf', '.docx']
        supported_description = {
            '.pdf': 'PDF dengan support gambar dan metadata',
            '.docx': 'Word Document dengan support tabel, gambar, dan formatting'
        }
        
        if file_extension not in supported_extensions:
            logger.error(f"Unsupported file format: {file_extension}")
            supported_list = ", ".join([f"{ext} ({supported_description[ext]})" for ext in supported_extensions])
            raise HTTPException(
                status_code=400,
                detail=f"Format file tidak didukung. Format yang didukung: {supported_list}"
            )
        
        # Generate ID unik untuk dokumen
        document_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_FOLDER, f"{document_id}{file_extension}")
        logger.info(f"Saving file to: {file_path}")
        
        # Pastikan direktori upload ada
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Simpan file
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info(f"File saved successfully: {file_path}")
        except Exception as save_error:
            logger.error(f"Error saving file: {save_error}", exc_info=True)
            raise HTTPException(
                status_code=500, 
                detail=f"Gagal menyimpan file: {str(save_error)}"
            )
        
        # Ekstrak teks dari dokumen dengan metode advanced
        logger.info("Extracting text from document with advanced method...")
        document_text = await extract_text_from_document(file_path)
        logger.info(f"Text extraction completed, length: {len(document_text)} chars")
        
        # Analisis tambahan struktur dokumen
        analysis_info = ""
        try:
            if file_extension == '.docx':
                analysis = await analyze_docx_structure(file_path)
                analysis_info = f"\n=== INFO DOKUMEN ===\nParagraf: {analysis.get('paragraphs', 0)}, Tabel: {analysis.get('tables', 0)}, Gambar: {analysis.get('images', 0)}, Heading: {analysis.get('headings', 0)}\n"
            elif file_extension == '.pdf':
                analysis = await analyze_pdf_structure(file_path)
                analysis_info = f"\n=== INFO DOKUMEN ===\nHalaman: {analysis.get('pages', 0)}, Gambar: {analysis.get('images', 0)}, Panjang teks: {analysis.get('text_length', 0)} karakter\n"
        except Exception as analysis_error:
            logger.warning(f"Analysis failed but text extraction succeeded: {analysis_error}")
        
        # Gabungkan teks dokumen dengan info analisis
        full_content = document_text + analysis_info

        response = DocumentResponse(
            document_id=document_id,
            content=full_content,  # Return full content with analysis
            filename=file.filename
        )
        
        logger.info(f"Document processed successfully: {document_id}")
        
        # Add document metadata to library
        document_metadata = DocumentMetadata(
            document_id=document_id,
            filename=f"{document_id}{file_extension}",
            original_filename=file.filename,
            upload_date=datetime.now().isoformat(),
            file_size=os.path.getsize(file_path),
            file_type=file_extension,
            content_preview=full_content[:200],  # First 200 chars as preview
            analysis_summary={},  # Empty summary for now
            is_active=True  # Set as active document
        )
        
        document_library.add_document(document_metadata)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saat mengunggah dokumen: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error saat mengunggah dokumen: {str(e)}")

# Document Library API Endpoints

@app.get("/api/documents")
async def get_document_library():
    """Get all documents in the library"""
    try:
        documents = document_library.get_all_documents()
        
        # Convert to response format
        response_docs = []
        for doc in documents:
            response_docs.append({
                "document_id": doc.document_id,
                "filename": doc.original_filename,
                "upload_date": doc.upload_date,
                "file_size": doc.file_size,
                "file_type": doc.file_type,
                "content_preview": doc.content_preview,
                "is_active": doc.is_active,
                "analysis_summary": doc.analysis_summary
            })
        
        # Sort by upload date (newest first)
        response_docs.sort(key=lambda x: x["upload_date"], reverse=True)
        
        return {
            "documents": response_docs,
            "total_count": len(response_docs),
            "active_document": next((doc for doc in response_docs if doc["is_active"]), None)
        }
        
    except Exception as e:
        logger.error(f"Error getting document library: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting documents: {str(e)}")

@app.get("/api/documents/{document_id}")
async def get_document_details(document_id: str):
    """Get detailed information about a specific document"""
    try:
        doc = document_library.get_document(document_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get full content
        file_path = os.path.join(UPLOAD_FOLDER, doc.filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Document file not found")
        
        # Extract full content again (cached version)
        full_content = await extract_text_from_document(file_path)
        
        return {
            "document_id": doc.document_id,
            "filename": doc.original_filename,
            "upload_date": doc.upload_date,
            "file_size": doc.file_size,
            "file_type": doc.file_type,
            "content": full_content,
            "content_preview": doc.content_preview,
            "is_active": doc.is_active,
            "analysis_summary": doc.analysis_summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document details: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting document: {str(e)}")

@app.post("/api/documents/{document_id}/select")
async def select_document(document_id: str):
    """Select a document as the active document for chat"""
    try:
        success = document_library.set_active_document(document_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get the newly active document
        active_doc = document_library.get_active_document()
        
        return {
            "success": True,
            "message": f"Document '{active_doc.original_filename}' selected successfully",
            "active_document": {
                "document_id": active_doc.document_id,
                "filename": active_doc.original_filename,
                "file_type": active_doc.file_type,
                "upload_date": active_doc.upload_date,
                "content_preview": active_doc.content_preview
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error selecting document: {e}")
        raise HTTPException(status_code=500, detail=f"Error selecting document: {str(e)}")

@app.delete("/api/documents/{document_id}")
async def delete_document_from_library(document_id: str):
    """Delete a document from the library"""
    try:
        # Get document info before deletion
        doc = document_library.get_document(document_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        success = document_library.delete_document(document_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete document")
        
        return {
            "success": True,
            "message": f"Document '{doc.original_filename}' deleted successfully",
            "deleted_document_id": document_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@app.post("/api/documents/clear-selection")
async def clear_document_selection():
    """Clear active document selection"""
    try:
        documents = document_library.get_all_documents()
        
        # Set all documents as inactive
        for doc in documents:
            doc.is_active = False
        
        document_library.save_metadata(documents)
        
        return {
            "success": True,
            "message": "Document selection cleared"
        }
        
    except Exception as e:
        logger.error(f"Error clearing document selection: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing selection: {str(e)}")

@app.get("/api/documents/active")
async def get_active_document():
    """Get the currently active document"""
    try:
        active_doc = document_library.get_active_document()
        
        if not active_doc:
            return {
                "active_document": None,
                "message": "No active document selected"
            }
        
        # Get full content
        file_path = os.path.join(UPLOAD_FOLDER, active_doc.filename)
        if not os.path.exists(file_path):
            return {
                "active_document": None,
                "message": "Active document file not found"
            }
        
        # Extract full content again (cached version)
        full_content = await extract_text_from_document(file_path)
        
        return {
            "active_document": {
                "document_id": active_doc.document_id,
                "filename": active_doc.original_filename,
                "file_type": active_doc.file_type,
                "upload_date": active_doc.upload_date,
                "content_preview": active_doc.content_preview,
                "content": full_content
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting active document: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting active document: {str(e)}")

# Simplified Response Processing Classes
class ResponseCache:
    """Simple in-memory cache for responses"""
    
    def __init__(self, max_size: int = 100, ttl_minutes: int = 30):
        self.cache: Dict[str, Dict] = {}
        self.max_size = max_size
        self.ttl = timedelta(minutes=ttl_minutes)
    
    def _generate_key(self, prompt: str, context: str = None) -> str:
        """Generate cache key"""
        content = f"{prompt}:{context or ''}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, prompt: str, context: str = None) -> Optional[str]:
        """Get cached response"""
        key = self._generate_key(prompt, context)
        
        if key in self.cache:
            entry = self.cache[key]
            if datetime.now() - entry['timestamp'] < self.ttl:
                return entry['response']
            else:
                del self.cache[key]
        
        return None
    
    def set(self, prompt: str, response: str, context: str = None):
        """Cache a response"""
        key = self._generate_key(prompt, context)
        
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]
        
        self.cache[key] = {
            'response': response,
            'timestamp': datetime.now()
        }
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()

class ResponseMonitor:
    """Simple response quality monitor"""
    
    def clean_response(self, text: str) -> str:
        """Basic response cleaning"""
        if not text or text.strip() == "":
            return "Maaf, tidak ada jawaban yang dapat saya berikan."
        
        cleaned_text = text.strip()
        
        # Limit length if extremely long
        if len(cleaned_text) > 2000:
            cleaned_text = cleaned_text[:2000] + "..."
        
        return cleaned_text
