from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import httpx
import logging
import json
import asyncio
import os
import shutil
import uuid
from typing import AsyncGenerator, Optional, List, Dict, Any
import tempfile
from dataclasses import dataclass
from enum import Enum
import docx2txt  # Alternatif untuk python-docx, lebih mudah diinstal
# Alternatif untuk PyPDF2 yang pure Python
import PyPDF4  # Library pure Python untuk PDF
import hashlib
from datetime import datetime, timedelta
import time

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

# AI Model Configuration System
class ModelType(Enum):
    LLAMA3_8B = "llama3:8b"
    LLAMA3_LATEST = "llama3:latest"
    LLAMA3_70B = "llama3:70b"
    MISTRAL = "mistral:latest"
    CODELLAMA = "codellama:latest"

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
    """Simplified AI Model optimizer - single balanced configuration"""
    
    def __init__(self):
        # Single optimized configuration for llama3:8b
        self.config = ModelConfig(
            model_name="llama3:8b",
            temperature=0.7,        # Good balance of creativity and consistency
            top_p=0.9,             # Flexible sampling
            top_k=50,              # Moderate token choices
            num_ctx=4096,          # Standard context window
            num_predict=400,       # Medium-length responses
            repeat_penalty=1.1,    # Light repetition penalty
            num_thread=-1,         # Use all available threads
            stop_tokens=["\n\n\n", "Human:", "Assistant:", "PERTANYAAN:", "JAWABAN:", "User:", "Dokai:"]
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
                
                # Performance optimizations
                "num_gpu": 1,
                "low_vram": False,
                "f16_kv": True,
                "num_batch": 256,    # Balanced batch size
                "numa": False,       # Disable NUMA for speed
                "mlock": True,       # Lock model in memory
                "seed": -1,          # Random seed for variety
                "num_gqa": -1        # Use default optimal setting
            }
        }

# Initialize optimizer
ai_optimizer = AIModelOptimizer()

async def stream_response(response: httpx.Response) -> AsyncGenerator[bytes, None]:
    try:
        logger.info("Starting to stream response...")
        
        full_response = ""
        last_sent_length = 0
        loop_detected = False
        
        # Process the response line by line
        async for line in response.aiter_lines():
            if not line:
                continue
                
            try:
                data = json.loads(line)
                
                # If this is a response chunk, process it properly
                if "response" in data:
                    chunk = data["response"]
                    if chunk.strip():  # Only process non-empty responses
                        full_response += chunk
                        
                        # Check for loops in the accumulated response
                        if response_monitor.is_response_looping(full_response):
                            logger.warning("Loop detected in response, stopping generation")
                            loop_detected = True
                            # Send error message
                            yield json.dumps({
                                "error": "Response loop detected. Please try again with a different question."
                            }).encode() + b"\n"
                            break
                        
                        # Only send new content (not the full accumulated response)
                        new_content = full_response[last_sent_length:]
                        if new_content.strip():
                            yield json.dumps({"response": new_content}).encode() + b"\n"
                            last_sent_length = len(full_response)
                
                # If this is an error, report it
                elif "error" in data:
                    logger.error(f"Ollama error: {data['error']}")
                    yield json.dumps({"error": data['error']}).encode() + b"\n"
                    
                # If this is a completion signal, log it
                elif "done" in data and data.get("done"):
                    if not loop_detected:
                        # Clean the final response
                        cleaned_response = response_monitor.clean_response(full_response)
                        if cleaned_response != full_response:
                            logger.info("Response was cleaned for quality")
                        logger.info("Response generation complete")
                        logger.info(f"Total response length: {len(full_response)} characters")
                    break
                    
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Error in stream_response: {e}")
        yield json.dumps({"error": str(e)}).encode() + b"\n"

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

        # Enhanced prompt engineering with conversation history
        prompt_start = time.time()
        if request.context:
            # Process document with enhanced intelligence and conversation history
            enhanced_context = doc_processor.enhance_context(request.message, request.context)
            optimized_prompt = prompt_engineer.create_document_prompt(
                request.message, 
                enhanced_context,
                request.conversation_history or []
            )
            logger.info(f"Using enhanced document context with {len(enhanced_context)} characters")
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
        cached_response = response_cache.get(request.message, context=request.context)
        cache_time = (time.time() - cache_start) * 1000
        logger.info(f"⚡ [TIMING] Cache check: {cache_time:.2f}ms")
        
        if cached_response:
            total_time = (time.time() - start_time) * 1000
            logger.info(f"🎯 [TIMING] CACHE HIT! Total response time: {total_time:.2f}ms")
            return JSONResponse({
                "response": cached_response,
                "status": "complete",
                "cached": True,
                "timing": {
                    "total_ms": total_time,
                    "source": "cache"
                }
            })
        
        # Update model activity and check if reload needed
        preload_start = time.time()
        model_preloader.update_activity()
        if model_preloader.should_reload():
            await model_preloader.preload_model("llama3:8b")
        preload_time = (time.time() - preload_start) * 1000
        logger.info(f"⚡ [TIMING] Model preload check: {preload_time:.2f}ms")
        
        # Prepare request payload
        payload_start = time.time()
        
        # Set reasonable timeout for llama3:8b (2 minutes)
        timeout_seconds = 120.0
        
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            # Get optimized payload
            request_payload = ai_optimizer.get_optimized_payload(optimized_prompt)
            request_payload = ollama_optimizer.get_optimized_payload(request_payload)
            
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
                response_cache.set(request.message, cleaned_response, request.context)
                
                # Monitor performance
                performance_monitor.add_response_time(ollama_time)
                
                processing_time = (time.time() - processing_start) * 1000
                total_time = (time.time() - start_time) * 1000
                
                # Add performance monitoring
                perf_status = performance_monitor.get_performance_status()
                
                logger.info(f"⚡ [TIMING] Response processing: {processing_time:.2f}ms")
                logger.info(f"🎯 [TIMING] TOTAL REQUEST TIME: {total_time:.2f}ms ({total_time/1000:.2f}s)")
                logger.info(f"📊 [PERFORMANCE] Status: {perf_status['status']} - {perf_status['message']}")
                
                if perf_status['status'] in ['slow', 'very_slow']:
                    logger.warning("🐌 [PERFORMANCE] Slow response detected!")
                    for rec in perf_status['recommendations']:
                        logger.warning(f"💡 [RECOMMENDATION] {rec}")
                
                logger.info(f"Final cleaned response length: {len(cleaned_response)} characters")
                
                return JSONResponse({
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
                            "preload_check": preload_time,
                            "payload_prep": payload_time,
                            "ollama_request": ollama_time,
                            "response_processing": processing_time
                        }
                    }
                })
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
    """Ekstrak teks dari file dokumen (PDF atau DOCX)."""
    try:
        # Log untuk debugging
        logger.info(f"Extracting text from document: {file_path}")
        file_extension = os.path.splitext(file_path)[1].lower()
        logger.info(f"File extension: {file_extension}")
        
        if file_extension == '.pdf':
            return extract_text_from_pdf(file_path)
        elif file_extension == '.docx':
            return extract_text_from_docx(file_path)
        else:
            logger.error(f"Format file tidak didukung: {file_extension}")
            return "Format file tidak didukung."
    except Exception as e:
        logger.error(f"Error saat mengekstrak teks dari dokumen: {e}", exc_info=True)
        return f"Error saat mengekstrak teks: {str(e)}"

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
    """Ekstrak teks dari file DOCX."""
    try:
        text = docx2txt.process(file_path)
        return text
    except Exception as e:
        logger.error(f"Error saat mengekstrak teks dari DOCX: {e}")
        return f"Error saat membaca DOCX: {str(e)}"

# Endpoint untuk mengunggah dan memproses dokumen
@app.post("/api/upload_document", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload dan proses dokumen (PDF atau DOCX)."""
    logger.info(f"Menerima permintaan upload file: {file.filename}")
    
    try:
        # Validasi tipe file
        if not file.filename:
            logger.error("Filename is empty")
            raise HTTPException(
                status_code=400, 
                detail="Filename tidak boleh kosong"
            )
            
        file_extension = os.path.splitext(file.filename)[1].lower()
        logger.info(f"File extension: {file_extension}")
        
        if file_extension not in ['.pdf', '.docx']:
            logger.error(f"Unsupported file format: {file_extension}")
            raise HTTPException(
                status_code=400,
                detail="Format file tidak didukung. Hanya PDF dan DOCX yang diizinkan."
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
        
        # Ekstrak teks dari dokumen
        logger.info("Extracting text from document...")
        document_text = await extract_text_from_document(file_path)
        logger.info(f"Text extraction completed, length: {len(document_text)} chars")
        
        response = DocumentResponse(
            document_id=document_id,
            content=document_text,  # Return full content instead of truncated
            filename=file.filename
        )
        
        logger.info(f"Document processed successfully: {document_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saat mengunggah dokumen: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error saat mengunggah dokumen: {str(e)}")

# Endpoint untuk memeriksa status server
@app.get("/api/status")
async def get_server_status():
    """Memeriksa status server dan dukungan format dokumen."""
    try:
        # Periksa dukungan format dokumen
        has_pdf_support = True
        has_docx_support = True
        requirements_installed = True
        
        try:
            # Cek dukungan PDF dengan PyPDF4
            try:
                import PyPDF4
            except ImportError:
                try:
                    # Fallback ke pdfplumber jika ada
                    import pdfplumber
                except ImportError:
                    try:
                        # Fallback ke PyPDF2 jika ada
                        import PyPDF2
                    except ImportError:
                        has_pdf_support = False
                        requirements_installed = False
            
        except ImportError:
            has_pdf_support = False
            requirements_installed = False
            
        try:
            # Cek dukungan DOCX dengan docx2txt
            try:
                import docx2txt
            except ImportError:
                try:
                    import docx
                except ImportError:
                    has_docx_support = False
                    requirements_installed = False
        except ImportError:
            has_docx_support = False
            requirements_installed = False
            
        return {
            "running": True,
            "version": "1.0.0",
            "message": "Server berjalan dengan baik",
            "hasPdfSupport": has_pdf_support,
            "hasDocxSupport": has_docx_support,
            "requirementsInstalled": requirements_installed
        }
    except Exception as e:
        logger.error(f"Error saat memeriksa status server: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "running": True,
                "version": "1.0.0",
                "message": f"Error: {str(e)}",
                "hasPdfSupport": False,
                "hasDocxSupport": False,
                "requirementsInstalled": False
            }
        )

@app.post("/api/install_dependencies")
async def install_dependencies():
    """Endpoint untuk menginstall dependensi yang diperlukan."""
    try:
        # Pada kasus nyata, sebaiknya gunakan subprocess untuk menginstall dependensi
        # Namun untuk keamanan, sebaiknya fungsi ini diimplementasikan dengan hati-hati
        # Di contoh ini kita hanya akan mengembalikan status palsu
        
        return {
            "success": True,
            "message": "Dependensi berhasil diinstall (simulasi)"
        }
    except Exception as e:
        logger.error(f"Error saat menginstall dependensi: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Error: {str(e)}"
            }
        )

# AI Model Management Endpoints

@app.get("/api/models/available")
async def get_available_models():
    """Get list of available AI models from Ollama"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models = response.json()
                return {
                    "success": True,
                    "models": models.get("models", []),
                    "recommended": [
                        {"name": "llama3:latest", "size": "4.7GB", "description": "Best general purpose model"},
                        {"name": "llama3:8b", "size": "4.7GB", "description": "Balanced performance"},
                        {"name": "mistral:latest", "size": "4.1GB", "description": "Fast and efficient"}
                    ]
                }
            else:
                return {"success": False, "error": "Failed to fetch models from Ollama"}
    except Exception as e:
        logger.error(f"Error fetching available models: {e}")
        return {"success": False, "error": str(e)}

@app.get("/api/models/performance")
async def get_model_performance():
    """Get current model performance metrics"""
    return {
        "current_config": {
            "model": "llama3:8b",
            "modes": {
                "fast": "Quick responses, 2K context",
                "balanced": "Balanced quality, 4K context", 
                "quality": "Best quality, 8K context"
            }
        },
        "performance_tips": [
            "Use 'fast' mode for quick Q&A",
            "Use 'balanced' for general chat",
            "Use 'quality' for complex document analysis",
            "Ensure adequate RAM (8GB+ recommended)",
            "GPU acceleration improves speed significantly"
        ]
    }

@app.post("/api/models/optimize")
async def optimize_model_settings(settings: dict):
    """Optimize model settings based on user hardware"""
    try:
        # This would analyze system specs and suggest optimal settings
        ram_gb = settings.get("ram_gb", 8)
        has_gpu = settings.get("has_gpu", False)
        cpu_cores = settings.get("cpu_cores", 4)
        
        if ram_gb < 8:
            recommendation = "fast"
            message = "Limited RAM detected. Using fast mode for better performance."
        elif ram_gb >= 16 and has_gpu:
            recommendation = "quality"
            message = "High-end system detected. Quality mode recommended."
        else:
            recommendation = "balanced"
            message = "Balanced configuration for your system."
            
        return {
            "success": True,
            "recommended_mode": recommendation,
            "message": message,
            "optimizations": {
                "num_thread": min(cpu_cores, 8),
                "use_gpu": has_gpu,
                "batch_size": 1024 if ram_gb >= 16 else 512
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# Document Intelligence System
class DocumentProcessor:
    """Advanced document processing for better context understanding"""
    
    def __init__(self):
        self.chunk_size = 1000
        self.overlap = 200
    
    def split_document(self, text: str) -> List[str]:
        """Split document into overlapping chunks for better context"""
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.overlap):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())
        return chunks
    
    def find_relevant_chunks(self, query: str, chunks: List[str], max_chunks: int = 3) -> List[str]:
        """Find most relevant chunks based on keyword matching"""
        query_words = set(query.lower().split())
        chunk_scores = []
        
        for chunk in chunks:
            chunk_words = set(chunk.lower().split())
            score = len(query_words.intersection(chunk_words))
            chunk_scores.append((chunk, score))
        
        # Sort by relevance score
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, score in chunk_scores[:max_chunks] if score > 0]
    
    def enhance_context(self, query: str, full_context: str) -> str:
        """Create enhanced context with relevant chunks (balanced optimization)"""
        
        # Balanced context limits for optimal performance
        max_context_length = 2500  # Reasonable context size
        max_chunks = 4             # Good balance of coverage
        
        if len(full_context) <= max_context_length:
            return full_context
        
        chunks = self.split_document(full_context)
        relevant_chunks = self.find_relevant_chunks(query, chunks, max_chunks=max_chunks)
        
        if not relevant_chunks:
            # If no relevant chunks found, use first chunk only
            relevant_chunks = chunks[:1]
        
        # Join chunks and limit total length
        combined_context = "\n\n".join(relevant_chunks)
        
        if len(combined_context) > max_context_length:
            combined_context = combined_context[:max_context_length] + "..."
            
        return combined_context

# Advanced Prompt Engineering System
class PromptEngineer:
    """Intelligent prompt engineering for better responses"""
    
    def __init__(self):
        self.conversation_history = []
    
    def create_document_prompt(self, query: str, context: str, conversation_history: List[Dict] = None) -> str:
        """Create optimized prompt with conversation history"""
        
        # Balanced context limits for optimal performance
        limited_context = context[:4000]   # Reasonable context size
        conversation_context = ""
        if conversation_history:
            last_exchanges = conversation_history[-3:] if conversation_history else []
            for msg in last_exchanges:
                conversation_context += f"{msg.get('sender', 'User')}: {msg.get('content', '')[:300]}...\n"
        
        # Optimized prompt for document analysis
        prompt = f"""DOKUMEN:
{limited_context}

RIWAYAT PERCAKAPAN:
{conversation_context}

PERTANYAAN: {query}

INSTRUKSI: Analisis dokumen dan berikan jawaban yang informatif dan detail berdasarkan konteks dokumen. Jelaskan dengan lengkap dan jelas.

JAWABAN:"""
        
        return prompt
    
    def create_general_prompt(self, query: str, conversation_history: List[Dict] = None) -> str:
        """Create optimized general conversation prompt with history"""
        
        conversation_context = ""
        if conversation_history:
            # Last 3 exchanges for balanced performance
            last_exchanges = conversation_history[-3:] if conversation_history else []
            for msg in last_exchanges:
                conversation_context += f"{msg.get('sender', 'User')}: {msg.get('content', '')[:150]}...\n"
        
        # Optimized prompt for natural conversation
        prompt = f"""{conversation_context}

Pertanyaan: {query}
Jawaban:"""
        
        return prompt
    
    def _analyze_query_type(self, query: str) -> str:
        """Analyze the type of query to tailor response"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['apa', 'apakah', 'what', 'is']):
            return "DEFINISI/KONSEP"
        elif any(word in query_lower for word in ['bagaimana', 'cara', 'how', 'steps']):
            return "PROSEDUR/TUTORIAL"
        elif any(word in query_lower for word in ['mengapa', 'kenapa', 'why', 'alasan']):
            return "PENJELASAN/ANALISIS"
        elif any(word in query_lower for word in ['kapan', 'when', 'waktu']):
            return "TEMPORAL"
        elif any(word in query_lower for word in ['dimana', 'where', 'lokasi']):
            return "LOKASI/TEMPAT"
        elif any(word in query_lower for word in ['berapa', 'jumlah', 'how many', 'how much']):
            return "KUANTITATIF"
        elif any(word in query_lower for word in ['bandingkan', 'compare', 'versus', 'vs']):
            return "PERBANDINGAN"
        else:
            return "UMUM"

# Initialize processors
doc_processor = DocumentProcessor()
prompt_engineer = PromptEngineer()

# Root endpoint
@app.get("/")
async def read_root():
    return {"message": "Welcome to FastAPI"}

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"An unexpected error occurred: {str(exc)}"}
    )

# Document Intelligence Endpoints

@app.post("/api/document/analyze")
async def analyze_document(document_id: str, query: str):
    """Analyze document content for specific query"""
    try:
        # This would retrieve document from storage
        # For now, return analysis capabilities
        
        analysis = {
            "document_id": document_id,
            "query": query,
            "analysis_type": prompt_engineer._analyze_query_type(query),
            "recommended_mode": "quality" if len(query) > 50 else "balanced",
            "context_strategy": "chunk_based" if len(query) > 20 else "full_context"
        }
        
        return analysis
    except Exception as e:
        logger.error(f"Error analyzing document: {e}")
        return {"error": str(e)}

@app.get("/api/intelligence/stats")
async def get_intelligence_stats():
    """Get current intelligence system statistics"""
    return {
        "document_processor": {
            "chunk_size": doc_processor.chunk_size,
            "overlap": doc_processor.overlap,
            "features": ["smart_chunking", "relevance_scoring", "context_enhancement"]
        },
        "prompt_engineer": {
            "query_types": ["DEFINISI/KONSEP", "PROSEDUR/TUTORIAL", "PENJELASAN/ANALISIS", 
                          "TEMPORAL", "LOKASI/TEMPAT", "KUANTITATIF", "PERBANDINGAN", "UMUM"],
            "modes": ["fast", "balanced", "quality"],
            "features": ["query_analysis", "mode_adaptation", "anti_repetition"]
        },
        "ai_optimizer": {
            "anti_loop_features": ["repeat_penalty", "presence_penalty", "frequency_penalty", 
                                 "mirostat", "enhanced_stop_tokens"],
            "sampling_improvements": ["tfs_z", "typical_p", "optimized_temperature"],
            "performance_features": ["memory_locking", "batch_optimization", "gpu_acceleration"]
        }
    }

# Speed Optimization Endpoints

@app.post("/api/benchmark")
async def benchmark_model():
    """Benchmark AI model performance across different modes"""
    benchmark_results = []
    
    test_queries = [
        "Hello, how are you?",
        "What is artificial intelligence?",
        "Explain machine learning briefly."
    ]
    
    for mode in ["fast", "balanced", "quality"]:
        for query in test_queries:
            start_time = time.time()
            
            try:
                request = ChatRequest(
                    message=query,
                    response_mode=mode
                )
                
                # This would call the actual chat endpoint
                # For benchmark, we'll simulate a simplified version
                result_time = (time.time() - start_time) * 1000
                
                benchmark_results.append({
                    "mode": mode,
                    "query": query,
                    "time_ms": result_time,
                    "status": "simulated"
                })
                
            except Exception as e:
                benchmark_results.append({
                    "mode": mode,
                    "query": query,
                    "time_ms": -1,
                    "error": str(e)
                })
    
    return {
        "benchmark_results": benchmark_results,
        "recommendations": {
            "fastest_mode": "fast",
            "optimal_for_chat": "balanced",
            "best_quality": "quality"
        }
    }

@app.get("/api/performance/stats")
async def get_performance_stats():
    """Get current performance statistics"""
    return {
        "optimizations": {
            "caching": {
                "enabled": True,
                "cache_size": len(response_cache.cache),
                "max_size": response_cache.max_size,
                "ttl_minutes": response_cache.ttl.total_seconds() / 60
            },
            "model_preloading": {
                "enabled": True,
                "model_loaded": model_preloader.model_loaded,
                "last_activity": model_preloader.last_activity.isoformat()
            },
            "ai_settings": {
                "temperature": ai_optimizer.config.temperature,
                "max_tokens": ai_optimizer.config.num_predict,
                "context_size": ai_optimizer.config.num_ctx,
                "top_k": ai_optimizer.config.top_k
            }
        },
        "tips": [
            "llama3:8b model is optimized for balanced speed and quality",
            "Cache will speed up repeated questions",
            "Model preloading reduces first-request latency",
            "Smaller documents and shorter questions process faster"
        ]
    }

@app.get("/api/performance/monitor")
async def get_performance_monitor():
    """Get current performance monitoring data and recommendations"""
    return performance_monitor.get_performance_status()

@app.post("/api/performance/reset")
async def reset_performance_monitor():
    """Reset performance monitoring data"""
    performance_monitor.response_times.clear()
    return {"message": "Performance monitor reset successfully"}

# Performance Monitor for detecting slow Ollama responses
class PerformanceMonitor:
    """Monitor Ollama performance and suggest optimizations"""
    
    def __init__(self):
        self.response_times = []
        self.slow_response_threshold = 30000  # 30 seconds
        self.very_slow_threshold = 60000      # 1 minute
        
    def add_response_time(self, response_time_ms: float):
        """Add a response time measurement"""
        self.response_times.append({
            'time': response_time_ms,
            'timestamp': datetime.now()
        })
        
        # Keep only last 10 measurements
        if len(self.response_times) > 10:
            self.response_times.pop(0)
    
    def is_performance_degraded(self) -> bool:
        """Check if performance is consistently poor"""
        if len(self.response_times) < 3:
            return False
            
        recent_times = [r['time'] for r in self.response_times[-3:]]
        return all(t > self.slow_response_threshold for t in recent_times)
    
    def get_performance_status(self) -> Dict[str, Any]:
        """Get current performance status and recommendations"""
        if not self.response_times:
            return {"status": "no_data", "message": "No performance data yet"}
        
        latest = self.response_times[-1]
        avg_time = sum(r['time'] for r in self.response_times) / len(self.response_times)
        
        if latest['time'] < 5000:  # Under 5 seconds
            status = "excellent"
            message = f"Great performance! Last response: {latest['time']:.0f}ms"
            recommendations = ["Continue using current settings"]
        elif latest['time'] < 15000:  # Under 15 seconds
            status = "good"
            message = f"Good performance. Last response: {latest['time']:.0f}ms"
            recommendations = ["Performance is acceptable", "Consider fast mode for quicker responses"]
        elif latest['time'] < 45000:  # Under 45 seconds
            status = "slow"
            message = f"Slow performance detected. Last response: {latest['time']:.0f}ms"
            recommendations = [
                "Try using fast mode",
                "Restart Ollama service if problem persists",
                "Check system resources (RAM/CPU)"
            ]
        else:  # Over 45 seconds
            status = "very_slow"
            message = f"Very slow performance! Last response: {latest['time']:.0f}ms"
            recommendations = [
                "URGENT: Restart Ollama service immediately",
                "Check for multiple Ollama processes: tasklist | findstr ollama",
                "Consider using a smaller model like phi3:mini",
                "Check system RAM usage",
                "Use shorter questions and smaller documents"
            ]
        
        return {
            "status": status,
            "message": message,
            "latest_time_ms": latest['time'],
            "average_time_ms": avg_time,
            "recommendations": recommendations,
            "degraded": self.is_performance_degraded()
        }

# Initialize performance monitor
performance_monitor = PerformanceMonitor()

# AI Response Caching System
class ResponseCache:
    """Simple in-memory cache for AI responses to improve speed"""
    
    def __init__(self, max_size: int = 100, ttl_minutes: int = 30):
        self.cache: Dict[str, Dict] = {}
        self.max_size = max_size
        self.ttl = timedelta(minutes=ttl_minutes)
    
    def _generate_key(self, prompt: str, context: str = None) -> str:
        """Generate cache key from prompt and context"""
        content = f"{prompt}:{context or ''}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, prompt: str, context: str = None) -> Optional[str]:
        """Get cached response if available and not expired"""
        key = self._generate_key(prompt, context)
        
        if key in self.cache:
            entry = self.cache[key]
            if datetime.now() - entry['timestamp'] < self.ttl:
                logger.info(f"Cache hit for prompt: {prompt[:50]}...")
                return entry['response']
            else:
                # Remove expired entry
                del self.cache[key]
        
        return None
    
    def set(self, prompt: str, response: str, context: str = None):
        """Cache a response"""
        key = self._generate_key(prompt, context)
        
        # Remove oldest entries if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]
        
        self.cache[key] = {
            'response': response,
            'timestamp': datetime.now()
        }
        logger.info(f"Cached response for prompt: {prompt[:50]}...")
    
    def clear(self):
        """Clear all cached responses"""
        self.cache.clear()
        logger.info("Cache cleared")

# Response Quality Monitor
class ResponseMonitor:
    """Monitor and filter AI responses for quality"""
    
    def __init__(self):
        self.repetition_threshold = 3
        self.max_response_length = 1000
    
    def is_response_looping(self, text: str) -> bool:
        """Detect if response is in a loop"""
        if not text:
            return False
        
        # Check for common loop patterns
        loop_patterns = [
            "HaiHaiiii", "IsIsiIsi", "dokdokumen", 
            "baik-baik-baik", "Haiii,Haiii"
        ]
        
        for pattern in loop_patterns:
            if pattern in text:
                return True
                
        # Check for character repetition
        if len(set(text[:20])) < 3:  # If first 20 chars have less than 3 unique chars
            return True
            
        # Check for excessive repetition
        words = text.split()
        if len(words) > 5:
            first_word = words[0]
            if text.count(first_word) > len(words) // 2:
                return True
                
        return False
    
    def clean_response(self, text: str) -> str:
        """Clean and format response (less aggressive)"""
        if not text or text.strip() == "":
            return "Maaf, saya tidak dapat memberikan jawaban yang tepat. Silakan coba pertanyaan yang berbeda."
            
        # Remove loop patterns (only extreme cases)
        loop_patterns = [
            "HaiHaiiii", "IsIsiIsi", "Haiii,Haiii"
        ]
        
        cleaned_text = text.strip()
        
        # Only clean if there are obvious loop patterns
        for pattern in loop_patterns:
            if pattern in cleaned_text:
                # If loop detected, return error message
                return "Maaf, terjadi kesalahan dalam pemrosesan. Silakan coba lagi dengan pertanyaan yang berbeda."
        
        # Basic cleaning only
        cleaned_text = cleaned_text.strip()
        
        # Limit length only if extremely long
        if len(cleaned_text) > 2000:  # Increased from 1000
            cleaned_text = cleaned_text[:2000] + "..."
            
        return cleaned_text if cleaned_text else "Maaf, tidak ada jawaban yang dapat saya berikan."

# Model Preloading System for Speed Optimization
class ModelPreloader:
    """Preload and warm up AI models for faster response times"""
    
    def __init__(self):
        self.model_loaded = False
        self.last_activity = datetime.now()
    
    async def preload_model(self, model_name: str = "llama3:8b"):
        """Preload the model to reduce first-request latency"""
        try:
            logger.info(f"Preloading model: {model_name}")
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                # Send a simple prompt to load the model into memory
                warmup_payload = {
                    "model": model_name,
                    "prompt": "Hello",
                    "stream": False,
                    "options": {
                        "num_predict": 1,
                        "temperature": 0.1
                    }
                }
                
                response = await client.post(
                    "http://localhost:11434/api/generate",
                    json=warmup_payload,
                    timeout=60.0
                )
                
                if response.status_code == 200:
                    self.model_loaded = True
                    self.last_activity = datetime.now()
                    logger.info(f"Model {model_name} preloaded successfully")
                    return True
                else:
                    logger.error(f"Failed to preload model: {response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error preloading model: {e}")
            return False
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()
    
    def should_reload(self, minutes_threshold: int = 30) -> bool:
        """Check if model should be reloaded due to inactivity"""
        return (datetime.now() - self.last_activity).total_seconds() > (minutes_threshold * 60)

# Ollama Connection Optimization
class OllamaOptimizer:
    """Optimize Ollama connections for speed"""
    
    def __init__(self):
        self.keep_alive_duration = "5m"  # Keep model loaded for 5 minutes
        self.last_used = {}
    
    async def warm_up_model(self, model_name: str = "llama3:8b"):
        """Warm up model with keep-alive"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Send keep-alive request
                payload = {
                    "model": model_name,
                    "prompt": "",
                    "keep_alive": self.keep_alive_duration,
                    "options": {
                        "num_predict": 0
                    }
                }
                
                response = await client.post(
                    "http://localhost:11434/api/generate",
                    json=payload,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    self.last_used[model_name] = datetime.now()
                    logger.info(f"🔥 Model {model_name} warmed up with keep-alive")
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to warm up model: {e}")
            return False
    
    def get_optimized_payload(self, base_payload: dict) -> dict:
        """Add keep-alive and other optimizations to payload"""
        optimized = base_payload.copy()
        
        # Add keep-alive to prevent model unloading
        optimized["keep_alive"] = self.keep_alive_duration
        
        # Simple optimization - balanced stop tokens
        optimized["options"].update({
            "stop": ["\n\n\n", "Human:", "User:", "Assistant:", "PERTANYAAN:", "JAWABAN:"],
        })
        
        return optimized

# Initialize all components
ai_optimizer = AIModelOptimizer()
response_cache = ResponseCache(max_size=200, ttl_minutes=60)  # 1 hour TTL
response_monitor = ResponseMonitor()
model_preloader = ModelPreloader()
ollama_optimizer = OllamaOptimizer()
performance_monitor = PerformanceMonitor()

# Emergency Ollama Management
class OllamaEmergencyManager:
    """Emergency management for stuck Ollama processes"""
    
    def __init__(self):
        self.last_restart = datetime.now() - timedelta(minutes=10)  # Allow immediate restart
        self.restart_cooldown = timedelta(minutes=2)  # 2 minute cooldown between restarts
    
    async def emergency_restart_ollama(self) -> bool:
        """Emergency restart of Ollama service when it's stuck"""
        try:
            # Check cooldown
            if datetime.now() - self.last_restart < self.restart_cooldown:
                logger.warning("Ollama restart on cooldown, skipping")
                return False
            
            logger.warning("🚨 EMERGENCY: Attempting to restart Ollama service...")
            
            # Kill existing Ollama processes
            import subprocess
            try:
                subprocess.run(['taskkill', '/f', '/im', 'ollama.exe'], 
                             capture_output=True, check=False)
                logger.info("Killed existing Ollama processes")
            except Exception as e:
                logger.warning(f"Failed to kill Ollama processes: {e}")
            
            # Wait a moment
            await asyncio.sleep(2)
            
            # Start Ollama service (assuming it's in PATH)
            try:
                subprocess.Popen(['ollama', 'serve'], 
                               creationflags=subprocess.CREATE_NEW_CONSOLE)
                logger.info("Started new Ollama service")
                
                # Wait for service to start
                await asyncio.sleep(5)
                
                # Test if it's responding
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get("http://localhost:11434/api/tags")
                    if response.status_code == 200:
                        logger.info("✅ Ollama restart successful!")
                        self.last_restart = datetime.now()
                        return True
                    else:
                        logger.error("Ollama not responding after restart")
                        return False
                        
            except Exception as e:
                logger.error(f"Failed to start Ollama service: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Emergency restart failed: {e}")
            return False
    
    def can_restart(self) -> bool:
        """Check if we can restart Ollama (not on cooldown)"""
        return datetime.now() - self.last_restart >= self.restart_cooldown

# Initialize emergency manager
ollama_emergency = OllamaEmergencyManager()

# Startup and Shutdown Events
@app.on_event("startup")
async def startup_event():
    """Initialize and preload models on startup"""
    logger.info("🚀 Starting FastAPI application with ultra-speed optimizations...")
    
    # Preload model for faster responses
    await model_preloader.preload_model("llama3:8b")
    
    # Warm up Ollama with keep-alive
    await ollama_optimizer.warm_up_model("llama3:8b")
    
    logger.info("⚡ Application startup complete with speed optimizations")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down FastAPI application...")
    response_cache.clear()

# Entry point untuk menjalankan server
if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting FastAPI AI Chat Backend Server...")
    print("📍 Server akan berjalan di: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔄 Hot reload: Enabled")
    print("⚡ Mode: Development")
    print("\n" + "="*50)
    print("💡 Untuk menghentikan server: Ctrl+C")
    print("="*50 + "\n")
    
    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Server dihentikan oleh user")
    except Exception as e:
        print(f"\n❌ Error saat menjalankan server: {e}")
        print("💡 Pastikan port 8000 tidak digunakan aplikasi lain")
        print("💡 Coba jalankan: netstat -ano | findstr :8000")

# Ollama Health Check and Mode Recommendation
@app.get("/api/ollama/health")
async def check_ollama_health():
    """Check Ollama connectivity and suggest optimal mode based on response time"""
    try:
        start_time = time.time()
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Test with a simple request
            test_payload = {
                "model": "llama3:8b",
                "prompt": "Hi",
                "stream": False,
                "options": {
                    "num_predict": 5,
                    "temperature": 0.1
                }
            }
            
            response = await client.post(
                "http://localhost:11434/api/generate",
                json=test_payload,
                timeout=30.0
            )
            
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                # Suggest mode based on response time
                if response_time < 2000:  # Under 2 seconds
                    suggested_mode = "quality"
                    message = f"Excellent performance ({response_time:.0f}ms)! All modes available."
                elif response_time < 5000:  # Under 5 seconds
                    suggested_mode = "balanced"
                    message = f"Good performance ({response_time:.0f}ms). Balanced or Fast mode recommended."
                else:  # Over 5 seconds
                    suggested_mode = "fast"
                    message = f"Slower performance ({response_time:.0f}ms). Fast mode recommended for better experience."
                
                return {
                    "status": "healthy",
                    "response_time_ms": response_time,
                    "suggested_mode": suggested_mode,
                    "message": message,
                    "recommendations": {
                        "fast": "Use for quick Q&A (3+ min timeout)",
                        "balanced": "Use for normal chat (5+ min timeout)",
                        "quality": "Use for detailed analysis (10+ min timeout)"
                    }
                }
            else:
                return {
                    "status": "error",
                    "error": f"Ollama returned status {response.status_code}",
                    "suggestion": "Check if Ollama is properly running with llama3:8b model"
                }
                
    except httpx.ConnectError:
        return {
            "status": "disconnected",
            "error": "Cannot connect to Ollama",
            "suggestion": "Start Ollama service: 'ollama serve' then 'ollama pull llama3:8b'"
        }
    except httpx.TimeoutException:
        return {
            "status": "timeout",
            "error": "Ollama health check timed out",
            "suggestion": "Ollama is running but very slow. Try restarting Ollama service."
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "suggestion": "Check Ollama installation and model availability"
        }
