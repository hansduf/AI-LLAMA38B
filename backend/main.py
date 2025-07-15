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
            # SAFETY: Limit document context size to prevent timeout
            if len(request.context) > 10000:  # Max 10KB context
                logger.warning(f"⚠️ Large document truncated: {len(request.context)} → 10000 chars")
                request.context = request.context[:10000] + "..."
            
            # Process document with enhanced intelligence and conversation history
            try:
                enhanced_context = doc_processor.enhance_context(request.message, request.context)
                optimized_prompt = prompt_engineer.create_document_prompt(
                    request.message, 
                    enhanced_context,
                    request.conversation_history or []
                )
                logger.info(f"Using enhanced document context with {len(enhanced_context)} characters")
            except Exception as e:
                logger.error(f"Document processing failed: {e}")
                # Fallback to simple context
                enhanced_context = request.context[:1500] + "..." if len(request.context) > 1500 else request.context
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
                response_cache.set(request.message, cleaned_response, request.context)
                
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
            # Cek dukungan PDF
            import PyPDF4
            import fitz  # PyMuPDF
        except ImportError as e:
            has_pdf_support = False
            requirements_installed = False
            logger.warning(f"PDF support libraries missing: {e}")
            
        try:
            # Cek dukungan DOCX
            import docx2txt
            from docx import Document
        except ImportError as e:
            has_docx_support = False
            requirements_installed = False
            logger.warning(f"DOCX support libraries missing: {e}")
            
        # Check Ollama connection
        ollama_status = "disconnected"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get("http://localhost:11434/api/tags")
                if response.status_code == 200:
                    ollama_status = "connected"
                else:
                    ollama_status = "error"
        except:
            ollama_status = "disconnected"
        
        # Get performance status
        performance_status = performance_monitor.get_simple_status()
        
        return {
            "running": True,
            "version": "1.0.0",
            "message": "Server berjalan dengan baik",
            "hasPdfSupport": has_pdf_support,
            "hasDocxSupport": has_docx_support,
            "requirementsInstalled": requirements_installed,
            "ollama_status": ollama_status,
            "performance": performance_status
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

# AI Model Management - Removed unused endpoints

# Document Intelligence System
class DocumentProcessor:
    """Advanced document processing for better context understanding"""
    
    def __init__(self):
        self.chunk_size = 1000
        self.overlap = 200
    
    def split_document(self, text: str) -> List[str]:
        """Split document into overlapping chunks for better context"""
        if not text or len(text.strip()) == 0:
            return []
        
        # Safety limits to prevent infinite loops
        if len(text) > 50000:  # Max 50KB text processing
            text = text[:50000] + "..."
        
        chunks = []
        step_size = max(1, self.chunk_size - self.overlap)  # Ensure step_size is at least 1
        
        for i in range(0, len(text), step_size):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())
            
            # Safety break to prevent infinite loops
            if len(chunks) >= 50:  # Max 50 chunks
                break
                
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
        """Create enhanced context with relevant chunks (GAMING LAPTOP optimization)"""
        
        # GENEROUS limits for gaming laptop with 22GB RAM
        max_context_length = 2000  # Much larger context
        max_chunks = 3             # More chunks for better analysis
        
        # Safety check - prevent processing huge documents
        if not full_context or len(full_context.strip()) == 0:
            return ""
        
        # With 22GB RAM, we can process larger contexts
        if len(full_context) <= max_context_length:
            return full_context.strip()
        
        try:
            # Full processing for gaming laptop
            chunks = self.split_document(full_context)
            
            # With good RAM, process more chunks
            if len(chunks) > 30:  # Increased from 10
                chunks = chunks[:30]
            
            relevant_chunks = self.find_relevant_chunks(query, chunks, max_chunks=max_chunks)
            
            if not relevant_chunks:
                # Use first chunks if no relevance found
                relevant_chunks = chunks[:max_chunks] if chunks else []
            
            # Join chunks with generous length
            combined_context = "\n\n".join(relevant_chunks)
            
            if len(combined_context) > max_context_length:
                combined_context = combined_context[:max_context_length] + "..."
                
            return combined_context.strip()
            
        except Exception as e:
            # Fallback: return larger truncated context
            logger.warning(f"Document processing failed: {e}")
            return full_context[:max_context_length] + "..." if len(full_context) > max_context_length else full_context

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
# Performance stats endpoint removed - not used by frontend
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
# Remove the entire ModelPreloader and OllamaOptimizer classes

# Initialize all components
ai_optimizer = AIModelOptimizer()
response_cache = ResponseCache(max_size=200, ttl_minutes=60)  # 1 hour TTL
response_monitor = ResponseMonitor()
# Initialize simple performance monitor
performance_monitor = SimplePerformanceMonitor()

# Startup and Shutdown Events
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("🚀 Starting FastAPI application...")
    logger.info("✅ Document processing system ready")
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

# Endpoint untuk analisis detail dokumen
@app.post("/api/analyze_document")
async def analyze_document_details(file: UploadFile = File(...)):
    """Analisis detail dokumen tanpa menyimpan file."""
    logger.info(f"Analyzing document: {file.filename}")
    
    try:
        # Validasi tipe file
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename tidak boleh kosong")
            
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension not in ['.pdf', '.docx']:
            raise HTTPException(
                status_code=400,
                detail="Format file tidak didukung. Hanya PDF dan DOCX yang diizinkan."
            )
        
        # Buat file temporary untuk analisis
        temp_id = str(uuid.uuid4())
        temp_path = os.path.join(UPLOAD_FOLDER, f"temp_{temp_id}{file_extension}")
        
        try:
            # Simpan temporary
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Analisis dengan fungsi advanced
            analysis_result = {}
            
            if file_extension == '.docx':
                analysis_result = await analyze_docx_structure(temp_path)
            elif file_extension == '.pdf':
                analysis_result = await analyze_pdf_structure(temp_path)
            
            analysis_result.update({
                "filename": file.filename,
                "file_type": file_extension,
                "file_size": os.path.getsize(temp_path)
            })
            
            return analysis_result
            
        finally:
            # Hapus file temporary
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error analyzing document: {str(e)}")

async def analyze_docx_structure(file_path: str) -> dict:
    """Analisis struktur detail dokumen DOCX."""
    try:
        doc = Document(file_path)
        
        analysis = {
            "document_type": "DOCX",
            "paragraphs": len([p for p in doc.paragraphs if p.text.strip()]),
            "tables": len(doc.tables),
            "images": 0,
            "headings": 0,
            "text_length": 0,
            "styles_used": set(),
            "has_header_footer": False
        }
        
        # Analisis paragraf dan style
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                analysis["text_length"] += len(paragraph.text)
                analysis["styles_used"].add(paragraph.style.name)
                
                if paragraph.style.name.startswith('Heading'):
                    analysis["headings"] += 1
        
        # Hitung gambar
        for rel in doc.part.rels.values():
            if "image" in rel.target_ref:
                analysis["images"] += 1
        
        # Konversi set ke list untuk JSON serialization
        analysis["styles_used"] = list(analysis["styles_used"])
        
        # Deteksi header/footer
        try:
            for section in doc.sections:
                if section.header.paragraphs or section.footer.paragraphs:
                    analysis["has_header_footer"] = True
                    break
        except:
            pass
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing DOCX structure: {e}")
        return {"error": str(e)}

async def analyze_pdf_structure(file_path: str) -> dict:
    """Analisis struktur detail dokumen PDF."""
    try:
        pdf_document = fitz.open(file_path)
        
        analysis = {
            "document_type": "PDF",
            "pages": len(pdf_document),
            "images": 0,
            "text_length": 0,
            "has_bookmarks": False,
            "has_forms": False,
            "metadata": {}
        }
        
        # Ekstrak metadata
        metadata = pdf_document.metadata
        analysis["metadata"] = {
            "title": metadata.get('title', ''),
            "author": metadata.get('author', ''),
            "subject": metadata.get('subject', ''),
            "creator": metadata.get('creator', ''),
            "producer": metadata.get('producer', ''),
            "creation_date": metadata.get('creationDate', ''),
            "modification_date": metadata.get('modDate', '')
        }
        
        # Analisis per halaman
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            
            # Hitung teks
            text = page.get_text()
            analysis["text_length"] += len(text)
            
            # Hitung gambar
            image_list = page.get_images()
            analysis["images"] += len(image_list)
        
        # Cek bookmark
        try:
            toc = pdf_document.get_toc()
            analysis["has_bookmarks"] = len(toc) > 0
        except:
            pass
        
        pdf_document.close()
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing PDF structure: {e}")
        return {"error": str(e)}
