#!/usr/bin/env python3
"""
Test timeout optimization untuk document analysis
"""

import requests
import time
import json

def test_document_analysis():
    """Test document analysis dengan timeout optimization"""
    
    print("🧪 TESTING DOCUMENT ANALYSIS - Timeout Optimization")
    print("="*60)
    
    # Test 1: Upload dokumen
    print("\n1️⃣ Upload test document...")
    
    # Buat file text sederhana untuk test
    test_content = "Artificial Intelligence (AI) adalah teknologi yang memungkinkan mesin untuk meniru kecerdasan manusia. AI digunakan dalam berbagai aplikasi seperti pengenalan suara, computer vision, dan pemrosesan bahasa alami."
    
    # Simulasi upload dengan content langsung
    document_context = test_content
    
    print(f"✅ Document ready: {len(document_context)} characters")
    
    # Test 2: Chat dengan document context
    print(f"\n2️⃣ Test chat dengan document context...")
    
    test_cases = [
        ("Apa itu AI?", "fast"),
        ("Jelaskan aplikasi AI", "balanced"),
    ]
    
    for question, mode in test_cases:
        print(f"\n📝 Test: '{question}' (mode: {mode})")
        
        payload = {
            "message": question,
            "context": document_context,
            "response_mode": mode
        }
        
        try:
            start = time.time()
            response = requests.post(
                "http://localhost:8000/api/chat",
                json=payload,
                timeout=120  # 2 menit client timeout
            )
            duration = time.time() - start
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result.get("response", "")
                timing = result.get("timing", {})
                
                print(f"✅ Status: OK ({duration:.1f}s)")
                print(f"📏 Response length: {len(ai_response)} characters")
                print(f"⏱️ Server timing: {timing.get('total_ms', 0)/1000:.1f}s")
                print(f"📝 Response: {ai_response[:150]}...")
                
                if duration < 60:  # Under 1 minute
                    print("🎯 EXCELLENT: Fast response time!")
                elif duration < 90:  # Under 1.5 minutes
                    print("✅ GOOD: Acceptable response time")
                else:
                    print("⚠️ SLOW: Still slow but completed")
                    
            elif response.status_code == 504:
                print("❌ TIMEOUT: Still timing out!")
                print("💡 Suggestion: Increase server timeout or optimize model")
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                print(f"Response: {response.text}")
                
        except requests.exceptions.Timeout:
            print("⏰ CLIENT TIMEOUT: Request exceeded 2 minutes")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n{'='*60}")
    print("📊 ANALISIS HASIL:")
    print("✅ Jika semua test berhasil < 60s: Optimasi berhasil!")
    print("⚠️ Jika masih 60-90s: Perlu optimasi lebih lanjut")
    print("❌ Jika masih timeout: Perlu restart Ollama atau cek konfigurasi")
    
    print(f"\n💡 TIPS OPTIMASI:")
    print("- Fast mode: Target < 30s untuk dokumen pendek")
    print("- Balanced mode: Target < 60s untuk analisis sedang")
    print("- Quality mode: Target < 120s untuk analisis mendalam")

if __name__ == "__main__":
    test_document_analysis()
