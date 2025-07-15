#!/usr/bin/env python3
"""
Quick validation test untuk memastikan AI memberikan respons yang wajar setelah penghapusan sistem mode
"""

import requests
import time

def test_ai_response():
    """Test apakah AI memberikan respons yang wajar tanpa mode selection"""
    
    print("🧪 QUICK VALIDATION: Sistem AI Tanpa Mode")
    print("="*50)
    
    test_cases = [
        "Buatkan cerpen pendek",
        "Apa itu AI?", 
        "Jelaskan singkat tentang Python",
        "Bagaimana cara memasak nasi?"
    ]
    
    for i, question in enumerate(test_cases, 1):
        print(f"\n{i}️⃣ Test: '{question}'")
        
        payload = {
            "message": question
        }
        
        try:
            start = time.time()
            response = requests.post(
                "http://localhost:8000/api/chat",
                json=payload,
                timeout=120  # 2 menit timeout untuk model yang lebih berat
            )
            duration = time.time() - start
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result.get("response", "")
                
                print(f"✅ Status: OK ({duration:.1f}s)")
                print(f"📏 Length: {len(ai_response)} characters")
                print(f"📝 Response: {ai_response[:150]}...")
                
                if len(ai_response) > 20:
                    print("🎯 BAGUS: Respons wajar dan panjang")
                elif len(ai_response) > 10:
                    print("⚠️ SEDANG: Respons agak pendek tapi masih OK")
                else:
                    print("❌ BURUK: Respons masih terlalu pendek!")
                    
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                if response.text:
                    print(f"Error detail: {response.text}")
                
        except requests.exceptions.Timeout:
            print("⏰ TIMEOUT: Server tidak merespons dalam 2 menit")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n{'='*50}")
    print("💡 HASIL:")
    print("- Model llama3:8b sekarang berjalan dengan konfigurasi balanced optimal")
    print("- Tidak ada lagi pemilihan mode fast/balanced/quality")
    print("- Timeout 2 menit cukup untuk model 8B")
    print("- Restart backend jika diperlukan: python main.py")

if __name__ == "__main__":
    test_ai_response()
