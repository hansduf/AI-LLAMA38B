import asyncio
import httpx
import psutil
import time
import subprocess
import sys
import os

class SystemDiagnostic:
    """Comprehensive system diagnostic for Ollama timeout issues"""
    
    def __init__(self):
        self.issues = []
        self.recommendations = []
    
    def check_system_resources(self):
        """Check RAM, CPU, and disk usage"""
        print("🔍 CHECKING SYSTEM RESOURCES...")
        
        # RAM usage
        memory = psutil.virtual_memory()
        ram_usage = memory.percent
        ram_available = memory.available / (1024**3)  # GB
        
        print(f"💾 RAM Usage: {ram_usage:.1f}% (Available: {ram_available:.1f}GB)")
        
        if ram_usage > 90:
            self.issues.append(f"HIGH RAM USAGE: {ram_usage:.1f}%")
            self.recommendations.append("Close other applications to free RAM")
        elif ram_available < 2:
            self.issues.append(f"LOW AVAILABLE RAM: {ram_available:.1f}GB")
            self.recommendations.append("Need at least 4GB free RAM for Ollama")
        
        # CPU usage
        cpu_usage = psutil.cpu_percent(interval=2)
        print(f"🖥️  CPU Usage: {cpu_usage:.1f}%")
        
        if cpu_usage > 80:
            self.issues.append(f"HIGH CPU USAGE: {cpu_usage:.1f}%")
            self.recommendations.append("High CPU usage detected - close CPU-intensive apps")
        
        # Disk usage
        disk = psutil.disk_usage('C:')
        disk_free = disk.free / (1024**3)  # GB
        print(f"💽 Disk Free: {disk_free:.1f}GB")
        
        if disk_free < 5:
            self.issues.append(f"LOW DISK SPACE: {disk_free:.1f}GB")
            self.recommendations.append("Free up disk space (Ollama needs temp space)")
    
    def check_ollama_processes(self):
        """Check for multiple or stuck Ollama processes"""
        print("\n🔍 CHECKING OLLAMA PROCESSES...")
        
        ollama_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'cpu_percent']):
            try:
                if 'ollama' in proc.info['name'].lower():
                    memory_mb = proc.info['memory_info'].rss / (1024**2)
                    ollama_processes.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'memory_mb': memory_mb
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        print(f"🤖 Found {len(ollama_processes)} Ollama processes:")
        total_memory = 0
        for proc in ollama_processes:
            print(f"   PID {proc['pid']}: {proc['name']} ({proc['memory_mb']:.1f}MB)")
            total_memory += proc['memory_mb']
        
        if len(ollama_processes) > 1:
            self.issues.append(f"MULTIPLE OLLAMA PROCESSES: {len(ollama_processes)}")
            self.recommendations.append("Kill all Ollama processes and restart fresh")
        
        if total_memory > 8000:  # Over 8GB
            self.issues.append(f"HIGH OLLAMA MEMORY: {total_memory:.1f}MB")
            self.recommendations.append("Ollama using too much memory - restart needed")
    
    def check_network_connectivity(self):
        """Check if Ollama port is accessible"""
        print("\n🔍 CHECKING OLLAMA CONNECTIVITY...")
        
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex(('127.0.0.1', 11434))
            sock.close()
            
            if result == 0:
                print("✅ Ollama port 11434 is accessible")
            else:
                print("❌ Ollama port 11434 is not accessible")
                self.issues.append("OLLAMA PORT NOT ACCESSIBLE")
                self.recommendations.append("Start Ollama service: ollama serve")
        except Exception as e:
            print(f"❌ Network check failed: {e}")
            self.issues.append("NETWORK CHECK FAILED")
    
    async def test_ollama_models(self):
        """Test different models for performance"""
        print("\n🔍 TESTING OLLAMA MODELS...")
        
        models_to_test = [
            {"name": "llama3:8b", "size": "4.7GB"},
            {"name": "phi3:mini", "size": "1.7GB"},
            {"name": "qwen2:0.5b", "size": "0.5GB"}
        ]
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Check available models
                response = await client.get("http://localhost:11434/api/tags")
                if response.status_code == 200:
                    available_models = response.json().get("models", [])
                    available_names = [m["name"] for m in available_models]
                    
                    print("📋 Available models:")
                    for model in available_models:
                        print(f"   - {model['name']}")
                    
                    # Test smallest available model
                    for test_model in models_to_test:
                        if test_model["name"] in available_names:
                            await self.test_model_speed(test_model["name"])
                            break
                    else:
                        print("⚠️ No suitable test models found")
                        self.recommendations.append("Install smaller model: ollama pull phi3:mini")
                else:
                    print("❌ Cannot fetch model list")
                    self.issues.append("CANNOT FETCH MODELS")
        except Exception as e:
            print(f"❌ Model test failed: {e}")
            self.issues.append("MODEL TEST FAILED")
    
    async def test_model_speed(self, model_name):
        """Test specific model speed"""
        print(f"\n🧪 Testing {model_name}...")
        
        ultra_minimal_payload = {
            "model": model_name,
            "prompt": "Hi",
            "stream": False,
            "options": {
                "num_predict": 1,
                "temperature": 0.01,
                "top_k": 1,
                "num_ctx": 32
            }
        }
        
        try:
            start_time = time.time()
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    "http://localhost:11434/api/generate",
                    json=ultra_minimal_payload
                )
                end_time = time.time()
                response_time = (end_time - start_time) * 1000
                
                if response.status_code == 200:
                    result = response.json()
                    ai_response = result.get("response", "")
                    print(f"✅ {model_name}: {response_time:.0f}ms - '{ai_response}'")
                    
                    if response_time < 5000:
                        print(f"🎯 {model_name} is WORKING! Use this model.")
                        self.recommendations.append(f"Switch to {model_name} for better performance")
                    else:
                        print(f"⚠️ {model_name} still slow: {response_time:.0f}ms")
                else:
                    print(f"❌ {model_name} error: {response.status_code}")
                    
        except httpx.TimeoutException:
            print(f"⏱️ {model_name} TIMEOUT - model is stuck")
            self.issues.append(f"{model_name.upper()} TIMEOUT")
        except Exception as e:
            print(f"❌ {model_name} test error: {e}")
    
    def check_windows_performance(self):
        """Check Windows-specific performance issues"""
        print("\n🔍 CHECKING WINDOWS PERFORMANCE...")
        
        try:
            # Check Windows Defender
            print("🛡️ Checking Windows Defender...")
            # Note: This is a simplified check
            
            # Check power plan
            try:
                result = subprocess.run(['powercfg', '/getactivescheme'], 
                                      capture_output=True, text=True, timeout=5)
                if 'High performance' not in result.stdout:
                    self.recommendations.append("Set Windows to High Performance power plan")
                    print("⚡ Consider High Performance power plan")
            except:
                pass
            
            # Check if running on battery
            battery = psutil.sensors_battery()
            if battery and not battery.power_plugged:
                self.issues.append("RUNNING ON BATTERY")
                self.recommendations.append("Connect to power adapter for better performance")
                print("🔋 Running on battery - connect power adapter")
            
        except Exception as e:
            print(f"Windows check failed: {e}")
    
    def generate_report(self):
        """Generate final diagnostic report"""
        print("\n" + "="*60)
        print("📊 DIAGNOSTIC REPORT")
        print("="*60)
        
        if not self.issues:
            print("✅ No critical issues detected")
            print("💡 The timeout issue may be:")
            print("   - Ollama model corruption")
            print("   - Ollama version bug")
            print("   - System-level issue")
        else:
            print("🚨 ISSUES DETECTED:")
            for i, issue in enumerate(self.issues, 1):
                print(f"   {i}. {issue}")
        
        print("\n💡 RECOMMENDED ACTIONS:")
        if self.recommendations:
            for i, rec in enumerate(self.recommendations, 1):
                print(f"   {i}. {rec}")
        
        print("\n🚀 EMERGENCY SOLUTIONS (try in order):")
        print("   1. ollama pull phi3:mini  # Smaller, faster model")
        print("   2. Restart computer")
        print("   3. Reinstall Ollama")
        print("   4. Use online AI temporarily")
        
        print("\n📋 MANUAL FIXES:")
        print("   - Close all other applications")
        print("   - Run: taskkill /f /im ollama.exe")
        print("   - Run: ollama serve")
        print("   - Wait 30 seconds before testing")

async def main():
    print("🚨 COMPREHENSIVE OLLAMA DIAGNOSTIC")
    print("="*50)
    
    diag = SystemDiagnostic()
    
    # Run all diagnostic checks
    diag.check_system_resources()
    diag.check_ollama_processes()
    diag.check_network_connectivity()
    diag.check_windows_performance()
    await diag.test_ollama_models()
    
    # Generate final report
    diag.generate_report()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Diagnostic cancelled by user")
    except Exception as e:
        print(f"\n❌ Diagnostic failed: {e}")
        print("💡 Try manual restart: taskkill /f /im ollama.exe && ollama serve")
