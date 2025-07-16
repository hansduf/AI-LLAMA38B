@echo off
echo 🚀 DOKAI CHAT BACKEND STARTUP
echo ===============================
cd /d "g:\NExtJS\pyreact\backend"
echo 📁 Working directory: %CD%
echo.

echo 🐍 Testing Python...
"C:/Users/USER/AppData/Local/Programs/Python/Python312/python.exe" --version
if %ERRORLEVEL% neq 0 (
    echo ❌ Python not found!
    pause
    exit /b 1
)

echo.
echo 🔧 Running diagnostics...
"C:/Users/USER/AppData/Local/Programs/Python/Python312/python.exe" start_debug.py

echo.
echo 🛑 Server stopped
pause
