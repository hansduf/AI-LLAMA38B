@echo off
cd /d "g:\NExtJS\pyreact\backend"
echo 🔍 Checking if port 8000 is available...
netstat -an | findstr :8000
echo.
echo 🚀 Starting server on port 8001...
"C:/Users/USER/AppData/Local/Programs/Python/Python312/python.exe" -m uvicorn main:app --reload --host 0.0.0.0 --port 8001
pause
