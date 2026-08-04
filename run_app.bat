taskkill /f /im node.exe >nul 2>&1

@echo off

echo Starting Python Model
cd /d "%~dp0\src"
python ML.py

echo Starting Backend Server
cd /d "%~dp0\server"
start cmd /k "npm install && node server.js"

echo Starting Frontend
cd /d "%~dp0"
start cmd /k "npm install && npm run dev"

timeout /t 5

echo Opening Web App
start http://localhost:5173

echo.
echo Close this window when finished to stop the app.
pause
taskkill /f /im node.exe
