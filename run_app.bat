@echo off
setlocal
cd /d "%~dp0"

if not exist "data\results\model_results.json" (
  echo Creating the first AAPL model results...
  python -m ml.model
  if errorlevel 1 (
    echo The model could not run. Check the Python packages and internet connection.
    pause
    exit /b 1
  )
)

echo Starting Backend Server
start "AI Finance API" cmd /k "cd /d ""%~dp0server"" && npm start"

echo Starting Frontend
start "AI Finance UI" cmd /k "cd /d ""%~dp0client"" && npm run dev"

timeout /t 5

echo Opening Web App
start http://localhost:5173
endlocal
