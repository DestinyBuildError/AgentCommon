start "agent-backend" cmd /k ".venv\Scripts\python.exe -m uvicorn app.main:app --port 8000"

echo [OK] Backend  : http://127.0.0.1:8000  (window: agent-backend)
pause