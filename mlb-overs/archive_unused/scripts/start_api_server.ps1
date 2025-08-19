# from your project root
.\.venv\Scripts\Activate.ps1
python -m pip install -U uvicorn fastapi

# Kill anything on :8000 (optional)
netstat -ano | findstr :8000
# take the last column PID and:
taskkill /PID <thatPID> /F

# run uvicorn
python -m uvicorn mlb_overs.api.app:app --reload --host 127.0.0.1 --port 8000
