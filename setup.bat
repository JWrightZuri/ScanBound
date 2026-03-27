@echo off
echo [1/3] Creating Virtual Environment...
python -m venv venv
echo [2/3] Activating Environment...
call .\venv\Scripts\activate
echo [3/3] Installing Dependencies...
pip install -r requirements.txt
echo.
echo Setup Complete! To start, run: .\venv\Scripts\activate then python main.py
pause