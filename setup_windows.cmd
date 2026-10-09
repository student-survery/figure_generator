@echo off
cd /d "%~dp0"
python -m venv .venv
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo Setup complete. Open start_windows.cmd.
pause
exit /b 0
:failed
echo Setup failed. Check Python installation and network access.
pause
exit /b 1
