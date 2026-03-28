@echo off


REM Change the current directory to the location of this batch file
cd /d "%~dp0"

REM Activate the virtual environment
call ".venv/Scripts/activate.bat"

REM Run the python script
python train_ppo.py

pause
