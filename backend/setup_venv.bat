@echo off
echo Creating Python virtual environment for backend...

REM Create virtual environment
python -m venv venv

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Install requirements
pip install -r requirements.txt

echo.
echo Virtual environment setup complete!
echo To activate: call venv\Scripts\activate.bat
echo To deactivate: deactivate
echo To run server: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
pause