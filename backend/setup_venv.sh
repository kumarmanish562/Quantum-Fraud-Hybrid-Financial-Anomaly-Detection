#!/bin/bash
echo "Creating Python virtual environment for backend..."

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

echo ""
echo "Virtual environment setup complete!"
echo "To activate: source venv/bin/activate"
echo "To deactivate: deactivate"
echo "To run server: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"