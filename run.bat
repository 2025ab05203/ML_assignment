@echo off
echo ========================================
echo ML Assignment 2 - Quick Start Script
echo ========================================
echo.

echo Step 1: Installing dependencies...
pip install -r requirements.txt
echo.

echo Step 2: Training all 6 models...
cd model
python ml_models_training.py
cd ..
echo.

echo Step 3: Starting Streamlit app...
echo.
echo The app will open in your browser at http://localhost:8501
echo Press Ctrl+C to stop the server
echo.
streamlit run app.py
