@echo off
if not exist venv (
    echo Virtual environment not found. Please set it up first.
    pause
    exit
)
call venv\Scripts\activate
echo Starting Personal Assistant (Local Ollama)...
echo Ensure 'ollama serve' is running and you have a model pulled
python -m src.main
pause
