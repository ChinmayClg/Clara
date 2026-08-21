@echo off
if not exist venv (
    echo Virtual environment not found. Please set it up first.
    pause
    exit
)
call venv\Scripts\activate
echo Starting Personal AI Assistant (Groq / Ollama)...
echo Ensure 'ollama serve' is running if you are using local fallback
python -m src.main
pause
