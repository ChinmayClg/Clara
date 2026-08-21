# Personal Desktop Assistant

A Python-based hybrid AI assistant that integrates ultra-fast cloud inference via **Groq** with seamless offline fallback using **Ollama**. It can control your desktop apps, write code, and help you with tasks using voice or text input.

## ✨ Features

- **Hybrid AI Architecture**: Uses Groq for ultra-low latency processing with fallback to local Ollama.
- **Voice & Text Input**: Use voice commands or type your requests
- **Desktop Control**: Open applications, URLs, and search the web
- **Code Generation**: Write and read files with project context awareness
- **Conversation Persistence**: Save and load conversation history
- **Keyboard Shortcuts**: Quick access to common functions
- **Command History**: ↑/↓ arrow keys to cycle through previous commands
- **Live Progress**: Real-time progress bar and tool activity feed
- **Structured Logging**: Track all activities with detailed logs
- **Performance Tracking**: Automatic timing of key operations
- **TTS Caching**: Repeated phrases play instantly
- **Configurable**: Easy customization through `src/config.py` or `.env`

## 🎯 Requirements

- Python 3.8+
- Groq API Key for cloud inference
- Ollama with a code-capable model (default: `qwen2.5-coder:7b`) for local execution
- Windows (for `start` command support)
- Microphone (for voice input)

## 🚀 Setup

1. **Install Dependencies**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure AI Providers**:
   - **Groq (Recommended for Speed)**: 
     - Get an API key from [console.groq.com](https://console.groq.com/keys)
     - Set `ASSISTANT_USE_GROQ=true` and `GROQ_API_KEY=your_key` in `.env`
     - *(Highly Recommended: Also complete the Ollama setup below so your assistant can fall back to local processing if Groq hits rate limits or your internet drops!)*
   - **Ollama (For 100% Local Privacy or Fallback)**:
     - Download from [ollama.ai](https://ollama.ai)
     - Pull a model: `ollama pull qwen2.5-coder:7b`
     - Start Ollama: `ollama serve`
     - Set `ASSISTANT_USE_GROQ=false` in `.env` (if using exclusively local)

3. **Advanced Configuration**:
   - Edit `.env` to override model names without changing code:
     ```env
     ASSISTANT_GROQ_PRIMARY_MODEL=openai/gpt-oss-120b
     ASSISTANT_PRIMARY_MODEL=qwen2.5-coder:7b
     ```
   - Or edit `src/config.py` for full customization

## 💻 Usage

### Starting the Assistant

**Option 1**: Double-click `run.bat`

**Option 2**: Command line
```bash
.\venv\Scripts\activate
python -m src.main
```

### Using the Assistant

#### Voice Commands
1. Click **"🎤 Listen"** or press **Ctrl+L**
2. Speak your command:
   - *"Open Calculator"* → Opens the Calculator app
   - *"Search for Python tutorials"* → Opens Google search
   - *"Write a hello world script in test.py"* → Creates a Python file

#### Text Commands
1. Type your command in the text box
2. Press **Enter** or click **Send**

#### Keyboard Shortcuts
- **Ctrl+L**: Start listening for voice input
- **Ctrl+H**: View conversation history
- **Ctrl+S**: Stop speaking (interrupt TTS)
- **Ctrl+Q**: Quit application
- **↑/↓**: Cycle through command history

#### Conversation Management
- **💾 Save**: Save current conversation to JSON file
- **📂 Load**: Load a previous conversation
- **📜 History**: View recent messages in a popup

## 📁 Project Structure

```
Assistant/
├── src/
│   ├── brain.py              # LLM integration and tool calling
│   ├── system_controller.py  # System operations (apps, files, URLs)
│   ├── voice_engine.py       # Speech recognition and TTS (with caching)
│   ├── gui.py                # CustomTkinter GUI (with progress bar)
│   ├── code_context.py       # Project context gathering
│   ├── config.py             # Configuration settings
│   ├── main.py               # Entry point
│   └── utils/
│       ├── logger.py          # Structured logging
│       └── conversation_manager.py  # Save/load conversations
├── conversations/             # Saved conversation files
├── logs/                      # Application logs
├── tests/                     # Test files
├── .env                       # Model overrides (optional)
├── requirements.txt           # Python dependencies
└── run.bat                    # Windows launcher
```

## ⚙️ Configuration

### 🧠 Bring Your Own Model
The assistant is designed to be model-agnostic. You can easily upgrade or swap models if you have the hardware or API access!

### Via `.env` (recommended for model changes)
```env
# Groq Settings
ASSISTANT_USE_GROQ=true
GROQ_API_KEY=your_api_key_here
ASSISTANT_GROQ_PRIMARY_MODEL=openai/gpt-oss-120b

# Ollama Local Settings
ASSISTANT_PRIMARY_MODEL=qwen2.5-coder:7b
```

### Via `src/config.py` (full control)
```python
# LLM Settings
MAX_HISTORY = 20  # Messages to keep in memory

# Voice Settings
LISTEN_TIMEOUT = 4  # Seconds
VOICE_NAME = "en-US-AriaNeural"

# Safety Settings
SAFE_DIRS = [PROJECT_ROOT, Path.home() / "Documents"]
```

## 🛠️ Available Tools

The assistant can use these tools:
- **open_app**: Open desktop applications
- **open_url**: Open websites in browser
- **search_web**: Google search
- **write_file**: Create/overwrite files
- **read_file**: Read file contents
- **create_folder**: Create directories
- **list_dir**: List directory contents
- **run_command**: Execute shell commands
- **gather_context**: Read project files for context

## 🐛 Troubleshooting

### Microphone Issues
- **"API unavailable"**: Check your internet connection (Google Speech API)
- **"Timeout"**: Speak louder or adjust `LISTEN_TIMEOUT` in config
- **Not hearing you**: Check microphone permissions

### Ollama Issues
- **"Failed to connect to Ollama"**: 
  - Ensure Ollama is running: `ollama serve`
  - Check if model is installed: `ollama list`
  - Pull model if needed: `ollama pull qwen2.5-coder:7b`

### File Operation Errors
- **"Cannot write files outside allowed directories"**:
  - Add your target directory to `SAFE_DIRS` in `config.py`

### Performance
- **Slow responses**: 
  - Use a faster/smaller model or reduce `MAX_HISTORY`
  - Check `logs/assistant.log` for `[PERF]` timing entries
- **Memory issues**: 
  - Reduce `MAX_HISTORY` in config
  - Restart the assistant periodically

## 📝 Logs

Logs are saved to `logs/assistant.log` with automatic rotation (10MB max, 3 backups).

Performance timings are logged with `[PERF]` prefix:
```
[PERF] process_command took 3.2s
```

## 🔐 Security Notes

- File operations are restricted to directories in `SAFE_DIRS`
- Dangerous commands (delete, format, shutdown) are blocked
- Voice input uses Google Speech API (requires internet)
- AI processing runs **100% locally** if using Ollama, or via secure cloud API if using Groq
- Conversations are stored locally in `conversations/`

## 📄 License

This project is for personal use.

## 🙏 Acknowledgments

- Powered by [Groq](https://groq.com) and [Ollama](https://ollama.ai) (Hybrid LLM Architecture)
- UI built with [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter)
- Speech recognition via [Google Speech API](https://cloud.google.com/speech-to-text)
- TTS via [Edge-TTS](https://github.com/rany2/edge-tts)
