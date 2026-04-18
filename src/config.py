"""
Configuration settings for Personal Desktop Assistant.
Modify these values to customize behavior without changing core code.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

class Config:
    """Centralized configuration for the assistant"""
    
    # ==================== LLM Settings ====================
    # PRIMARY_MODEL: used for tool calling (needs to be good at structured output)
    # FAST_MODEL: used for streaming follow-up responses (no tools needed)
    # Models are auto-detected — if preferred model isn't available, the best installed one is used
    PRIMARY_MODEL = os.getenv("ASSISTANT_PRIMARY_MODEL", "qwen2.5-coder:14b")
    FAST_MODEL = os.getenv("ASSISTANT_FAST_MODEL", "qwen2.5-coder:7b")
    FALLBACK_MODEL = os.getenv("ASSISTANT_FALLBACK_MODEL", "qwen2.5-coder:7b")
    MAX_HISTORY = 20                   # Maximum conversation messages to keep in memory
    LLM_MAX_RETRIES = 3               # Retry attempts for LLM calls
    LLM_RETRY_DELAY = 1.0            # Base delay in seconds (doubles each retry)
    STREAM_RESPONSES = True           # Stream LLM responses token-by-token
    
    # ==================== API Provider Settings ====================
    USE_GROQ = os.getenv("ASSISTANT_USE_GROQ", "false").lower() == "true"
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_PRIMARY_MODEL = os.getenv("ASSISTANT_GROQ_PRIMARY_MODEL", "llama-3.3-70b-versatile")
    GROQ_FAST_MODEL = os.getenv("ASSISTANT_GROQ_FAST_MODEL", "llama-3.1-8b-instant")
    # Free tier TPM (tokens-per-minute) limit — Groq counts input + max_tokens against this
    # Set to 0 to disable limiting (for paid/dev tier users)
    GROQ_TPM_LIMIT = int(os.getenv("ASSISTANT_GROQ_TPM_LIMIT", "12000"))
    
    # When to use the full PRIMARY_MODEL (requires tool keywords in user message)
    CODING_INDICATORS = [
        'write_file', 'read_file', 'gather_context',  # direct tool names
        'write code', 'write a script', 'create a program', 'build an app',
        'modify the code', 'edit the file', 'refactor', 'debug this',
        'read the file', 'show me the code', 'what does this code',
        'improve', 'fix', 'help me', 'enhance', 'update the', 'add a', 'add to',
    ]
    
    # ==================== Voice Settings ====================
    LISTEN_TIMEOUT = 4        # Seconds to wait for speech to start
    PHRASE_TIME_LIMIT = 15    # Maximum seconds for a single phrase (longer commands)
    LISTEN_RETRIES = 2        # Number of retry attempts for speech recognition
    
    # TTS (Text-to-Speech) Settings
    VOICE_NAME = "en-US-AriaNeural"  # Microsoft Edge TTS Voice
    VOICE_RATE = "+0%"     # Speech rate adjustment
    VOICE_PITCH = "+0Hz"   # Speech pitch adjustment
    VOICE_VOLUME = "+0%"   # Volume adjustment
    
    # ==================== Code Context Settings ====================
    MAX_FILE_SIZE = 12000          # Max characters per file
    MAX_TOTAL_CONTEXT_SIZE = 40000  # Max total characters across all files (16GB RAM can handle this)
    MAX_CONTEXT_FILES = 30         # Max number of files to include in context
    IGNORE_DIRS = {'.git', '__pycache__', 'venv', 'node_modules', '.idea', '.vscode', 'conversations', 'logs'}
    IGNORE_EXTS = {'.pyc', '.png', '.jpg', '.jpeg', '.exe', '.dll', '.so', '.iso', '.zip', '.tar', '.gz', '.log'}
    # File extensions considered relevant for code context
    CODE_EXTS = {'.py', '.js', '.jsx', '.ts', '.tsx', '.html', '.css', '.json', '.md', '.yaml', '.yml', '.toml', '.cfg', '.ini', '.sh', '.bat'}
    
    # ==================== GUI Settings ====================
    WINDOW_SIZE = "900x650"
    THEME = "Dark"          # "Dark" or "Light"
    COLOR_THEME = "blue"    # "blue", "green", "dark-blue"
    
    # ==================== File Paths ====================
    PROJECT_ROOT = Path(__file__).parent.parent
    CONVERSATIONS_DIR = PROJECT_ROOT / "conversations"
    LOGS_DIR = PROJECT_ROOT / "logs"
    
    # ==================== Logging Settings ====================
    LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_TO_FILE = True
    LOG_TO_CONSOLE = True
    LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT = 3
    
    # ==================== Safety Settings ====================
    # Directories where file operations are allowed
    SAFE_DIRS = [
        PROJECT_ROOT,
        Path.home(),          # Allow anywhere under user home
        Path.home() / "Documents",
        Path.home() / "Desktop",
        Path.home() / "Desktop" / "Testing",
        Path.home() / "Downloads",
    ]
    
    # Commands that require user confirmation
    DANGEROUS_KEYWORDS = ['delete', 'remove', 'format', 'shutdown', 'restart']
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist"""
        cls.CONVERSATIONS_DIR.mkdir(exist_ok=True)
        cls.LOGS_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def is_safe_path(cls, filepath: str) -> bool:
        """Check if a file path is within allowed directories"""
        try:
            path = Path(filepath).resolve()
            return any(path.is_relative_to(safe_dir) for safe_dir in cls.SAFE_DIRS)
        except (ValueError, OSError):
            return False
    
    @classmethod
    def needs_primary_model(cls, text: str) -> bool:
        """Determine if a request needs the full PRIMARY_MODEL"""
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in cls.CODING_INDICATORS)
