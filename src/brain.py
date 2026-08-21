import re
import time
import json
import ollama
from pathlib import Path
from src.system_controller import SystemController
from src.code_context import CodeContext
from src.config import Config
from src.utils.logger import setup_logger
from src.utils.conversation_manager import ConversationManager

logger = setup_logger(__name__)


def _timed(func):
    """Decorator to log execution time of key functions."""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"[PERF] {func.__name__} took {elapsed:.2f}s")
        return result
    wrapper.__name__ = func.__name__
    return wrapper


class Brain:
    def __init__(self):
        self.system = SystemController()
        self.context = CodeContext()
        self.conversation_manager = ConversationManager()
        
        # Use configuration
        self.fast_model = Config.FAST_MODEL
        self.primary_model = Config.PRIMARY_MODEL
        self.fallback_model = Config.FALLBACK_MODEL
        self.max_history = Config.MAX_HISTORY
        self.model = self.primary_model  # alias for test compatibility
        
        # Callbacks (set by GUI)
        self.on_stream_token = None
        self.on_tool_activity = None  # Called with (tool_name, args_summary) for live GUI updates
        self.on_blueprint_ready = None  # Called with (blueprint_text, user_request) for GUI approval
        self._ran_npx_create_react = False  # Track if React scaffolding succeeded
        self._user_wants_react = False  # Track if user explicitly asked for React
        self._active_project_root = None  # Track the project root for path resolution
        self._read_files = set()  # Track files that have been read (for read-before-write)
        
        # Groq API key management
        self.groq_api_keys = Config.get_groq_api_keys()
        self.current_groq_key_index = 0
        
        # Blueprint approval state
        self._pending_blueprint = None   # The blueprint text awaiting user approval
        self._pending_request = None     # The original user request for the pending blueprint
        self._pending_is_react = False   # Whether the pending request is a React project
        
        # Validate Ollama connection
        self._validate_llm_connection()
        
        logger.info(f"Brain initialized — fast: {self.fast_model}, primary: {self.primary_model}")
        
        # Get actual Desktop path
        desktop_path = str(Path.home() / "Desktop")
        documents_path = str(Path.home() / "Documents")
        
        # Build system prompt — use a concise version for Groq (70B model needs less hand-holding)
        if Config.USE_GROQ:
            system_prompt = f"""You are a helpful, friendly desktop assistant. You can EXECUTE tasks using tools AND answer general questions conversationally.

WHEN TO USE TOOLS: Use tools (write_file, read_file, gather_context, run_command, create_folder, list_dir, open_app, open_url, search_web) when the user asks you to perform an ACTION on their computer — like creating files, opening apps, writing code, etc.
- Always use tools to perform actions. Never output raw code blocks in your response.
- Before overwriting an existing file, call read_file first to see its current content.
- Write complete, production-quality code with proper imports, error handling, and styling.
- No placeholders, TODO comments, or skeleton code — implement all logic fully.
- For web apps, use vanilla HTML + CSS + JS unless the user explicitly asks for React.
- For existing projects: call gather_context(target_path) first, then write_file to improve.

WHEN TO JUST TALK: When the user asks a general question, asks for suggestions, ideas, explanations, advice, opinions, or just wants to chat — respond directly with text. Do NOT use any tools for these. Examples: "suggest me some names", "what is Python", "tell me a joke", "explain recursion", "help me brainstorm ideas", "which language is best for games".

PROACTIVE & IMMERSIVE: When the user gives a vague or open-ended request (like "make a game" or "build me an app"), YOU decide the best, most impressive option. Don't make something boring or generic — pick something visually stunning and fun. For example:
- "make a game" → build a polished, visually impressive game (like a space shooter with particle effects, or a sleek puzzle game) — not a bare-bones blank page.
- "build an app" → choose an exciting, useful app concept with rich UI.
- Always go above and beyond. Make the user say "wow".

PERSONALITY: Be concise (1-2 sentences for actions, longer for knowledge questions), natural, and friendly.

PATHS:
- Desktop: {desktop_path}/
- Documents: {documents_path}/

HONESTY: If you don't know something, say so. Never fabricate paths, URLs, or data you haven't verified."""
        else:
            system_prompt = f"""You are a helpful, friendly desktop assistant. You help users by EXECUTING tasks with tools AND by answering questions conversationally.

WHEN TO USE TOOLS (actions on the computer):
You MUST use the provided JSON tools (like `write_file`) to execute actions. 
NEVER output raw code blocks (like ```html ... ```) in your response. 
If the user asks you to create or modify code, you MUST respond by calling the `write_file` tool.

WHEN TO JUST TALK (general questions, advice, brainstorming):
When the user asks a general question, wants suggestions, ideas, explanations, definitions, opinions, or just wants to chat — respond directly with text. Do NOT call any tools.
Examples of conversational requests: "suggest me some names", "what is machine learning", "tell me a joke", "explain recursion", "help me brainstorm", "what should I name my project", "give me ideas for...".

PROACTIVE & IMMERSIVE: When the user gives a vague or open-ended request (like "make a game" or "build me an app"), YOU decide the best, most impressive option. Don't make something boring or generic — pick something visually stunning and fun. For example:
- "make a game" → build a polished, visually impressive game (like a space shooter with particle effects, or a sleek puzzle game with animations) — not a bare-bones blank page.
- "build an app" → choose an exciting, useful app concept with rich UI, dark theme, and smooth animations.
- Always go above and beyond. Make the user say "wow".

PERSONALITY & SPOKEN CONVERSATION RULES:
- Since your text is read aloud by a Text-to-Speech Voice Engine, DO NOT use markdown formatting like **bold**, *italics*, or code fences in your conversational response.
- Speak like a human co-worker: use natural contractions (I'm, you're, we'll) and casual transitions (Got it, Sure, Hmm, Oh).
- Keep sentences short, punchy, and easy to listen to.
- Avoid robotic phrases like "As an AI..." or "I have processed your request."
- For ACTIONS: Your primary job is EXECUTING tasks with TOOLS. You must call tools to perform actions.
- For QUESTIONS: Answer helpfully, clearly, and conversationally. You can give longer, detailed answers, but format them for speech.
- Be concise but CONVERSATIONAL. When you finish tasks (like coding a file), DO NOT say "Done" or "Completed 3 tasks."
- Instead, say something natural and friendly like: "I've just scaffolded the new React app for you!" or "I've updated the styles in App.css, let me know how it looks."
- Be helpful: if a task needs multiple steps, do them all.

RULES:
1. ALWAYS use tools to complete ACTION tasks. Call gather_context, write_file, create_folder, run_command, etc.
2. When the user says "make/build/create/improve/fix", use write_file to write code.
3. For web search, only use search_web when the user explicitly asks to search the web.
4. NEVER overwrite an existing file using write_file without calling read_file FIRST.
5. Write REAL, COMPLETE, WORKING code. Include proper imports, error handling, styling.
6. Web UI should be modern and beautiful: dark mode, CSS animations, responsive layout, hover effects.
7. Keep your spoken responses SHORT (1-2 sentences) for tool actions but natural and conversational. For general questions, give helpful detailed answers.
8. For existing projects: call gather_context(target_path) FIRST to understand the code, then write_file to improve it.
9. Use the EXACT project/folder name the user specified.
10. Choose the best language/stack for the task (Python, Node.js, HTML/CSS/JS, etc.). For web apps, use vanilla HTML + CSS + JS. Only use React if the user explicitly says "React".
11. If using React: run_command("npx create-react-app <name>") FIRST, then edit src/ files.
12. DO NOT USE PLACEHOLDERS. Comments like "TODO", "rest of code here", or "fetch API here" are STRICTLY PROHIBITED. You must implement all logic fully.

PATHS:
- Desktop: {desktop_path}/
- Documents: {documents_path}/

When the user asks you to improve/fix a project, you MUST:
1. Call gather_context(target_path) to read the project
2. Identify what to improve
3. Use write_file to rewrite the improved files with COMPLETE code

CODE QUALITY:
- Complete imports, error handling, async/await with try/catch.
- HTML: full layout (nav, sidebar, main, footer). CSS: colors, spacing, typography, responsive.
- JS: real DOM manipulation, event listeners, fetch calls, data rendering.
- If modifying: rewrite the ENTIRE file, not just changed parts.

HONESTY (CRITICAL — DO NOT HALLUCINATE):
- If you do NOT know something, say "I'm not sure" or "I don't know." NEVER fabricate facts, statistics, or answers.
- NEVER invent file contents or paths that you have not actually read with read_file or gather_context.
- If you are unsure if a file or folder exists, use list_dir or read_file FIRST — do NOT guess.
- NEVER make up URLs, API endpoints, or data that you cannot verify.
- When the user asks a factual question you're uncertain about, be honest about your uncertainty."""

        self.messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Define tools with improved descriptions for better model tool selection
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "open_app",
                    "description": "Launch a desktop application on Windows. Use when the user asks to open, start, or launch an app like Chrome, Notepad, Calculator, etc.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "app_name": {
                                "type": "string",
                                "description": "The name of the application to open (e.g., 'calculator', 'chrome', 'notepad')."
                            }
                        },
                        "required": ["app_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "take_screenshot",
                    "description": "Take a screenshot of the user's current screen and save it to their Desktop. Use when the user asks you to take a screenshot or capture the screen.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {
                                "type": "string",
                                "description": "(Optional) The name of the file to save as (e.g., 'screenshot.png'). Leave empty for auto-generated timestamp."
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "open_url",
                    "description": "Open a website URL in the default browser. Use when the user wants to visit or navigate to a specific URL.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The URL or website to open (e.g., 'youtube.com', 'https://github.com')."
                            }
                        },
                        "required": ["url"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "gather_context",
                    "description": "Scan an entire project directory to understand its structure, tech stack, API endpoints, and source code. Use this FIRST when the user mentions a project path or asks to improve/fix/modify an existing project. Returns a comprehensive project overview.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target_path": {
                                "type": "string",
                                "description": "The file or directory path to scan (e.g., 'Desktop/my-project', 'src/components'). Use Desktop/ or Documents/ prefix for user folders."
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Search the web using Google. Use ONLY when the user explicitly asks to search or look something up online. Never use for coding tasks.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query (e.g., 'python tutorials', 'latest news')."
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Create or overwrite a file with complete content. Always write the ENTIRE file — never partial updates. If the file already exists, you MUST call read_file first to see its current content before overwriting. Write production-quality code with no placeholders.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "The path to the file to write (e.g., 'Desktop/my-project/index.html', 'src/app.py')."
                            },
                            "content": {
                                "type": "string",
                                "description": "The COMPLETE file content to write. Must be full, working code with imports, error handling, and styling."
                            }
                        },
                        "required": ["file_path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read and return the contents of a file. Use BEFORE calling write_file on an existing file to understand its current state. Required before modifying any code.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "The path to the file to read (e.g., 'Desktop/my-project/src/config.py')."
                            }
                        },
                        "required": ["file_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_folder",
                    "description": "Create a directory and any parent directories needed. Use before write_file when the target directory doesn't exist yet.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "folder_path": {
                                "type": "string",
                                "description": "Path of the folder to create (e.g., 'Desktop/my-project', 'Desktop/my-project/src/components')."
                            }
                        },
                        "required": ["folder_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_dir",
                    "description": "List files and folders in a directory. Use to discover what exists before reading or writing. Helps verify paths and project structure.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "dir_path": {
                                "type": "string",
                                "description": "Path of the directory to list (e.g., '.', 'Desktop/my-project', 'src/')."
                            }
                        },
                        "required": ["dir_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "description": "Execute a shell command (PowerShell on Windows) and return stdout+stderr. Use for: installing packages (pip install, npm install), running scripts (python app.py, node server.js), starting dev servers, or any system command.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The command to run (e.g., 'python hello.py', 'pip install flask', 'npm run dev')."
                            }
                        },
                        "required": ["command"]
                    }
                }
            }
        ]
    
    def _validate_llm_connection(self):
        """Validate LLM connection (Ollama or Groq)"""
        if Config.USE_GROQ:
            if not self.groq_api_keys:
                logger.error("USE_GROQ is True but no Groq API keys found!")
                raise RuntimeError("Groq API key is missing. Please set GROQ_API_KEYS in your .env file.")
            logger.info("Using Groq API for LLM connection.")
            return

        try:
            models_response = ollama.list()
            
            # Handle different response formats
            if hasattr(models_response, 'models'):
                models_list = models_response.models
            elif isinstance(models_response, dict):
                models_list = models_response.get('models', [])
            else:
                models_list = models_response
            
            available_models = []
            for m in models_list:
                if hasattr(m, 'model'):
                    available_models.append(m.model)
                elif isinstance(m, dict) and 'model' in m:
                    available_models.append(m['model'])
                elif isinstance(m, dict) and 'name' in m:
                    available_models.append(m['name'])
            
            logger.info(f"Available Ollama models: {available_models}")
            
            if not available_models:
                raise RuntimeError("No models found. Run: ollama pull qwen2.5-coder:14b")
            
            # Auto-select: use configured model if available, otherwise pick best available
            if self.primary_model not in available_models:
                old_model = self.primary_model
                best = self._pick_best_model(available_models)
                self.primary_model = best
                self.model = best  # Keep alias in sync
                logger.warning(f"Primary model '{old_model}' not found. Auto-selected: {best}")
            
            # For fast model: prefer a smaller model for speed (streaming responses)
            if self.fast_model not in available_models:
                self.fast_model = self._pick_best_model(available_models, prefer_small=True)
            if self.fallback_model not in available_models:
                self.fallback_model = self.primary_model
            
            logger.info(f"Using model: {self.primary_model} (fast: {self.fast_model})")
                
        except RuntimeError:
            raise
        except Exception as e:
            error_msg = f"Failed to connect to Ollama server. Is it running? Error: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _select_model(self, text):
        """Select the best model for the given request.
        Since fast-path already handles simple commands, everything here needs the primary model."""
        return self.primary_model
    
    def _trim_history(self):
        """Trim message history with smart compression.
        
        Strategy:
        - System prompt: always kept
        - Recent user/assistant turns (last 6 pairs): always kept verbatim
        - Tool call/result messages: compressed to 1-line summaries
        - Old messages: compressed to brief summaries
        """
        if len(self.messages) <= self.max_history:
            return
        
        system_msg = self.messages[0]
        
        # Keep last N messages verbatim (the active conversation)
        keep_recent = min(self.max_history - 2, 12)
        recent_messages = self.messages[-keep_recent:]
        
        # Compress older messages
        old_messages = self.messages[1:-keep_recent]
        
        summary_parts = []
        for msg in old_messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            if role == 'user':
                summary_parts.append(f"User: {content[:80]}")
            elif role == 'assistant' and content:
                summary_parts.append(f"AI: {content[:80]}")
            elif role == 'tool' and content:
                # Aggressively compress tool results to just the outcome
                if 'Successfully wrote' in content:
                    summary_parts.append(f"Tool: {content[:60]}")
                elif 'Error' in content[:20]:
                    summary_parts.append(f"Tool error: {content[:60]}")
                # Skip verbose gather_context / read_file outputs entirely
            # Skip system nudges — they're only relevant in the moment
        
        if summary_parts:
            # Keep only the last 10 summary items to avoid ballooning
            context_summary = "CONVERSATION CONTEXT (earlier messages summarized):\n" + "\n".join(summary_parts[-10:])
            context_msg = {"role": "system", "content": context_summary}
            self.messages = [system_msg, context_msg] + recent_messages
        else:
            self.messages = [system_msg] + recent_messages
        
        logger.debug(f"Trimmed history to {len(self.messages)} messages (compressed {len(old_messages)} old msgs)")
    
    def save_conversation(self, name=None):
        """Save current conversation to file"""
        try:
            filepath = self.conversation_manager.save_conversation(self.messages, name)
            logger.info(f"Conversation saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")
            return None
    
    def load_conversation(self, filepath):
        """Load conversation from file"""
        try:
            self.messages = self.conversation_manager.load_conversation(filepath)
            logger.info(f"Loaded conversation from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load conversation: {e}")
            return False
    
    def get_conversation_history(self, limit=10):
        """Get recent messages from conversation"""
        return self.messages[-limit:]

    def _fast_path(self, text):
        """
        Handles simple commands immediately without LLM latency.
        Returns response string or None to fall through to LLM.
        """
        text_lower = text.lower().strip()
        
        # --- Greetings ---
        greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 'howdy']
        if text_lower.rstrip('!. ') in greetings:
            return "Hey! How can I help you?"
        
        # --- "What time is it" / "What's the date" ---
        if any(x in text_lower for x in ['what time', 'current time', "what's the time"]):
            from datetime import datetime
            return f"It's {datetime.now().strftime('%I:%M %p')}."
        if any(x in text_lower for x in ['what date', "today's date", "what day"]):
            from datetime import datetime
            return f"Today is {datetime.now().strftime('%A, %B %d, %Y')}."
        
        # --- "Open [app]" ---
        # IMPORTANT: Use re.match (not re.search) so we only match at the START of the sentence.
        # re.search would match "start" in "the player is dead at the start of game" → open_app("of game")
        match = re.match(r"(?:please\s+)?(?:can you\s+)?(?:open|start|launch)\s+(.+)", text_lower)
        if match:
            potential_app = match.group(1).strip()
            
            # Clean politeness words BEFORE complexity check
            # (prevents "for me" triggering the " for " skip-word)
            potential_app = re.sub(r"\b(please|now|for me|can you|could you)\b", "", potential_app).strip()
            potential_app = potential_app.strip(".,!?")
            
            # Skip complex commands — let LLM handle them
            if any(x in potential_app for x in [" and ", " then ", " with ", " for ", " like ", " on "]) or len(potential_app.split()) > 4:
                logger.debug(f"[Fast Path] Skipped complex command: {potential_app}")
                return None

            app_name = potential_app
            
            # Check if this looks like a URL (has a domain TLD or http scheme)
            # e.g., "youtube.com", "https://github.com", "www.reddit.com"
            if re.search(r'(https?://|[\w.-]+\.\w{2,})', app_name):
                logger.info(f"[Fast Path] Opening URL: {app_name}")
                return self.system.open_url(app_name)
            
            logger.info(f"[Fast Path] Opening: {app_name}")
            return self.system.open_app(app_name)
        
        # NOTE: search_web removed from fast path — let the LLM decide when to search.
        
        # --- "Go to [url]" / "Open [url]" ---
        url_match = re.search(r"(?:go to|visit|navigate to)\s+([\w.-]+\.[a-z]{2,}(?:/\S*)?)", text_lower)
        if url_match:
            url = url_match.group(1).strip()
            logger.info(f"[Fast Path] Opening URL: {url}")
            return self.system.open_url(url)
        
        return None
    
    def _is_conversational(self, text):
        """
        Detect if a user request is purely conversational (no computer action needed).
        These should be answered directly without tools.
        
        Examples: "suggest me some names", "what is Python", "tell me a joke",
                  "explain recursion", "who invented the internet", "give me ideas",
                  "which language can you make a game in the best"
        """
        text_lower = text.lower().strip()
        
        # --- PRIORITY 1: Question-framing patterns ---
        # If the sentence is structured as a QUESTION (starts with WH-word, ends with ?,
        # or uses "can you tell me" framing), it's conversational REGARDLESS of action words inside.
        # e.g. "which language can you make a game in" is a question, not a command.
        question_patterns = [
            r'^(?:what|which|who|whom|whose|where|when|why|how)\b',  # WH-questions
            r'^(?:is |are |do |does |did |can |could |would |should |will |shall )\b',  # yes/no questions
            r'^(?:can you tell|could you tell|do you know|can you explain|could you explain)\b',  # polite questions
            r'^(?:what\'s|whats|who\'s|where\'s|how\'s|when\'s)\b',  # contracted WH-questions
        ]
        
        is_question_framed = any(re.search(p, text_lower) for p in question_patterns)
        ends_with_question = text.rstrip().endswith('?')
        
        if is_question_framed or ends_with_question:
            # Even in question form, some are actually action requests:
            # "can you open Chrome" → action, "can you create a website" → action
            # Filter: only if the question is asking for INFO, not requesting an action
            action_request_in_question = re.search(
                r'^(?:can you |could you |will you |would you )?'
                r'(?:please )?'
                r'(?:open|launch|start|create|build|make|write|fix|debug|run|install|download|delete|remove|update|modify|go to|visit|navigate)\b',
                text_lower
            )
            if not action_request_in_question:
                logger.debug(f"[Brain] Question-framing detected as conversational: {text[:60]}")
                return True
        
        # --- PRIORITY 2: File/folder path references → NOT conversational ---
        has_path_reference = bool(re.search(r'(?:Desktop|Documents)[/\\]|\w+\.(?:py|js|html|css|jsx|tsx|json)', text, re.IGNORECASE))
        if has_path_reference:
            return False
        
        # --- PRIORITY 3: Action keywords → NOT conversational ---
        action_keywords = [
            'open', 'launch', 'start', 'run',  # app/command actions
            'create', 'build', 'make', 'develop', 'scaffold', 'generate', 'implement',  # creation
            'write', 'code', 'program', 'refactor', 'solve', 'rewrite',  # coding
            'fix', 'debug', 'update', 'modify', 'change', 'improve', 'enhance', 'add', 'insert', 'put', 'edit', 'append', 'replace',  # modification
            'install', 'download', 'delete', 'remove', 'uninstall',  # system actions
            'read file', 'read_file', 'write_file', 'gather_context',  # direct tool names
            'search the web', 'search online', 'google',  # web search
            'go to', 'visit', 'navigate to',  # URL navigation
            'take screenshot', 'screenshot', 'capture screen', 'take a screenshot', 'capture the screen',  # screenshot actions
        ]
        
        # Use word boundaries to avoid false positives (e.g., "start" in "startup")
        for kw in action_keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                return False
        
        # --- PRIORITY 4: Conversational patterns ---
        conversational_patterns = [
            r'^(?:suggest|recommend|give me|list|tell me|name)\b',  # suggestions/advice
            r'^(?:explain|describe|define|clarify|elaborate)\b',  # explanations
            r'^(?:help me (?:understand|think|brainstorm|decide|choose|pick))\b',  # brainstorming
            r'\b(?:ideas? for|suggestions? for|advice|opinion|thoughts? on|think about)\b',  # idea requests
            r'^(?:thank|thanks|okay|ok|cool|great|nice|awesome|perfect|got it|understood|alright)\b',  # acknowledgements
            r'\b(?:meaning of|difference between|pros and cons|advantages|disadvantages)\b',  # knowledge
            r'^(?:hey |hi |hello )?(?:can you |could you )?(?:suggest|recommend|give|tell|share|provide)\b',  # request patterns
            r'\b(?:best (?:way|language|framework|tool|option|practice)|recommend|between .+ and )\b',  # comparison/recommendation
            
            # Small talk & feedback
            r'\b(?:good job|well done|excellent|amazing|good boy|good girl|nice work|great work|awesome|beautiful|perfect)\b',
            r'\b(?:how are you|how is it going|how was your|how are things|whats up|what\'s up|sup|wassup)\b',
            r'\b(?:my day|my night|my morning|i feel|i am doing|i had a)\b',
            
            # Additional conversational edge cases
            r'^(?:hello|hi|hey|greetings|morning|afternoon|evening|goodnight|good night|bye|goodbye|see ya|cya)\b',
            r'\b(?:who are you|what are you|who made you|are you a(?:n)? (?:ai|robot|human))\b',
            r'\b(?:joke|funny|laugh|story|poem|sing|riddle)\b',
            r'^(?:sorry|my bad|my fault|apologies|excuse me)\b',
            r'\b(?:i love you|i hate you|you suck|you are (?:smart|dumb|stupid|cool|awesome|great|amazing))\b',
            r'\b(?:what(?: do)? you think|do you know|can you talk|can we chat)\b',
            r'^(?:yes|no|maybe|probably|definitely|of course|sure thing|absolutely|not really)\b',
        ]
        
        for pattern in conversational_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # --- PRIORITY 5: Short questions without action verbs ---
        words = text_lower.split()
        if len(words) <= 8:
            if text.rstrip().endswith('?'):
                return True
        
        return False

    def _analyze_request(self, text):
        """
        Pre-analyze user request to extract project context.
        Returns a dict with project_name, source_path, target_type, target_path.
        Returns None if the request doesn't involve project creation/modification.
        """
        text_lower = text.lower()
        
        result = {
            'project_name': None,
            'source_path': None,
            'target_type': None,  # 'frontend', 'backend', 'fullstack'
            'target_path': None,
        }
        
        # Check if this is a project/coding task
        build_keywords = ['create', 'build', 'make', 'develop', 'code', 'design',
                          'frontend', 'backend', 'web app', 'website', 'project',
                          'read my', 'gather', 'generate', 'change', 'update', 
                          'fix', 'modify', 'add', 'edit']
        has_keywords = any(kw in text_lower for kw in build_keywords)
        
        # Extract path references (Desktop/X/backend, Desktop/X/frontend, etc.)
        path_match = re.search(r'(?:Desktop|Documents)[/\\]([\w\-\.]+)(?:[/\\][\w\-\.]+)*', text, re.IGNORECASE)
        
        if not has_keywords and not path_match:
            return None
        if path_match:
            full_path = path_match.group(0)
            parts = re.split(r'[/\\]', path_match.group(1))
            
            # First part after Desktop/ is usually the project name
            result['project_name'] = parts[0]
            
            # Default: If they give a path, assume they want us to look at it and use it as target
            result['source_path'] = full_path
            result['target_path'] = full_path
            
            # Detect direction
            if 'backend' in full_path.lower():
                if any(x in text_lower for x in ['frontend', 'front end', 'front-end', 'client', 'ui']):
                    result['target_type'] = 'frontend'
                    result['target_path'] = re.sub(r'backend', 'frontend', full_path, flags=re.IGNORECASE)
            elif 'frontend' in full_path.lower() or 'client' in full_path.lower():
                if any(x in text_lower for x in ['backend', 'back end', 'back-end', 'server', 'api']):
                    result['target_type'] = 'backend'
                    result['target_path'] = re.sub(r'(?:frontend|client)', 'backend', full_path, flags=re.IGNORECASE)
        
        # Extract project name from "called X", "named X", "folder X"
        if not result['project_name']:
            name_match = re.search(r'(?:called|named|folder|project)\s+["\']?([\w\-\.]+)["\']?', text, re.IGNORECASE)
            if name_match:
                result['project_name'] = name_match.group(1)
        
        # Detect "like X" pattern (web app like YouTube) — note: project name is NOT the "like" reference
        like_match = re.search(r'like\s+(\w+)', text, re.IGNORECASE)
        if like_match:
            result['target_type'] = result.get('target_type') or 'fullstack'
        
        # If we have a project name but no target type, infer from keywords
        if result['project_name'] and not result['target_type']:
            if 'frontend' in text_lower or 'front end' in text_lower:
                result['target_type'] = 'frontend'
            elif 'backend' in text_lower or 'back end' in text_lower or 'server' in text_lower:
                result['target_type'] = 'backend'
            elif any(x in text_lower for x in ['web app', 'website', 'app', 'project', 'full']):
                result['target_type'] = 'fullstack'
        
        # Only return if we found something useful
        if result['project_name'] or result['source_path']:
            logger.info(f"[Brain] Analyzed request: {result}")
            return result
        
        return None

    def _call_llm_with_retry(self, model, messages, tools=None):
        """
        Call LLM (Ollama or Groq) with retry logic and fallback model.
        Supports streaming if Config.STREAM_RESPONSES is True and self.on_stream_token is set.
        """
        models_to_try = [model]
        if model != self.fallback_model:
            models_to_try.append(self.fallback_model)
        
        last_error = None
        force_ollama = False
        
        for current_model in models_to_try:
            keys_tried = 0
            
            # Using a while loop so rate limit key-swaps don't consume general "attempt" counts
            attempt = 0
            while attempt < Config.LLM_MAX_RETRIES:
                try:
                    if Config.USE_GROQ and len(self.groq_api_keys) > 0 and not force_ollama:
                        return self._execute_groq_call(current_model, messages, tools)
                    
                    # Streaming mode for Ollama
                    if Config.STREAM_RESPONSES and self.on_stream_token and not tools:
                        return self._stream_response(current_model, messages)
                    
                    # Normal mode (or tool-calling which doesn't support streaming well)
                    response = ollama.chat(
                        model=current_model,
                        messages=messages,
                        tools=tools,
                        options={"temperature": 0.1, "num_predict": 4096}
                    )
                    return response
                    
                except Exception as e:
                    last_error = e
                    error_str = str(e).lower()
                    
                    # Handle rate limit errors (Groq 429)
                    if Config.USE_GROQ and not force_ollama and ('429' in str(e) or 'rate_limit' in error_str or 'rate limit' in error_str):
                        keys_tried += 1
                        if keys_tried < len(self.groq_api_keys):
                            # Swap to the next key and immediately retry
                            self.current_groq_key_index = (self.current_groq_key_index + 1) % len(self.groq_api_keys)
                            logger.info(f"Rate limit hit. Swapping to Groq API key index {self.current_groq_key_index} and retrying...")
                            continue # Do NOT increment attempt
                        else:
                            logger.warning("All Groq API keys exhausted! Falling back to local Ollama...")
                            force_ollama = True
                            if current_model != self.fallback_model:
                                break # Move to the fallback model in the outer loop
                            else:
                                continue # We are already on the fallback model, just loop around and it will use Ollama
                    
                    delay = Config.LLM_RETRY_DELAY * (2 ** attempt)
                    logger.warning(f"LLM call failed (model={current_model}, attempt {attempt+1}): {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                    attempt += 1
            
            logger.warning(f"All retries exhausted for model '{current_model}'. Trying fallback...")
        
        raise last_error

    def _estimate_message_tokens(self, messages, tools=None):
        """Rough token estimate for Groq messages. ~4 chars per token for English text."""
        total_chars = sum(len(str(m.get('content', ''))) for m in messages)
        token_estimate = total_chars // 4
        # Add overhead for tool schemas (~50 tokens per tool definition)
        if tools:
            token_estimate += len(tools) * 50
        return token_estimate
    
    def _cap_max_tokens_for_groq(self, messages, desired_max_tokens, tools=None):
        """Cap max_tokens so input + output stays within Groq's TPM limit.
        Returns the capped max_tokens value."""
        if not Config.GROQ_TPM_LIMIT or Config.GROQ_TPM_LIMIT <= 0:
            return desired_max_tokens  # No limit (paid tier)
        
        input_tokens = self._estimate_message_tokens(messages, tools)
        safety_buffer = 500  # Reserve for overhead/variance in token counting
        available = Config.GROQ_TPM_LIMIT - input_tokens - safety_buffer
        
        if available < 1000:
            # Input alone is near/over the limit — use minimum viable output
            logger.warning(f"[Brain] Groq input tokens (~{input_tokens}) near TPM limit ({Config.GROQ_TPM_LIMIT}). Capping output to 1000 tokens.")
            return 1000
        
        capped = min(desired_max_tokens, available)
        if capped < desired_max_tokens:
            logger.info(f"[Brain] Groq token budget: input~{input_tokens} + output {capped} = ~{input_tokens + capped} (limit: {Config.GROQ_TPM_LIMIT})")
        return capped

    def _execute_groq_call(self, model, messages, tools=None):
        """Execute chat completion using the Groq API with optimized parameters."""
        from groq import Groq
        if not self.groq_api_keys:
            raise ValueError("No Groq API keys available")
            
        current_key = self.groq_api_keys[self.current_groq_key_index]
        client = Groq(api_key=current_key)
        
        # Override the model string with the Groq counterpart
        if model == self.primary_model or model == self.fallback_model:
            groq_model = Config.GROQ_PRIMARY_MODEL
        else:
            groq_model = Config.GROQ_FAST_MODEL
        
        # Task-specific temperature and max_tokens (Improvement #3 & #4)
        if tools:
            # Tool-calling: fully deterministic for reliable tool selection
            temperature = 0.0
            desired_max_tokens = 4096  # Tool decisions are short
        else:
            # Streaming/code generation: slightly creative, much more output room
            temperature = 0.2
            desired_max_tokens = 16384  # Code generation needs space for full files
        
        # Cap max_tokens to fit within Groq's TPM limit (prevents 413 errors)
        max_tokens = self._cap_max_tokens_for_groq(messages, desired_max_tokens, tools)
        
        # Sanitize messages for Groq API (Groq strictly requires 'arguments' to be a string)
        import copy
        import json
        safe_messages = copy.deepcopy(messages)
        for m in safe_messages:
            # Remove tool_calls completely if it is None (prevents Groq 400 nullable error)
            if 'tool_calls' in m and m['tool_calls'] is None:
                del m['tool_calls']
            elif m.get('tool_calls'):
                for tc in m['tool_calls']:
                    # Groq requires 'type': 'function' and a valid string 'id' for every tool call
                    if 'type' not in tc:
                        tc['type'] = 'function'
                    if 'id' not in tc or not tc['id']:
                        tc['id'] = 'call_extracted'
                    if isinstance(tc.get('function', {}).get('arguments'), dict):
                        tc['function']['arguments'] = json.dumps(tc['function']['arguments'])
            
        if Config.STREAM_RESPONSES and self.on_stream_token and not tools:
            # Streaming conversation: use warmer temperature for natural responses
            stream = client.chat.completions.create(
                model=groq_model,
                messages=safe_messages,
                temperature=0.5,
                max_tokens=max_tokens,
                stream=True
            )
            full_content = ""
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_content += token
                    if self.on_stream_token:
                        self.on_stream_token(token)
            return {
                'message': {
                    'role': 'assistant',
                    'content': full_content,
                    'tool_calls': None
                }
            }
        
        kwargs = {
            "model": groq_model,
            "messages": safe_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
            
        # Groq tool-calling
        try:
            response = client.chat.completions.create(**kwargs)
            msg = response.choices[0].message
        except Exception as e:
            error_str = str(e)
            if "failed_generation" in error_str:
                logger.warning(f"Groq API parsing failed due to Llama tool tag mismatch. Attempting manual extraction.")
                
                # Extract the failed_generation value from the error JSON
                # Format: 'failed_generation': '<function=open_app{"app_name": "chrome"}</function>\n'
                err_content = None
                try:
                    # The error body is Python dict repr with single quotes, not valid JSON
                    # Use ast.literal_eval to parse it
                    import ast
                    err_body_match = re.search(r"(\{.*\})\s*$", error_str, re.DOTALL)
                    if err_body_match:
                        err_data = ast.literal_eval(err_body_match.group(1))
                        err_content = err_data.get('error', {}).get('failed_generation', '')
                except Exception:
                    pass
                
                if not err_content:
                    # Fallback: regex extract
                    fg_match = re.search(r"'failed_generation':\s*'(.*?)'(?:\s*\})", error_str, re.DOTALL)
                    if fg_match:
                        err_content = fg_match.group(1)
                
                if err_content:
                    logger.info(f"Extracted failed_generation: {err_content[:100]}...")
                    
                    # Match <function=name{...}</function> OR <function=name {...} </function>
                    # Note: Llama often omits the space between name and JSON args
                    func_match = re.search(r'<function=(\w+)\s*(.*?)\s*</function>', err_content, re.DOTALL)
                    if func_match:
                        tool_name = func_match.group(1)
                        args_str = func_match.group(2).strip()
                        
                        # Find the JSON block in the arguments string
                        args_match = re.search(r'(\{.*\})', args_str, re.DOTALL)
                        if args_match:
                            try:
                                args = json.loads(args_match.group(1))
                                logger.info(f"Successfully recovered tool call: {tool_name}({args})")
                                return {
                                    'message': {
                                        'role': 'assistant',
                                        'content': '',
                                        'tool_calls': [{
                                            'id': 'call_recovered',
                                            'function': {
                                                'name': tool_name,
                                                'arguments': args
                                            }
                                        }]
                                    }
                                }
                            except Exception as json_e:
                                logger.error(f"Failed to parse recovered tool JSON: {json_e}")
                    else:
                        logger.error(f"Could not match <function=...> pattern in: {err_content[:100]}")
                else:
                    logger.error(f"Could not extract failed_generation from error")
            raise e
        
        # Format Groq's response to look exactly like Ollama's response
        # Improvement #1: Include tool_call_id for Groq API compliance
        formatted_tool_calls = None
        if msg.tool_calls:
            formatted_tool_calls = []
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except Exception:
                    args = tc.function.arguments
                formatted_tool_calls.append({
                    'id': tc.id,  # Required by Groq for tool result messages
                    'type': 'function',
                    'function': {
                        'name': tc.function.name,
                        'arguments': tc.function.arguments
                    }
                })
        
        return {
            'message': {
                'role': 'assistant',
                'content': msg.content or "",
                'tool_calls': formatted_tool_calls
            }
        }
    
    def _stream_response(self, model, messages):
        """Stream response token-by-token using Ollama, calling on_stream_token callback"""
        full_content = ""
        
        for chunk in ollama.chat(model=model, messages=messages, stream=True, options={"temperature": 0.1, "num_predict": 4096}):
            token = chunk.get('message', {}).get('content', '')
            if token:
                full_content += token
                if self.on_stream_token:
                    self.on_stream_token(token)
        
        # Return a response dict matching the non-streaming format
        return {
            'message': {
                'role': 'assistant',
                'content': full_content,
                'tool_calls': None
            }
        }

    def _extract_fallback_tool_calls(self, content):
        """Extract tool calls from raw JSON in LLM content (fallback for models that don't use native tool calling)"""
        if not content or "{" not in content or "name" not in content:
            return None
        
        logger.debug(f"Fallback parsing for JSON in content: {content[:80]}...")
        
        # Known tool parameter names for array→dict conversion
        known_tools_params = {
            'open_app': ['app_name'],
            'open_url': ['url'],
            'search_web': ['query'],
            'write_file': ['file_path', 'content'],
            'read_file': ['file_path'],
            'create_folder': ['folder_path'],
            'list_dir': ['dir_path'],
            'run_command': ['command'],
            'gather_context': ['target_path'],
        }
        
        # Pre-process: replace backtick-quoted strings with double-quoted
        cleaned = re.sub(r'`([^`]*)`', lambda m: '"' + m.group(1).replace('"', '\\"').replace('\n', '\\n') + '"', content)
        
        # Robust JSON extractor using brace counting
        objects = []
        stack = []
        start_index = -1
        
        for i, char in enumerate(cleaned):
            if char == '{':
                if not stack:
                    start_index = i
                stack.append(char)
            elif char == '}':
                if stack:
                    stack.pop()
                    if not stack:
                        json_str = cleaned[start_index:i+1]
                        try:
                            obj = json.loads(json_str)
                            objects.append(obj)
                        except json.JSONDecodeError:
                            pass
        
        fallback_calls = []
        for data in objects:
            if not isinstance(data, dict) or "name" not in data:
                continue
            
            # Accept both "arguments" and "args" (model uses either)
            raw_args = data.get("arguments") or data.get("args")
            if raw_args is None:
                continue
            
            tool_name = data["name"]
            
            # Convert array-style args to dict using known param names
            # e.g. ["src/App.js", "<code>"] → {"file_path": "src/App.js", "content": "<code>"}
            if isinstance(raw_args, list) and tool_name in known_tools_params:
                param_names = known_tools_params[tool_name]
                args_dict = {}
                for idx, param in enumerate(param_names):
                    if idx < len(raw_args):
                        args_dict[param] = raw_args[idx]
                raw_args = args_dict
            
            if raw_args:
                fallback_calls.append({
                    'function': {
                        'name': tool_name,
                        'arguments': raw_args
                    }
                })
        
        if fallback_calls:
            logger.info(f"Found {len(fallback_calls)} fallback tool calls.")
            return fallback_calls
        
        return None
    
    def _extract_function_style_tool_calls(self, content):
        """
        Extract tool calls written as function_name("arg1", "arg2") in LLM text.
        Handles cases where the LLM outputs tool calls as numbered lists:
          1. create_folder("path")
          2. write_file("path", "content")
        """
        if not content:
            return None
        
        known_tools = {
            'open_app': ['app_name'],
            'open_url': ['url'],
            'search_web': ['query'],
            'write_file': ['file_path', 'content'],
            'read_file': ['file_path'],
            'create_folder': ['folder_path'],
            'list_dir': ['dir_path'],
            'run_command': ['command'],
            'gather_context': ['target_path'],
            'take_screenshot': ['filename'],
        }
        
        tool_names_pattern = '|'.join(re.escape(t) for t in known_tools.keys())
        pattern = re.compile(
            r'(?:^|\n)\s*(?:\d+[\.)\-]\s*)?(' + tool_names_pattern + r')\s*\(',
            re.MULTILINE
        )
        
        if not pattern.search(content):
            return None
        
        logger.debug(f"Function-style parsing for content: {content[:80]}...")
        
        calls = []
        for match in pattern.finditer(content):
            tool_name = match.group(1)
            start = match.end()  # position right after '('
            
            # Find matching closing paren with string-aware depth tracking
            depth = 1
            pos = start
            in_string = None
            
            while pos < len(content) and depth > 0:
                c = content[pos]
                
                if in_string:
                    if c == '\\' and pos + 1 < len(content):
                        pos += 2  # skip escaped char inside string
                        continue
                    if c == in_string:
                        in_string = None
                else:
                    if c in ('"', "'"):
                        in_string = c
                    elif c == '(':
                        depth += 1
                    elif c == ')':
                        depth -= 1
                
                pos += 1
            
            if depth != 0:
                continue
            
            args_str = content[start:pos - 1].strip()
            param_names = known_tools[tool_name]
            
            # Extract quoted string arguments with escape handling
            args_values = []
            i = 0
            while i < len(args_str) and len(args_values) < len(param_names):
                if args_str[i] in ('"', "'"):
                    quote = args_str[i]
                    j = i + 1
                    value_parts = []
                    while j < len(args_str):
                        if args_str[j] == '\\' and j + 1 < len(args_str):
                            next_c = args_str[j + 1]
                            if next_c == 'n':
                                value_parts.append('\n')
                            elif next_c == 't':
                                value_parts.append('\t')
                            elif next_c == '\\':
                                value_parts.append('\\')
                            elif next_c == quote:
                                value_parts.append(quote)
                            else:
                                # Keep as-is (e.g. \U in Windows paths)
                                value_parts.append(args_str[j:j + 2])
                            j += 2
                            continue
                        if args_str[j] == quote:
                            break
                        value_parts.append(args_str[j])
                        j += 1
                    args_values.append(''.join(value_parts))
                    i = j + 1
                else:
                    i += 1
            
            # Build args dict from extracted values
            args_dict = {}
            for k, param_name in enumerate(param_names):
                if k < len(args_values):
                    args_dict[param_name] = args_values[k]
            
            if args_dict:
                calls.append({
                    'function': {
                        'name': tool_name,
                        'arguments': args_dict
                    }
                })
        
        if calls:
            logger.info(f"Found {len(calls)} function-style tool calls.")
            return calls
        
        return None
    
    def _resolve_project_path(self, file_path):
        """Resolve a relative path against the active project root, if set.
        
        When the model gives paths like 'src/App.js' instead of 'spotify-clone/src/App.js',
        this ensures they land in the right project directory.
        """
        if not file_path or not self._active_project_root:
            return file_path
        
        p = Path(file_path)
        
        # Already absolute — check if it's already under the project root
        if p.is_absolute():
            return file_path
        
        # Check if the path already starts with the project folder name
        project_name = Path(self._active_project_root).name
        parts = Path(file_path).parts
        if parts and parts[0].lower() == project_name.lower():
            # Already prefixed (e.g. 'spotify-clone/src/App.js')
            return file_path
        
        # Relative path — resolve against project root
        candidate = Path(self._active_project_root) / file_path
        # Check if the parent directory exists or the file already exists in the project
        if candidate.parent.exists() or candidate.exists():
            logger.info(f"[Brain] Resolved '{file_path}' → '{candidate}'")
            return str(candidate)
        
        # Parent doesn't exist yet, but if it's a common project path (src/, public/, etc.)
        # still resolve against the project root
        common_dirs = ('src', 'public', 'lib', 'components', 'pages', 'styles', 'css', 'js',
                       'routes', 'models', 'controllers', 'middleware', 'utils', 'config',
                       'assets', 'images', 'tests', 'test')
        if parts and parts[0].lower() in common_dirs:
            logger.info(f"[Brain] Resolved common dir '{file_path}' → '{candidate}'")
            return str(candidate)
        
        return file_path

    def _repair_write_file_calls(self, tool_calls, message_content):
        """If write_file has empty content but code block exists in message, extract it"""
        for tool in tool_calls:
            if tool['function']['name'] == 'write_file':
                args = tool['function']['arguments']
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        continue
                
                content_arg = args.get('content', '').strip()
                
                if len(content_arg) < 10 and "```" in (message_content or ''):
                    logger.info("[Brain] Detected empty tool content but found code block in message. Repairing...")
                    code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", message_content, re.DOTALL)
                    if code_blocks:
                        best_block = max(code_blocks, key=len)
                        args['content'] = best_block
                        tool['function']['arguments'] = args
                        logger.info(f"[Brain] Repaired content with {len(best_block)} chars of code.")
    
    def _execute_tool(self, function_name, args):
        """Execute a single tool call and return the result"""
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Fire tool activity callback for live GUI updates
        if self.on_tool_activity:
            # Build a brief summary of the args for display
            if isinstance(args, dict):
                summary = args.get('file_path') or args.get('folder_path') or args.get('app_name') or args.get('command') or args.get('query') or args.get('url') or args.get('dir_path') or args.get('target_path') or ''
            else:
                summary = str(args)[:40]
            try:
                self.on_tool_activity(function_name, str(summary))
            except Exception:
                pass  # GUI callback errors must never break tool execution
        
        logger.info(f"Calling tool: {function_name} with {args}")
        
        # Ensure args is a dictionary to prevent .get() AttributeErrors
        if not isinstance(args, dict):
            args = {}
        
        try:
            if function_name == 'open_app':
                return self.system.open_app(args.get('app_name'))
            elif function_name == 'open_url':
                return self.system.open_url(args.get('url'))
            elif function_name == 'take_screenshot':
                return self.system.take_screenshot(args.get('filename'))
            elif function_name == 'search_web':
                return self.system.search_web(args.get('query'))
            elif function_name == 'write_file':
                fp = args.get('file_path', '')
                # Resolve relative paths against active project root
                fp = self._resolve_project_path(fp)
                # React guard: only activate if user explicitly asked for React
                if (self._user_wants_react 
                    and not self._ran_npx_create_react 
                    and re.search(r'src[/\\](?:App|index)\.(js|jsx|tsx?)$', fp)):
                    logger.warning(f"[Brain] Blocked React write without npx scaffold: {fp}")
                    return (
                        "ERROR: You must scaffold the React project FIRST. "
                        "Run: run_command('cd C:\\Users\\Chinmay\\Desktop && npx create-react-app <project-name>') "
                        "BEFORE writing any files. Do NOT create_folder or write_file first — npx does that for you. "
                        "Only call run_command with npx. Nothing else."
                    )
                # Read-before-write guard: warn if overwriting existing file without reading it first
                resolved_fp = Path(fp)
                if not resolved_fp.is_absolute():
                    # Try to resolve for existence check
                    if self._active_project_root:
                        check_path = Path(self._active_project_root) / fp
                    else:
                        check_path = Path.home() / 'Desktop' / fp
                    resolved_fp = check_path
                
                if resolved_fp.exists() and str(resolved_fp) not in self._read_files:
                    logger.warning(f"[Brain] Blocked blind write to existing file: {fp}")
                    try:
                        current_content = self.system.read_file(fp)
                        # Assume the AI implicitly reads it now to prevent infinite loops
                        self._read_files.add(str(resolved_fp))
                        return (
                            f"ERROR: BLIND WRITE BLOCKED. You tried to overwrite '{fp}' without reading it first.\n"
                            f"I have blocked this action and automatically fetched the file's current content for you:\n\n"
                            f"```\n{current_content}\n```\n\n"
                            f"Please review the code, and then call write_file again with the COMPLETE updated file content."
                        )
                    except Exception as e:
                        return f"ERROR: Blocked blind write to '{fp}', but failed to read it for context: {e}"
                
                # If it's a new file, or we've now read it, allow the write.
                self._read_files.add(str(resolved_fp))
                return self.system.write_file(fp, args.get('content'))
            elif function_name == 'read_file':
                fp = args.get('file_path', '')
                fp = self._resolve_project_path(fp)
                self._read_files.add(str(Path(fp).resolve()) if Path(fp).is_absolute() else fp)
                result = self.system.read_file(fp)
                # Also track the resolved path from the result
                if 'Successfully' not in str(result)[:20]:
                    # read_file returns content, track the resolved path
                    if self._active_project_root:
                        self._read_files.add(str(Path(self._active_project_root) / fp))
                return result
            elif function_name == 'create_folder':
                fp = args.get('folder_path', '')
                fp = self._resolve_project_path(fp)
                return self.system.create_folder(fp)
            elif function_name == 'list_dir':
                return self.system.list_dir(args.get('dir_path', '.'))
            elif function_name == 'run_command':
                cmd = args.get('command', '')
                result = self.system.run_command(cmd)
                # Track npx create-react-app success (not just attempt)
                if ('create-react-app' in cmd or 'create-vite' in cmd):
                    if 'conflict' not in str(result).lower() and 'error' not in str(result).lower()[:50]:
                        self._ran_npx_create_react = True
                        logger.info("[Brain] React scaffold succeeded — unlocking React file writes.")
                    else:
                        logger.warning(f"[Brain] React scaffold FAILED — guard remains active.")
                return result
            elif function_name == 'gather_context':
                target_path = args.get('target_path')
                # Set the active project root from the target path
                if target_path:
                    try:
                        resolved = self.system._resolve_path(target_path) if hasattr(self.system, '_resolve_path') else None
                        if resolved and Path(resolved).is_dir():
                            self._active_project_root = str(resolved)
                            logger.info(f"[Brain] Active project root set to: {self._active_project_root}")
                    except FileNotFoundError:
                        logger.warning(f"[Brain] Could not pre-resolve target_path '{target_path}' for active project root tracking.")
                result = self.context.gather_context(target_path=target_path) if target_path else self.context.gather_context()
                # Auto-detect existing React/framework projects
                result_lower = str(result).lower()
                if ('"react"' in result_lower or "'react'" in result_lower 
                    or 'from react' in result_lower or 'import react' in result_lower):
                    self._ran_npx_create_react = True
                    logger.info("[Brain] Detected existing React project from gather_context — React writes allowed.")
                return result
            else:
                return f"Unknown tool: {function_name}"
        except Exception as e:
            logger.error(f"Tool {function_name} failed: {e}", exc_info=True)
            return f"Error executing {function_name}: {e}"
    
    def _check_skeleton_code(self, file_path, content):
        """
        Check if written code is skeleton/placeholder.
        Only flags files with placeholder comments (TODO, "add here", etc.).
        Short-but-functional code is NOT flagged — length alone never triggers.
        Returns a warning string if skeleton detected, None otherwise.
        """
        if not content or not file_path:
            return None
        
        ext = Path(file_path).suffix.lower()
        
        # Placeholder patterns that indicate skeleton code
        placeholder_patterns = [
            r'//\s*(TODO|FIXME|add .* here|rest of|implement|placeholder)',
            r'#\s*(TODO|FIXME|add .* here|rest of|implement|placeholder)',
            r'<!--\s*(content|add|placeholder|TODO)',
            r'\{/\*\s*(content|add|placeholder|TODO|main content|goes here)',  # JSX comments
            r'//\s*\.\.\.\s*$',
            r'#\s*\.\.\.\s*$',
        ]
        
        # Only flag if placeholder patterns are found
        has_placeholder = False
        for pattern in placeholder_patterns:
            if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                has_placeholder = True
                break
        
        if not has_placeholder:
            return None
        
        basename = Path(file_path).name
        warning = f"WARNING: {basename} contains placeholder/TODO comments"
        
        # Add length context if the file is also suspiciously short (or if it's JUST short)
        min_lengths = {'.html': 600, '.css': 400, '.js': 300, '.py': 200}
        min_len = min_lengths.get(ext, 0)
        
        # If it doesn't have placeholders, but is EXTREMELY short, flag it anyway
        if not has_placeholder and min_len and len(content) < min_len:
            warning = f"WARNING: {basename} is too short ({len(content)} chars) and appears to be incomplete skeleton code."
            return warning
            
        if not has_placeholder:
            return None
            
        if min_len and len(content) < min_len:
            warning += f" and is very short ({len(content)} chars)"
        
        return warning
    
    def _generate_project_blueprint(self, user_request):
        """
        PHASE 1 of creation: Generate a detailed project blueprint/specification.
        
        Instead of sending the raw user request ("make a mario game") directly to
        the code generator, we first ask the LLM to think deeply about what the
        project needs — features, mechanics, visual design, file structure, etc.
        
        This blueprint then becomes the detailed prompt for Phase 2 (code generation),
        resulting in dramatically better output.
        """
        logger.info("[Brain] Phase 1: Generating project blueprint...")
        
        text_lower = user_request.lower()
        is_python = 'python' in text_lower or '.py' in text_lower
        is_game = any(kw in text_lower for kw in ['game', 'chess', 'sudoku', 'quiz', 'puzzle', 'shooter', 'platformer', 'snake', 'tetris', 'pong', 'mario'])
        
        if is_game:
            planning_prompt = [
                {"role": "system", "content": (
                    "You are a senior game designer and developer. The user wants to create a game.\n"
                    "Your job is to create a DETAILED BLUEPRINT for this game.\n\n"
                    "CRITICAL RULES:\n"
                    "- Organize the code logically. Use multiple files if it improves readability and structure (e.g., separating logic, assets, and UI).\n"
                    "- Do NOT over-engineer the architecture, but ensure it is scalable and highly polished.\n\n"
                    "Include ALL of the following in your blueprint:\n"
                    "1. GAME CONCEPT: What the game is, core gameplay loop, how to win/lose\n"
                    "2. IMPLEMENTATION LOGIC: Game loop logic, physics math, state management, collision algorithms\n"
                    "3. VISUAL DESIGN: Art style, color palette, background, character/sprite descriptions, particle effects, animations\n"
                    "4. UI/HUD: Score display, health bar, start screen, game over screen, pause menu\n"
                    "5. SOUND & FEEDBACK: Visual feedback for actions (screen shake, flash effects, particle bursts)\n"
                    "6. POLISH DETAILS: Smooth animations, easing functions, responsive design, performance optimizations\n\n"
                    "Be SPECIFIC and DETAILED. For example, don't say 'add enemies' — say 'enemies spawn every 3 seconds from the right side, "
                    "move left at 2px/frame, have a red glow effect, and explode into 8 particles when destroyed.'\n\n"
                    "Output ONLY the blueprint, no code. Keep it under 400 words but make every word count."
                )},
                {"role": "user", "content": user_request}
            ]
        elif is_python:
            planning_prompt = [
                {"role": "system", "content": (
                    "You are a senior Python developer. The user wants to create a Python project.\n"
                    "Your job is to create a DETAILED BLUEPRINT for this project.\n\n"
                    "CRITICAL RULES:\n"
                    "- Organize the code logically into appropriate modules/files (e.g., separating logic, data models, and CLI/UI).\n"
                    "- Do NOT over-engineer, but ensure the architecture is clean and maintainable.\n\n"
                    "Include ALL of the following:\n"
                    "1. PROJECT PURPOSE: What it does, who it's for, core features\n"
                    "2. IMPLEMENTATION LOGIC: Step-by-step logic, core algorithms, and data flow\n"
                    "3. KEY FEATURES: List every feature with specific technical implementation details\n"
                    "4. ERROR HANDLING: Edge cases, input validation, user-friendly error messages\n"
                    "5. USER EXPERIENCE: CLI interface design, output formatting, colors (if applicable)\n"
                    "6. LIBRARIES: Which specific third-party or built-in libraries to use and why\n\n"
                    "Be SPECIFIC. Don't say 'handle errors' — say 'validate user input is a positive integer, "
                    "catch FileNotFoundError with a helpful message showing the expected path.'\n\n"
                    "Output ONLY the blueprint, no code. Keep it under 400 words."
                )},
                {"role": "user", "content": user_request}
            ]
        else:
            # Web app / website
            planning_prompt = [
                {"role": "system", "content": (
                    "You are a senior web developer and UI/UX designer. The user wants to create a web project.\n"
                    "Your job is to create a DETAILED BLUEPRINT for this project.\n\n"
                    "CRITICAL RULES:\n"
                    "- Organize code cleanly (e.g., separating HTML, modular CSS, and modular JS files if complex).\n"
                    "- Focus heavily on high-end design, animations, and robust implementation logic.\n\n"
                    "Include ALL of the following:\n"
                    "1. PROJECT CONCEPT: What it is, core features, target user experience\n"
                    "2. IMPLEMENTATION LOGIC: How to manage state, DOM manipulation logic, API calls, or local storage\n"
                    "3. PAGE LAYOUT: Header, navigation, main content sections, sidebar (if any), footer\n"
                    "4. VISUAL DESIGN: Color scheme (specific hex codes), typography, spacing, dark/light theme\n"
                    "5. INTERACTIVE ELEMENTS: Buttons, forms, modals, dropdowns, hover effects\n"
                    "6. RESPONSIVE DESIGN: Mobile vs desktop layout differences, breakpoints\n"
                    "7. POLISH: CSS animations, transitions, glassmorphism, gradients, micro-interactions\n\n"
                    "Be SPECIFIC. Don't say 'add a nice background' — say 'radial gradient from #1a1a2e to #16213e, "
                    "with a subtle animated grid pattern overlay at 10% opacity.'\n\n"
                    "Output ONLY the blueprint, no code. Keep it under 400 words."
                )},
                {"role": "user", "content": user_request}
            ]
        
        try:
            saved_stream_callback = self.on_stream_token
            self.on_stream_token = None
            
            try:
                # Always use Groq for blueprint generation (better quality than local Qwen)
                logger.info("[Brain] Using Groq for blueprint generation...")
                response = self._call_llm_with_retry(self.primary_model, planning_prompt)
                blueprint = response.get('message', {}).get('content', '')
            finally:
                self.on_stream_token = saved_stream_callback
            
            if blueprint.strip():
                logger.info(f"[Brain] Blueprint generated ({len(blueprint)} chars)")
                logger.debug(f"[Brain] Blueprint preview: {blueprint[:200]}...")
                return blueprint.strip()
            else:
                logger.warning("[Brain] Blueprint generation returned empty — using raw request")
                return None
                
        except Exception as e:
            error_str = str(e).lower()
            # On rate limit, skip blueprint silently — code gen will still work (or fail with a clear message)
            if '429' in str(e) or 'rate_limit' in error_str or 'rate limit' in error_str:
                logger.warning(f"[Brain] Blueprint skipped due to rate limit — will use raw request")
            else:
                logger.warning(f"[Brain] Blueprint generation failed: {e} — falling back to raw request")
            return None
    
    def _force_execute_creation(self, user_request):
        """
        Two-phase creation with user approval:
        Phase 1: Generate a detailed project blueprint via Groq → show to user (if requested)
        Phase 2 (after approval or if bypassed): Use the blueprint/request to generate high-quality code
        """
        text_lower = user_request.lower()
        wants_blueprint = any(kw in text_lower for kw in ['blueprint', 'plan', 'design first', 'architect'])
        
        if not wants_blueprint:
            logger.info("[Brain] Bypassing blueprint phase (not explicitly requested)")
            return self._execute_creation_with_blueprint(user_request, None)
            
        logger.info("[Brain] Phase 1: Generating blueprint for user approval")
        
        # --- PHASE 1: Generate blueprint and ask for approval ---
        blueprint = self._generate_project_blueprint(user_request)
        
        if blueprint:
            # Store the pending blueprint and request for later
            self._pending_blueprint = blueprint
            self._pending_request = user_request
            self._pending_is_react = False
            
            # Notify the GUI to show the blueprint for approval
            if self.on_blueprint_ready:
                self.on_blueprint_ready(blueprint, user_request)
            
            return f"📋 Here's my blueprint for your project:\n\n{blueprint}\n\n✅ Type 'yes' or 'ok' to approve, or tell me what to change."
        else:
            # Blueprint generation failed — fall back to direct creation
            logger.warning("[Brain] Blueprint failed, falling back to direct creation")
            return self._execute_creation_with_blueprint(user_request, None)
    
    def approve_blueprint(self):
        """
        Called when the user approves the pending blueprint.
        Proceeds to Phase 2: code generation using the approved blueprint.
        """
        if not self._pending_blueprint or not self._pending_request:
            return "No pending blueprint to approve."
        
        blueprint = self._pending_blueprint
        user_request = self._pending_request
        is_react = self._pending_is_react
        
        # Clear pending state
        self._pending_blueprint = None
        self._pending_request = None
        self._pending_is_react = False
        
        if is_react:
            return self._execute_react_creation_with_blueprint(user_request, blueprint)
        else:
            return self._execute_creation_with_blueprint(user_request, blueprint)
    
    def reject_blueprint(self, feedback):
        """
        Called when the user rejects or wants changes to the blueprint.
        Re-generates the blueprint incorporating user feedback.
        """
        if not self._pending_request:
            return "No pending blueprint to modify."
        
        original_request = self._pending_request
        
        # Clear pending state
        self._pending_blueprint = None
        self._pending_request = None
        self._pending_is_react = False
        
        # Re-run creation with the feedback appended
        modified_request = f"{original_request}. Additional requirements: {feedback}"
        return self._force_execute_creation(modified_request)
    
    def _execute_creation_with_blueprint(self, user_request, blueprint):
        """
        Phase 2 of creation: Generate code using an approved blueprint.
        This is the actual code generation step that writes files to disk.
        """
        logger.info("[Brain] Phase 2: Generating code from approved blueprint")
        
        desktop_path = str(Path.home() / "Desktop")
        project_name = self._extract_project_name(user_request)
        project_path = f"{desktop_path}/{project_name}"
        
        # Detect if this is a Python/Java/non-web request
        text_lower_req = user_request.lower()
        is_python = 'python' in text_lower_req or '.py' in text_lower_req
        is_java = 'java' in text_lower_req.split() or 'in java' in text_lower_req
        
        if is_java:
            file_examples = (
                "=== FILE: Main.java ===\n"
                "(complete Java code here)\n\n"
            )
            file_rules = (
                "- Create the main Java file (e.g. Main.java)\n"
                "- Ensure the public class name exactly matches the filename\n"
            )
            is_backend = True
        elif is_python:
            file_examples = (
                "=== FILE: main.py ===\n"
                "(complete Python code here)\n\n"
            )
            file_rules = (
                "- Create the main Python file (main.py or app.py)\n"
                "- Add any helper files if needed\n"
            )
            is_backend = True
        else:
            file_examples = (
                "=== FILE: index.html ===\n"
                "(complete HTML code here)\n\n"
                "=== FILE: style.css ===\n"
                "(complete CSS code here)\n\n"
                "=== FILE: script.js ===\n"
                "(complete JS code here)\n\n"
            )
            file_rules = (
                "- Create at least: index.html, style.css, script.js\n"
            )
            is_backend = False
        
        # Build blueprint context for the code generation prompt
        blueprint_context = ""
        if blueprint:
            blueprint_context = (
                f"\n\nDETAILED PROJECT BLUEPRINT (follow this closely):\n"
                f"─────────────────────────────────────────────────\n"
                f"{blueprint}\n"
                f"─────────────────────────────────────────────────\n\n"
                f"Implement ALL features described in the blueprint above. "
                f"The blueprint is your specification — follow it precisely.\n"
            )
        
        code_prompt = [
            {"role": "system", "content": (
                "You are an expert code generator. The user wants you to create a project.\n"
                "Output the COMPLETE file contents for each file.\n"
                f"{blueprint_context}"
                "\nIMPORTANT FORMAT — you MUST use this EXACT format for EACH file:\n\n"
                f"{file_examples}"
                "Rules:\n"
                f"{file_rules}"
                "- Write COMPLETE, WORKING, PRODUCTION-QUALITY code for each file\n"
                "- Do NOT use placeholders, TODO comments, or skeleton code\n"
                "- Do NOT wrap code in markdown code fences (no ```)\n"
                "- ONLY output the === FILE: name === headers and the code\n"
                "- Implement ALL features from the blueprint — every detail matters\n"
                "- Make it visually stunning with modern CSS, smooth animations, and polished UI\n"
            )},
            {"role": "user", "content": user_request}
        ]
        
        try:
            # IMPORTANT: Temporarily disable streaming so raw code doesn't
            # appear in the chat. We want to show only the clean summary.
            saved_stream_callback = self.on_stream_token
            self.on_stream_token = None
            
            try:
                response = self._call_llm_with_retry(self.primary_model, code_prompt)
            finally:
                self.on_stream_token = saved_stream_callback
            
            raw_content = response.get('message', {}).get('content', '')
            
            if not raw_content.strip():
                return "I had trouble generating the code. Please try again with a simpler request."
            
            logger.info(f"[Brain] Got code response ({len(raw_content)} chars)")
            
            # Create the project folder
            self.system.create_folder(project_path)
            logger.info(f"[Brain] Created project folder: {project_path}")
            
            # Parse files from the response using multiple strategies
            files_written = self._parse_and_write_files(raw_content, project_path)
            
            if files_written:
                # Remember this project for follow-up modifications
                self._last_created_project = project_path
                run_hint = ""
                if is_java:
                    main_file = "Main.java"
                    java_files = [f for f in files_written if f.endswith('.java')]
                    if "Main.java" not in files_written and java_files:
                        main_file = java_files[0]
                    class_name = main_file.replace('.java', '')
                    run_hint = f"Compile and run with: javac {main_file} && java {class_name}"
                elif is_python:
                    main_file = "main.py"
                    if "app.py" in files_written: main_file = "app.py"
                    elif "main.py" not in files_written:
                        py_files = [f for f in files_written if f.endswith('.py')]
                        if py_files: main_file = py_files[0]
                    run_hint = f"Run it with: python {main_file}"
                else:
                    if "index.html" in files_written:
                        run_hint = "Open index.html in your browser to try it!"
                
                summary_text = f"Created '{project_name}' on your Desktop with: {', '.join(files_written)}."
                if run_hint:
                    summary_text += f" {run_hint}"
                    
                self.messages.append({"role": "assistant", "content": summary_text})
                return summary_text
            else:
                logger.error(f"[Brain] Could not parse any files from response:\n{raw_content[:500]}")
                return "I generated code but couldn't parse it into files. Please try again."
                
        except Exception as e:
            logger.error(f"[Brain] Force-execute creation failed: {e}", exc_info=True)
            error_str = str(e)
            if '429' in error_str or 'rate_limit' in error_str.lower() or 'rate limit' in error_str.lower():
                wait_match = re.search(r'try again in (\d+h\d+m[\d.]+s|\d+m[\d.]+s|\d+s)', error_str, re.IGNORECASE)
                wait_time = wait_match.group(1) if wait_match else "a while"
                return (
                    f"⏳ I've hit the Groq API rate limit. "
                    f"Please try again in {wait_time}, or switch to local Ollama "
                    f"by setting ASSISTANT_USE_GROQ=false in your .env file."
                )
            return f"Sorry, I encountered an error while creating the project: {e}"
    
    def _force_execute_modification(self, user_request, project_path):
        """
        Read existing project files, ask the model to improve them,
        then write the improved files back.
        
        Smart file selection:
        - If the user mentions specific files (e.g. "fix App.jsx"), only read those
        - Otherwise, read all code files (but skip large/lock files)
        - For React projects, also check src/ subdirectory
        """
        logger.info(f"[Brain] Force-executing modification on '{project_path}'")
        
        import os
        
        # --- Smart file detection: extract mentioned filenames from user request ---
        mentioned_files = re.findall(
            r'(\w+\.(?:html|css|jsx?|tsx?|py|json|vue))',
            user_request, re.IGNORECASE
        )
        
        # Build list of directories to scan (project root + src/ for React)
        scan_dirs = [project_path]
        src_path = os.path.join(project_path, 'src')
        if os.path.isdir(src_path):
            scan_dirs.append(src_path)
        
        # Read files
        existing_files = {}
        try:
            available_files = []
            file_paths = {}
            for scan_dir in scan_dirs:
                if not os.path.exists(scan_dir):
                    continue
                for fname in os.listdir(scan_dir):
                    fpath = os.path.join(scan_dir, fname)
                    if not os.path.isfile(fpath):
                        continue
                    if not fname.endswith(('.html', '.css', '.js', '.jsx', '.tsx', '.py', '.json', '.vue')):
                        continue
                    if 'lock' in fname.lower() or os.path.getsize(fpath) > 15000:
                        continue
                    
                    rel_name = fname if scan_dir == project_path else f"src/{fname}"
                    available_files.append(rel_name)
                    file_paths[rel_name] = fpath

            # Ask LLM to select files if none were explicitly mentioned and there are multiple options
            if not mentioned_files and len(available_files) > 2:
                logger.info(f"[Brain] Too many files ({len(available_files)}), asking LLM to filter based on request...")
                selection_prompt = [
                    {"role": "system", "content": "You are a file selector. Based on the user's request, identify WHICH files need to be modified. Output ONLY a comma-separated list of filenames from the provided available files list. Output nothing else. If unsure, output the 1-2 most likely files."},
                    {"role": "user", "content": f"Available files: {', '.join(available_files)}\n\nUser request: {user_request}"}
                ]
                try:
                    # Use fast model to save tokens/rate limits for simple tasks like file routing
                    resp = self._call_llm_with_retry(self.fast_model, selection_prompt)
                    selected_text = resp.get('message', {}).get('content', '')
                    logger.info(f"[Brain] LLM suggested files: {selected_text}")
                    for f in available_files:
                        if f in selected_text or f.split('/')[-1] in selected_text:
                            mentioned_files.append(f)
                except Exception as e:
                    logger.warning(f"[Brain] File selection failed: {e}")

            # Read selected or all files
            for rel_name in available_files:
                fpath = file_paths[rel_name]
                fname = rel_name.split('/')[-1]
                
                if mentioned_files:
                    if not any(m.lower() == rel_name.lower() or m.lower() == fname.lower() for m in mentioned_files):
                        continue
                
                content = self.system.read_file(fpath)
                if not content.startswith('Error'):
                    existing_files[rel_name] = content
        except Exception as e:
            logger.error(f"[Brain] Could not read project files: {e}")
            return f"Could not read the project files at '{project_path}': {e}"
        
        if not existing_files:
            if mentioned_files:
                return f"Could not find {', '.join(mentioned_files)} in the project. Check the filename and try again."
            return f"No code files found in '{project_path}' to modify."
        
        logger.info(f"[Brain] Reading {len(existing_files)} file(s) for modification: {list(existing_files.keys())}")
        
        # Build context of selected files
        files_context = '\n'.join(
            f'=== FILE: {name} ===\n{content}\n' 
            for name, content in existing_files.items()
        )
        
        # Determine if this is a targeted single-file edit
        is_single_file = len(existing_files) == 1
        file_list_str = ', '.join(existing_files.keys())
        
        if is_single_file:
            output_instruction = (
                f"Rewrite the file below with the improvements applied.\n"
                f"Output the COMPLETE updated file using this format:\n\n"
                f"=== FILE: {list(existing_files.keys())[0]} ===\n"
                f"(complete improved code here)\n"
            )
        else:
            output_instruction = (
                "Rewrite ALL files below with the improvements applied.\n\n"
                "IMPORTANT FORMAT — you MUST use this EXACT format for EACH file:\n\n"
                "=== FILE: filename.ext ===\n"
                "(complete improved code here)\n"
            )
        
        # Ask model to improve
        code_prompt = [
            {"role": "system", "content": (
                f"You are a code improver. The user has an existing project and wants improvements.\n"
                f"Below are the current file(s): {file_list_str}\n\n"
                f"{output_instruction}\n"
                "Rules:\n"
                "- Write COMPLETE files, not just the changed parts\n"
                "- Do NOT wrap code in markdown code fences (no ```)\n"
                "- ONLY output the === FILE: name === headers and the code\n"
                "- Make the code production-quality with proper imports, error handling, and styling\n"
            )},
            {"role": "user", "content": (
                f"Here are the current project files:\n\n{files_context}\n\n"
                f"User request: {user_request}"
            )}
        ]
        
        try:
            saved_stream_callback = self.on_stream_token
            self.on_stream_token = None
            
            try:
                response = self._call_llm_with_retry(self.primary_model, code_prompt)
            finally:
                self.on_stream_token = saved_stream_callback
            
            raw_content = response.get('message', {}).get('content', '')
            
            if not raw_content.strip():
                return "I had trouble improving the code. Please try again."
            
            # Write files — try project root first, then src/ for React projects
            files_written = self._parse_and_write_files(raw_content, project_path)
            
            # If files have src/ prefix or we're in a React project, also try writing to src/
            if not files_written and os.path.isdir(src_path):
                files_written = self._parse_and_write_files(raw_content, src_path)
            
            if files_written:
                summary = f"Updated {', '.join(files_written)} in your project. Refresh to see the changes!"
                self.messages.append({"role": "assistant", "content": summary})
                return summary
            else:
                return "I tried to improve the project but couldn't parse the updated code. Please try again."
                
        except Exception as e:
            logger.error(f"[Brain] Force-execute modification failed: {e}", exc_info=True)
            error_str = str(e)
            if '429' in error_str or 'rate_limit' in error_str.lower() or 'rate limit' in error_str.lower():
                wait_match = re.search(r'try again in (\d+h\d+m[\d.]+s|\d+m[\d.]+s|\d+s)', error_str, re.IGNORECASE)
                wait_time = wait_match.group(1) if wait_match else "a while"
                return (
                    f"⏳ I've hit the Groq API rate limit. "
                    f"Please try again in {wait_time}, or switch to local Ollama "
                    f"by setting ASSISTANT_USE_GROQ=false in your .env file."
                )
            return f"Sorry, I encountered an error while improving the project: {e}"
    
    def _force_execute_with_fallback(self, user_request):
        """Smart dispatcher: modify existing project if one exists, otherwise create new."""
        if hasattr(self, '_last_created_project'):
            logger.info(f"[Brain] Routing to modification for '{self._last_created_project}'")
            return self._force_execute_modification(user_request, self._last_created_project)
        else:
            logger.info("[Brain] No recent project — routing to creation")
            return self._force_execute_creation(user_request)
    
    def _force_execute_react_creation(self, user_request):
        """
        Handle React/Vue/Next.js project creation with blueprint approval (if requested).
        Phase 1: Generate blueprint via Groq and show to user.
        Phase 2 (after approval or bypassed): Scaffold + generate code.
        """
        text_lower = user_request.lower()
        wants_blueprint = any(kw in text_lower for kw in ['blueprint', 'plan', 'design first', 'architect'])
        
        if not wants_blueprint:
            logger.info("[Brain] Bypassing React blueprint phase (not explicitly requested)")
            return self._execute_react_creation_with_blueprint(user_request, None)
            
        logger.info("[Brain] Phase 1 (React): Generating blueprint for user approval")
        
        blueprint = self._generate_project_blueprint(user_request)
        
        if blueprint:
            self._pending_blueprint = blueprint
            self._pending_request = user_request
            self._pending_is_react = True
            
            if self.on_blueprint_ready:
                self.on_blueprint_ready(blueprint, user_request)
            
            return f"📋 Here's my blueprint for your project:\n\n{blueprint}\n\n✅ Type 'yes' or 'ok' to approve, or tell me what to change."
        else:
            logger.warning("[Brain] Blueprint failed, falling back to direct React creation")
            return self._execute_react_creation_with_blueprint(user_request, None)
    
    def _execute_react_creation_with_blueprint(self, user_request, blueprint):
        """
        Phase 2 of React creation: Scaffold the project and generate code
        using the approved blueprint.
        """
        import os
        logger.info("[Brain] Phase 2 (React): Generating code from approved blueprint")
        
        desktop_path = str(Path.home() / "Desktop")
        project_name = self._extract_project_name(user_request)
        project_path = f"{desktop_path}/{project_name}"
        text_lower = user_request.lower()
        
        # Detect which framework
        use_npx = False
        if 'next' in text_lower or 'next.js' in text_lower:
            scaffold_cmd = f'npx -y create-next-app@latest "{project_path}" --yes'
            framework = 'Next.js'
            src_subdir = 'src'
            main_file = 'page.js'
            css_file = 'page.module.css'
            use_npx = True
        elif 'vue' in text_lower:
            scaffold_cmd = f'npx -y create-vue@latest "{project_path}" --default'
            framework = 'Vue'
            src_subdir = 'src'
            main_file = 'App.vue'
            css_file = 'style.css'
            use_npx = True
        else:
            # React + Vite — scaffold INLINE (no npx needed)
            framework = 'React (Vite)'
            src_subdir = 'src'
            main_file = 'App.jsx'
            css_file = 'App.css'
        
        try:
            if use_npx:
                logger.info(f"[Brain] Running scaffold: {scaffold_cmd}")
                result = self.system.run_command(scaffold_cmd)
                logger.info(f"[Brain] Scaffold result: {result[:200]}")
                
                if not os.path.isdir(project_path):
                    logger.error(f"[Brain] Scaffold failed — project directory not created: {project_path}")
                    return f"Failed to create {framework} project. Output: {result[:300]}"
            else:
                logger.info(f"[Brain] Creating Vite+React scaffold inline at: {project_path}")
                self._create_vite_react_scaffold(project_path, project_name)
                logger.info(f"[Brain] Vite scaffold created successfully")
            
            self._last_created_project = project_path
            self._ran_npx_create_react = True
            
            blueprint_context = ""
            if blueprint:
                blueprint_context = (
                    f"\n\nDETAILED PROJECT BLUEPRINT (follow this closely):\n"
                    f"─────────────────────────────────────────────────\n"
                    f"{blueprint}\n"
                    f"─────────────────────────────────────────────────\n\n"
                    f"Implement ALL features described in the blueprint above.\n"
                )
            
            logger.info(f"[Brain] Generating custom {framework} components...")
            saved_stream_callback = self.on_stream_token
            self.on_stream_token = None
            
            try:
                code_prompt = [
                    {"role": "system", "content": (
                        f"You are a senior {framework} developer. A new {framework} project exists at: {project_path}\n\n"
                        f"Generate the COMPLETE custom component code to fulfill the user's request.\n"
                        f"The project uses React with Vite, so write modern React code with hooks.\n"
                        f"{blueprint_context}"
                        f"\nCRITICAL FORMAT — use this EXACT format for EACH file:\n\n"
                        f"=== FILE: {main_file} ===\n"
                        f"(complete component code with all features)\n\n"
                        f"=== FILE: {css_file} ===\n"
                        f"(complete, beautiful CSS with dark theme, gradients, hover effects)\n\n"
                        f"IMPORTANT RULES:\n"
                        f"- Do NOT include 'src/' prefix in filenames — just use '{main_file}', '{css_file}'\n"
                        f"- Write COMPLETE, WORKING, BEAUTIFUL code — not a skeleton\n"
                        f"- Implement ALL features from the blueprint — every detail matters\n"
                        f"- Use modern CSS: dark theme (#121212 background), gradients, smooth transitions, hover effects, proper spacing\n"
                        f"- Use proper React imports and JSX syntax\n"
                        f"- Do NOT wrap code in markdown code fences (no ```)\n"
                        f"- ONLY output === FILE: === headers and the code\n"
                        f"- You may create additional component files if needed (e.g. Sidebar.jsx, Player.jsx)\n"
                    )},
                    {"role": "user", "content": user_request}
                ]
                
                response = self._call_llm_with_retry(self.primary_model, code_prompt)
                raw_content = response.get('message', {}).get('content', '')
                
                if raw_content.strip():
                    logger.info(f"[Brain] Got custom code response ({len(raw_content)} chars)")
                    
                    src_path = os.path.join(project_path, src_subdir)
                    if not os.path.exists(src_path):
                        os.makedirs(src_path, exist_ok=True)
                    
                    files_written = self._parse_and_write_files(raw_content, src_path)
                    
                    if files_written:
                        logger.info(f"[Brain] Successfully wrote custom components: {files_written}")
                    else:
                        logger.warning(f"[Brain] Custom code could not be parsed. Raw starts with: {raw_content[:200]}")
                else:
                    logger.warning("[Brain] Custom code generation returned empty — defaults remain")
                    
            except Exception as e:
                logger.error(f"[Brain] Custom code generation failed (scaffold still exists): {e}", exc_info=True)
            finally:
                self.on_stream_token = saved_stream_callback
            
            summary = (
                f"Created {framework} project '{project_name}' on your Desktop!\n"
                f"To run it:\n"
                f"  cd Desktop/{project_name}\n"
                f"  npm install && npm run dev"
            )
            self.messages.append({"role": "assistant", "content": summary})
            return summary
            
        except Exception as e:
            logger.error(f"[Brain] React creation failed: {e}", exc_info=True)
            return f"Error creating {framework} project: {e}"
    
    def _create_vite_react_scaffold(self, project_path, project_name):
        """
        Create a Vite + React project scaffold instantly, with no npx or network.
        This writes the exact same files that 'npx create-vite --template react' would.
        """
        import os
        
        # Create directory structure
        os.makedirs(os.path.join(project_path, 'src'), exist_ok=True)
        os.makedirs(os.path.join(project_path, 'public'), exist_ok=True)
        
        # 1. package.json
        self.system.write_file(f"{project_path}/package.json", f'''{{"name": "{project_name}",
  "private": true,
  "version": "0.0.0",
  "type": "module",
  "scripts": {{
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  }},
  "dependencies": {{
    "react": "^18.3.1",
    "react-dom": "^18.3.1"
  }},
  "devDependencies": {{
    "@types/react": "^18.3.3",
    "@types/react-dom": "^18.3.0",
    "@vitejs/plugin-react": "^4.3.1",
    "vite": "^5.4.0"
  }}
}}
''')
        
        # 2. vite.config.js
        self.system.write_file(f"{project_path}/vite.config.js", """import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
})
""")
        
        # 3. index.html (Vite entry point — lives in project root, not public/)
        self.system.write_file(f"{project_path}/index.html", f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{project_name.replace('-', ' ').title()}</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
""")
        
        # 4. src/main.jsx (React entry point)
        self.system.write_file(f"{project_path}/src/main.jsx", """import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
""")
        
        # 5. src/index.css (global reset styles)
        self.system.write_file(f"{project_path}/src/index.css", """* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen,
    Ubuntu, Cantarell, 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  background: #121212;
  color: #ffffff;
}
""")
        
        # 6. src/App.jsx (placeholder — will be overwritten by LLM)
        self.system.write_file(f"{project_path}/src/App.jsx", """import './App.css'

function App() {
  return (
    <div className="App">
      <h1>Loading...</h1>
    </div>
  )
}

export default App
""")
        
        # 7. src/App.css (placeholder — will be overwritten by LLM)
        self.system.write_file(f"{project_path}/src/App.css", """/* Will be replaced with custom styles */
.App {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
}
""")
        
        logger.info(f"[Brain] Vite+React scaffold created: 7 files in {project_path}")
    
    def _extract_project_name(self, user_request):
        """Extract project name from user request text."""
        text_lower = user_request.lower()
        
        # Try explicit naming: "called X", "named X"
        m = re.search(r'(?:called|named)\s+["\']?([\w\-\.]+)', user_request, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        
        # Try "create a X game/app/clone" (greedy match for multi-word names like tic-tac-toe)
        m = re.search(r'(?:create|build|make|get)\s+(?:me\s+)?(?:a\s+)?(?:new\s+)?(.+?)\s+(?:web\s*app|app|website|game|clone|project|page|in\s+python|script|program)', text_lower)
        if m:
            name = m.group(1).strip()
            name = re.sub(r'\b(simple|basic|cool|nice|like|similar|chrome|google|similar to|view of)\b', '', name).strip()
            if name and len(name) > 1:
                return name.replace(' ', '-')
        
        # Try to find a noun-like keyword after creation verbs
        m = re.search(r'(?:create|build|make|get)\s+(?:me\s+)?(?:a\s+)?(?:new\s+)?([\w-]+)', text_lower)
        if m:
            name = m.group(1).strip()
            # Filter out generic words
            if name not in ('me', 'a', 'an', 'the', 'my', 'new', 'simple', 'basic'):
                return name
        
        return "my-project"
    
    def _parse_and_write_files(self, raw_content, project_path):
        """Parse code response and write files. Handles multiple output formats."""
        files_written = []
        
        # Safety: if project_path already ends in /src, strip src/ prefix from filenames
        # to prevent writing to src/src/App.js
        base_is_src = project_path.rstrip('/\\').endswith('src')
        
        def _safe_path(filename):
            """Strip redundant src/ prefix if base is already src/"""
            if base_is_src and filename.startswith('src/'):
                filename = filename[4:]  # Remove 'src/' prefix
                logger.debug(f"[Brain] Stripped redundant 'src/' prefix → {filename}")
            return filename
        
        # Strategy 1: === FILE: name === or === FILENAME: name ===
        file_blocks = re.split(r'===\s*(?:FILE|FILENAME):\s*(.+?)\s*===', raw_content)
        if len(file_blocks) > 1:
            for i in range(1, len(file_blocks), 2):
                if i + 1 < len(file_blocks):
                    filename = _safe_path(file_blocks[i].strip())
                    content = file_blocks[i + 1].strip()
                    # Strip opening markdown fence (```css, ```jsx, etc.)
                    content = re.sub(r'^```\w*\n?', '', content)
                    # Strip closing markdown fence AND any trailing text/explanation after it
                    # (models often add "This code provides..." after the closing ```)
                    content = re.sub(r'\n?```[\s\S]*', '', content)
                    if content:
                        self.system.write_file(f"{project_path}/{filename}", content)
                        files_written.append(filename)
                        logger.info(f"[Brain] Wrote (=== pattern): {filename} ({len(content)} chars)")
            if files_written:
                return files_written
        
        # Strategy 2: ### FILENAME: name or ### name or #### name
        header_blocks = re.split(r'#{2,4}\s*(?:FILENAME:\s*)?(\S+\.(?:html|css|jsx?|tsx?|py|json|vue))', raw_content)
        if len(header_blocks) > 1:
            for i in range(1, len(header_blocks), 2):
                if i + 1 < len(header_blocks):
                    filename = _safe_path(header_blocks[i].strip())
                    content = header_blocks[i + 1].strip()
                    # Extract code from markdown fences
                    code_match = re.search(r'```\w*\n(.*?)```', content, re.DOTALL)
                    if code_match:
                        content = code_match.group(1).strip()
                    if content:
                        self.system.write_file(f"{project_path}/{filename}", content)
                        files_written.append(filename)
                        logger.info(f"[Brain] Wrote (### pattern): {filename} ({len(content)} chars)")
            if files_written:
                return files_written
        
        # Strategy 3: **filename** followed by code block
        bold_pattern = re.findall(r'\*\*(\S+\.(?:html|css|jsx?|tsx?|py|json|vue))\*\*.*?```\w*\n(.*?)```', raw_content, re.DOTALL)
        if bold_pattern:
            for filename, code in bold_pattern:
                if code.strip():
                    self.system.write_file(f"{project_path}/{filename}", code.strip())
                    files_written.append(filename)
                    logger.info(f"[Brain] Wrote (bold pattern): {filename} ({len(code)} chars)")
            if files_written:
                return files_written
        
        # Strategy 4: code blocks with filename on the line before
        lines = raw_content.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            # Check if this line mentions a filename
            fname_match = re.search(r'(\w+\.(?:html|css|jsx?|tsx?|py|json|vue))', line)
            # Check if the next line starts a code block
            if fname_match and i + 1 < len(lines) and lines[i + 1].strip().startswith('```'):
                filename = _safe_path(fname_match.group(1))
                # Find the end of the code block
                code_lines = []
                j = i + 2  # skip the ``` line
                while j < len(lines) and not lines[j].strip().startswith('```'):
                    code_lines.append(lines[j])
                    j += 1
                if code_lines:
                    content = '\n'.join(code_lines)
                    self.system.write_file(f"{project_path}/{filename}", content)
                    files_written.append(filename)
                    logger.info(f"[Brain] Wrote (context pattern): {filename} ({len(content)} chars)")
                i = j + 1
                continue
            i += 1
        if files_written:
            return files_written
        
        # Strategy 5 (last resort): code blocks with language hints
        code_blocks = re.findall(r'```(\w+)?\n(.*?)```', raw_content, re.DOTALL)
        ext_map = {'html': 'index.html', 'css': 'style.css', 'javascript': 'script.js', 
                   'js': 'script.js', 'jsx': 'App.jsx', 'tsx': 'App.tsx', 'python': 'app.py'}
        used_names = set()
        for lang, code in code_blocks:
            lang = (lang or '').lower()
            filename = ext_map.get(lang)
            if not filename or filename in used_names or not code.strip():
                continue
            used_names.add(filename)
            self.system.write_file(f"{project_path}/{filename}", code.strip())
            files_written.append(filename)
            logger.info(f"[Brain] Wrote (lang-hint): {filename} ({len(code)} chars)")
        
        return files_written
    
    def _needs_followup(self, tool_calls):
        """Check if tool results need a follow-up LLM call for a natural answer"""
        followup_tools = {'read_file', 'gather_context'}
        return any(t['function']['name'] in followup_tools for t in tool_calls)

    def _extract_api_details(self, context_output):
        """
        Extract API base URL, routes, and model fields from gather_context output.
        Returns a formatted string to inject into the nudge, or empty string.
        """
        if not context_output:
            return ''
        
        details = []
        
        # Extract port number from context (e.g., PORT = 5000, listen(5000))
        port_match = re.search(r'(?:PORT|port)\s*(?:=|:|\|\|)\s*(\d{4,5})', context_output)
        port = port_match.group(1) if port_match else '5000'
        
        # Extract route prefixes (e.g., app.use('/api/users', ...))
        route_prefixes = re.findall(r"app\.use\(['\"](/\S+?)['\"]", context_output)
        
        # Build base URL
        base_url = f'http://localhost:{port}'
        details.append(f'API BASE URL: {base_url}')
        
        # Extract all route endpoints from the API SUMMARY section
        api_routes = []
        route_matches = re.findall(r'-\s+(GET|POST|PUT|PATCH|DELETE)\s+(/\S+)', context_output)
        if route_matches:
            # Combine route prefixes with route paths
            for method, path in route_matches:
                # Try to find which prefix this route belongs to
                full_path = path
                if route_prefixes and not path.startswith('/api'):
                    # Check source file context to match prefix
                    for prefix in route_prefixes:
                        if path == '/' or path.startswith('/:'):
                            full_path = prefix + ('' if path == '/' else path)
                            break
                        elif not path.startswith('/api'):
                            full_path = prefix + path
                            break
                api_routes.append(f'  {method} {base_url}{full_path}')
            
            details.append('API ENDPOINTS (use these EXACT URLs in fetch()):')
            details.extend(api_routes)
        
        # Extract model/schema field names
        model_fields = {}
        # Look for schema definitions with field names
        schema_blocks = re.findall(r'(\w+)Schema\s*=\s*new\s+mongoose\.Schema\(\{([\s\S]*?)\}\s*,', context_output)
        for model_name, fields_block in schema_blocks:
            fields = re.findall(r'(\w+)\s*:\s*\{', fields_block)
            if fields:
                model_fields[model_name] = fields
        
        # Also try simpler patterns
        if not model_fields:
            model_defs = re.findall(r'(?:const|let|var)\s+(\w+)Schema.*?(\w+)\s*:\s*\{.*?type:', context_output, re.DOTALL)
        
        if model_fields:
            details.append('MODEL FIELDS (use these exact field names in your code):')
            for model, fields in model_fields.items():
                details.append(f'  {model}: {", ".join(fields)}')
        
        return '\n'.join(details) if len(details) > 1 else ''

    def _process_conversational_response(self, text):
        """Handle a purely conversational interaction without tools."""
        logger.info(f"[Brain] Processing conversational response for: {text[:40]}")
        try:
            self.messages.append({"role": "user", "content": text})
            self._trim_history()
            
            # Use streaming for natural, conversational responses (no tools)
            model = self._select_model(text)
            response = self._call_llm_with_retry(model, self.messages)
            content = response.get('message', {}).get('content', '')
            
            if content.strip():
                self.messages.append({"role": "assistant", "content": content})
                return content
            else:
                return "Hmm, I'm not sure how to respond to that. Could you rephrase?"
        except Exception as e:
            error_str = str(e).lower()
            if "not found" in error_str and ("404" in error_str or "ollama" in error_str):
                msg = f"Cloud API rate limits exceeded, and local fallback failed because the model '{self.fallback_model}' is not installed. Please run 'ollama pull {self.fallback_model}' in your terminal."
                logger.error(msg)
                return msg
            logger.error(f"Conversational response failed: {e}", exc_info=True)
            return f"Sorry, something went wrong: {e}"

    @_timed
    def process_command(self, text):
        """Process a user command — fast path, blueprint approval, conversational path, then LLM with failsafe multi-turn tool calling"""
        
        # 0. Check for pending blueprint approval FIRST
        if self._pending_blueprint:
            text_lower = text.lower().strip()
            approval_words = ['yes', 'ok', 'okay', 'sure', 'approve', 'go ahead', 'do it', 
                              'looks good', 'perfect', 'great', 'good', 'yep', 'yeah', 'y',
                              'go for it', 'proceed', 'lets go', "let's go", 'start']
            rejection_words = ['no', 'cancel', 'stop', 'nevermind', 'never mind', 'nah', 'nope', 'abort']
            
            if any(text_lower == w or text_lower.startswith(w + ' ') for w in approval_words):
                logger.info("[Brain] User approved blueprint — proceeding to code generation")
                return self.approve_blueprint()
            elif any(text_lower == w or text_lower.startswith(w + ' ') for w in rejection_words):
                logger.info("[Brain] User rejected blueprint — cancelling")
                self._pending_blueprint = None
                self._pending_request = None
                self._pending_is_react = False
                return "Blueprint cancelled. Let me know if you'd like to try something different!"
            else:
                # User gave feedback — regenerate the blueprint with their changes
                logger.info(f"[Brain] User gave blueprint feedback: {text[:60]}")
                return self.reject_blueprint(text)
        
        # 1. Try Fast Path for simple commands (no LLM needed)
        fast_response = self._fast_path(text)
        if fast_response:
            return fast_response
        
        # 2. Conversational path: general questions, suggestions, advice — no tools needed
        if self._is_conversational(text):
            logger.info(f"[Brain] Conversational request detected — responding without tools")
            return self._process_conversational_response(text)

        # 3. Use LLM with failsafe multi-turn tool calling
        try:
            # Reset per-request flags
            self._user_wants_react = False
            self._ran_npx_create_react = False
            self._active_project_root = None
            self._read_files = set()  # Reset read tracking per request
            
            # Detect if user explicitly asked for React/framework
            text_lower = text.lower()
            if any(fw in text_lower for fw in ['react', 'vue', 'angular', 'next.js', 'next js', 'vite']):
                self._user_wants_react = True
                logger.info("[Brain] User explicitly requested a framework (React/Vue/Angular).")
            
            # --- FIX-MODE DETECTION (ported from GeminiBrain) ---
            # When user asks to fix/debug, inject a read-first directive so the model
            # doesn't blindly rewrite files from memory — it must read the actual state first.
            fix_keywords = ['fix', 'debug', 'has issues', 'has errors', 'has bugs', 'broken',
                            'not working', 'doesnt work', "doesn't work", 'logical error',
                            'something wrong', 'there is an error']
            if any(kw in text_lower for kw in fix_keywords):
                logger.info("[Brain] Fix-mode detected — will inject read-first directive")
                self.messages.append({"role": "user", "content": text})
                self.messages.append({
                    "role": "system",
                    "content": (
                        "IMPORTANT: Before fixing anything, you MUST call read_file on the relevant files "
                        "to see their current content. If you don't know which files exist, call list_dir first. "
                        "Do NOT write code from memory — read the actual files first."
                    )
                })
                # Don't add user message again below — it's already added
                text_lower = text_lower  # keep for routing checks below
            
            # --- EARLY ROUTING: Skip tool loop for simple web app creation ---
            # The 7B Ollama model can't reliably make structured tool calls.
            # For creation requests, bypass the tool loop and generate code directly.
            creation_keywords = ['create', 'build', 'make', 'develop', 'generate', 'scaffold', 'get me']
            # Strong creation verbs that clearly indicate a NEW project (not "make X red")
            strong_creation_verbs = ['create', 'build', 'develop', 'scaffold', 'generate']
            modification_keywords = ['improve', 'fix', 'update', 'modify', 'change', 'enhance', 
                                     'better', 'upgrade', 'redesign', 'add to', 'edit',
                                     'add a', 'add the', 'remove', 'delete', 'make the',
                                     'make it', 'make its', 'set the', 'put a', 'insert']
            target_keywords = ['game', 'app', 'website', 'web app', 'webapp', 'clone', 'page', 'site',
                               'in python', 'script', 'program', 'tool', 'calculator', 'project',
                               'sudoku', 'chess', 'quiz', 'todo', 'timer', 'clock', 'bot']
            
            is_creation = any(kw in text_lower for kw in creation_keywords)
            is_strong_creation = any(kw in text_lower for kw in strong_creation_verbs)
            is_modification = any(kw in text_lower for kw in modification_keywords)
            has_target = any(kw in text_lower for kw in target_keywords)
            
            has_create_noun = bool(re.match(r'(?:create|build|make|get)\s+(?:me\s+)?(?:a\s+)?(?:new\s+)?\w+', text_lower))
            
            # --- CLEAR CONTEXT DETECTION ---
            reset_keywords = ['new task', 'new project', 'start over', 'clear context', 'forget previous', 'different project']
            if any(kw in text_lower for kw in reset_keywords):
                logger.info("[Brain] User requested new task/project — clearing active project context")
                self._last_created_project = None
                self._active_project_root = None
                # If the entire message is just the reset command, return immediately
                if len(text_lower.split()) <= 4 and any(text_lower.strip() == kw for kw in reset_keywords):
                    return "Got it! I've cleared the previous project from my memory. What would you like to do next?"
            
            # --- PRIORITY 1: If a recent project exists, prefer MODIFICATION ---
            # This prevents "make the background black" from creating a new project.
            # Only route to creation if user explicitly names a NEW, different project.
            if hasattr(self, '_last_created_project') and self._last_created_project:
                import os
                if os.path.isdir(self._last_created_project):
                    # Check if user explicitly wants a NEW, differently-named project
                    extracted_name = self._extract_project_name(text)
                    current_name = Path(self._last_created_project).name
                    names_different = (extracted_name != 'my-project' 
                                       and extracted_name.lower() != current_name.lower())
                    
                    # Break out of modification lock if they specify a new language or explicitly say "new"
                    is_new_stack = bool(re.search(r'\bin\s+(java|python|react|vue|angular|c\+\+|c#|ruby|go|rust)\b', text_lower))
                    is_explicitly_new = 'new' in text_lower.split()
                    
                    if is_strong_creation and has_target and (names_different or is_new_stack or is_explicitly_new) and not is_modification:
                        # User explicitly wants a NEW project
                        logger.info(f"[Brain] New project requested (names_diff={names_different}, new_stack={is_new_stack}, explicit_new={is_explicitly_new})")
                        self._last_created_project = None  # Clear it so it routes purely to creation
                        if self._user_wants_react:
                            self.messages.append({"role": "user", "content": text})
                            return self._force_execute_react_creation(text)
                        # Fall through to creation below
                    else:
                        # Before forcing modification, catch any residual conversational inputs that slipped through
                        if not is_modification and not has_target and len(text_lower.split()) < 15 and not bool(re.search(r'\b(code|file|function|class|style|color|text|ui)\b', text_lower)):
                            logger.info(f"[Brain] Active project exists, but input seems conversational — routing to conversational handler")
                            return self._process_conversational_response(text)
                            
                        # Default: modify the existing project
                        logger.info(f"[Brain] Active project exists — routing to modification for '{self._last_created_project}'")
                        self.messages.append({"role": "user", "content": text})
                        return self._force_execute_modification(text, self._last_created_project)
            
            # --- PRIORITY 2: No recent project — route based on keywords ---
            # Route React/framework projects to the React handler
            if is_creation and self._user_wants_react and not is_modification:
                logger.info("[Brain] React/framework creation detected — routing to React handler")
                self.messages.append({"role": "user", "content": text})
                return self._force_execute_react_creation(text)
            
            # Only route to creation if it's genuinely a NEW project request
            if is_creation and (has_target or has_create_noun) and not is_modification and not self._user_wants_react:
                logger.info("[Brain] Creation request detected — routing directly to force-execute (bypassing tool loop)")
                self.messages.append({"role": "user", "content": text})
                return self._force_execute_creation(text)
            
            # Route modifications to the modification handler (no recent project, but explicit path)
            if is_modification and hasattr(self, '_last_created_project'):
                logger.info(f"[Brain] Modification request detected for '{self._last_created_project}'")
                self.messages.append({"role": "user", "content": text})
                return self._force_execute_modification(text, self._last_created_project)
            
            # Analyze request for project context
            project_context = self._analyze_request(text)
            
            self.messages.append({"role": "user", "content": text})
            
            # Inject project context if detected
            if project_context:
                ctx_parts = []
                if project_context['project_name']:
                    ctx_parts.append(f'Project name: "{project_context["project_name"]}"')
                    ctx_parts.append(f'ALL folders and files MUST use the name "{project_context["project_name"]}". Do NOT use any other name.')
                if project_context['source_path']:
                    ctx_parts.append(f'Source to read: {project_context["source_path"]}')
                if project_context['target_path']:
                    ctx_parts.append(f'Target to create: {project_context["target_path"]}')
                if project_context['target_type']:
                    ctx_parts.append(f'Task: Create {project_context["target_type"]} code')
                
                ctx_msg = "PROJECT CONTEXT (follow strictly):\n" + "\n".join(ctx_parts)
                self.messages.append({"role": "system", "content": ctx_msg})
                logger.info(f"[Brain] Injected project context: {ctx_msg}")
            
            self._trim_history()
            
            # Select model
            model = self._select_model(text)
            
            all_results = []  # Track all tool results across iterations
            executed_calls = set()  # Track (tool_name, args_key) to prevent duplicates
            skeleton_files = []  # Track files that appear to be skeleton/placeholder code
            max_iterations = 10
            last_call_signature = None  # Loop guard
            repeated_call_count = 0
            
            for iteration in range(max_iterations):
                logger.info(f"[Brain] Tool-calling loop iteration {iteration + 1}/{max_iterations}")
                
                # Call LLM with tools
                response = self._call_llm_with_retry(model, self.messages, tools=self.tools)
                self.messages.append(response['message'])
                
                # Check for tool calls
                tool_calls = response['message'].get('tool_calls')
                content = response['message'].get('content', '')
                
                if not tool_calls:
                    tool_calls = self._extract_fallback_tool_calls(content)
                if not tool_calls:
                    tool_calls = self._extract_function_style_tool_calls(content)
                
                # No tool calls — LLM is done
                if not tool_calls:
                    # --- Refusal detector ---
                    # Small models sometimes refuse with short responses like
                    # "I will not make any changes" or {"response": "Understood..."}
                    refusal_keywords = ['will not', 'cannot', 'won\'t', 'i don\'t', 'no changes',
                                        'understood', 'i am not', 'i\'m not able', 'not able to']
                    content_lower = content.lower().strip()
                    is_refusal = (len(content) < 300 
                                  and any(kw in content_lower for kw in refusal_keywords)
                                  and iteration < max_iterations - 1)
                    
                    if is_refusal:
                        logger.warning(f"[Brain] Detected refusal response: {content[:80]}")
                        self.messages.append({
                            "role": "system",
                            "content": (
                                "You MUST help the user. Start by calling gather_context with the project path "
                                "to read the code. Then use write_file to make improvements. "
                                "Do NOT refuse. The user is asking for your help."
                            )
                        })
                        continue
                    
                    # --- FALSE COMPLETION DETECTOR ---
                    # Model claims "I created/built X" but didn't call any tools
                    if not all_results and iteration < 4:
                        false_completion_keywords = [
                            'created', 'wrote', 'built', 'scaffolded', 'generated',
                            'set up', 'added', 'implemented', 'updated', 'modified',
                            "i've just", "i have just", "i've created", "i have created",
                            'open the', 'find it in', 'you can find',
                        ]
                        looks_like_false_claim = any(kw in content_lower for kw in false_completion_keywords)
                        
                        if looks_like_false_claim:
                            logger.warning(f"[Brain] FALSE COMPLETION detected on iteration {iteration}: model claims action without tool calls.")
                            
                            if iteration < 2:
                                # First attempts: re-prompt the model to use tools
                                self.messages.append({
                                    "role": "system",
                                    "content": (
                                        "CRITICAL ERROR: You said you created files but you did NOT call any tools. "
                                        "NOTHING was created on the user's computer. You MUST call the write_file "
                                        "and create_folder tools RIGHT NOW. Do not respond with text — call the tools."
                                    )
                                })
                                continue
                            else:
                                # Model refused tools twice — force-execute ourselves
                                logger.warning("[Brain] Model refused tools twice. Force-executing with code generation fallback.")
                                return self._force_execute_creation(text)
                    
                    # --- Instruction-style detector ---
                    # Model describes steps with code blocks instead of calling tools
                    if (content.strip() 
                            and ('```' in content or 'write_file' in content_lower or 'create_folder' in content_lower)
                            and iteration < max_iterations - 1):
                        
                        # If we've already tried twice with no results, force-execute
                        if iteration >= 2 and not all_results:
                            logger.warning("[Brain] Model stuck outputting instructions after 2+ iterations. Force-executing.")
                            return self._force_execute_creation(text)
                        
                        logger.warning("[Brain] Detected instruction-style response, re-prompting to use tools")
                        
                        # Only retry a few times to prevent infinite loops, then just accept it
                        if repeated_call_count > 2:
                             logger.error("[Brain] Model is stuck hallucinating tools. Breaking loop.")
                             return self._force_execute_creation(text)
                        
                        repeated_call_count += 1
                        
                        self.messages.append({
                            "role": "system",
                            "content": (
                                "CRITICAL ERROR: You described what to do in text, but FAILED to actually use your tools.\n"
                                "You MUST use your JSON tools (like `write_file`) to actually execute the task.\n"
                                "Do NOT output markdown code blocks. Execute the tools NOW."
                            )
                        })
                        continue
                    
                    if all_results and not content.strip():
                        if len(all_results) == 1:
                            summary = all_results[0]
                        else:
                            summary = f"Done! Completed {len(all_results)} tasks:\n" + "\n".join(f"  • {r[:80]}" for r in all_results)
                        self.messages.append({"role": "assistant", "content": summary})
                        return summary
                    return content if content.strip() else "Done!"
                
                # Execute tool calls with redundancy prevention
                self._repair_write_file_calls(tool_calls, content)
                
                has_context_tool = False
                has_write_tool = False
                skipped_any = False
                
                for tool in tool_calls:
                    function_name = tool['function']['name']
                    args = tool['function']['arguments']
                    
                    # Create a hashable key for dedup
                    if isinstance(args, dict):
                        # For write_file: only compare file_path, not content
                        # (model may write different content to same file when fixing)
                        if function_name == 'write_file':
                            args_key = args.get('file_path', '')
                        else:
                            args_key = json.dumps(args, sort_keys=True)
                    else:
                        args_key = str(args)
                    call_key = (function_name, args_key)
                    
                    # --- Loop guard: detect repeated identical tool calls ---
                    if call_key == last_call_signature:
                        repeated_call_count += 1
                        if repeated_call_count >= 4:
                            logger.warning(f"[Brain] Aborting loop: tool '{function_name}' called {repeated_call_count} times on same target.")
                            summary = f"I tried multiple times but couldn't complete this task. The model kept repeating the same action. Please try breaking it into smaller steps."
                            self.messages.append({"role": "assistant", "content": summary})
                            return summary
                        # On 2nd repeat of write_file, nudge model to read the file first
                        if repeated_call_count == 2 and function_name == 'write_file':
                            fp = args.get('file_path', '') if isinstance(args, dict) else ''
                            logger.warning(f"[Brain] write_file repeated for '{fp}' — injecting read-first nudge")
                            self.messages.append({
                                "role": "system",
                                "content": (
                                    f"STOP: You are writing to '{fp}' again, but your previous attempt had issues. "
                                    f"Call read_file('{fp}') FIRST to see what's actually in the file NOW, "
                                    f"then fix the specific problem. Do NOT rewrite the whole file from scratch."
                                )
                            })
                    else:
                        last_call_signature = call_key
                        repeated_call_count = 1
                    
                    # Skip redundant calls (same tool + same args)
                    if call_key in executed_calls and function_name in ('gather_context', 'read_file', 'list_dir'):
                        logger.warning(f"[Brain] Skipping redundant call: {function_name}({args_key[:60]})")
                        skip_msg = {
                            "role": "tool",
                            "content": f"ALREADY DONE — you already called {function_name} with these arguments. "
                                       f"Do NOT call it again. Instead, use create_folder and write_file to create the project files NOW.",
                        }
                        # Groq requires tool_call_id on all tool messages
                        if Config.USE_GROQ and tool.get('id'):
                            skip_msg["tool_call_id"] = tool['id']
                        self.messages.append(skip_msg)
                        skipped_any = True
                        continue
                        
                    # Stop hallucinated 'MyProject' loops
                    if function_name == 'gather_context' and 'MyProject' in args_key and 'MyProject' not in text:
                        logger.warning(f"[Brain] Blocked hallucinated gather_context('MyProject')")
                        block_msg = {
                            "role": "tool",
                            "content": "ERROR: Do NOT guess folder names like 'MyProject'. "
                                       "If you do not know the path, ask the user."
                        }
                        # Groq requires tool_call_id on all tool messages
                        if Config.USE_GROQ and tool.get('id'):
                            block_msg["tool_call_id"] = tool['id']
                        self.messages.append(block_msg)
                        skipped_any = True
                        continue

                    executed_calls.add(call_key)
                    
                    tool_result = self._execute_tool(function_name, args)
                    all_results.append(str(tool_result))
                    
                    if function_name in ('gather_context', 'read_file'):
                        has_context_tool = True
                    if function_name in ('write_file', 'create_folder'):
                        has_write_tool = True
                    
                    # Check for skeleton code in written files
                    if function_name == 'write_file':
                        written_content = args.get('content', '') if isinstance(args, dict) else ''
                        skeleton_warning = self._check_skeleton_code(
                            args.get('file_path', '') if isinstance(args, dict) else '', 
                            written_content
                        )
                        if skeleton_warning:
                            skeleton_files.append(skeleton_warning)
                            logger.warning(f"[Brain] {skeleton_warning}")
                    
                    tool_msg = {
                        "role": "tool",
                        "content": str(tool_result),
                    }
                    # Groq requires tool_call_id on all tool result messages
                    if Config.USE_GROQ and tool.get('id'):
                        tool_msg["tool_call_id"] = tool['id']
                    self.messages.append(tool_msg)
                
                # Inject project-specific nudge after context gathering
                if has_context_tool and project_context:
                    pname = project_context.get('project_name', 'the project')
                    tpath = project_context.get('target_path', '')
                    ttype = project_context.get('target_type', 'the code')
                    
                    # Extract API details from the context output
                    api_details = self._extract_api_details(all_results[-1] if all_results else '')
                    
                    nudge = (f'You have read the code for "{pname}". '
                             f'DO NOT describe or summarize it. DO NOT call gather_context again. '
                             f'IMMEDIATELY use create_folder and write_file to create the {ttype} files. ')
                    if tpath:
                        nudge += f'Create all files under "{tpath}". '
                    nudge += f'Use the project name "{pname}" for ALL folder and file paths.\n\n'
                    
                    # Add extracted API details
                    if api_details:
                        nudge += api_details + '\n\n'
                    
                    nudge += 'Write COMPLETE, WORKING code. Use only vanilla HTML/CSS/JS unless the user asked for React.'
                    
                    self.messages.append({"role": "system", "content": nudge})
                    logger.info(f"[Brain] Injected project-specific nudge with API details for '{pname}'")
                elif has_context_tool:
                    # Generic nudge if no project context
                    nudge = ("You have now read the project code above. "
                             "DO NOT describe or summarize it. DO NOT call gather_context again. "
                             "IMMEDIATELY use create_folder and write_file tools to "
                             "create the files based on the API endpoints and models you found. "
                             "Write COMPLETE, WORKING code for each file.")
                    self.messages.append({"role": "system", "content": nudge})
                    logger.info("[Brain] Injected generic action nudge")
                
                # If all calls were skipped (redundant loop), force-inject a stronger directive
                if skipped_any and not has_context_tool and not has_write_tool:
                    force = ("STOP calling gather_context or read_file blindly. "
                             "If a project or file was not found, YOU MUST STOP AND TELL THE USER. "
                             "Do NOT keep guessing folder names. "
                             "Otherwise, if you have the context, your ONLY next action should be create_folder or write_file.")
                    self.messages.append({"role": "system", "content": force})
                    logger.info("[Brain] Injected force-action directive after skipped redundant calls")
                
                # Nudge after file-writing: keep LLM using tools for remaining project files
                if has_write_tool and not has_context_tool:
                    # Build quality feedback if skeleton files were detected
                    if skeleton_files:
                        quality_msg = (
                            "CRITICAL QUALITY ISSUE — the following files are incomplete, too short, or contain placeholders:\n"
                            + "\n".join(f"  • {w}" for w in skeleton_files) + "\n"
                            "You MUST rewrite these files immediately with COMPLETE, REAL, WORKING code. "
                            "No placeholders, no TODO comments, no skeleton code. "
                            "Do NOT be lazy. You are a senior engineer. "
                            "Each HTML file should have full page structure with real content sections. "
                            "Each CSS file needs complete styling (colors, spacing, typography, layout, hover effects). "
                            "Each JS file needs actual working logic (DOM manipulation, event handlers, fetch calls, data rendering). "
                            "Write the FULL file content using write_file."
                        )
                        self.messages.append({"role": "system", "content": quality_msg})
                        logger.info(f"[Brain] Injected quality rewrite directive for {len(skeleton_files)} skeleton file(s)")
                        skeleton_files.clear()
                        continue # Force tool loop restart to rewrite
                    else:
                        write_nudge = (
                            "Good — files created. If the project needs MORE files "
                            "(backend server, additional pages, config files, package.json, etc.), "
                            "use create_folder and write_file to create them NOW. "
                            "Do NOT describe or explain what to do — just DO IT with tools. "
                            "If ALL files are complete, respond with a brief confirmation only."
                        )
                        self.messages.append({"role": "system", "content": write_nudge})
                        logger.info("[Brain] Injected post-write nudge to continue using tools")
                
                logger.info(f"[Brain] Iteration {iteration + 1}: executed {len(tool_calls)} tool(s), total actions: {len(all_results)}")
            
            # Hit iteration limit
            logger.warning(f"[Brain] Hit max iterations ({max_iterations}). Returning summary.")
            if all_results:
                if len(all_results) == 1:
                    summary = all_results[0]
                else:
                    summary = f"Done! Completed {len(all_results)} tasks:\n" + "\n".join(f"  • {r[:80]}" for r in all_results)
                self.messages.append({"role": "assistant", "content": summary})
                return summary
            
            return "I wasn't able to complete the task. Could you try rephrasing?"

        except Exception as e:
            logger.error(f"Error processing command: {e}", exc_info=True)
            error_str = str(e)
            
            # Friendly rate limit message
            if '429' in error_str or 'rate_limit' in error_str.lower() or 'rate limit' in error_str.lower():
                # Try to extract wait time from error message
                wait_match = re.search(r'try again in (\d+h\d+m[\d.]+s|\d+m[\d.]+s|\d+s)', error_str, re.IGNORECASE)
                wait_time = wait_match.group(1) if wait_match else "a while"
                return (
                    f"⏳ I've hit the Groq API rate limit for today. "
                    f"Please try again in {wait_time}, or you can switch to local Ollama "
                    f"by setting ASSISTANT_USE_GROQ=false in your .env file."
                )
            
            if "not found" in error_str:
                return f"Model not found. Did you run 'ollama pull {self.primary_model}'?"
            return f"Sorry, something went wrong: {e}"
