import customtkinter as ctk
import threading
from tkinter import filedialog, messagebox
from src.voice_engine import VoiceEngine
from src.brain import Brain
from src.config import Config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# ─── Premium Dark Palette ───────────────────────────────────────────────────
COLORS = {
    "bg_dark":       "#1a1b26",   # Main background
    "bg_sidebar":    "#16161e",   # Sidebar background
    "bg_header":     "#1f2335",   # Header bar
    "bg_input":      "#1f2335",   # Input bar background
    "bg_chat":       "#1a1b26",   # Chat area background
    "bubble_user":   "#292e42",   # User message bubble
    "bubble_ai":     "#24283b",   # AI message bubble
    "accent":        "#7aa2f7",   # Primary accent (soft blue)
    "accent_hover":  "#89b4fa",   # Accent hover
    "accent_green":  "#9ece6a",   # Status: ready
    "accent_yellow": "#e0af68",   # Status: thinking
    "accent_red":    "#f7768e",   # Status: error
    "text_primary":  "#c0caf5",   # Primary text
    "text_muted":    "#565f89",   # Muted / secondary text
    "text_bright":   "#ffffff",   # Bright text
    "border":        "#292e42",   # Subtle borders
    "btn_bg":        "#24283b",   # Button resting background
    "btn_hover":     "#414868",   # Button hover
}

FONT_FAMILY = "Segoe UI"


class AssistantApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Personal Assistant")
        self.geometry(Config.WINDOW_SIZE)
        self.configure(fg_color=COLORS["bg_dark"])
        self.minsize(700, 500)

        self.voice_engine = VoiceEngine()
        self.brain = Brain()

        # Wire up streaming callback
        self.brain.on_stream_token = self._on_stream_token
        # Wire up tool activity callback for real-time tool feed
        self.brain.on_tool_activity = self._on_tool_activity
        # Wire up blueprint approval callback
        self.brain.on_blueprint_ready = self._on_blueprint_ready
        self._streaming_active = False
        self._blueprint_pending = False  # Track if a blueprint is awaiting approval

        self.is_processing = False  # Prevent concurrent processing
        self._input_history = []    # Command history for ↑/↓ navigation
        self._history_index = -1

        # Status animation state
        self._status_dot_pulse = False
        self._pulse_after_id = None

        self.setup_ui()
        self.setup_keyboard_shortcuts()

        logger.info("Assistant GUI initialized")

    # ═══════════════════════════════════════════════════════════════════════
    #  UI Setup
    # ═══════════════════════════════════════════════════════════════════════

    def setup_ui(self):
        # ── Root layout: sidebar (fixed) + main panel (expand) ──
        self.grid_columnconfigure(0, weight=0, minsize=68)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_main_panel()

        # Welcome message
        self._post_system("Welcome! You can:")
        self._post_system("  • Click 🎤 or press Ctrl+L to use voice")
        self._post_system("  • Type commands below and press Enter")
        self._post_system("  • Press Ctrl+H for history · Ctrl+S stop speech · Ctrl+Q quit")
        self._post_system("  • Use ↑ / ↓ to cycle through command history\n")

    # ── Sidebar ────────────────────────────────────────────────────────────

    def _build_sidebar(self):
        sidebar = ctk.CTkFrame(
            self, width=68, corner_radius=0,
            fg_color=COLORS["bg_sidebar"],
            border_width=0,
        )
        sidebar.grid(row=0, column=0, sticky="ns")
        sidebar.grid_propagate(False)

        # App icon / branding at top
        brand_label = ctk.CTkLabel(
            sidebar, text="🤖", font=(FONT_FAMILY, 26),
            text_color=COLORS["accent"],
        )
        brand_label.pack(pady=(18, 24))

        # Sidebar buttons
        btn_defs = [
            ("🎤", "Listen\nCtrl+L", self.start_listening_thread),
            ("💾", "Save", self.save_conversation),
            ("📂", "Load", self.load_conversation),
            ("📜", "History\nCtrl+H", self.show_history),
        ]

        self._sidebar_buttons = []
        for icon, tooltip, cmd in btn_defs:
            btn = ctk.CTkButton(
                sidebar, text=icon, width=48, height=48,
                font=(FONT_FAMILY, 20),
                fg_color=COLORS["btn_bg"],
                hover_color=COLORS["btn_hover"],
                corner_radius=12,
                border_width=0,
                command=cmd,
            )
            btn.pack(pady=4)
            self._sidebar_buttons.append(btn)

            # Small tooltip-style label beneath button
            tip = ctk.CTkLabel(
                sidebar, text=tooltip, font=(FONT_FAMILY, 8),
                text_color=COLORS["text_muted"],
            )
            tip.pack(pady=(0, 8))

        # Spacer
        spacer = ctk.CTkFrame(sidebar, fg_color="transparent", height=10)
        spacer.pack(fill="x", expand=True)

        # Conversation mode toggle at bottom of sidebar
        self.conversational_mode_var = ctk.BooleanVar(value=False)
        conv_label = ctk.CTkLabel(
            sidebar, text="♻️", font=(FONT_FAMILY, 20),
            text_color=COLORS["text_muted"],
        )
        conv_label.pack(pady=(0, 2))
        self.conversational_switch = ctk.CTkSwitch(
            sidebar, text="",
            variable=self.conversational_mode_var,
            onvalue=True, offvalue=False,
            width=40,
            progress_color=COLORS["accent"],
            button_color=COLORS["text_primary"],
            fg_color=COLORS["btn_bg"],
        )
        self.conversational_switch.pack(pady=(0, 4))
        conv_tip = ctk.CTkLabel(
            sidebar, text="Conv\nMode", font=(FONT_FAMILY, 8),
            text_color=COLORS["text_muted"],
        )
        conv_tip.pack(pady=(0, 16))

    # ── Main Panel ─────────────────────────────────────────────────────────

    def _build_main_panel(self):
        main = ctk.CTkFrame(self, fg_color=COLORS["bg_dark"], corner_radius=0)
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(1, weight=1)   # chat area expands

        self._build_header(main)
        self._build_chat_area(main)
        self._build_input_bar(main)

    # ── Header Bar ─────────────────────────────────────────────────────────

    def _build_header(self, parent):
        header = ctk.CTkFrame(
            parent, height=52, corner_radius=0,
            fg_color=COLORS["bg_header"],
            border_width=0,
        )
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)
        header.grid_columnconfigure(0, weight=1)
        header.grid_columnconfigure(1, weight=0)

        # Title
        title = ctk.CTkLabel(
            header, text="Personal Assistant",
            font=(FONT_FAMILY, 16, "bold"),
            text_color=COLORS["text_bright"],
        )
        title.grid(row=0, column=0, padx=16, pady=14, sticky="w")

        # Status frame (dot + text)
        status_frame = ctk.CTkFrame(header, fg_color="transparent")
        status_frame.grid(row=0, column=1, padx=16, pady=14, sticky="e")

        self._status_dot = ctk.CTkLabel(
            status_frame, text="●", font=(FONT_FAMILY, 14),
            text_color=COLORS["accent_green"],
        )
        self._status_dot.pack(side="left", padx=(0, 6))

        self.label = ctk.CTkLabel(
            status_frame, text="Ready",
            font=(FONT_FAMILY, 12),
            text_color=COLORS["text_muted"],
        )
        self.label.pack(side="left")

    # ── Chat Area ──────────────────────────────────────────────────────────

    def _build_chat_area(self, parent):
        self.log_area = ctk.CTkTextbox(
            parent, corner_radius=0,
            fg_color=COLORS["bg_chat"],
            text_color=COLORS["text_primary"],
            font=(FONT_FAMILY, 13),
            border_width=0,
            wrap="word",
        )
        self.log_area.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)
        self.log_area.configure(state="disabled")

        # Configure text tags for styled messages
        self.log_area._textbox.tag_configure(
            "user_prefix",
            foreground=COLORS["accent"],
            font=(FONT_FAMILY, 13, "bold"),
        )
        self.log_area._textbox.tag_configure(
            "ai_prefix",
            foreground=COLORS["accent_green"],
            font=(FONT_FAMILY, 13, "bold"),
        )
        self.log_area._textbox.tag_configure(
            "system_text",
            foreground=COLORS["text_muted"],
            font=(FONT_FAMILY, 12, "italic"),
        )
        self.log_area._textbox.tag_configure(
            "error_text",
            foreground=COLORS["accent_red"],
            font=(FONT_FAMILY, 13, "bold"),
        )
        self.log_area._textbox.tag_configure(
            "user_text",
            foreground=COLORS["accent_hover"],
            font=(FONT_FAMILY, 13),
        )
        self.log_area._textbox.tag_configure(
            "ai_text",
            foreground=COLORS["text_primary"],
            font=(FONT_FAMILY, 13),
        )
        self.log_area._textbox.tag_configure(
            "blueprint_text",
            foreground=COLORS["accent_yellow"],
            font=(FONT_FAMILY, 12),
        )
        self.log_area._textbox.tag_configure(
            "blueprint_header",
            foreground=COLORS["accent"],
            font=(FONT_FAMILY, 13, "bold"),
        )

    # ── Input Bar ──────────────────────────────────────────────────────────

    def _build_input_bar(self, parent):
        input_bar = ctk.CTkFrame(
            parent, height=56, corner_radius=0,
            fg_color=COLORS["bg_input"],
            border_width=0,
        )
        input_bar.grid(row=2, column=0, sticky="ew")
        input_bar.grid_propagate(False)
        input_bar.grid_columnconfigure(1, weight=1)

        # Mic button
        self.mic_button = ctk.CTkButton(
            input_bar, text="🎤", width=40, height=40,
            font=(FONT_FAMILY, 18),
            fg_color=COLORS["btn_bg"],
            hover_color=COLORS["accent"],
            corner_radius=20,
            border_width=0,
            command=self.start_listening_thread,
        )
        self.mic_button.grid(row=0, column=0, padx=(12, 6), pady=8)

        # Text input
        self.text_input = ctk.CTkEntry(
            input_bar,
            placeholder_text="Type a message…",
            font=(FONT_FAMILY, 13),
            height=40,
            corner_radius=20,
            fg_color=COLORS["bubble_user"],
            text_color=COLORS["text_primary"],
            placeholder_text_color=COLORS["text_muted"],
            border_color=COLORS["border"],
            border_width=1,
        )
        self.text_input.grid(row=0, column=1, padx=4, pady=8, sticky="ew")
        self.text_input.bind("<Return>", self.process_text_input)

        # Send button
        self.send_button = ctk.CTkButton(
            input_bar, text="➤", width=40, height=40,
            font=(FONT_FAMILY, 18),
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            text_color=COLORS["bg_dark"],
            corner_radius=20,
            border_width=0,
            command=lambda: self.process_text_input(None),
        )
        self.send_button.grid(row=0, column=2, padx=(6, 12), pady=8)

    # ═══════════════════════════════════════════════════════════════════════
    #  Keyboard Shortcuts
    # ═══════════════════════════════════════════════════════════════════════

    def setup_keyboard_shortcuts(self):
        """Set up keyboard shortcuts"""
        self.bind("<Control-l>", lambda e: self.start_listening_thread())
        self.bind("<Control-L>", lambda e: self.start_listening_thread())
        self.bind("<Control-h>", lambda e: self.show_history())
        self.bind("<Control-H>", lambda e: self.show_history())
        self.bind("<Control-s>", lambda e: self.stop_speaking())
        self.bind("<Control-S>", lambda e: self.stop_speaking())
        self.bind("<Control-q>", lambda e: self.quit())
        self.bind("<Control-Q>", lambda e: self.quit())

        # Input history navigation
        self.text_input.bind("<Up>", self._history_up)
        self.text_input.bind("<Down>", self._history_down)

        logger.debug("Keyboard shortcuts configured")

    # ═══════════════════════════════════════════════════════════════════════
    #  Input History
    # ═══════════════════════════════════════════════════════════════════════

    def _history_up(self, event):
        """Navigate to previous command in history"""
        if not self._input_history:
            return
        if self._history_index < len(self._input_history) - 1:
            self._history_index += 1
        self.text_input.delete(0, 'end')
        self.text_input.insert(0, self._input_history[-(self._history_index + 1)])

    def _history_down(self, event):
        """Navigate to next command in history"""
        if self._history_index > 0:
            self._history_index -= 1
            self.text_input.delete(0, 'end')
            self.text_input.insert(0, self._input_history[-(self._history_index + 1)])
        elif self._history_index == 0:
            self._history_index = -1
            self.text_input.delete(0, 'end')

    # ═══════════════════════════════════════════════════════════════════════
    #  Voice Listening
    # ═══════════════════════════════════════════════════════════════════════

    def start_listening_thread(self):
        """Start voice listening in a separate thread"""
        if self.is_processing:
            self.update_status("Already processing…")
            return

        threading.Thread(target=self.listen_process, daemon=True).start()

    def listen_process(self):
        """Voice listening process"""
        self.is_processing = True
        self.update_status("Listening…")

        text = self.voice_engine.listen()

        if text == "timeout":
            self.update_status("Timeout — no speech detected")
            self.is_processing = False
            # Resume silently if in conversational mode
            if self.conversational_mode_var.get():
                self.start_listening_thread()
            return
        elif text == "API unavailable":
            self.update_status("Speech recognition API unavailable")
            self._post_error("Check your internet connection")
            self.is_processing = False
            # Force disable conversational mode to prevent infinite loops on error
            if self.conversational_mode_var.get():
                self.conversational_switch.deselect()
                self._post_system("Conversational Mode disabled due to API error.")
            return
        elif text:
            self._post_user(text)
            self.process_command(text)
        else:
            self.update_status("Could not understand you")
            self.is_processing = False
            # Resume silently if in conversational mode
            if self.conversational_mode_var.get():
                self.start_listening_thread()

    # ═══════════════════════════════════════════════════════════════════════
    #  Text Input Processing
    # ═══════════════════════════════════════════════════════════════════════

    def process_text_input(self, event):
        """Process text input from the entry field"""
        if self.is_processing:
            return

        text = self.text_input.get().strip()
        if not text:
            return

        # Save to input history
        self._input_history.append(text)
        if len(self._input_history) > 50:  # Cap history at 50 entries
            self._input_history = self._input_history[-50:]
        self._history_index = -1

        self.text_input.delete(0, 'end')
        self._post_user(text)

        threading.Thread(target=self.process_command, args=(text,), daemon=True).start()

    # ═══════════════════════════════════════════════════════════════════════
    #  Streaming & Tool Callbacks
    # ═══════════════════════════════════════════════════════════════════════

    def _on_stream_token(self, token):
        """Callback for streaming tokens — update log area in real-time"""
        if not self._streaming_active:
            self._streaming_active = True
            # Start the "AI: " prefix on first token
            self.after(0, self._append_tagged, "AI: ", "ai_prefix")

        self.after(0, self._append_tagged, token, "ai_text")

    def _append_to_log(self, text):
        """Append text to the log area without adding a newline (used for streaming)"""
        self.log_area.configure(state="normal")
        self.log_area.insert("end", text)
        self.log_area.configure(state="disabled")
        self.log_area.see("end")

    def _append_tagged(self, text, tag):
        """Append text with a specific tag (for colored/styled output)"""
        self.log_area.configure(state="normal")
        self.log_area._textbox.insert("end", text, tag)
        self.log_area.configure(state="disabled")
        self.log_area.see("end")

    def _on_tool_activity(self, tool_name, args_summary):
        """Callback for tool activity — show real-time tool execution in status bar"""
        icons = {
            'write_file': '📝', 'read_file': '📖', 'create_folder': '📁',
            'list_dir': '📂', 'run_command': '⚙️', 'gather_context': '🔍',
            'open_app': '🚀', 'open_url': '🌐', 'search_web': '🔎',
        }
        icon = icons.get(tool_name, '🔧')
        self.update_status(f"{icon} {tool_name}({args_summary[:40]})")

    def _on_blueprint_ready(self, blueprint_text, user_request):
        """Callback when a blueprint is ready for user approval"""
        self._blueprint_pending = True
        self.update_status("📋 Blueprint ready — approve or give feedback")
        logger.info(f"[GUI] Blueprint ready for approval ({len(blueprint_text)} chars)")

    # ═══════════════════════════════════════════════════════════════════════
    #  Command Processing
    # ═══════════════════════════════════════════════════════════════════════

    def process_command(self, text):
        """Process a command (voice or text)"""
        self.is_processing = True
        self._streaming_active = False
        
        # Show appropriate status based on context
        if self._blueprint_pending:
            self.update_status("Processing blueprint response…")
        else:
            self.update_status("Thinking…")

        try:
            response = self.brain.process_command(text)

            # Check if a blueprint is now pending (response contains blueprint)
            if self.brain._pending_blueprint:
                self._blueprint_pending = True
                self.update_status("📋 Blueprint ready — type 'yes' to approve")
            else:
                self._blueprint_pending = False

            # If we were streaming, just add a newline to finish
            if self._streaming_active:
                self.after(0, self._append_to_log, "\n")
            else:
                # Non-streamed response (fast-path or tool results)
                self._post_ai(response)

            # Prepare optional callback for conversational mode
            def resume_listening():
                if self.conversational_mode_var.get():
                    self.start_listening_thread()

            # Speak short conversational responses only (skip status messages / file paths)
            if self._should_speak(response):
                self.voice_engine.speak(response, callback=resume_listening)
            else:
                logger.info("Response is a status/path message, skipping TTS")
                if self.conversational_mode_var.get():
                    # Still need to resume listening immediately if we skipped speaking
                    self.after(500, resume_listening)

            if not self._blueprint_pending:
                self.update_status("Ready")
        except Exception as e:
            error_msg = f"Error: {e}"
            self._post_error(error_msg)
            self.update_status("Error occurred")
            logger.error(f"Command processing error: {e}", exc_info=True)
        finally:
            self.is_processing = False
            self._streaming_active = False

    @staticmethod
    def _should_speak(text: str) -> bool:
        """Return True only if the response looks like a conversational reply worth speaking."""
        if len(text) > 500:
            return False
        # Skip status messages and technical output
        skip_prefixes = (
            "Successfully", "Opening", "Error", "Command", "Stopping",
            "Failed", "Directory", "Contents of", "Completed", "Stopped:"
        )
        if any(text.startswith(p) for p in skip_prefixes):
            return False
        # Skip if it contains a file path (backslash = Windows path)
        if "\\" in text or "://" in text:
            return False
        return True

    # ═══════════════════════════════════════════════════════════════════════
    #  Styled Message Posting Helpers
    # ═══════════════════════════════════════════════════════════════════════

    def _post_user(self, text):
        """Post a user message with styled formatting"""
        def _do():
            self.log_area.configure(state="normal")
            self.log_area._textbox.insert("end", "  You  ", "user_prefix")
            self.log_area._textbox.insert("end", f"  {text}\n", "user_text")
            self.log_area.configure(state="disabled")
            self.log_area.see("end")
        self.after(0, _do)

    def _post_ai(self, text):
        """Post an AI message with styled formatting"""
        def _do():
            self.log_area.configure(state="normal")
            self.log_area._textbox.insert("end", "  AI  ", "ai_prefix")
            self.log_area._textbox.insert("end", f"  {text}\n", "ai_text")
            self.log_area.configure(state="disabled")
            self.log_area.see("end")
        self.after(0, _do)

    def _post_system(self, text):
        """Post a system/info message"""
        def _do():
            self.log_area.configure(state="normal")
            self.log_area._textbox.insert("end", f"  {text}\n", "system_text")
            self.log_area.configure(state="disabled")
            self.log_area.see("end")
        self.after(0, _do)

    def _post_error(self, text):
        """Post an error message"""
        def _do():
            self.log_area.configure(state="normal")
            self.log_area._textbox.insert("end", f"  ✖ {text}\n", "error_text")
            self.log_area.configure(state="disabled")
            self.log_area.see("end")
        self.after(0, _do)

    # ═══════════════════════════════════════════════════════════════════════
    #  Status Updates (with animated dot)
    # ═══════════════════════════════════════════════════════════════════════

    def update_status(self, text):
        """Update status label and dot color (thread-safe)"""
        def _do():
            self.label.configure(text=text)

            # Update dot color based on status
            if "error" in text.lower() or "✖" in text:
                self._status_dot.configure(text_color=COLORS["accent_red"])
                self._stop_pulse()
            elif "ready" in text.lower():
                self._status_dot.configure(text_color=COLORS["accent_green"])
                self._stop_pulse()
            else:
                # Thinking / working / listening — animate
                self._status_dot.configure(text_color=COLORS["accent_yellow"])
                self._start_pulse()

        self.after(0, _do)
        logger.debug(f"Status: {text}")

    def _start_pulse(self):
        """Start pulsing the status dot between bright and dim"""
        if self._pulse_after_id is not None:
            return  # Already pulsing
        self._status_dot_pulse = True
        self._do_pulse(True)

    def _stop_pulse(self):
        """Stop the pulsing animation"""
        self._status_dot_pulse = False
        if self._pulse_after_id is not None:
            self.after_cancel(self._pulse_after_id)
            self._pulse_after_id = None
        self._status_dot.configure(font=(FONT_FAMILY, 14))

    def _do_pulse(self, big):
        """Pulse animation step"""
        if not self._status_dot_pulse:
            return
        size = 16 if big else 10
        self._status_dot.configure(font=(FONT_FAMILY, size))
        self._pulse_after_id = self.after(500, self._do_pulse, not big)

    # ═══════════════════════════════════════════════════════════════════════
    #  Legacy update_log (for backwards compatibility)
    # ═══════════════════════════════════════════════════════════════════════

    def update_log(self, text):
        """Update log area with a full line (thread-safe) — maps to styled helpers"""
        if text.startswith("You: "):
            self._post_user(text[5:])
        elif text.startswith("AI: "):
            self._post_ai(text[4:])
        elif text.startswith("ERROR:"):
            self._post_error(text[6:].strip())
        elif text.startswith("SYSTEM:"):
            self._post_system(text[7:].strip())
        else:
            self._post_system(text)

    def _update_log_sync(self, text):
        """Sync update to log area — must be called on main thread"""
        self.log_area.configure(state="normal")
        self.log_area.insert("end", text + "\n")
        self.log_area.configure(state="disabled")
        self.log_area.see("end")

    # ═══════════════════════════════════════════════════════════════════════
    #  Conversation Save / Load / History
    # ═══════════════════════════════════════════════════════════════════════

    def save_conversation(self):
        """Save current conversation to file"""
        try:
            filepath = self.brain.save_conversation()
            if filepath:
                messagebox.showinfo("Success", f"Conversation saved to:\n{filepath}")
                logger.info(f"Conversation saved via GUI: {filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save conversation:\n{e}")
            logger.error(f"Save conversation error: {e}")

    def load_conversation(self):
        """Load conversation from file"""
        try:
            filepath = filedialog.askopenfilename(
                title="Load Conversation",
                initialdir=Config.CONVERSATIONS_DIR,
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )

            if filepath:
                success = self.brain.load_conversation(filepath)
                if success:
                    messagebox.showinfo("Success", "Conversation loaded successfully!")
                    self._post_system("─── Conversation Loaded ───")
                    logger.info(f"Conversation loaded via GUI: {filepath}")
                else:
                    messagebox.showerror("Error", "Failed to load conversation")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load conversation:\n{e}")
            logger.error(f"Load conversation error: {e}")

    def show_history(self):
        """Show recent conversation history in a styled popup"""
        try:
            history = self.brain.get_conversation_history(limit=10)

            history_window = ctk.CTkToplevel(self)
            history_window.title("Conversation History")
            history_window.geometry("550x450")
            history_window.configure(fg_color=COLORS["bg_dark"])

            # Header inside popup
            popup_header = ctk.CTkLabel(
                history_window,
                text="📜  Conversation History",
                font=(FONT_FAMILY, 16, "bold"),
                text_color=COLORS["text_bright"],
            )
            popup_header.pack(padx=16, pady=(16, 8), anchor="w")

            # Separator line
            sep = ctk.CTkFrame(
                history_window, height=1,
                fg_color=COLORS["border"],
            )
            sep.pack(fill="x", padx=16, pady=(0, 8))

            # Text widget
            text_widget = ctk.CTkTextbox(
                history_window,
                fg_color=COLORS["bg_chat"],
                text_color=COLORS["text_primary"],
                font=(FONT_FAMILY, 12),
                corner_radius=8,
                border_width=1,
                border_color=COLORS["border"],
            )
            text_widget.pack(padx=16, pady=(0, 16), fill="both", expand=True)

            # Configure tags for history popup
            text_widget._textbox.tag_configure(
                "role_user", foreground=COLORS["accent"], font=(FONT_FAMILY, 12, "bold"))
            text_widget._textbox.tag_configure(
                "role_ai", foreground=COLORS["accent_green"], font=(FONT_FAMILY, 12, "bold"))
            text_widget._textbox.tag_configure(
                "role_other", foreground=COLORS["text_muted"], font=(FONT_FAMILY, 12, "bold"))

            for msg in history:
                role = msg.get('role', 'unknown').upper()
                content = msg.get('content', '')
                tag = "role_user" if role == "USER" else ("role_ai" if role == "ASSISTANT" else "role_other")
                text_widget._textbox.insert("end", f"{role}: ", tag)
                text_widget._textbox.insert("end", f"{content}\n\n")

            text_widget.configure(state="disabled")
            logger.debug("Displayed conversation history")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to show history:\n{e}")
            logger.error(f"Show history error: {e}")

    # ═══════════════════════════════════════════════════════════════════════
    #  Speech Control
    # ═══════════════════════════════════════════════════════════════════════

    def stop_speaking(self):
        """Stop the TTS engine"""
        try:
            self.voice_engine.stop()
            # If the user interrupts, intentionally disable conversational mode
            if self.conversational_mode_var.get():
                self.conversational_switch.deselect()
                self._post_system("Conversational Mode paused by user interrupt.")
            self.update_status("Stopped speaking")
            logger.info("TTS stopped by user")
        except Exception as e:
            logger.error(f"Failed to stop speaking: {e}")
