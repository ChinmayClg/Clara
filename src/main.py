import customtkinter as ctk
from src.gui import AssistantApp
from src.config import Config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def main():
    # Ensure necessary directories exist
    Config.ensure_directories()
    
    # Set appearance from config
    ctk.set_appearance_mode(Config.THEME)
    ctk.set_default_color_theme(Config.COLOR_THEME)
    
    logger.info("Starting Personal Desktop Assistant (Local Ollama Brain)...")
    
    app = AssistantApp()
    app.mainloop()

if __name__ == "__main__":
    main()
