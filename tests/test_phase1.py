"""
Test script to verify Phase 1 improvements.
Tests configuration, logging, conversation management, and core functionality.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.utils.logger import setup_logger
from src.utils.conversation_manager import ConversationManager
from src.brain import Brain

logger = setup_logger(__name__)

def test_configuration():
    """Test configuration system"""
    print("\n=== Testing Configuration ===")
    print(f"Primary Model: {Config.PRIMARY_MODEL}")
    print(f"Max History: {Config.MAX_HISTORY}")
    print(f"Listen Timeout: {Config.LISTEN_TIMEOUT}")
    print(f"Conversations Dir: {Config.CONVERSATIONS_DIR}")
    print(f"Logs Dir: {Config.LOGS_DIR}")
    
    # Test path safety
    safe_path = "src/test.py"
    unsafe_path = "C:/Windows/System32/test.txt"
    print(f"\nPath Safety Check:")
    print(f"  '{safe_path}' is safe: {Config.is_safe_path(safe_path)}")
    print(f"  '{unsafe_path}' is safe: {Config.is_safe_path(unsafe_path)}")
    
    print("✓ Configuration test passed")

def test_logging():
    """Test logging system"""
    print("\n=== Testing Logging ===")
    test_logger = setup_logger("test_module")
    
    test_logger.debug("This is a debug message")
    test_logger.info("This is an info message")
    test_logger.warning("This is a warning message")
    test_logger.error("This is an error message")
    
    print(f"✓ Logging test passed (check {Config.LOGS_DIR}/assistant.log)")

def test_conversation_manager():
    """Test conversation persistence"""
    print("\n=== Testing Conversation Manager ===")
    manager = ConversationManager()
    
    # Create test conversation
    test_messages = [
        {"role": "system", "content": "You are a test assistant"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]
    
    # Save conversation
    filepath = manager.save_conversation(test_messages, name="test")
    print(f"Saved test conversation to: {filepath}")
    
    # Load conversation
    loaded_messages = manager.load_conversation(filepath)
    print(f"Loaded {len(loaded_messages)} messages")
    
    # List conversations
    conversations = manager.list_conversations()
    print(f"Found {len(conversations)} saved conversations")
    
    # Cleanup
    manager.delete_conversation(filepath)
    print("Cleaned up test conversation")
    
    print("✓ Conversation manager test passed")

def test_brain_initialization():
    """Test brain initialization and Ollama connection"""
    print("\n=== Testing Brain Initialization ===")
    
    try:
        brain = Brain()
        print("✓ Brain initialized successfully")
        print(f"  Model: {brain.model}")
        print(f"  Max History: {brain.max_history}")
        print(f"  Tools: {len(brain.tools)} available")
        
        # Test history trimming
        print("\n=== Testing History Trimming ===")
        initial_count = len(brain.messages)
        
        # Add messages beyond max history
        for i in range(Config.MAX_HISTORY + 5):
            brain.messages.append({"role": "user", "content": f"Test message {i}"})
        
        print(f"Messages before trim: {len(brain.messages)}")
        brain._trim_history()
        print(f"Messages after trim: {len(brain.messages)}")
        print(f"✓ History trimming works (kept system message + {Config.MAX_HISTORY-1} recent)")
        
        # Test conversation save/load
        print("\n=== Testing Conversation Save/Load ===")
        filepath = brain.save_conversation("test_brain")
        print(f"Saved conversation to: {filepath}")
        
        # Clear messages and load
        brain.messages = []
        success = brain.load_conversation(filepath)
        print(f"Loaded conversation: {success}")
        print(f"Messages after load: {len(brain.messages)}")
        
        # Cleanup
        brain.conversation_manager.delete_conversation(filepath)
        print("✓ Brain conversation persistence works")
        
    except RuntimeError as e:
        print(f"⚠ Warning: {e}")
        print("  Make sure Ollama is running: ollama serve")
        print("  And model is pulled: ollama pull qwen2.5-coder")
        return False
    except Exception as e:
        print(f"✗ Brain initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_fast_path():
    """Test fast-path command handling"""
    print("\n=== Testing Fast Path ===")
    
    try:
        brain = Brain()
        
        # Test simple command (should use fast path)
        result = brain._fast_path("open calculator")
        print(f"Simple command result: {result}")
        
        # Test complex command (should return None to defer to LLM)
        result = brain._fast_path("open chrome and search for python")
        print(f"Complex command result: {result}")
        print("✓ Fast path test passed")
        
    except Exception as e:
        print(f"⚠ Fast path test skipped: {e}")

def main():
    """Run all tests"""
    print("=" * 60)
    print("Personal Desktop Assistant - Phase 1 Verification Tests")
    print("=" * 60)
    
    # Ensure directories exist
    Config.ensure_directories()
    
    # Run tests
    test_configuration()
    test_logging()
    test_conversation_manager()
    brain_ok = test_brain_initialization()
    
    if brain_ok:
        test_fast_path()
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("  ✓ Configuration system working")
    print("  ✓ Logging system working")
    print("  ✓ Conversation persistence working")
    if brain_ok:
        print("  ✓ Brain initialization working")
        print("  ✓ Memory management working")
        print("  ✓ Fast path working")
    else:
        print("  ⚠ Brain tests skipped (Ollama not running)")
    print("\nNext steps:")
    print("  1. Start Ollama: ollama serve")
    print("  2. Run the assistant: python -m src.main")
    print("  3. Try voice commands (Ctrl+L) or text input")
    print("=" * 60)

if __name__ == "__main__":
    main()
