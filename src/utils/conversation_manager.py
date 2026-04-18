"""
Conversation persistence manager.
Handles saving and loading conversation history.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from src.config import Config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class ConversationManager:
    """Manages saving and loading conversation history"""
    
    def __init__(self):
        Config.ensure_directories()
        self.conversations_dir = Config.CONVERSATIONS_DIR
    
    def save_conversation(self, messages: List[Dict], name: Optional[str] = None) -> str:
        """
        Save conversation to JSON file.
        
        Args:
            messages: List of message dictionaries
            name: Optional custom name, otherwise uses timestamp
            
        Returns:
            Path to saved file
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{timestamp}.json" if name else f"conversation_{timestamp}.json"
            filepath = self.conversations_dir / filename
            
            conversation_data = {
                "timestamp": timestamp,
                "message_count": len(messages),
                "messages": messages
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Conversation saved to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")
            raise
    
    def load_conversation(self, filepath: str) -> List[Dict]:
        """
        Load conversation from JSON file.
        
        Args:
            filepath: Path to conversation file
            
        Returns:
            List of message dictionaries
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            messages = data.get('messages', [])
            logger.info(f"Loaded conversation with {len(messages)} messages from {filepath}")
            return messages
            
        except Exception as e:
            logger.error(f"Failed to load conversation from {filepath}: {e}")
            raise
    
    def list_conversations(self, limit: int = 10) -> List[Dict[str, str]]:
        """
        List available conversation files.
        
        Args:
            limit: Maximum number of conversations to return
            
        Returns:
            List of dicts with 'name', 'path', and 'timestamp'
        """
        try:
            conversations = []
            
            for filepath in sorted(
                self.conversations_dir.glob("*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )[:limit]:
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    conversations.append({
                        'name': filepath.stem,
                        'path': str(filepath),
                        'timestamp': data.get('timestamp', 'Unknown'),
                        'message_count': data.get('message_count', 0)
                    })
                except Exception as e:
                    logger.warning(f"Skipping invalid conversation file {filepath}: {e}")
                    continue
            
            return conversations
            
        except Exception as e:
            logger.error(f"Failed to list conversations: {e}")
            return []
    
    def delete_conversation(self, filepath: str) -> bool:
        """
        Delete a conversation file.
        
        Args:
            filepath: Path to conversation file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            Path(filepath).unlink()
            logger.info(f"Deleted conversation: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete conversation {filepath}: {e}")
            return False
