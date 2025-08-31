import sqlite3
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional

class ChatManager:
    def __init__(self, db_path: str = "scout_chats.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize the chat database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chats (
                    id TEXT PRIMARY KEY,
                    session_id TEXT UNIQUE,
                    title TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    messages TEXT
                )
            """)
            conn.commit()
    
    def create_chat(self, session_id: Optional[str] = None) -> str:
        """Create a new chat and return its ID."""
        chat_id = str(uuid.uuid4())
        if not session_id:
            session_id = str(uuid.uuid4())
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO chats (id, session_id, title, messages) VALUES (?, ?, ?, ?)",
                (chat_id, session_id, "New Chat", "[]")
            )
            conn.commit()
        return chat_id
    
    def get_chats(self) -> List[Dict]:
        """Get all chats ordered by update time."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT id, session_id, title, updated_at FROM chats ORDER BY updated_at DESC"
            )
            return [
                {"id": row[0], "session_id": row[1], "title": row[2], "updated_at": row[3]}
                for row in cursor.fetchall()
            ]
    
    def get_chat(self, chat_id: str) -> Optional[Dict]:
        """Get a specific chat by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT id, session_id, title, messages FROM chats WHERE id = ?",
                (chat_id,)
            )
            row = cursor.fetchone()
            if row:
                return {
                    "id": row[0],
                    "session_id": row[1], 
                    "title": row[2],
                    "messages": json.loads(row[3])
                }
        return None
    
    def update_chat(self, chat_id: str, messages: List, title: Optional[str] = None):
        """Update chat messages and optionally title."""
        # Convert ChatMessage objects to serializable dicts
        serializable_messages = []
        for msg in messages:
            if hasattr(msg, 'role') and hasattr(msg, 'content'):
                # It's a ChatMessage object
                msg_dict = {
                    "role": msg.role,
                    "content": msg.content,
                    "metadata": getattr(msg, 'metadata', None)
                }
                serializable_messages.append(msg_dict)
            else:
                # It's already a dict
                serializable_messages.append(msg)
        
        with sqlite3.connect(self.db_path) as conn:
            if title:
                conn.execute(
                    "UPDATE chats SET messages = ?, title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (json.dumps(serializable_messages), title, chat_id)
                )
            else:
                conn.execute(
                    "UPDATE chats SET messages = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (json.dumps(serializable_messages), chat_id)
                )
            conn.commit()
    
    def delete_chat(self, chat_id: str):
        """Delete a chat."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
            conn.commit()
    
    def get_session_id(self, chat_id: str) -> Optional[str]:
        """Get session_id for a chat."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT session_id FROM chats WHERE id = ?", (chat_id,))
            row = cursor.fetchone()
            return row[0] if row else None