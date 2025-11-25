"""Memory layer using Mem0."""

from typing import Dict, List, Optional, Any
from mem0 import Memory
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class SessionMemory:
    """Manages session memory using Mem0."""

    def __init__(self):
        """Initialize Mem0 memory."""
        config = {
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": "evoc_deap_sessions",
                    "path": "/tmp/chroma_db"
                }
            }
        }

        if settings.mem0_api_key:
            config["api_key"] = settings.mem0_api_key

        try:
            self.memory = Memory.from_config(config)
            logger.info("Mem0 memory initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize Mem0: {e}. Using in-memory fallback.")
            self.memory = None
            self._fallback_memory: Dict[str, List[Dict]] = {}

    def add_interaction(
        self,
        session_id: str,
        operation: str,
        details: Dict[str, Any],
        result: str
    ) -> None:
        """Store an interaction in memory."""
        message = f"Operation: {operation}\nDetails: {details}\nResult: {result}"

        try:
            if self.memory:
                self.memory.add(
                    messages=[{"role": "user", "content": message}],
                    user_id=session_id
                )
            else:
                # Fallback
                if session_id not in self._fallback_memory:
                    self._fallback_memory[session_id] = []
                self._fallback_memory[session_id].append({
                    "operation": operation,
                    "details": details,
                    "result": result
                })
        except Exception as e:
            logger.error(f"Failed to add memory: {e}")

    def get_history(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve session history."""
        try:
            if self.memory:
                memories = self.memory.get_all(user_id=session_id)
                return memories[-limit:] if memories else []
            else:
                # Fallback
                return self._fallback_memory.get(session_id, [])[-limit:]
        except Exception as e:
            logger.error(f"Failed to get history: {e}")
            return []

    def search_similar(
        self,
        session_id: str,
        query: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar past interactions."""
        try:
            if self.memory:
                results = self.memory.search(
                    query=query,
                    user_id=session_id,
                    limit=limit
                )
                return results
            else:
                # Simple fallback search
                history = self._fallback_memory.get(session_id, [])
                return history[-limit:]
        except Exception as e:
            logger.error(f"Failed to search memory: {e}")
            return []

    def add_preference(self, session_id: str, preference: str) -> None:
        """Store user preference."""
        message = f"User preference: {preference}"
        try:
            if self.memory:
                self.memory.add(
                    messages=[{"role": "system", "content": message}],
                    user_id=session_id,
                    metadata={"type": "preference"}
                )
            else:
                if session_id not in self._fallback_memory:
                    self._fallback_memory[session_id] = []
                self._fallback_memory[session_id].append({
                    "type": "preference",
                    "content": preference
                })
        except Exception as e:
            logger.error(f"Failed to add preference: {e}")

    def get_preferences(self, session_id: str) -> List[str]:
        """Get stored preferences for a session."""
        try:
            if self.memory:
                memories = self.memory.get_all(user_id=session_id)
                return [
                    m.get("content", "")
                    for m in memories
                    if m.get("metadata", {}).get("type") == "preference"
                ]
            else:
                history = self._fallback_memory.get(session_id, [])
                return [
                    h["content"]
                    for h in history
                    if h.get("type") == "preference"
                ]
        except Exception as e:
            logger.error(f"Failed to get preferences: {e}")
            return []


# Global memory instance
session_memory = SessionMemory()
