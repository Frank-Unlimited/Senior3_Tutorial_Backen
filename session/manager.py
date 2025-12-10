"""Session manager for Biology Tutorial Workflow.

This module provides session management with support for:
- In-memory storage (default)
- Optional Redis storage for persistence
"""
import asyncio
from typing import Optional, Dict, Any
from .models import Session, TaskStatus, TaskState, TutoringStyle, ConversationState


class SessionManager:
    """Manages tutoring sessions with in-memory or Redis storage."""

    def __init__(self, redis_url: Optional[str] = None):
        """Initialize session manager.
        
        Args:
            redis_url: Optional Redis URL for persistent storage.
                      If None, uses in-memory storage.
        """
        self._sessions: Dict[str, Session] = {}
        self._redis_url = redis_url
        self._redis = None
        self._lock = asyncio.Lock()

    async def _init_redis(self) -> None:
        """Initialize Redis connection if configured."""
        if self._redis_url and self._redis is None:
            try:
                import aioredis
                self._redis = await aioredis.from_url(self._redis_url)
            except ImportError:
                print("Warning: aioredis not installed, using in-memory storage")
            except Exception as e:
                print(f"Warning: Failed to connect to Redis: {e}, using in-memory storage")

    async def create_session(self, session_id: str) -> Session:
        """Create a new tutoring session.
        
        Args:
            session_id: Unique identifier for the session
            
        Returns:
            Newly created Session instance
        """
        async with self._lock:
            session = Session(session_id=session_id)
            self._sessions[session_id] = session
            return session

    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session if found, None otherwise
        """
        return self._sessions.get(session_id)

    async def update_session(self, session_id: str, **kwargs) -> None:
        """Update session attributes.
        
        Args:
            session_id: Session identifier
            **kwargs: Attributes to update (e.g., user_thinking, tutoring_style)
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            
            for key, value in kwargs.items():
                if hasattr(session, key):
                    setattr(session, key, value)
                else:
                    raise AttributeError(f"Session has no attribute: {key}")

    async def update_task_status(
        self,
        session_id: str,
        task_name: str,
        status: TaskStatus,
        result: Any = None,
        error: Optional[str] = None
    ) -> None:
        """Update the status of a background task.
        
        Args:
            session_id: Session identifier
            task_name: Name of the task (e.g., 'vision_extraction')
            status: New status for the task
            result: Optional result data (for COMPLETED status)
            error: Optional error message (for FAILED status)
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            
            task = session.get_task(task_name)
            
            if status == TaskStatus.RUNNING:
                task.start()
            elif status == TaskStatus.COMPLETED:
                task.complete(result)
            elif status == TaskStatus.FAILED:
                task.fail(error or "Unknown error")
            else:
                task.status = status

    async def set_conversation_state(
        self,
        session_id: str,
        state: ConversationState
    ) -> None:
        """Update the conversation state.
        
        Args:
            session_id: Session identifier
            state: New conversation state
        """
        await self.update_session(session_id, conversation_state=state)

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str
    ) -> None:
        """Add a message to the conversation history.
        
        Args:
            session_id: Session identifier
            role: Message role ('user' or 'assistant')
            content: Message content
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            
            session.messages.append({
                "role": role,
                "content": content
            })

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session was deleted, False if not found
        """
        async with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

    async def get_all_sessions(self) -> Dict[str, Session]:
        """Get all active sessions.
        
        Returns:
            Dictionary of session_id to Session
        """
        return dict(self._sessions)

    async def cleanup_old_sessions(self, max_age_seconds: int = 3600) -> int:
        """Remove sessions older than max_age_seconds.
        
        Args:
            max_age_seconds: Maximum session age in seconds
            
        Returns:
            Number of sessions removed
        """
        import time
        current_time = time.time()
        removed = 0
        
        async with self._lock:
            sessions_to_remove = [
                sid for sid, session in self._sessions.items()
                if current_time - session.created_at > max_age_seconds
            ]
            for sid in sessions_to_remove:
                del self._sessions[sid]
                removed += 1
        
        return removed
