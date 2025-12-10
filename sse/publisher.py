"""SSE (Server-Sent Events) publisher for Biology Tutorial Workflow.

This module provides real-time event publishing to connected clients with:
- Multiple subscribers per session
- Pending event storage for reconnection recovery
- Chronological event ordering
"""
import asyncio
import time
import json
from typing import Dict, Set, Any, Optional, List
from dataclasses import dataclass, field


@dataclass
class SSEEvent:
    """An SSE event to be sent to clients."""
    type: str
    data: Any
    timestamp: float = field(default_factory=time.time)

    def to_sse_format(self) -> str:
        """Format event for SSE transmission."""
        payload = {
            "type": self.type,
            "data": self.data,
            "timestamp": self.timestamp
        }
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type,
            "data": self.data,
            "timestamp": self.timestamp
        }


class SSEPublisher:
    """SSE event publisher with reconnection support."""

    def __init__(self, max_pending_events: int = 100):
        """Initialize SSE publisher.
        
        Args:
            max_pending_events: Maximum pending events to store per session
        """
        self._subscribers: Dict[str, Set[asyncio.Queue]] = {}
        self._pending_events: Dict[str, List[SSEEvent]] = {}
        self._max_pending = max_pending_events
        self._lock = asyncio.Lock()

    async def subscribe(self, session_id: str) -> asyncio.Queue:
        """Subscribe to events for a session.
        
        Creates a new queue for receiving events. If there are pending
        events from a previous disconnection, they will be delivered
        immediately upon subscription.
        
        Args:
            session_id: Session to subscribe to
            
        Returns:
            Queue that will receive SSEEvent objects
        """
        async with self._lock:
            if session_id not in self._subscribers:
                self._subscribers[session_id] = set()
            
            queue: asyncio.Queue = asyncio.Queue()
            self._subscribers[session_id].add(queue)
            
            # Deliver pending events from disconnection period
            if session_id in self._pending_events:
                pending = self._pending_events[session_id]
                # Sort by timestamp to ensure chronological order
                pending.sort(key=lambda e: e.timestamp)
                for event in pending:
                    await queue.put(event)
                del self._pending_events[session_id]
            
            return queue

    async def unsubscribe(self, session_id: str, queue: asyncio.Queue) -> None:
        """Unsubscribe from session events.
        
        Args:
            session_id: Session to unsubscribe from
            queue: The queue to remove
        """
        async with self._lock:
            if session_id in self._subscribers:
                self._subscribers[session_id].discard(queue)
                # Clean up empty subscriber sets
                if not self._subscribers[session_id]:
                    del self._subscribers[session_id]

    async def publish(
        self,
        session_id: str,
        event_type: str,
        data: Any
    ) -> None:
        """Publish an event to all subscribers of a session.
        
        If no subscribers are connected, the event is stored as pending
        and will be delivered when a client reconnects.
        
        Args:
            session_id: Target session
            event_type: Type of event (e.g., 'question_extracted')
            data: Event payload data
        """
        event = SSEEvent(type=event_type, data=data)
        
        async with self._lock:
            subscribers = self._subscribers.get(session_id, set())
            
            if subscribers:
                # Deliver to all connected subscribers
                for queue in subscribers:
                    try:
                        await queue.put(event)
                    except Exception:
                        # Queue might be closed, ignore
                        pass
            else:
                # No subscribers, store as pending
                if session_id not in self._pending_events:
                    self._pending_events[session_id] = []
                
                pending = self._pending_events[session_id]
                pending.append(event)
                
                # Limit pending events to prevent memory issues
                if len(pending) > self._max_pending:
                    pending.pop(0)

    async def publish_task_completed(
        self,
        session_id: str,
        task_name: str,
        result: Any
    ) -> None:
        """Convenience method to publish task completion event.
        
        Args:
            session_id: Target session
            task_name: Name of completed task
            result: Task result data
        """
        event_type_map = {
            "vision_extraction": "question_extracted",
            "exam_points": "exam_points_ready",
            "deep_solution": "solution_ready",
            "knowledge_points": "knowledge_ready",
            "logic_chain": "logic_chain_ready",
        }
        event_type = event_type_map.get(task_name, f"{task_name}_completed")
        await self.publish(session_id, event_type, result)

    async def publish_task_failed(
        self,
        session_id: str,
        task_name: str,
        error: str
    ) -> None:
        """Convenience method to publish task failure event.
        
        Args:
            session_id: Target session
            task_name: Name of failed task
            error: Error message
        """
        await self.publish(session_id, "task_failed", {
            "task": task_name,
            "error": error
        })

    async def publish_session_complete(self, session_id: str) -> None:
        """Publish session completion event.
        
        Args:
            session_id: Target session
        """
        await self.publish(session_id, "session_complete", {
            "message": "All tasks completed"
        })

    def has_subscribers(self, session_id: str) -> bool:
        """Check if a session has active subscribers.
        
        Args:
            session_id: Session to check
            
        Returns:
            True if there are active subscribers
        """
        return bool(self._subscribers.get(session_id))

    def get_pending_count(self, session_id: str) -> int:
        """Get count of pending events for a session.
        
        Args:
            session_id: Session to check
            
        Returns:
            Number of pending events
        """
        return len(self._pending_events.get(session_id, []))

    async def clear_session(self, session_id: str) -> None:
        """Clear all data for a session.
        
        Args:
            session_id: Session to clear
        """
        async with self._lock:
            self._subscribers.pop(session_id, None)
            self._pending_events.pop(session_id, None)
