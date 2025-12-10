"""Session data models for Biology Tutorial Workflow.

This module defines the core data structures for managing tutoring sessions,
including task states, tutoring styles, and session context.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List
import time


class TaskStatus(Enum):
    """Status of a background task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TutoringStyle(Enum):
    """User's preferred tutoring style."""
    GUIDED = "guided"      # 引导式辅导 - 一步一步引导
    DIRECT = "direct"      # 直接解答 - 直接给出答案


class ConversationState(Enum):
    """State of the conversation flow."""
    INITIAL = "initial"                    # 初始状态
    AWAITING_THINKING = "awaiting_thinking"  # 等待用户说明思考过程
    AWAITING_STYLE = "awaiting_style"        # 等待用户选择辅导方式
    TUTORING = "tutoring"                    # 辅导进行中
    COMPLETED = "completed"                  # 辅导完成


@dataclass
class TaskState:
    """State of a single background task."""
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    def start(self) -> None:
        """Mark task as started."""
        self.status = TaskStatus.RUNNING
        self.started_at = time.time()

    def complete(self, result: Any) -> None:
        """Mark task as completed with result."""
        self.status = TaskStatus.COMPLETED
        self.result = result
        self.completed_at = time.time()

    def fail(self, error: str) -> None:
        """Mark task as failed with error."""
        self.status = TaskStatus.FAILED
        self.error = error
        self.completed_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at
        }


@dataclass
class Session:
    """A tutoring session containing all context and state."""
    session_id: str
    created_at: float = field(default_factory=time.time)
    
    # Frontend model configuration (API keys and model selection)
    frontend_model_config: Optional[Dict[str, Any]] = None
    
    # User input
    image_data: Optional[bytes] = None
    user_thinking: Optional[str] = None
    user_confusion: Optional[str] = None
    tutoring_style: Optional[TutoringStyle] = None
    
    # Conversation state
    conversation_state: ConversationState = ConversationState.INITIAL
    messages: List[Dict[str, str]] = field(default_factory=list)
    
    # Task states - all background processing tasks
    tasks: Dict[str, TaskState] = field(default_factory=lambda: {
        "vision_extraction": TaskState(),
        "exam_points": TaskState(),
        "deep_solution": TaskState(),
        "knowledge_points": TaskState(),
        "logic_chain": TaskState(),
    })
    
    # Results from completed tasks
    question_text: Optional[str] = None
    exam_points: Optional[List[str]] = None
    solution: Optional[str] = None
    knowledge_points: Optional[List[str]] = None
    common_mistakes: Optional[List[str]] = None
    logic_chain_steps: Optional[List[str]] = None  # 解题步骤列表
    thinking_pattern: Optional[str] = None  # 思维模式

    def get_task(self, task_name: str) -> TaskState:
        """Get a task state by name."""
        if task_name not in self.tasks:
            raise KeyError(f"Unknown task: {task_name}")
        return self.tasks[task_name]

    def is_all_tasks_completed(self) -> bool:
        """Check if all tasks are completed (success or failure)."""
        return all(
            task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
            for task in self.tasks.values()
        )

    def is_ready_for_tutoring(self) -> bool:
        """Check if we have enough data to start tutoring."""
        return (
            self.question_text is not None
            and self.tutoring_style is not None
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "user_thinking": self.user_thinking,
            "user_confusion": self.user_confusion,
            "tutoring_style": self.tutoring_style.value if self.tutoring_style else None,
            "conversation_state": self.conversation_state.value,
            "tasks": {name: task.to_dict() for name, task in self.tasks.items()},
            "question_text": self.question_text,
            "exam_points": self.exam_points,
            "solution": self.solution,
            "knowledge_points": self.knowledge_points,
            "common_mistakes": self.common_mistakes,
            "logic_chain_steps": self.logic_chain_steps,
            "thinking_pattern": self.thinking_pattern,
        }
