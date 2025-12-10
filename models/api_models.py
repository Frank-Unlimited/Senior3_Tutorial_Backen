"""API request and response models for Biology Tutorial Workflow."""
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class TutoringStyleEnum(str, Enum):
    """Tutoring style enumeration for API."""
    GUIDED = "guided"
    DIRECT = "direct"


class ModelConfigRequest(BaseModel):
    """Model configuration from frontend."""
    vision_model: Optional[str] = Field(None, description="Vision model ID")
    vision_api_key: Optional[str] = Field(None, description="Vision model API key")
    deep_model: Optional[str] = Field(None, description="Deep thinking model ID")
    deep_api_key: Optional[str] = Field(None, description="Deep thinking model API key")
    quick_model: Optional[str] = Field(None, description="Quick model ID")
    quick_api_key: Optional[str] = Field(None, description="Quick model API key")


class CreateSessionRequest(BaseModel):
    """Request for session creation."""
    models: Optional[ModelConfigRequest] = Field(None, description="Optional model configuration")


class CreateSessionResponse(BaseModel):
    """Response for session creation."""
    session_id: str = Field(..., description="Unique session identifier")
    greeting: str = Field(..., description="Initial greeting message")


class UploadImageResponse(BaseModel):
    """Response for image upload."""
    status: str = Field(..., description="Processing status")
    message: str = Field(..., description="Status message")


class SendMessageRequest(BaseModel):
    """Request for sending a message."""
    content: str = Field(..., description="Message content")
    tutoring_style: Optional[TutoringStyleEnum] = Field(
        None, description="Optional tutoring style selection"
    )


class SendMessageResponse(BaseModel):
    """Response for message sending."""
    content: str = Field(..., description="AI response content")
    is_final: bool = Field(False, description="Whether this is the final response")


class TaskStatusInfo(BaseModel):
    """Status information for a single task."""
    status: str = Field(..., description="Task status")
    result: Optional[Any] = Field(None, description="Task result if completed")
    error: Optional[str] = Field(None, description="Error message if failed")


class TaskStatusResponse(BaseModel):
    """Response for session status query."""
    session_id: str = Field(..., description="Session identifier")
    conversation_state: str = Field(..., description="Current conversation state")
    tasks: Dict[str, str] = Field(..., description="Status of all tasks")
    task_errors: Optional[Dict[str, str]] = Field(None, description="Error messages for failed tasks")
    has_question: bool = Field(..., description="Whether question is extracted")
    has_solution: bool = Field(..., description="Whether solution is ready")
    question_text: Optional[str] = Field(None, description="Extracted question text if ready")
    exam_points: Optional[List[str]] = Field(None, description="Exam points if ready")
    knowledge_points: Optional[List[str]] = Field(None, description="Knowledge points if ready")
    logic_chain_steps: Optional[List[str]] = Field(None, description="Solving steps list for guided tutoring")
    thinking_pattern: Optional[str] = Field(None, description="Thinking pattern for this type of problem")


class SSEEventData(BaseModel):
    """SSE event data structure."""
    type: str = Field(..., description="Event type")
    data: Any = Field(..., description="Event payload")
    timestamp: float = Field(..., description="Event timestamp")
