"""API routes for Biology Tutorial Workflow.

This module defines all REST API endpoints and SSE event streaming.
"""
import uuid
import json
import asyncio
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse

from models.api_models import (
    CreateSessionRequest,
    CreateSessionResponse,
    UploadImageResponse,
    SendMessageRequest,
    SendMessageResponse,
    TaskStatusResponse
)
from session.manager import SessionManager
from session.models import ConversationState
from sse.publisher import SSEPublisher
from workflow.biology_tutor import BiologyTutorWorkflow


router = APIRouter(prefix="/api", tags=["Biology Tutorial"])


# Dependency injection helpers
def get_session_manager(request: Request) -> SessionManager:
    """Get session manager from app state."""
    return request.app.state.session_manager


def get_sse_publisher(request: Request) -> SSEPublisher:
    """Get SSE publisher from app state."""
    return request.app.state.sse_publisher


def get_workflow(request: Request) -> BiologyTutorWorkflow:
    """Get workflow from app state."""
    return request.app.state.workflow


# Greeting generator
def generate_greeting() -> str:
    """Generate warm greeting message."""
    return (
        "你好呀~ 我是你的生物辅导姐姐 🌸\n\n"
        "看到你遇到了一道题目呢，别担心，姐姐来帮你！\n\n"
        "请先上传你的错题图片吧，我会仔细帮你分析的~ 📸"
    )


@router.post("/session", response_model=CreateSessionResponse)
async def create_session(
    request: CreateSessionRequest = None,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """Create a new tutoring session.
    
    Optionally accepts model configuration from frontend.
    Returns a session ID and initial greeting message.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    session_id = str(uuid.uuid4())
    session = await session_manager.create_session(session_id)
    
    logger.info(f"📝 [API] 创建会话: {session_id}")
    logger.info(f"   Request: {request}")
    
    # Store model config in session if provided
    if request and request.models:
        logger.info(f"   📦 收到前端模型配置:")
        logger.info(f"      Vision: {request.models.vision_model}, Key: {request.models.vision_api_key[:10] if request.models.vision_api_key else 'None'}...")
        logger.info(f"      Deep: {request.models.deep_model}, Key: {request.models.deep_api_key[:10] if request.models.deep_api_key else 'None'}...")
        logger.info(f"      Quick: {request.models.quick_model}, Key: {request.models.quick_api_key[:10] if request.models.quick_api_key else 'None'}...")
        
        await session_manager.update_session(
            session_id,
            frontend_model_config={
                "vision_model": request.models.vision_model,
                "vision_api_key": request.models.vision_api_key,
                "deep_model": request.models.deep_model,
                "deep_api_key": request.models.deep_api_key,
                "quick_model": request.models.quick_model,
                "quick_api_key": request.models.quick_api_key,
            }
        )
    else:
        logger.info(f"   ⚠️ 未收到前端模型配置，将使用默认配置")
    
    greeting = generate_greeting()
    
    return CreateSessionResponse(
        session_id=session_id,
        greeting=greeting
    )


@router.post("/session/{session_id}/image", response_model=UploadImageResponse)
async def upload_image(
    session_id: str,
    file: UploadFile = File(...),
    session_manager: SessionManager = Depends(get_session_manager),
    workflow: BiologyTutorWorkflow = Depends(get_workflow)
):
    """Upload an error question image for analysis.
    
    Starts background vision extraction immediately.
    """
    # Verify session exists
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read image data
    image_data = await file.read()
    
    # Start background processing
    await workflow.process_image(
        session_id,
        image_data,
        mime_type=file.content_type
    )
    
    # Update conversation state
    await session_manager.set_conversation_state(
        session_id, ConversationState.INITIAL
    )
    
    return UploadImageResponse(
        status="processing",
        message="图片收到啦~ 正在分析中，请稍等哦~ 🔍"
    )


@router.post("/session/{session_id}/message", response_model=SendMessageResponse)
async def send_message(
    session_id: str,
    request: SendMessageRequest,
    session_manager: SessionManager = Depends(get_session_manager),
    workflow: BiologyTutorWorkflow = Depends(get_workflow)
):
    """Send a message in the tutoring session.
    
    Handles conversation flow based on current state.
    """
    # Verify session exists
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Process message through workflow
    response = await workflow.process_message(session_id, request.content)
    
    # Check if session is complete
    session = await session_manager.get_session(session_id)
    is_final = session.conversation_state == ConversationState.COMPLETED
    
    return SendMessageResponse(
        content=response,
        is_final=is_final
    )


@router.get("/session/{session_id}/events")
async def subscribe_events(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
    sse_publisher: SSEPublisher = Depends(get_sse_publisher)
):
    """Subscribe to SSE events for a session.
    
    Returns a streaming response with real-time task updates.
    """
    # Verify session exists
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    async def event_generator():
        """Generate SSE events."""
        queue = await sse_publisher.subscribe(session_id)
        try:
            # Send initial connection event
            yield f"data: {json.dumps({'type': 'connected', 'data': {'session_id': session_id}})}\n\n"
            
            while True:
                try:
                    # Wait for events with timeout for keepalive
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield event.to_sse_format()
                except asyncio.TimeoutError:
                    # Send keepalive ping
                    yield f": keepalive\n\n"
        except asyncio.CancelledError:
            # Client disconnected
            await sse_publisher.unsubscribe(session_id, queue)
        except Exception:
            await sse_publisher.unsubscribe(session_id, queue)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/session/{session_id}/status", response_model=TaskStatusResponse)
async def get_status(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """Get current status of a tutoring session.
    
    Returns status of all background tasks and available results.
    """
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Collect error messages for failed tasks
    task_errors = {}
    for name, task in session.tasks.items():
        if task.error:
            task_errors[name] = task.error
    
    return TaskStatusResponse(
        session_id=session_id,
        conversation_state=session.conversation_state.value,
        tasks={name: task.status.value for name, task in session.tasks.items()},
        task_errors=task_errors if task_errors else None,
        has_question=session.question_text is not None,
        has_solution=session.solution is not None,
        question_text=session.question_text,
        exam_points=session.exam_points,
        knowledge_points=session.knowledge_points,
        logic_chain_steps=session.logic_chain_steps,
        thinking_pattern=session.thinking_pattern
    )


@router.post("/session/{session_id}/chat", response_model=SendMessageResponse)
async def general_chat(
    session_id: str,
    request: SendMessageRequest,
    session_manager: SessionManager = Depends(get_session_manager),
    workflow: BiologyTutorWorkflow = Depends(get_workflow)
):
    """Handle general chat messages (non-image analysis).
    
    Uses the quick model to respond to general biology questions.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Verify session exists
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        # Get quick model for this session
        quick_model = workflow._get_model_for_session(session, 'quick')
        
        # Create a simple chat prompt
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一位温柔的大姐姐，擅长辅导高三学生的生物学习。
你的特点是：
- 说话温柔有耐心，经常用"呢"、"哦"、"呀"等语气词
- 善于鼓励学生，即使学生答错也会先肯定他们的思考
- 解释问题时会用生动的比喻和例子
- 会关心学生的学习状态和情绪

请用温柔、专业的方式回答学生的生物学问题。"""),
            ("human", "{question}")
        ])
        
        chain = prompt | quick_model | StrOutputParser()
        
        logger.info(f"💬 [Chat] 处理普通聊天: {request.content[:50]}...")
        
        response = await chain.ainvoke({"question": request.content})
        
        logger.info(f"✅ [Chat] 回复生成完成")
        
        return SendMessageResponse(
            content=response,
            is_final=False
        )
        
    except Exception as e:
        logger.error(f"❌ [Chat] 聊天失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # 检查是否是鉴权错误
        error_str = str(e).lower()
        if "401" in error_str or "unauthorized" in error_str or "authentication" in error_str or "invalid api key" in error_str:
            raise HTTPException(
                status_code=401, 
                detail="API 鉴权失败：API Key 无效或已过期，请在设置中检查模型的 API Key"
            )
        elif "403" in error_str or "forbidden" in error_str:
            raise HTTPException(
                status_code=403,
                detail="API 访问被拒绝：请检查 API Key 权限或账户状态"
            )
        else:
            raise HTTPException(status_code=500, detail=str(e))


@router.delete("/session/{session_id}")
async def delete_session(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
    sse_publisher: SSEPublisher = Depends(get_sse_publisher)
):
    """Delete a tutoring session.
    
    Cleans up session data and SSE subscriptions.
    """
    deleted = await session_manager.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    
    await sse_publisher.clear_session(session_id)
    
    return {"status": "deleted", "session_id": session_id}
