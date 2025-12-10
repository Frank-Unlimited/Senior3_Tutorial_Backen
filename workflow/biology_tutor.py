"""Main Biology Tutor Workflow orchestrating all chains.

This module implements the BiologyTutorWorkflow class that coordinates
all background tasks with proper async execution and SSE notifications.
"""
import asyncio
import logging
from typing import Optional, Any, Dict
from langchain_core.runnables import RunnableParallel

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from config.settings import Settings
from session.manager import SessionManager
from session.models import TaskStatus, TutoringStyle, ConversationState, Phase2State
from sse.publisher import SSEPublisher
from workflow.model_factory import ModelFactory
from workflow.chains.vision_chain import create_vision_chain
from workflow.chains.solution_chain import create_solution_chain
from workflow.chains.exam_points_chain import create_exam_points_chain
from workflow.chains.knowledge_chain import create_knowledge_chain
from workflow.chains.logic_chain import create_logic_chain
from workflow.phase2_workflow import Phase2Workflow


class BiologyTutorWorkflow:
    """Main workflow orchestrator for biology tutoring."""

    def __init__(
        self,
        settings: Settings,
        session_manager: SessionManager,
        sse_publisher: SSEPublisher
    ):
        """Initialize the workflow.
        
        Args:
            settings: Application settings
            session_manager: Session manager instance
            sse_publisher: SSE publisher instance
        """
        self.settings = settings
        self.session_manager = session_manager
        self.sse = sse_publisher
        
        # Initialize models
        self._init_models()
        
        # Initialize chains
        self._init_chains()

    def _init_models(self) -> None:
        """Initialize AI models from settings."""
        logger.info("🔧 初始化模型...")
        logger.info(f"  Vision Model: {self.settings.vision_model.model_name}")
        logger.info(f"  Vision API Base: {self.settings.vision_model.api_base}")
        logger.info(f"  Vision API Key: {self.settings.vision_model.api_key[:10]}...")
        
        self.vision_model = ModelFactory.create_vision_model(
            self.settings.vision_model
        )
        self.deep_model = ModelFactory.create(
            self.settings.deep_thinking_model
        )
        self.quick_model = ModelFactory.create(
            self.settings.quick_model
        )
        logger.info("✅ 模型初始化完成")

    def _init_chains(self) -> None:
        """Initialize LangChain chains with default models."""
        self.vision_chain = create_vision_chain(self.vision_model)
        self.solution_chain = create_solution_chain(
            self.deep_model,
            self.settings.persona_prompt
        )
        self.exam_points_chain = create_exam_points_chain(self.quick_model)
        self.knowledge_chain = create_knowledge_chain(self.quick_model)
        self.logic_chain_extractor = create_logic_chain(self.quick_model)
        
        # Initialize Phase 2 workflow
        self.phase2_workflow = Phase2Workflow(
            self.settings,
            self.session_manager,
            self.sse,
            self.deep_model
        )
    
    def _get_model_for_session(self, session, model_type: str):
        """Get model for session, using frontend config if available.
        
        Args:
            session: Session object
            model_type: 'vision', 'deep', or 'quick'
            
        Returns:
            Model instance (either from frontend config or default)
        """
        if not session or not session.frontend_model_config:
            # Use default models
            if model_type == 'vision':
                return self.vision_model
            elif model_type == 'deep':
                return self.deep_model
            else:
                return self.quick_model
        
        config = session.frontend_model_config
        logger.info(f"🔧 [Session] 使用前端模型配置: {model_type}")
        
        # Map model_type to config keys
        model_key = f"{model_type}_model"
        api_key_key = f"{model_type}_api_key"
        
        model_id = config.get(model_key)
        api_key = config.get(api_key_key)
        
        # If no API key provided, use default model
        if not api_key:
            logger.info(f"   未提供 {model_type} API Key，使用默认模型")
            if model_type == 'vision':
                return self.vision_model
            elif model_type == 'deep':
                return self.deep_model
            else:
                return self.quick_model
        
        logger.info(f"   Model ID: {model_id}")
        logger.info(f"   API Key: {api_key[:10]}..." if api_key else "   API Key: None")
        
        # Create dynamic model based on model_id
        return self._create_dynamic_model(model_id, api_key, model_type)
    
    def _create_dynamic_model(self, model_id: str, api_key: str, model_type: str):
        """Create a model dynamically based on model ID and API key.
        
        Args:
            model_id: Model identifier (e.g., 'doubao-vision', 'gpt-4')
            api_key: API key for the model
            model_type: 'vision', 'deep', or 'quick'
            
        Returns:
            Model instance
        """
        from langchain_openai import ChatOpenAI
        
        # Determine provider and actual model name based on model_id
        if model_id.startswith('doubao'):
            # Doubao models
            model_map = {
                'doubao-vision': 'doubao-1-5-vision-pro-32k-250115',
                'doubao-pro': 'doubao-1-5-pro-32k-250115',
                'doubao-lite': 'doubao-1-5-lite-32k-250115',
            }
            model_name = model_map.get(model_id, model_id)
            api_base = "https://ark.cn-beijing.volces.com/api/v3"
        elif model_id.startswith('gpt'):
            # OpenAI models
            model_map = {
                'gpt-4': 'gpt-4-turbo-preview',
                'gpt-4-vision': 'gpt-4-vision-preview',
                'gpt-3.5': 'gpt-3.5-turbo',
            }
            model_name = model_map.get(model_id, model_id)
            api_base = "https://api.openai.com/v1"
        elif model_id.startswith('claude'):
            # Anthropic models (via OpenAI-compatible API)
            model_map = {
                'claude-3': 'claude-3-opus-20240229',
                'claude-vision': 'claude-3-opus-20240229',
                'claude-instant': 'claude-instant-1.2',
            }
            model_name = model_map.get(model_id, model_id)
            api_base = "https://api.anthropic.com/v1"
        elif model_id == 'deepseek':
            model_name = "deepseek-chat"
            api_base = "https://api.deepseek.com/v1"
        else:
            # Default to doubao
            model_name = model_id
            api_base = "https://ark.cn-beijing.volces.com/api/v3"
        
        # Set temperature and max_tokens based on model type
        if model_type == 'vision':
            temperature = 0.3
            max_tokens = 2048
        elif model_type == 'deep':
            temperature = 0.7
            max_tokens = 8192
        else:  # quick
            temperature = 0.5
            max_tokens = 1024
        
        logger.info(f"🤖 [Dynamic] 创建动态模型: {model_name}")
        logger.info(f"   API Base: {api_base}")
        logger.info(f"   Temperature: {temperature}")
        
        return ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            openai_api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def process_image(
        self,
        session_id: str,
        image_data: bytes,
        mime_type: str = "image/jpeg"
    ) -> None:
        """Process uploaded image - starts background vision extraction.
        
        This method returns immediately after starting the background task.
        
        Args:
            session_id: Session identifier
            image_data: Raw image bytes
            mime_type: Image MIME type
        """
        # Store image in session
        await self.session_manager.update_session(
            session_id,
            image_data=image_data
        )
        
        # Start background vision extraction
        asyncio.create_task(
            self._run_vision_extraction(session_id, image_data, mime_type)
        )

    async def _run_vision_extraction(
        self,
        session_id: str,
        image_data: bytes,
        mime_type: str
    ) -> None:
        """Run vision extraction in background.
        
        Args:
            session_id: Session identifier
            image_data: Raw image bytes
            mime_type: Image MIME type
        """
        logger.info(f"🖼️ [Vision] 开始视觉理解任务 session={session_id}")
        logger.info(f"🖼️ [Vision] 图片大小: {len(image_data)} bytes, MIME: {mime_type}")
        
        # Update task status to running
        await self.session_manager.update_task_status(
            session_id, "vision_extraction", TaskStatus.RUNNING
        )
        
        try:
            # Get session to check for frontend model config
            session = await self.session_manager.get_session(session_id)
            
            # Get vision model (dynamic or default)
            vision_model = self._get_model_for_session(session, 'vision')
            
            # Create vision chain with the model
            vision_chain = create_vision_chain(vision_model)
            
            logger.info("🖼️ [Vision] 调用视觉模型 API...")
            # Run vision chain
            question_text = await vision_chain.ainvoke({
                "image_data": image_data,
                "mime_type": mime_type
            })
            
            logger.info(f"✅ [Vision] 视觉理解成功! 提取文本长度: {len(question_text)}")
            logger.debug(f"✅ [Vision] 提取内容: {question_text[:200]}...")
            
            # Store result in session
            await self.session_manager.update_session(
                session_id,
                question_text=question_text
            )
            
            # Update task status
            await self.session_manager.update_task_status(
                session_id, "vision_extraction", TaskStatus.COMPLETED,
                result=question_text
            )
            
            # Notify frontend
            logger.info("📤 [Vision] 发送 SSE 通知...")
            await self.sse.publish_task_completed(
                session_id, "vision_extraction",
                {"question_text": question_text}
            )
            
            # Start parallel analysis tasks
            await self._start_parallel_analysis(session_id, question_text)
            
        except Exception as e:
            logger.error(f"❌ [Vision] 视觉理解失败: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # 检查是否是鉴权错误
            error_str = str(e).lower()
            error_msg = str(e)
            if "401" in error_str or "unauthorized" in error_str or "authentication" in error_str or "invalid api key" in error_str:
                error_msg = "API 鉴权失败：API Key 无效或已过期，请在设置中检查视觉模型的 API Key"
            elif "403" in error_str or "forbidden" in error_str:
                error_msg = "API 访问被拒绝：请检查 API Key 权限或账户状态"
            elif "rate limit" in error_str or "429" in error_str:
                error_msg = "请求太频繁，请稍后重试"
            
            # Update task status to failed
            await self.session_manager.update_task_status(
                session_id, "vision_extraction", TaskStatus.FAILED,
                error=error_msg
            )
            
            # Notify frontend of failure
            await self.sse.publish_task_failed(
                session_id, "vision_extraction", error_msg
            )

    async def _start_parallel_analysis(
        self,
        session_id: str,
        question_text: str
    ) -> None:
        """Start parallel exam points and solution tasks after vision extraction.
        
        Uses LangChain's RunnableParallel for concurrent execution of:
        - exam_points: Quick summary of exam points (fast)
        - deep_solution: Detailed solution generation (slow)
        
        Both tasks run in parallel based on the extracted question text.
        User input (thinking process, tutoring style) is collected separately
        and will be used in Phase 2 for personalized tutoring.
        
        Args:
            session_id: Session identifier
            question_text: Extracted question text
        """
        logger.info(f"🔀 [Parallel] 启动并行分析任务 session={session_id}")
        
        # Get session for context
        session = await self.session_manager.get_session(session_id)
        if not session:
            logger.error(f"❌ [Parallel] 会话不存在: {session_id}")
            return
        
        # Update both task statuses to RUNNING
        await self.session_manager.update_task_status(
            session_id, "exam_points", TaskStatus.RUNNING
        )
        await self.session_manager.update_task_status(
            session_id, "deep_solution", TaskStatus.RUNNING
        )
        
        # Get models for this session (dynamic or default)
        quick_model = self._get_model_for_session(session, 'quick')
        deep_model = self._get_model_for_session(session, 'deep')
        
        # Create chains with session-specific models
        exam_points_chain = create_exam_points_chain(quick_model)
        solution_chain = create_solution_chain(deep_model, self.settings.persona_prompt)
        
        # Create RunnableParallel for concurrent execution
        parallel_chain = RunnableParallel(
            exam_points=exam_points_chain,
            solution=solution_chain
        )
        
        # Start parallel execution in background
        asyncio.create_task(
            self._run_parallel_analysis(session_id, question_text, parallel_chain)
        )
    
    async def _run_parallel_analysis(
        self,
        session_id: str,
        question_text: str,
        parallel_chain: RunnableParallel
    ) -> None:
        """Execute parallel analysis using RunnableParallel.
        
        This method runs exam_points and solution chains concurrently,
        then handles results and triggers subsequent tasks.
        
        Args:
            session_id: Session identifier
            question_text: Question text
            parallel_chain: RunnableParallel instance
        """
        import time
        start_time = time.time()
        
        logger.info(f"⏱️ [Parallel] 开始并行执行...")
        
        try:
            # Execute both chains in parallel
            results = await parallel_chain.ainvoke({
                "question": question_text
            })
            
            elapsed = time.time() - start_time
            logger.info(f"✅ [Parallel] 并行执行完成，耗时: {elapsed:.2f}s")
            
            # Handle exam points result (fast task)
            await self._handle_exam_points_result(session_id, results.get("exam_points", {}))
            
            # Handle solution result (slow task)
            await self._handle_solution_result(session_id, question_text, results.get("solution", ""))
            
        except Exception as e:
            logger.error(f"❌ [Parallel] 并行执行失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # 检查是否是鉴权错误
            error_str = str(e).lower()
            error_msg = str(e)
            if "401" in error_str or "unauthorized" in error_str or "authentication" in error_str or "invalid api key" in error_str:
                error_msg = "API 鉴权失败：API Key 无效或已过期，请在设置中检查模型的 API Key"
            elif "403" in error_str or "forbidden" in error_str:
                error_msg = "API 访问被拒绝：请检查 API Key 权限或账户状态"
            elif "rate limit" in error_str or "429" in error_str:
                error_msg = "请求太频繁，请稍后重试"
            
            # Mark both tasks as failed
            await self.session_manager.update_task_status(
                session_id, "exam_points", TaskStatus.FAILED, error=error_msg
            )
            await self.session_manager.update_task_status(
                session_id, "deep_solution", TaskStatus.FAILED, error=error_msg
            )
            
            # Notify frontend
            await self.sse.publish_task_failed(session_id, "exam_points", error_msg)
            await self.sse.publish_task_failed(session_id, "deep_solution", error_msg)
    
    async def _handle_exam_points_result(
        self,
        session_id: str,
        result: dict
    ) -> None:
        """Handle exam points extraction result.
        
        Args:
            session_id: Session identifier
            result: Exam points result dict
        """
        logger.info(f"📊 [ExamPoints] 处理考察点结果")
        
        try:
            # Store result in session
            await self.session_manager.update_session(
                session_id,
                exam_points=result.get("exam_points", [])
            )
            
            # Update task status
            await self.session_manager.update_task_status(
                session_id, "exam_points", TaskStatus.COMPLETED,
                result=result
            )
            
            # Notify frontend
            await self.sse.publish_task_completed(
                session_id, "exam_points", result
            )
            
            logger.info(f"✅ [ExamPoints] 考察点处理完成")
            
        except Exception as e:
            logger.error(f"❌ [ExamPoints] 处理失败: {e}")
            await self.session_manager.update_task_status(
                session_id, "exam_points", TaskStatus.FAILED, error=str(e)
            )
            await self.sse.publish_task_failed(session_id, "exam_points", str(e))
    
    async def _handle_solution_result(
        self,
        session_id: str,
        question_text: str,
        solution: str
    ) -> None:
        """Handle deep solution generation result.
        
        Args:
            session_id: Session identifier
            question_text: Original question text
            solution: Generated solution
        """
        logger.info(f"🧠 [Solution] 处理解答结果")
        
        try:
            # Store result in session
            await self.session_manager.update_session(
                session_id,
                solution=solution
            )
            
            # Update task status
            await self.session_manager.update_task_status(
                session_id, "deep_solution", TaskStatus.COMPLETED,
                result=solution
            )
            
            # Notify frontend
            await self.sse.publish_task_completed(
                session_id, "deep_solution",
                {"solution": solution}
            )
            
            logger.info(f"✅ [Solution] 解答处理完成")
            
            # Start knowledge and logic extraction
            await self._start_knowledge_extraction(
                session_id, question_text, solution
            )
            
        except Exception as e:
            logger.error(f"❌ [Solution] 处理失败: {e}")
            await self.session_manager.update_task_status(
                session_id, "deep_solution", TaskStatus.FAILED, error=str(e)
            )
            await self.sse.publish_task_failed(session_id, "deep_solution", str(e))

    async def _start_knowledge_extraction(
        self,
        session_id: str,
        question_text: str,
        solution: str
    ) -> None:
        """Start parallel knowledge and logic chain extraction using RunnableParallel.
        
        Uses LangChain's RunnableParallel for concurrent execution of:
        - knowledge_points: Knowledge points and common mistakes extraction
        - logic_chain: Solving logic chain extraction
        
        Args:
            session_id: Session identifier
            question_text: Question text
            solution: Generated solution
        """
        logger.info(f"🔀 [Knowledge] 启动知识点和逻辑链并行提取 session={session_id}")
        
        # Get session for model config
        session = await self.session_manager.get_session(session_id)
        
        # Update both task statuses to RUNNING
        await self.session_manager.update_task_status(
            session_id, "knowledge_points", TaskStatus.RUNNING
        )
        await self.session_manager.update_task_status(
            session_id, "logic_chain", TaskStatus.RUNNING
        )
        
        # Get quick model for this session (dynamic or default)
        quick_model = self._get_model_for_session(session, 'quick')
        
        # Create chains with session-specific model
        knowledge_chain = create_knowledge_chain(quick_model)
        logic_chain = create_logic_chain(quick_model)
        
        # Create RunnableParallel for concurrent execution
        knowledge_parallel_chain = RunnableParallel(
            knowledge=knowledge_chain,
            logic=logic_chain
        )
        
        # Start parallel execution in background
        asyncio.create_task(
            self._run_knowledge_parallel(
                session_id, question_text, solution, knowledge_parallel_chain
            )
        )
    
    async def _run_knowledge_parallel(
        self,
        session_id: str,
        question_text: str,
        solution: str,
        parallel_chain: RunnableParallel
    ) -> None:
        """Execute parallel knowledge and logic extraction.
        
        Args:
            session_id: Session identifier
            question_text: Question text
            solution: Generated solution
            parallel_chain: RunnableParallel instance
        """
        import time
        start_time = time.time()
        
        logger.info(f"⏱️ [Knowledge] 开始并行执行知识点和逻辑链提取...")
        
        try:
            # Execute both chains in parallel
            results = await parallel_chain.ainvoke({
                "question": question_text,
                "solution": solution
            })
            
            elapsed = time.time() - start_time
            logger.info(f"✅ [Knowledge] 并行执行完成，耗时: {elapsed:.2f}s")
            
            # Handle knowledge points result
            await self._handle_knowledge_result(session_id, results.get("knowledge", {}))
            
            # Handle logic chain result
            await self._handle_logic_result(session_id, results.get("logic", {}))
            
            # Check if all tasks complete and mark session as complete
            await self._check_session_complete(session_id)
            
        except Exception as e:
            logger.error(f"❌ [Knowledge] 并行执行失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Mark both tasks as failed
            await self.session_manager.update_task_status(
                session_id, "knowledge_points", TaskStatus.FAILED, error=str(e)
            )
            await self.session_manager.update_task_status(
                session_id, "logic_chain", TaskStatus.FAILED, error=str(e)
            )
            
            # Notify frontend
            await self.sse.publish_task_failed(session_id, "knowledge_points", str(e))
            await self.sse.publish_task_failed(session_id, "logic_chain", str(e))
    
    async def _handle_knowledge_result(
        self,
        session_id: str,
        result: dict
    ) -> None:
        """Handle knowledge points extraction result.
        
        Args:
            session_id: Session identifier
            result: Knowledge points result dict
        """
        logger.info(f"📚 [Knowledge] 处理知识点结果")
        
        try:
            # Store result in session
            # knowledge_points is now a list of strings, not objects
            await self.session_manager.update_session(
                session_id,
                knowledge_points=result.get("knowledge_points", []),
                common_mistakes=[cm.get("mistake", "") for cm in result.get("common_mistakes", [])]
            )
            
            # Update task status
            await self.session_manager.update_task_status(
                session_id, "knowledge_points", TaskStatus.COMPLETED,
                result=result
            )
            
            # Notify frontend
            await self.sse.publish_task_completed(
                session_id, "knowledge_points", result
            )
            
            logger.info(f"✅ [Knowledge] 知识点处理完成")
            
        except Exception as e:
            logger.error(f"❌ [Knowledge] 处理失败: {e}")
            await self.session_manager.update_task_status(
                session_id, "knowledge_points", TaskStatus.FAILED, error=str(e)
            )
            await self.sse.publish_task_failed(session_id, "knowledge_points", str(e))
    
    async def _handle_logic_result(
        self,
        session_id: str,
        result: dict
    ) -> None:
        """Handle logic chain extraction result.
        
        Args:
            session_id: Session identifier
            result: Logic chain result dict
        """
        logger.info(f"🔗 [Logic] 处理逻辑链结果")
        
        try:
            # Store result in session
            await self.session_manager.update_session(
                session_id,
                logic_chain_steps=result.get("steps", []),
                thinking_pattern=result.get("thinking_pattern", "")
            )
            
            # Update task status
            await self.session_manager.update_task_status(
                session_id, "logic_chain", TaskStatus.COMPLETED,
                result=result
            )
            
            # Notify frontend
            await self.sse.publish_task_completed(
                session_id, "logic_chain", result
            )
            
            logger.info(f"✅ [Logic] 逻辑链处理完成")
            
        except Exception as e:
            logger.error(f"❌ [Logic] 处理失败: {e}")
            await self.session_manager.update_task_status(
                session_id, "logic_chain", TaskStatus.FAILED, error=str(e)
            )
            await self.sse.publish_task_failed(session_id, "logic_chain", str(e))

    async def _check_session_complete(self, session_id: str) -> None:
        """Check if all background tasks are complete and notify.
        
        Args:
            session_id: Session identifier
        """
        session = await self.session_manager.get_session(session_id)
        if session and session.is_all_tasks_completed():
            await self.sse.publish_session_complete(session_id)
            
            # Check if Phase 1 is fully complete (tasks + user input)
            await self._check_phase1_complete(session_id)
    
    async def _check_phase1_complete(self, session_id: str) -> None:
        """Check if Phase 1 data collection is complete.
        
        Phase 1 requires:
        - All background tasks completed (vision, exam_points, solution, knowledge, logic)
        - User thinking process collected
        - User tutoring style selected
        
        Args:
            session_id: Session identifier
        """
        session = await self.session_manager.get_session(session_id)
        if not session:
            return
        
        # Check all 8 data points
        has_all_tasks = session.is_all_tasks_completed()
        has_user_thinking = session.user_thinking is not None
        has_tutoring_style = session.tutoring_style is not None
        
        if has_all_tasks and has_user_thinking and has_tutoring_style:
            # Phase 1 complete - log summary
            self._log_phase1_summary(session)
    
    def _log_phase1_summary(self, session: Any) -> None:
        """Log summary of all collected data at end of phase 1.
        
        Phase 1 collects:
        1. 题干 (question_text)
        2. 详细解答过程 (solution)
        3. 快速考察点总结 (exam_points)
        4. 知识点列表 (knowledge_points)
        5. 易错点列表 (common_mistakes)
        6. 解题逻辑链梳理 (logic_chain)
        7. 用户的思考过程 (user_thinking)
        8. 用户希望的辅导方式 (tutoring_style)
        """
        logger.info("=" * 60)
        logger.info("📋 [Phase 1 Complete] 第一阶段数据收集完成")
        logger.info("=" * 60)
        
        # 1. 题干
        logger.info(f"📝 1. 题干 (question_text):")
        logger.info(f"   {session.question_text[:200] if session.question_text else '未获取'}...")
        
        # 2. 详细解答过程
        logger.info(f"✅ 2. 详细解答过程 (solution):")
        logger.info(f"   {session.solution[:200] if session.solution else '未生成'}...")
        
        # 3. 快速考察点总结
        logger.info(f"📊 3. 快速考察点总结 (exam_points):")
        if session.exam_points:
            for i, point in enumerate(session.exam_points[:5], 1):
                logger.info(f"   {i}. {point}")
        else:
            logger.info("   未获取")
        
        # 4. 知识点列表
        logger.info(f"📚 4. 知识点列表 (knowledge_points):")
        if session.knowledge_points:
            for i, kp in enumerate(session.knowledge_points[:5], 1):
                logger.info(f"   {i}. {kp}")
        else:
            logger.info("   未获取")
        
        # 5. 易错点列表
        logger.info(f"⚠️ 5. 易错点列表 (common_mistakes):")
        if session.common_mistakes:
            for i, cm in enumerate(session.common_mistakes[:5], 1):
                logger.info(f"   {i}. {cm}")
        else:
            logger.info("   未获取")
        
        # 6. 解题逻辑链梳理
        logger.info(f"🔗 6. 解题步骤 (logic_chain_steps):")
        if session.logic_chain_steps:
            for i, step in enumerate(session.logic_chain_steps, 1):
                logger.info(f"   {i}. {step}")
        else:
            logger.info("   未获取")
        
        if session.thinking_pattern:
            logger.info(f"   思维模式: {session.thinking_pattern}")
        
        # 7. 用户的思考过程
        logger.info(f"💭 7. 用户的思考过程 (user_thinking):")
        logger.info(f"   {session.user_thinking if session.user_thinking else '用户未提供'}")
        
        # 8. 用户希望的辅导方式
        logger.info(f"🎯 8. 用户希望的辅导方式 (tutoring_style):")
        style_name = session.tutoring_style.value if session.tutoring_style else '未选择'
        style_desc = "引导式辅导" if style_name == "guided" else "直接解答" if style_name == "direct" else "未选择"
        logger.info(f"   {style_name} ({style_desc})")
        
        # Task status summary
        logger.info("-" * 60)
        logger.info("📈 任务状态汇总:")
        for task_name, task_state in session.tasks.items():
            status_emoji = "✅" if task_state.status.value == "completed" else "❌" if task_state.status.value == "failed" else "⏳"
            logger.info(f"   {status_emoji} {task_name}: {task_state.status.value}")
        
        logger.info("=" * 60)

    async def process_message(
        self,
        session_id: str,
        message: str
    ) -> str:
        """Process user message based on conversation state.
        
        Args:
            session_id: Session identifier
            message: User message content
            
        Returns:
            AI response message
        """
        session = await self.session_manager.get_session(session_id)
        if not session:
            return "抱歉，找不到你的会话呢，请重新开始吧~"
        
        # Add user message to history
        await self.session_manager.add_message(session_id, "user", message)
        
        # Handle based on conversation state
        state = session.conversation_state
        
        if state == ConversationState.INITIAL:
            # First message after image upload - ask about thinking
            # This handles both empty trigger message and actual user message
            response = self._generate_thinking_prompt()
            await self.session_manager.set_conversation_state(
                session_id, ConversationState.AWAITING_THINKING
            )
            
        elif state == ConversationState.AWAITING_THINKING:
            # Store thinking and ask about tutoring style
            await self.session_manager.update_session(
                session_id,
                user_thinking=message
            )
            response = self._generate_style_prompt()
            await self.session_manager.set_conversation_state(
                session_id, ConversationState.AWAITING_STYLE
            )
            
        elif state == ConversationState.AWAITING_STYLE:
            # Parse tutoring style choice and save it
            style = self._parse_tutoring_style(message)
            await self.session_manager.update_session(
                session_id,
                tutoring_style=style
            )
            
            logger.info(f"🎯 [Workflow] 用户选择辅导方式: {style.value}")
            
            # Update conversation state to TUTORING
            await self.session_manager.set_conversation_state(
                session_id, ConversationState.TUTORING
            )
            
            # Return empty - actual response will come from streaming
            response = ""
            
        elif state == ConversationState.TUTORING:
            # Phase 2 tutoring - handled by streaming endpoint
            response = ""
            
        else:
            # General tutoring conversation
            response = "好的，让我来帮你解答这个问题呢~"
        
        # Add AI response to history
        await self.session_manager.add_message(session_id, "assistant", response)
        
        return response

    def _generate_thinking_prompt(self) -> str:
        """Generate prompt asking about user's thinking."""
        return (
            "能告诉姐姐你是怎么思考这道题的吗？"
            "有什么地方让你感到困惑呢？\n\n"
            "不用担心说错哦，把你的想法告诉我就好~ 💕"
        )

    def _generate_style_prompt(self) -> str:
        """Generate prompt asking about tutoring style."""
        return (
            "嗯嗯，我明白你的想法了呢~ 👍\n\n"
            "那姐姐想问一下，你希望我怎么帮你呢？\n\n"
            "1️⃣ **引导式辅导** - 我会一步一步引导你思考，帮你自己找到答案\n"
            "2️⃣ **直接解答** - 我会直接给你详细的解答过程\n\n"
            "回复 1 或 2 告诉我吧~"
        )

    def _parse_tutoring_style(self, message: str) -> TutoringStyle:
        """Parse tutoring style from user message."""
        message = message.strip().lower()
        if "1" in message or "引导" in message:
            return TutoringStyle.GUIDED
        return TutoringStyle.DIRECT

    def _generate_tutoring_start_message(self, style: TutoringStyle) -> str:
        """Generate message when tutoring starts."""
        if style == TutoringStyle.GUIDED:
            return (
                "好的，那我们一起来探索这道题吧~ 🌟\n\n"
                "我会一步一步引导你，不用着急，慢慢来哦~\n"
                "如果后台分析完成了，我会马上告诉你的！"
            )
        else:
            return (
                "好的，那我来给你详细讲解这道题~ 📝\n\n"
                "请稍等一下，我正在整理解答过程...\n"
                "分析完成后会立刻告诉你哦！"
            )

    async def process_phase2_message_stream(
        self,
        session_id: str,
        message: str
    ):
        """Process Phase 2 message with streaming output.
        
        This method handles tutoring after user selects their preferred style.
        It delegates to Phase2Workflow for actual processing.
        
        Args:
            session_id: Session identifier
            message: User message
            
        Yields:
            Response text chunks
        """
        session = await self.session_manager.get_session(session_id)
        if not session:
            yield "抱歉，找不到你的会话呢，请重新开始吧~"
            return
        
        # Get model for this session
        deep_model = self._get_model_for_session(session, 'deep')
        
        # Create Phase2Workflow with session-specific model
        phase2 = Phase2Workflow(
            self.settings,
            self.session_manager,
            self.sse,
            deep_model
        )
        
        # Delegate to Phase2Workflow
        async for chunk in phase2.process_message_stream(session_id, message):
            yield chunk
