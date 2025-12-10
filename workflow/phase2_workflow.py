"""Phase 2 Workflow for guided tutoring.

This module implements the Phase 2 tutoring workflow, handling both
direct answer mode and guided step-by-step tutoring mode.
"""
import logging
from typing import AsyncGenerator, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import Settings
from session.manager import SessionManager
from session.models import Session, Phase2State, TutoringStyle, GuidedStep
from sse.publisher import SSEPublisher
from workflow.step_generator import StepGenerator
from workflow.step_guider import StepGuider

logger = logging.getLogger(__name__)


class Phase2Workflow:
    """Phase 2 tutoring workflow handler."""
    
    def __init__(
        self,
        settings: Settings,
        session_manager: SessionManager,
        sse_publisher: SSEPublisher,
        model
    ):
        """Initialize Phase 2 workflow.
        
        Args:
            settings: Application settings
            session_manager: Session manager instance
            sse_publisher: SSE publisher instance
            model: LangChain model for generation
        """
        self.settings = settings
        self.session_manager = session_manager
        self.sse = sse_publisher
        self.model = model
        
        self.step_generator = StepGenerator(model)
        self.step_guider = StepGuider(model, settings.persona_prompt)
    
    async def process_message_stream(
        self,
        session_id: str,
        message: str
    ) -> AsyncGenerator[str, None]:
        """Process Phase 2 message with streaming output.
        
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
        
        # Check for escape phrase first (only in guided mode)
        if (session.phase2_state == Phase2State.GUIDING_STEP and 
            self.step_guider.check_escape(message)):
            async for chunk in self._handle_escape(session_id, session):
                yield chunk
            return
        
        # Route based on phase2 state
        if session.phase2_state == Phase2State.AWAITING_MODE:
            async for chunk in self._handle_mode_selection(session_id, session, message):
                yield chunk
                
        elif session.phase2_state == Phase2State.GUIDING_STEP:
            async for chunk in self._handle_guided_response(session_id, session, message):
                yield chunk
                
        elif session.phase2_state == Phase2State.COMPLETED:
            async for chunk in self._handle_followup(session_id, session, message):
                yield chunk
        else:
            yield "好的，让我来帮你解答这个问题呢~"

    
    async def _handle_mode_selection(
        self,
        session_id: str,
        session: Session,
        message: str
    ) -> AsyncGenerator[str, None]:
        """Handle tutoring mode selection.
        
        Args:
            session_id: Session identifier
            session: Current session
            message: User message (should be "1" or "2" or "开始辅导")
            
        Yields:
            Response text chunks
        """
        # Check if user already selected a style (from Phase 1)
        if session.tutoring_style:
            style = session.tutoring_style
            logger.info(f"🎯 [Phase2] 使用已选择的辅导方式: {style.value}")
        else:
            # Parse style from message
            style = self._parse_style(message)
            # Update session with tutoring style
            await self.session_manager.update_session(
                session_id,
                tutoring_style=style
            )
            logger.info(f"🎯 [Phase2] 用户选择辅导方式: {style.value}")
        
        if style == TutoringStyle.DIRECT:
            # Direct answer mode
            async for chunk in self._output_direct_solution(session_id, session):
                yield chunk
        else:
            # Guided mode
            async for chunk in self._init_guided_mode(session_id, session):
                yield chunk
    
    def _parse_style(self, message: str) -> TutoringStyle:
        """Parse tutoring style from user message.
        
        Args:
            message: User message
            
        Returns:
            TutoringStyle enum value
        """
        message = message.strip().lower()
        if "1" in message or "引导" in message:
            return TutoringStyle.GUIDED
        return TutoringStyle.DIRECT
    
    async def _output_direct_solution(
        self,
        session_id: str,
        session: Session
    ) -> AsyncGenerator[str, None]:
        """Output complete solution directly.
        
        Args:
            session_id: Session identifier
            session: Current session
            
        Yields:
            Response text chunks
        """
        logger.info("📝 [Phase2] 输出直接解答...")
        
        yield "好的，那姐姐来给你详细讲解这道题~ 📝\n\n"
        yield "---\n\n"
        
        # Output the solution
        if session.solution:
            yield session.solution
        else:
            yield "抱歉呢，解答还在生成中，请稍等一下哦~"
        
        yield "\n\n---\n\n"
        
        # Output knowledge points if available
        if session.knowledge_points:
            yield "**📚 涉及的知识点：**\n"
            for kp in session.knowledge_points:
                yield f"- {kp}\n"
            yield "\n"
        
        # Output common mistakes if available
        if session.common_mistakes:
            yield "**⚠️ 常见易错点：**\n"
            for cm in session.common_mistakes:
                yield f"- {cm}\n"
            yield "\n"
        
        yield "---\n\n"
        yield "还有什么不明白的地方吗？可以继续问我哦~ 😊"
        
        # Update state
        await self.session_manager.update_session(
            session_id,
            phase2_state=Phase2State.COMPLETED
        )
    
    async def _init_guided_mode(
        self,
        session_id: str,
        session: Session
    ) -> AsyncGenerator[str, None]:
        """Initialize guided tutoring mode.
        
        Args:
            session_id: Session identifier
            session: Current session
            
        Yields:
            Response text chunks
        """
        logger.info("🎓 [Phase2] 初始化引导式辅导...")
        
        yield "好的，那我们一起来探索这道题吧~ 🌟\n\n"
        
        # Generate steps
        steps = await self.step_generator.generate_steps(
            session.question_text or "",
            session.solution or "",
            session.logic_chain_steps
        )
        
        logger.info(f"📋 [Phase2] 生成了 {len(steps)} 个引导步骤")
        
        # Output TODO list
        yield "**📋 解题步骤：**\n"
        for step in steps:
            yield f"{step.to_checkbox_str()}\n"
        yield "\n---\n\n"
        
        # Start first step - 输出开场白并立即生成第一个引导问题
        first_step = steps[0]
        opening = f"让我们从 **步骤1: {first_step.title}** 开始~\n\n"
        yield opening
        
        # 立即生成第一个引导问题（模拟用户说"开始"）
        first_question = ""
        async for chunk in self.step_guider.guide_step(
            first_step,
            "开始",  # 模拟用户触发
            [],  # 空历史
            question_text=session.question_text or "",
            solution=session.solution or "",
            knowledge_points=session.knowledge_points or [],
            all_steps=steps,
            skip_summary=True  # 初始化时跳过总结讲解
        ):
            first_question += chunk
            yield chunk
        
        # Initialize conversation history with opening and first question
        initial_history = [
            {"role": "assistant", "content": opening + first_question}
        ]
        
        # Store steps in session with initial history
        await self.session_manager.update_session(
            session_id,
            guided_steps=steps,
            current_step_index=0,
            phase2_state=Phase2State.GUIDING_STEP,
            step_conversation_history=initial_history
        )
        
        logger.info(f"✅ [Phase2] 初始化完成，已生成第一个引导问题")

    
    async def _handle_guided_response(
        self,
        session_id: str,
        session: Session,
        message: str
    ) -> AsyncGenerator[str, None]:
        """Handle user response during guided tutoring.
        
        Args:
            session_id: Session identifier
            session: Current session
            message: User message
            
        Yields:
            Response text chunks
        """
        # Get current step
        current_step = session.get_current_step()
        if not current_step:
            # All steps done, output summary
            async for chunk in self._output_summary(session_id, session):
                yield chunk
            return
        
        logger.info(f"📝 [Phase2] 当前步骤: {current_step.index + 1}, 历史记录数: {len(session.step_conversation_history)}")
        logger.info(f"📝 [Phase2] 用户消息: {message[:50]}...")
        
        # Add user message to step conversation history first
        session.step_conversation_history.append({
            "role": "user",
            "content": message
        })
        
        # Save user message to history immediately
        await self.session_manager.update_session(
            session_id,
            step_conversation_history=session.step_conversation_history
        )
        
        logger.info(f"📝 [Phase2] 保存后历史记录数: {len(session.step_conversation_history)}")
        
        # Evaluate if step is complete
        is_complete = await self.step_guider.evaluate_completion(
            current_step,
            message,
            session.step_conversation_history
        )
        
        if is_complete:
            # Step completed - but first summarize and explain the final answer
            logger.info(f"✅ [Phase2] 步骤完成，先总结讲解最后一次回答")
            response_text = ""
            
            # Summarize the final answer
            async for chunk in self.step_guider.summarize_and_explain(
                current_step,
                message,
                session.step_conversation_history,
                question_text=session.question_text or "",
                knowledge_points=session.knowledge_points or []
            ):
                response_text += chunk
                yield chunk
            
            # Add assistant response to history
            session.step_conversation_history.append({
                "role": "assistant",
                "content": response_text
            })
            
            # Save updated history
            await self.session_manager.update_session(
                session_id,
                step_conversation_history=session.step_conversation_history
            )
            
            yield "\n\n"
            
            # Then proceed to complete the step
            async for chunk in self._complete_step(session_id, session):
                yield chunk
        else:
            # Continue guiding - pass question, solution, knowledge points and all steps
            logger.info(f"🔄 [Phase2] 继续引导，传入历史记录: {session.step_conversation_history}")
            response_text = ""
            async for chunk in self.step_guider.guide_step(
                current_step,
                "",
                session.step_conversation_history,
                question_text=session.question_text or "",
                solution=session.solution or "",
                knowledge_points=session.knowledge_points or [],
                all_steps=session.guided_steps
            ):
                response_text += chunk
                yield chunk
            
            # Add assistant response to history
            session.step_conversation_history.append({
                "role": "assistant",
                "content": response_text
            })
            
            # Save updated history with assistant response
            await self.session_manager.update_session(
                session_id,
                step_conversation_history=session.step_conversation_history
            )
            logger.info(f"✅ [Phase2] 引导完成，最终历史记录数: {len(session.step_conversation_history)}")
    
    async def _complete_step(
        self,
        session_id: str,
        session: Session
    ) -> AsyncGenerator[str, None]:
        """Handle step completion.
        
        Args:
            session_id: Session identifier
            session: Current session
            
        Yields:
            Response text chunks
        """
        current_idx = session.current_step_index
        
        # Get the completed step before marking it complete
        completed_step = session.guided_steps[current_idx]
        
        # Mark step complete
        all_done = session.mark_current_step_complete()
        
        # Positive feedback
        feedbacks = [
            "太棒了！你理解得很好呢~ ✨",
            "非常好！这一步你掌握得很扎实~ 👍",
            "很棒呀！你的思路完全正确~ 🌟",
            "太厉害了！继续保持这个状态~ 💪",
        ]
        yield f"{feedbacks[current_idx % len(feedbacks)]}\n\n"
        
        # Output the completed step's full description
        yield f"**✅ 步骤{current_idx + 1}完成：{completed_step.title}**\n\n"
        yield f"{completed_step.description}\n\n"
        
        # Output updated TODO list
        yield "**📋 当前进度：**\n"
        for step in session.guided_steps:
            yield f"{step.to_checkbox_str()}\n"
        yield "\n---\n\n"
        
        if all_done:
            # All steps completed
            async for chunk in self._output_summary(session_id, session):
                yield chunk
        else:
            # Move to next step - 输出开场白并立即生成引导问题
            next_step = session.guided_steps[session.current_step_index]
            opening = f"接下来是 **步骤{session.current_step_index + 1}: {next_step.title}**~\n\n"
            yield opening
            
            # 立即生成引导问题（模拟用户说"继续"）
            next_question = ""
            async for chunk in self.step_guider.guide_step(
                next_step,
                "继续",  # 模拟用户触发
                [],  # 新步骤，空历史
                question_text=session.question_text or "",
                solution=session.solution or "",
                knowledge_points=session.knowledge_points or [],
                all_steps=session.guided_steps,
                skip_summary=True  # 新步骤开始时跳过总结讲解
            ):
                next_question += chunk
                yield chunk
            
            # Initialize new step's conversation history with opening and question
            new_history = [
                {"role": "assistant", "content": opening + next_question}
            ]
            
            # Save state with new history
            await self.session_manager.update_session(
                session_id,
                guided_steps=session.guided_steps,
                current_step_index=session.current_step_index,
                step_conversation_history=new_history
            )
            
            logger.info(f"➡️ [Phase2] 进入步骤 {session.current_step_index + 1}，已生成引导问题")
    
    async def _handle_escape(
        self,
        session_id: str,
        session: Session
    ) -> AsyncGenerator[str, None]:
        """Handle escape request - output full solution.
        
        Args:
            session_id: Session identifier
            session: Current session
            
        Yields:
            Response text chunks
        """
        logger.info("🚪 [Phase2] 处理跳出请求...")
        
        yield "没关系呢，有时候直接看答案也是一种学习方式~ 💕\n\n"
        yield "让姐姐来给你详细讲解吧：\n\n"
        
        # Mark all steps complete
        session.mark_all_steps_complete()
        
        # Save state
        await self.session_manager.update_session(
            session_id,
            guided_steps=session.guided_steps,
            current_step_index=session.current_step_index
        )
        
        # Output solution
        async for chunk in self._output_direct_solution(session_id, session):
            yield chunk
    
    async def _output_summary(
        self,
        session_id: str,
        session: Session
    ) -> AsyncGenerator[str, None]:
        """Output completion summary.
        
        Args:
            session_id: Session identifier
            session: Current session
            
        Yields:
            Response text chunks
        """
        logger.info("🎉 [Phase2] 输出完成总结...")
        
        yield "🎉 **太棒了！你完成了所有步骤！**\n\n"
        yield "让我们来回顾一下完整的解题过程：\n\n"
        
        # Output all steps with descriptions
        for step in session.guided_steps:
            yield f"{step.to_checkbox_str()}\n"
            yield f"   {step.description}\n\n"
        
        yield "---\n\n"
        
        # Output knowledge points
        if session.knowledge_points:
            yield "**📚 涉及的知识点：**\n"
            for kp in session.knowledge_points:
                yield f"- {kp}\n"
            yield "\n"
        
        # Output common mistakes
        if session.common_mistakes:
            yield "**⚠️ 常见易错点：**\n"
            for cm in session.common_mistakes:
                yield f"- {cm}\n"
            yield "\n"
        
        yield "---\n\n"
        yield "你做得很好呢！还想练习类似的题目吗？ 😊"
        
        # Update state
        await self.session_manager.update_session(
            session_id,
            phase2_state=Phase2State.COMPLETED
        )
    
    async def _handle_followup(
        self,
        session_id: str,
        session: Session,
        message: str
    ) -> AsyncGenerator[str, None]:
        """Handle follow-up questions after completion.
        
        Args:
            session_id: Session identifier
            session: Current session
            message: User message
            
        Yields:
            Response text chunks
        """
        # Simple follow-up handling - can be expanded
        yield "好的呢，你还有什么问题想问姐姐吗？\n\n"
        yield "如果想练习新的题目，可以上传新的图片哦~ 📷"
