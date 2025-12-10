"""Step Guider for guided tutoring.

This module manages the guiding process for individual steps,
including escape detection, step guidance, and completion evaluation.
"""
import logging
from typing import List, Dict, AsyncGenerator
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from session.models import GuidedStep

logger = logging.getLogger(__name__)


# Escape phrases that trigger direct answer output
ESCAPE_PHRASES = [
    "直接告诉我答案", "直接给我答案", "我不会", "不会做",
    "告诉我完整答案", "直接解答", "跳过引导", "跳过",
    "不想思考了", "直接说答案", "给我答案", "看答案",
    "放弃", "太难了", "想不出来", "不知道怎么做",
    "直接给答案", "完整答案", "全部答案"
]


# 总结讲解提示词：专门负责总结学生回答并完整讲解知识点
SUMMARY_EXPLANATION_PROMPT = """你是一位专业又亲切的生物学科辅导老师，擅长用生动易懂的方式讲解知识点。

{persona_prompt}

### 任务
学生刚刚回答了一个问题，你需要：
1. **简要总结学生的回答**（10-20字）
   - 如果正确或接近正确：给予肯定，如"是的"、"没错"、"对的"
   - 如果错误或不完整：温和指出，如"不太准确呢"、"还需要补充一下"

2. **完整讲解正确答案**（60-100字，这是核心）
   - 不管学生答对答错，都要将本步骤的正确答案/结论完整陈述一遍
   - 必须结合题干中的具体信息（生物名称、数值、实验条件等）
   - 必须结合"涉及知识点"进行详细解释
   - 讲解要详细清晰，让学生彻底理解这个知识点

### 解题上下文
- 原题目：{question_text}
- 涉及知识点：{knowledge_points}

### 当前步骤信息
- 步骤标题：{step_title}
- 步骤内容：{step_description}
- 本步骤核心知识点/正确结论：{expected_understanding}

### 对话记录
{conversation_history}

### 学生最新回答
"{user_message}"

### 输出示例

**示例1：学生回答正确**
学生回答："距离要离得更近"
你的输出：
```
是的，你的理解是对的呢~ ✨

若要观察到细胞较大且数量较少的物像，物镜离装片的距离应当离得更近。这是因为显微镜放大倍数越大，看到的细胞越大、数量越少，物镜离装片越近，光圈越大；放大倍数越小，看到的细胞越小、数量越多，物镜离装片越远，光圈越小。
```

**示例2：学生回答不完整**
学生回答："在叶子里"
你的输出：
```
你说的方向是对的，不过还可以更准确一些哦~

光合作用的场所是叶绿体。叶绿体是植物细胞中的一种细胞器，主要存在于叶片的叶肉细胞中。叶绿体内含有叶绿素，能够吸收光能，将二氧化碳和水转化为有机物，并释放氧气。这就是光合作用的完整过程呢。
```

**示例3：学生回答错误**
学生回答："细胞膜"
你的输出：
```
这个答案不太准确呢，让姐姐来帮你理清楚~

DNA主要存在于细胞核中。在真核细胞中，DNA与蛋白质结合形成染色体，储存在细胞核内。细胞核是遗传信息库，控制着细胞的生命活动。另外，线粒体和叶绿体中也含有少量DNA，但细胞核才是DNA的主要存在场所。
```

### 输出要求
- 只输出总结和讲解部分，不要提出新问题
- 语言亲切活泼，用词生动不生硬
- 控制在80-120字
"""


# 引导问题提示词：专门负责生成下一个引导问题
GUIDING_PROMPT = """你是一位专业又亲切的生物学科辅导老师，擅长用生动易懂的方式，带着学生一步步拆解生物题、吃透核心知识点。

{persona_prompt}

### 任务
学生刚刚完成了一轮学习，你需要根据情况决定下一步：
- 如果{student_reply_count} >= 3：鼓励学生进入下一步，不再提问
- 如果{student_reply_count} < 3：提出一个新的引导性问题，继续深化理解

### 核心规则
1.  **问题要求**（仅在{student_reply_count} < 3时适用）：
    - 每个步骤只提**一个引导性问题**，问题必须明确对应"涉及知识点"中的某一个具体知识点
    - **避免重复知识点**：查看对话记录，不要重复提问相同或相似的知识点
    - **问题设计原则**：
      * ❌ 禁止：直接问"答案是什么""结论是什么""选哪个选项"
      * ✅ 正确：问知识点的概念、原理、定义、公式、适用条件
    - 问题必须包含题目里的具体信息，严禁用"这个""那个""它"等指代词
    - 问题要详尽且生动，必须以？结尾

2.  **语气风格**：亲切活泼，像面对面辅导一样，用词生动不生硬

3.  **长度限制**：控制在30-50字

### 解题上下文
- 原题目：{question_text}
- 涉及知识点：{knowledge_points}

### 所有步骤TODO列表
{todo_list}

### 当前步骤信息
- 步骤序号：{step_index}
- 步骤标题：{step_title}
- 步骤内容：{step_description}
- 本步骤核心知识点/正确结论：{expected_understanding}

### 对话记录
{conversation_history}

### 学生最新回答
"{user_message}"

### 当前轮次
学生已回复{student_reply_count}次

### 输出要求
- 如果{student_reply_count} >= 3：输出鼓励语，如"很好呢！让我们继续下一步吧~ 💪"
- 如果{student_reply_count} < 3：输出一个新的引导性问题
- 只输出问题或鼓励语，不要重复讲解
- 控制在30-50字
"""


# 重写的评估提示词：放宽判断标准，意思重合即可
EVALUATION_PROMPT = """你是一位宽容的辅导老师，需要判断学生是否基本理解当前步骤的核心知识点。

### 当前步骤信息
- 步骤标题：{step_title}
- 步骤内容：{step_description}
- 本步骤核心知识点/正确结论：{expected_understanding}

### 对话历史
{conversation_history}

### 学生最新回复
{user_message}

### 判断标准（宽松）
1. 学生的回答只要与核心知识点/正确结论的**意思有重合**即可，不要求完全准确或表述完整
2. 学生提到了关键概念、关键数值、关键结论中的任何一个，就算理解
3. 学生的思路方向正确，即使细节有误，也算基本掌握
4. 如果对话记录显示老师已经直接告知答案，且学生表示理解或认可，也算完成

### 输出要求
仅回复"完成"或"继续"：
- 学生回答与答案意思有重合/方向正确：回复"完成"
- 学生完全答非所问/方向错误：回复"继续"
"""


class StepGuider:
    """Manages step-by-step guidance in tutoring."""
    
    def __init__(self, model, persona_prompt: str = ""):
        """Initialize with a language model.
        
        Args:
            model: LangChain chat model instance
            persona_prompt: Persona prompt for the tutor
        """
        self.model = model
        self.persona_prompt = persona_prompt
        
        # 总结讲解链
        self.summary_prompt = ChatPromptTemplate.from_template(SUMMARY_EXPLANATION_PROMPT)
        self.summary_chain = self.summary_prompt | model
        
        # 引导问题链
        self.guiding_prompt = ChatPromptTemplate.from_template(GUIDING_PROMPT)
        self.guiding_chain = self.guiding_prompt | model
        
        # 评估链
        self.evaluation_prompt = ChatPromptTemplate.from_template(EVALUATION_PROMPT)
        self.evaluation_chain = self.evaluation_prompt | model | StrOutputParser()
    
    def check_escape(self, message: str) -> bool:
        """Check if message contains escape phrases.
        
        Args:
            message: User message
            
        Returns:
            True if escape phrase detected
        """
        message_lower = message.lower().strip()
        for phrase in ESCAPE_PHRASES:
            if phrase in message_lower:
                logger.info(f"🚪 [StepGuider] 检测到跳出短语: {phrase}")
                return True
        return False
    
    async def summarize_and_explain(
        self,
        step: GuidedStep,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        question_text: str = "",
        knowledge_points: List[str] = None
    ) -> AsyncGenerator[str, None]:
        """Summarize student's answer and provide complete explanation.
        
        Args:
            step: Current guided step
            user_message: User's message
            conversation_history: Previous conversation in this step
            question_text: Original question text for context
            knowledge_points: List of knowledge points for this question
            
        Yields:
            Summary and explanation text chunks
        """
        # Format conversation history
        history_str = self._format_history(conversation_history)
        
        # Format knowledge points
        if knowledge_points:
            kp_str = "、".join(knowledge_points)
        else:
            kp_str = "（知识点信息未提供）"
        
        # Build prompt input
        prompt_input = {
            "persona_prompt": self.persona_prompt,
            "question_text": question_text or "（题目信息未提供）",
            "knowledge_points": kp_str,
            "step_title": step.title,
            "step_description": step.description,
            "expected_understanding": step.expected_understanding,
            "conversation_history": history_str,
            "user_message": user_message
        }
        
        logger.info(f"📝 [StepGuider] 总结讲解步骤 {step.index + 1}: {step.title}")
        
        # Stream summary and explanation
        async for chunk in self.summary_chain.astream(prompt_input):
            if hasattr(chunk, 'content') and chunk.content:
                yield chunk.content
    
    async def generate_next_question(
        self,
        step: GuidedStep,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        question_text: str = "",
        knowledge_points: List[str] = None,
        all_steps: List[GuidedStep] = None,
        student_reply_count: int = 0
    ) -> AsyncGenerator[str, None]:
        """Generate next guiding question or encouragement.
        
        Args:
            step: Current guided step
            user_message: User's message
            conversation_history: Previous conversation in this step
            question_text: Original question text for context
            knowledge_points: List of knowledge points for this question
            all_steps: All guided steps for TODO list display
            student_reply_count: Number of times student has replied in this step
            
        Yields:
            Next question or encouragement text chunks
        """
        # Format conversation history
        history_str = self._format_history(conversation_history)
        
        # Format knowledge points
        if knowledge_points:
            kp_str = "、".join(knowledge_points)
        else:
            kp_str = "（知识点信息未提供）"
        
        # Format TODO list
        if all_steps:
            todo_lines = []
            for s in all_steps:
                checkbox = "☑" if s.completed else "☐"
                todo_lines.append(f"{checkbox} 步骤{s.index + 1}: {s.title}")
            todo_str = "\n".join(todo_lines)
        else:
            todo_str = "（步骤列表未提供）"
        
        # Build prompt input
        prompt_input = {
            "persona_prompt": self.persona_prompt,
            "question_text": question_text or "（题目信息未提供）",
            "knowledge_points": kp_str,
            "todo_list": todo_str,
            "student_reply_count": student_reply_count,
            "step_index": step.index + 1,
            "step_title": step.title,
            "step_description": step.description,
            "expected_understanding": step.expected_understanding,
            "conversation_history": history_str,
            "user_message": user_message
        }
        
        logger.info(f"❓ [StepGuider] 生成引导问题，轮次: {student_reply_count}")
        
        # Stream next question or encouragement
        async for chunk in self.guiding_chain.astream(prompt_input):
            if hasattr(chunk, 'content') and chunk.content:
                yield chunk.content
    
    async def guide_step(
        self,
        step: GuidedStep,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        question_text: str = "",
        solution: str = "",
        knowledge_points: List[str] = None,
        all_steps: List[GuidedStep] = None,
        skip_summary: bool = False
    ) -> AsyncGenerator[str, None]:
        """Guide the current step with streaming output (two-stage approach).
        
        This method now uses a two-stage approach:
        1. First, summarize student's answer and provide complete explanation (optional)
        2. Then, generate next guiding question (if needed)
        
        Args:
            step: Current guided step
            user_message: User's message
            conversation_history: Previous conversation in this step
            question_text: Original question text for context
            solution: Complete solution for reference
            knowledge_points: List of knowledge points for this question
            all_steps: All guided steps for TODO list display
            skip_summary: If True, skip the summary/explanation stage (for initial question)
            
        Yields:
            Response text chunks
        """
        # Calculate student reply count in current step
        student_reply_count = sum(1 for msg in conversation_history if msg.get("role") == "user")
        
        logger.info(f"🎯 [StepGuider] 引导步骤 {step.index + 1}: {step.title}")
        logger.info(f"📝 [StepGuider] 用户消息: {user_message}")
        logger.info(f"🔢 [StepGuider] 当前步骤学生回复轮次: {student_reply_count}")
        logger.info(f"⏭️ [StepGuider] 跳过总结: {skip_summary}")
        
        # Stage 1: Summarize and explain (only if not skipped)
        if not skip_summary:
            async for chunk in self.summarize_and_explain(
                step, user_message, conversation_history, question_text, knowledge_points
            ):
                yield chunk
            
            # Add spacing between summary and next question
            yield "\n\n"
        
        # Stage 2: Generate next question or encouragement
        async for chunk in self.generate_next_question(
            step, user_message, conversation_history, question_text, 
            knowledge_points, all_steps, student_reply_count
        ):
            yield chunk
    
    async def evaluate_completion(
        self,
        step: GuidedStep,
        user_message: str,
        conversation_history: List[Dict[str, str]]
    ) -> bool:
        """Evaluate if user has completed the current step.
        
        Args:
            step: Current guided step
            user_message: User's latest message
            conversation_history: Previous conversation in this step
            
        Returns:
            True if step is completed
        """
        # Format conversation history
        history_str = self._format_history(conversation_history)
        
        # Build prompt input
        prompt_input = {
            "step_title": step.title,
            "step_description": step.description,
            "expected_understanding": step.expected_understanding,
            "conversation_history": history_str,
            "user_message": user_message
        }
        
        try:
            result = await self.evaluation_chain.ainvoke(prompt_input)
            is_complete = "完成" in result
            logger.info(f"📊 [StepGuider] 步骤评估结果: {'完成' if is_complete else '继续'}")
            return is_complete
        except Exception as e:
            logger.error(f"❌ [StepGuider] 评估失败: {e}")
            return False
    
    def _format_history(self, history: List[Dict[str, str]]) -> str:
        """Format conversation history for prompt.
        
        Args:
            history: List of message dicts with 'role' and 'content'
            
        Returns:
            Formatted history string
        """
        if not history:
            return "（这是这一步的第一次对话）"
        
        # 保留全部历史（不再排除最后一条，因为需要完整上下文判断学生意图）
        lines = []
        for msg in history[-6:]:  # 保留最近6条消息，避免上下文过长
            role = "学生" if msg.get("role") == "user" else "老师"
            content = msg.get("content", "")[:300]  # 截断长消息，控制上下文长度
            lines.append(f"{role}: {content}")
        
        return "\n".join(lines) if lines else "（这是这一步的第一次对话）"
