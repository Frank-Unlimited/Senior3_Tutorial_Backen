"""Step Generator for guided tutoring.

This module extracts solving steps from solutions and converts them
into guided steps for step-by-step tutoring.
"""
import logging
import re
from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from session.models import GuidedStep

logger = logging.getLogger(__name__)


STEP_GENERATION_PROMPT = """你是一位经验丰富的生物老师，需要将解题过程分解为清晰的引导步骤。

## 题目
{question}

## 完整解答
{solution}

请将解题过程分解为 3-7 个关键步骤。每个步骤需要包含：
1. 步骤标题（简短，10字以内）
2. 步骤描述（详细说明这一步要做什么，包含具体的知识点或计算过程）
3. 引导问题（必须是有明确答案的具体问题）
4. 标准答案（这个问题的正确答案）

请严格按照以下格式输出，每个步骤用 --- 分隔：

步骤1标题: [标题]
步骤1描述: [描述]
步骤1问题: [引导问题]
步骤1答案: [标准答案]
---
步骤2标题: [标题]
步骤2描述: [描述]
步骤2问题: [引导问题]
步骤2答案: [标准答案]
---
...

【重要】引导问题的要求：
- 必须是有明确答案的具体问题，不能是开放式问题
- 答案应该是具体的知识点、数值、概念名称或判断结论
- 避免使用"你怎么想""有什么想法""如何理解"等模糊问法
- 好的问题示例：
  * "食物链中，草属于哪个营养级？"（答案：第一营养级/生产者）
  * "根据能量传递效率10%-20%，第三营养级最多能获得多少能量？"（答案：具体数值）
  * "光合作用的场所是什么？"（答案：叶绿体）
  * "这个遗传图谱中，患病基因是显性还是隐性？"（答案：隐性）
- 不好的问题示例：
  * "你觉得这一步应该怎么做？"
  * "关于这个知识点，你有什么想法？"
  * "你能说说你的理解吗？"

注意：
- 步骤数量控制在 3-7 个
- 每个步骤要有明确的目标和可验证的答案
- 按照解题的逻辑顺序排列
"""


class StepGenerator:
    """Generates guided steps from solution."""
    
    def __init__(self, model):
        """Initialize with a language model.
        
        Args:
            model: LangChain chat model instance
        """
        self.model = model
        self.prompt = ChatPromptTemplate.from_template(STEP_GENERATION_PROMPT)
        self.chain = self.prompt | model | StrOutputParser()
    
    async def generate_steps(
        self,
        question: str,
        solution: str,
        logic_chain_steps: Optional[List[str]] = None
    ) -> List[GuidedStep]:
        """Generate guided steps from solution.
        
        Prioritizes logic_chain_steps if available and valid (3-7 steps).
        Otherwise extracts steps from solution using LLM.
        
        Args:
            question: The question text
            solution: The complete solution
            logic_chain_steps: Pre-extracted logic chain steps (optional)
            
        Returns:
            List of 3-7 GuidedStep objects
        """
        logger.info("🔧 [StepGenerator] 开始生成引导步骤...")
        
        # Try to use existing logic chain steps first
        if logic_chain_steps and 3 <= len(logic_chain_steps) <= 7:
            logger.info(f"📋 [StepGenerator] 使用现有逻辑链步骤: {len(logic_chain_steps)} 步")
            return self._convert_logic_steps(logic_chain_steps)
        
        # Otherwise extract from solution using LLM
        logger.info("🤖 [StepGenerator] 使用 LLM 从解答中提取步骤...")
        return await self._extract_from_solution(question, solution)
    
    def _convert_logic_steps(self, steps: List[str]) -> List[GuidedStep]:
        """Convert logic chain steps to guided steps.
        
        Args:
            steps: List of step descriptions
            
        Returns:
            List of GuidedStep objects
        """
        guided_steps = []
        for i, step in enumerate(steps):
            guided_steps.append(GuidedStep(
                index=i,
                title=self._extract_title(step),
                description=step,
                guiding_question=self._generate_simple_question(step, i),
                expected_understanding=step
            ))
        return guided_steps
    
    def _extract_title(self, step: str) -> str:
        """Extract a short title from step description.
        
        Args:
            step: Full step description
            
        Returns:
            Short title (max 10 chars)
        """
        # Try to extract first phrase or key concept
        step = step.strip()
        
        # Remove numbering if present
        step = re.sub(r'^[\d\.\)]+\s*', '', step)
        
        # Take first 10 characters or first phrase
        if '：' in step:
            title = step.split('：')[0]
        elif ':' in step:
            title = step.split(':')[0]
        elif '，' in step:
            title = step.split('，')[0]
        else:
            title = step[:15]
        
        return title[:10] + "..." if len(title) > 10 else title
    
    def _generate_simple_question(self, step: str, index: int) -> str:
        """Generate a specific guiding question for a step.
        
        Args:
            step: Step description
            index: Step index
            
        Returns:
            Guiding question string with concrete answer expected
        """
        # 从步骤描述中提取关键信息生成具体问题
        step_lower = step.lower()
        
        # 根据步骤内容生成具体问题
        if "营养级" in step or "食物链" in step:
            return f"在这条食物链中，{self._extract_title(step)}属于第几营养级？"
        elif "能量" in step:
            return f"根据能量传递效率，这一步需要计算的能量值是多少？"
        elif "光合作用" in step:
            return f"光合作用中，{self._extract_title(step)}发生在什么部位？"
        elif "呼吸作用" in step:
            return f"呼吸作用中，{self._extract_title(step)}的产物是什么？"
        elif "遗传" in step or "基因" in step:
            return f"根据遗传规律，{self._extract_title(step)}的基因型是什么？"
        elif "比例" in step or "概率" in step:
            return f"根据分析，这个比例/概率的具体数值是多少？"
        elif "判断" in step or "正确" in step or "错误" in step:
            return f"这个选项的说法是正确还是错误？请说出你的判断。"
        elif "分析" in step:
            return f"分析这一步，关键的结论是什么？"
        else:
            # 默认生成具体问题
            return f"关于{self._extract_title(step)}，正确的答案/结论是什么？"

    
    async def _extract_from_solution(
        self,
        question: str,
        solution: str
    ) -> List[GuidedStep]:
        """Extract steps from solution using LLM.
        
        Args:
            question: The question text
            solution: The complete solution
            
        Returns:
            List of GuidedStep objects
        """
        try:
            result = await self.chain.ainvoke({
                "question": question,
                "solution": solution
            })
            
            steps = self._parse_steps_output(result)
            
            # Ensure we have 3-7 steps
            if len(steps) < 3:
                logger.warning(f"⚠️ [StepGenerator] 步骤太少 ({len(steps)}), 使用默认步骤")
                steps = self._create_default_steps(solution)
            elif len(steps) > 7:
                logger.warning(f"⚠️ [StepGenerator] 步骤太多 ({len(steps)}), 截取前7步")
                steps = steps[:7]
            
            logger.info(f"✅ [StepGenerator] 生成了 {len(steps)} 个引导步骤")
            return steps
            
        except Exception as e:
            logger.error(f"❌ [StepGenerator] 步骤生成失败: {e}")
            return self._create_default_steps(solution)
    
    def _parse_steps_output(self, output: str) -> List[GuidedStep]:
        """Parse LLM output into GuidedStep objects.
        
        Args:
            output: Raw LLM output
            
        Returns:
            List of GuidedStep objects
        """
        steps = []
        
        # Split by separator
        step_blocks = output.split('---')
        
        for i, block in enumerate(step_blocks):
            block = block.strip()
            if not block:
                continue
            
            # Parse each field
            title = self._extract_field(block, r'步骤\d*标题[：:]\s*(.+?)(?:\n|$)')
            description = self._extract_field(block, r'步骤\d*描述[：:]\s*(.+?)(?:\n|$)')
            question = self._extract_field(block, r'步骤\d*问题[：:]\s*(.+?)(?:\n|$)')
            # 支持"答案"和"要点"两种格式
            understanding = self._extract_field(block, r'步骤\d*答案[：:]\s*(.+?)(?:\n|$)')
            if not understanding:
                understanding = self._extract_field(block, r'步骤\d*要点[：:]\s*(.+?)(?:\n|$)')
            
            if title or description:
                steps.append(GuidedStep(
                    index=len(steps),
                    title=title or f"步骤{len(steps)+1}",
                    description=description or block[:100],
                    guiding_question=question or self._generate_simple_question(description or block, len(steps)),
                    expected_understanding=understanding or description or block[:50]
                ))
        
        return steps
    
    def _extract_field(self, text: str, pattern: str) -> str:
        """Extract a field from text using regex.
        
        Args:
            text: Text to search
            pattern: Regex pattern
            
        Returns:
            Extracted value or empty string
        """
        match = re.search(pattern, text, re.MULTILINE)
        return match.group(1).strip() if match else ""
    
    def _create_default_steps(self, solution: str) -> List[GuidedStep]:
        """Create default steps when extraction fails.
        
        Args:
            solution: The solution text
            
        Returns:
            List of 3 default GuidedStep objects
        """
        # Split solution into roughly 3 parts
        lines = solution.split('\n')
        lines = [l for l in lines if l.strip()]
        
        if len(lines) < 3:
            lines = [solution]
        
        chunk_size = max(1, len(lines) // 3)
        
        default_steps = [
            GuidedStep(
                index=0,
                title="分析题目",
                description="首先我们需要仔细阅读题目，找出关键信息和已知条件。",
                guiding_question="题目中提到的核心概念是什么？请说出具体的生物学术语。",
                expected_understanding="识别出题目中的核心概念和关键条件"
            ),
            GuidedStep(
                index=1,
                title="运用知识",
                description="根据题目信息，运用相关的生物学知识进行分析。",
                guiding_question="解决这道题需要用到的关键知识点叫什么名称？",
                expected_understanding="正确说出相关知识点的名称"
            ),
            GuidedStep(
                index=2,
                title="得出结论",
                description="综合分析，得出最终答案。",
                guiding_question="根据分析，最终答案是什么？请给出具体的选项或数值。",
                expected_understanding="正确说出最终答案"
            ),
        ]
        
        return default_steps
