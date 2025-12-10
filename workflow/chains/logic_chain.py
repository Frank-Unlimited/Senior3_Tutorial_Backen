"""Logic chain for extracting solving logic and reasoning steps.

This chain analyzes the solution to extract a clear, step-by-step
logic chain for solving the problem.
"""
from typing import Any, Dict, List
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
import json
import re


LOGIC_CHAIN_PROMPT = """你是一位经验丰富的高三生物老师，擅长梳理解题逻辑。

请根据以下题目和解答，梳理出清晰的解题步骤列表。这个步骤列表将用于引导式辅导，帮助学生一步一步理解解题过程。

## 题目
{question}

## 解答
{solution}

请用以下JSON格式输出：
```json
{{
    "steps": [
        "第一步：阅读题目，识别关键信息（如：细胞类型、溶液浓度等）",
        "第二步：回忆相关知识点（如：渗透作用原理）",
        "第三步：分析题目条件与知识点的关联",
        "第四步：推导结论",
        "第五步：验证答案的合理性"
    ],
    "thinking_pattern": "这类题目的思维模式"
}}
```

要求：
1. 步骤要清晰、具体、可操作
2. 每个步骤都是一个完整的句子，描述学生应该做什么
3. 步骤数量控制在3-7步
4. 步骤顺序要符合解题的自然思维流程
5. 这些步骤将用于引导式辅导，所以要适合逐步引导学生思考"""


def create_logic_chain(quick_model: BaseChatModel) -> RunnableLambda:
    """Create a logic chain for extracting solving logic.
    
    Args:
        quick_model: Quick response LangChain model
        
    Returns:
        Runnable chain that takes question and solution, returns logic chain
    """
    
    async def extract_logic(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract solving logic chain.
        
        Args:
            inputs: Dict with 'question' and 'solution'
            
        Returns:
            Dict with 'logic_chain', 'thinking_pattern', 'summary'
        """
        question = inputs.get("question", "")
        solution = inputs.get("solution", "")
        
        # Create prompt
        prompt = ChatPromptTemplate.from_template(LOGIC_CHAIN_PROMPT)
        
        # Create chain
        chain = prompt | quick_model | StrOutputParser()
        
        # Generate logic chain
        result = await chain.ainvoke({
            "question": question,
            "solution": solution
        })
        
        # Parse JSON from result
        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', result, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = result
            
            parsed = json.loads(json_str)
            return {
                "steps": parsed.get("steps", []),
                "thinking_pattern": parsed.get("thinking_pattern", "")
            }
        except json.JSONDecodeError:
            # Fallback: try to extract steps from raw text
            lines = [line.strip() for line in result.split('\n') if line.strip()]
            steps = [line for line in lines if line.startswith('第') or line.startswith('步骤')]
            return {
                "steps": steps if steps else [result],
                "thinking_pattern": ""
            }
    
    return RunnableLambda(extract_logic)


def format_logic_chain_display(logic_result: Dict[str, Any]) -> str:
    """Format logic chain for display.
    
    Args:
        logic_result: Logic chain extraction result
        
    Returns:
        Formatted string for display
    """
    lines = ["## 解题步骤\n"]
    
    steps = logic_result.get("steps", [])
    for i, step in enumerate(steps, 1):
        lines.append(f"{i}. {step}")
    
    if logic_result.get("thinking_pattern"):
        lines.append(f"\n### 思维模式\n{logic_result['thinking_pattern']}")
    
    return "\n".join(lines)
