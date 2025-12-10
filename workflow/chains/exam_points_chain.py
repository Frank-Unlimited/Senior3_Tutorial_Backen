"""Exam points chain for quick summary of test focus areas.

This chain uses a quick model to summarize what knowledge points
the question is testing, without providing reasoning or solutions.
"""
from typing import Any, Dict, List
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
import json
import re


EXAM_POINTS_PROMPT = """你是一位经验丰富的高三生物老师，熟悉全国甲卷的考点分布。

请分析以下生物题目，快速总结出这道题考察的知识点。

## 题目
{question}

要求：
1. 只列出考察的知识点，不要解答题目
2. 不要进行推理分析
3. 按照重要程度排序
4. 每个知识点用一句话概括
5. 关联到高考考纲中的具体章节

请用以下JSON格式输出：
```json
{{
    "exam_points": [
        "知识点1：具体描述",
        "知识点2：具体描述"
    ],
    "chapter": "所属章节",
    "difficulty": "简单/中等/困难"
}}
```"""


def create_exam_points_chain(quick_model: BaseChatModel) -> RunnableLambda:
    """Create an exam points chain for quick knowledge point summary.
    
    Args:
        quick_model: Quick response LangChain model
        
    Returns:
        Runnable chain that takes question and returns exam points
    """
    
    async def summarize_exam_points(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize exam points from question.
        
        Args:
            inputs: Dict with 'question' text
            
        Returns:
            Dict with 'exam_points', 'chapter', 'difficulty'
        """
        question = inputs.get("question", "")
        
        # Create prompt
        prompt = ChatPromptTemplate.from_template(EXAM_POINTS_PROMPT)
        
        # Create chain
        chain = prompt | quick_model | StrOutputParser()
        
        # Generate summary
        result = await chain.ainvoke({"question": question})
        
        # Parse JSON from result
        try:
            # Extract JSON from markdown code block if present
            json_match = re.search(r'```json\s*(.*?)\s*```', result, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = result
            
            parsed = json.loads(json_str)
            return {
                "exam_points": parsed.get("exam_points", []),
                "chapter": parsed.get("chapter", "未知章节"),
                "difficulty": parsed.get("difficulty", "中等")
            }
        except json.JSONDecodeError:
            # Fallback: extract points from plain text
            lines = result.strip().split("\n")
            points = [line.strip("- •").strip() for line in lines if line.strip()]
            return {
                "exam_points": points[:5],  # Limit to 5 points
                "chapter": "未知章节",
                "difficulty": "中等"
            }
    
    return RunnableLambda(summarize_exam_points)


def validate_exam_points_content(result: Dict[str, Any]) -> bool:
    """Validate that exam points don't contain solutions.
    
    Args:
        result: Exam points result dict
        
    Returns:
        True if content is valid (no solutions)
    """
    # Solution indicators that shouldn't appear
    solution_indicators = [
        "答案", "选", "因为", "所以", "由此可知",
        "计算得", "代入", "解得", "故"
    ]
    
    exam_points = result.get("exam_points", [])
    for point in exam_points:
        point_lower = point.lower()
        for indicator in solution_indicators:
            if indicator in point_lower:
                return False
    
    return True
