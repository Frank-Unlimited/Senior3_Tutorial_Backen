"""Knowledge chain for extracting knowledge points and common mistakes.

This chain analyzes the solution to extract structured knowledge points
and common mistakes that students often make.
"""
from typing import Any, Dict, List
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
import json
import re


KNOWLEDGE_POINTS_PROMPT = """你是一位经验丰富的高三生物老师，擅长总结知识点和易错点。

请根据以下题目和解答，梳理出相关的知识点和易错点。

## 题目
{question}

## 解答
{solution}

请用以下JSON格式输出：
```json
{{
    "knowledge_points": [
        {{
            "name": "知识点名称",
            "description": "详细描述",
            "importance": "高/中/低"
        }}
    ],
    "common_mistakes": [
        {{
            "mistake": "常见错误描述",
            "reason": "错误原因",
            "correction": "正确理解"
        }}
    ],
    "related_topics": ["相关知识点1", "相关知识点2"]
}}
```

要求：
1. 知识点要具体、可操作
2. 易错点要结合学生常见的思维误区
3. 关联到高考考纲"""


def create_knowledge_chain(quick_model: BaseChatModel) -> RunnableLambda:
    """Create a knowledge chain for extracting knowledge points.
    
    Args:
        quick_model: Quick response LangChain model
        
    Returns:
        Runnable chain that takes question and solution, returns knowledge points
    """
    
    async def extract_knowledge(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract knowledge points and common mistakes.
        
        Args:
            inputs: Dict with 'question' and 'solution'
            
        Returns:
            Dict with 'knowledge_points', 'common_mistakes', 'related_topics'
        """
        question = inputs.get("question", "")
        solution = inputs.get("solution", "")
        
        # Create prompt
        prompt = ChatPromptTemplate.from_template(KNOWLEDGE_POINTS_PROMPT)
        
        # Create chain
        chain = prompt | quick_model | StrOutputParser()
        
        # Generate knowledge points
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
                "knowledge_points": parsed.get("knowledge_points", []),
                "common_mistakes": parsed.get("common_mistakes", []),
                "related_topics": parsed.get("related_topics", [])
            }
        except json.JSONDecodeError:
            # Fallback: return raw text as single knowledge point
            return {
                "knowledge_points": [{"name": "知识点", "description": result, "importance": "中"}],
                "common_mistakes": [],
                "related_topics": []
            }
    
    return RunnableLambda(extract_knowledge)
