"""Solution chain for generating detailed answers.

This chain uses a deep thinking model to generate comprehensive
solutions with support for guided and direct tutoring styles.
"""
from typing import Any, Dict
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel


SOLUTION_PROMPT = """你是一位温柔的大姐姐，擅长辅导高三学生的生物学习。现在需要给出这道题的详细解答。

{persona_prompt}

## 题目
{question}

请给出详细的解答：
1. 首先分析题目考察的知识点
2. 给出完整的解题过程
3. 解释每一步的原因
4. 总结解题方法和技巧
5. 指出常见的易错点

注意：
- 解答要清晰、完整、准确
- 可以用生动的比喻帮助理解
- 语气要温柔有耐心"""


def create_solution_chain(
    deep_model: BaseChatModel,
    persona_prompt: str = ""
) -> RunnableLambda:
    """Create a solution chain for generating detailed answers.
    
    This chain generates a complete solution based solely on the question text.
    User preferences (thinking process, tutoring style) are used in Phase 2
    for personalized tutoring delivery.
    
    Args:
        deep_model: Deep thinking LangChain model
        persona_prompt: AI persona prompt for style
        
    Returns:
        Runnable chain that takes question info and returns solution
    """
    
    async def generate_solution(inputs: Dict[str, Any]) -> str:
        """Generate detailed solution based on question text.
        
        Args:
            inputs: Dict with 'question' (required)
            
        Returns:
            Detailed solution text
        """
        question = inputs.get("question", "")
        
        # Create prompt
        prompt = ChatPromptTemplate.from_template(SOLUTION_PROMPT)
        
        # Create chain
        chain = prompt | deep_model | StrOutputParser()
        
        # Generate solution
        result = await chain.ainvoke({
            "persona_prompt": persona_prompt,
            "question": question
        })
        
        return result
    
    return RunnableLambda(generate_solution)


def format_solution_for_style(solution: str, style: str) -> str:
    """Format solution based on tutoring style.
    
    Args:
        solution: Raw solution text
        style: 'guided' or 'direct'
        
    Returns:
        Formatted solution
    """
    if style == "guided":
        # Add interactive markers for guided style
        lines = solution.split("\n")
        formatted_lines = []
        for line in lines:
            if line.strip().endswith("?") or line.strip().endswith("？"):
                # Questions should prompt for user response
                formatted_lines.append(line)
                formatted_lines.append("\n💭 *请思考一下这个问题...*\n")
            else:
                formatted_lines.append(line)
        return "\n".join(formatted_lines)
    else:
        return solution
