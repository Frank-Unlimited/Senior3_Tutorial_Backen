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


LOGIC_CHAIN_PROMPT = """你是一位经验丰富的高三生物老师，擅长将解题过程分解成详细的步骤。

你的任务是：将下面的完整解答分解成3-7个详细步骤，每个步骤都必须包含完整的信息。

## 题目
{question}

## 完整解答
{solution}

请将解答分解成详细步骤，每个步骤必须包含：
- 具体的生物学术语、名称、概念
- 具体的数值、数据、条件
- 具体的计算过程、判断依据
- 具体的结论、答案

请用以下JSON格式输出：
```json
{{
    "steps": [
        "步骤1的完整描述，包含所有具体信息",
        "步骤2的完整描述，包含所有具体信息",
        "步骤3的完整描述，包含所有具体信息"
    ],
    "thinking_pattern": "解题思路总结"
}}
```

## 示例1：能量流动计算题

**题目**：某生态系统中，草固定的太阳能为1000kJ，能量传递效率为10%-20%，求狐最多能获得多少能量？食物链为：草→兔→狐。

**正确的步骤分解**：
```json
{{
    "steps": [
        "识别食物链结构：草（生产者，第一营养级）→兔（初级消费者，第二营养级）→狐（次级消费者，第三营养级）",
        "提取已知条件：草固定太阳能1000kJ，相邻营养级间能量传递效率为10%-20%",
        "计算兔（第二营养级）最多获得的能量：1000kJ × 20% = 200kJ",
        "计算狐（第三营养级）最多获得的能量：200kJ × 20% = 40kJ",
        "得出最终答案：狐最多能获得40kJ能量"
    ],
    "thinking_pattern": "能量流动计算：识别营养级→提取数据→逐级计算能量传递"
}}
```

## 示例2：遗传判断题

**题目**：某遗传病在一个家系中的遗传情况如下：父母正常，生了一个患病女儿。判断该病是显性还是隐性？是常染色体遗传还是伴性遗传？

**正确的步骤分解**：
```json
{{
    "steps": [
        "分析家系特征：父母表型正常，女儿患病，说明父母都携带致病基因",
        "判断显隐性：正常的父母生出患病的孩子，说明该病为隐性遗传病",
        "判断遗传方式：女儿患病说明从父母双方各获得一个隐性致病基因，如果是伴X隐性遗传，父亲应该患病（XaY），但父亲正常，所以不是伴X隐性遗传",
        "得出结论：该病为常染色体隐性遗传病，父母基因型均为Aa，女儿基因型为aa"
    ],
    "thinking_pattern": "遗传病判断：分析家系→判断显隐性→排除伴性遗传→确定遗传方式"
}}
```

## 示例3：选择题判断

**题目**：关于细胞呼吸的叙述，正确的是（）
A. 有氧呼吸的场所是线粒体
B. 无氧呼吸不产生ATP
C. 有氧呼吸和无氧呼吸都产生CO2
D. 葡萄糖是细胞呼吸的唯一底物

**正确的步骤分解**：
```json
{{
    "steps": [
        "分析选项A：有氧呼吸分为三个阶段，第一阶段在细胞质基质，第二、三阶段在线粒体，所以A选项'场所是线粒体'不完全正确",
        "分析选项B：无氧呼吸第一阶段（葡萄糖分解为丙酮酸）会产生少量ATP（2个），所以B选项'不产生ATP'是错误的",
        "分析选项C：有氧呼吸产生CO2（在第二阶段），但无氧呼吸分为两种类型：产生乳酸的无氧呼吸不产生CO2，产生酒精的无氧呼吸才产生CO2，所以C选项'都产生CO2'是错误的",
        "分析选项D：细胞呼吸的底物除了葡萄糖，还可以是脂肪酸、氨基酸等有机物，所以D选项'唯一底物'是错误的",
        "得出答案：四个选项都不完全正确，如果必须选择，需要看题目具体要求"
    ],
    "thinking_pattern": "选择题判断：逐个分析选项→结合知识点判断正误→得出答案"
}}
```

## 核心要求

**✅ 必须做到**：
1. 每个步骤都包含完整的具体信息（名称、数值、条件、结论）
2. 每个步骤都是独立完整的陈述，不依赖其他步骤才能理解
3. 步骤数量：3-7步
4. 步骤按照解题的逻辑顺序排列

**❌ 严格禁止**：
1. 禁止出现"判断选项A"这样信息不完整的步骤，必须写成"分析选项A：有氧呼吸分为三个阶段..."
2. 禁止出现"提取题目信息"这样的空泛步骤，必须写成"提取已知条件：草固定太阳能1000kJ..."
3. 禁止出现"运用知识点"这样的抽象步骤，必须写成"运用能量传递效率知识点：相邻营养级间能量传递效率为10%-20%"
4. 禁止使用"第一步""第二步"作为步骤开头
5. 禁止出现"阅读题目""回忆知识点"这样的元认知步骤

现在请分解上面的题目和解答："""


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
