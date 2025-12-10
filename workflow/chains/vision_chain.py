"""Vision chain for extracting question text from images.

This chain uses a vision model to extract the question stem from
an uploaded image without solving the problem.
"""
import base64
import logging
from typing import Any, Dict
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)

VISION_EXTRACTION_PROMPT = """你是一个专业的题目提取助手。请仔细观察这张图片，提取其中的生物题目内容。

要求：
1. 只提取题干和选项（如果有），不要解答
2. 使用纯文本格式输出，保持原题的结构
3. 如果有图表，用文字描述图表内容
4. 如果有多道题，全部提取
5. 保持题目的完整性，不要遗漏任何信息

请直接输出提取的题目内容，不要添加任何解释或评论。"""


def create_vision_chain(vision_model: BaseChatModel) -> RunnableLambda:
    """Create a vision chain for question extraction.
    
    Args:
        vision_model: Vision-capable LangChain model
        
    Returns:
        Runnable chain that takes image_data and returns extracted text
    """
    
    async def extract_question(inputs: Dict[str, Any]) -> str:
        """Extract question text from image.
        
        Args:
            inputs: Dict with 'image_data' (bytes) or 'image_base64' (str)
            
        Returns:
            Extracted question text
        """
        logger.info("📷 [VisionChain] 开始处理图片...")
        
        # Get image data
        if "image_base64" in inputs:
            image_base64 = inputs["image_base64"]
            logger.info("📷 [VisionChain] 使用 base64 输入")
        elif "image_data" in inputs:
            image_data = inputs["image_data"]
            image_base64 = base64.b64encode(image_data).decode("utf-8")
            logger.info(f"📷 [VisionChain] 转换 bytes 到 base64, 长度: {len(image_base64)}")
        else:
            raise ValueError("Either 'image_data' or 'image_base64' must be provided")
        
        # Get mime type (default to jpeg)
        mime_type = inputs.get("mime_type", "image/jpeg")
        logger.info(f"📷 [VisionChain] MIME 类型: {mime_type}")
        
        # Create multimodal message
        message = HumanMessage(
            content=[
                {"type": "text", "text": VISION_EXTRACTION_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{image_base64}"
                    }
                }
            ]
        )
        
        logger.info("📷 [VisionChain] 调用视觉模型 API...")
        logger.info(f"📷 [VisionChain] 模型信息: {vision_model}")
        
        try:
            # Invoke vision model
            response = await vision_model.ainvoke([message])
            logger.info(f"✅ [VisionChain] API 调用成功!")
            logger.info(f"✅ [VisionChain] 响应类型: {type(response)}")
            logger.info(f"✅ [VisionChain] 响应内容长度: {len(response.content) if response.content else 0}")
            return response.content
        except Exception as e:
            logger.error(f"❌ [VisionChain] API 调用失败: {type(e).__name__}: {str(e)}")
            raise
    
    return RunnableLambda(extract_question)


def validate_extraction_result(text: str) -> bool:
    """Validate that extraction result doesn't contain solutions.
    
    Args:
        text: Extracted text to validate
        
    Returns:
        True if text appears to be just the question (no solution)
    """
    # Solution indicators that shouldn't appear in extraction
    solution_indicators = [
        "答案是", "答案为", "正确答案",
        "解析：", "解答：", "分析：",
        "所以选", "因此选", "故选",
        "综上所述", "由此可知"
    ]
    
    text_lower = text.lower()
    for indicator in solution_indicators:
        if indicator in text_lower:
            return False
    
    return True
