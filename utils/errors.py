"""Error handling module for Biology Tutorial Workflow.

This module defines custom exceptions, error codes, and error response models.
"""
from typing import Optional, Dict, Any
from enum import Enum


class ErrorCode(str, Enum):
    """Application error codes."""
    # Session errors
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    SESSION_EXPIRED = "SESSION_EXPIRED"
    
    # Image errors
    IMAGE_REQUIRED = "IMAGE_REQUIRED"
    INVALID_IMAGE_FORMAT = "INVALID_IMAGE_FORMAT"
    IMAGE_TOO_LARGE = "IMAGE_TOO_LARGE"
    
    # Model errors
    VISION_FAILED = "VISION_FAILED"
    SOLUTION_FAILED = "SOLUTION_FAILED"
    MODEL_TIMEOUT = "MODEL_TIMEOUT"
    MODEL_RATE_LIMITED = "MODEL_RATE_LIMITED"
    MODEL_AUTH_FAILED = "MODEL_AUTH_FAILED"
    
    # Configuration errors
    INVALID_CONFIG = "INVALID_CONFIG"
    MISSING_API_KEY = "MISSING_API_KEY"
    INVALID_API_KEY = "INVALID_API_KEY"
    
    # General errors
    INTERNAL_ERROR = "INTERNAL_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"


# Error messages in Chinese for user-friendly display
ERROR_MESSAGES = {
    ErrorCode.SESSION_NOT_FOUND: "找不到你的会话呢，请重新开始吧~",
    ErrorCode.SESSION_EXPIRED: "会话已过期，请重新开始哦~",
    ErrorCode.IMAGE_REQUIRED: "需要上传图片才能开始分析呢~",
    ErrorCode.INVALID_IMAGE_FORMAT: "图片格式不对哦，请上传 JPG 或 PNG 格式的图片~",
    ErrorCode.IMAGE_TOO_LARGE: "图片太大了呢，请上传小于 10MB 的图片~",
    ErrorCode.VISION_FAILED: "图片分析出了点问题，请重新上传试试~",
    ErrorCode.SOLUTION_FAILED: "生成解答时遇到了问题，请稍后重试~",
    ErrorCode.MODEL_TIMEOUT: "AI 思考太久了，请稍后重试~",
    ErrorCode.MODEL_RATE_LIMITED: "请求太频繁了，请稍等一下再试~",
    ErrorCode.MODEL_AUTH_FAILED: "API 鉴权失败，请检查 API Key 是否正确~",
    ErrorCode.INVALID_CONFIG: "系统配置有问题，请联系管理员~",
    ErrorCode.MISSING_API_KEY: "系统配置不完整，请联系管理员~",
    ErrorCode.INVALID_API_KEY: "API Key 无效或已过期，请在设置中更新~",
    ErrorCode.INTERNAL_ERROR: "服务器出了点小问题，请稍后重试~",
    ErrorCode.VALIDATION_ERROR: "输入的内容有问题，请检查一下~",
}


# HTTP status codes for each error
ERROR_STATUS_CODES = {
    ErrorCode.SESSION_NOT_FOUND: 404,
    ErrorCode.SESSION_EXPIRED: 410,
    ErrorCode.IMAGE_REQUIRED: 400,
    ErrorCode.INVALID_IMAGE_FORMAT: 400,
    ErrorCode.IMAGE_TOO_LARGE: 413,
    ErrorCode.VISION_FAILED: 500,
    ErrorCode.SOLUTION_FAILED: 500,
    ErrorCode.MODEL_TIMEOUT: 504,
    ErrorCode.MODEL_RATE_LIMITED: 429,
    ErrorCode.MODEL_AUTH_FAILED: 401,
    ErrorCode.INVALID_CONFIG: 500,
    ErrorCode.MISSING_API_KEY: 500,
    ErrorCode.INVALID_API_KEY: 401,
    ErrorCode.INTERNAL_ERROR: 500,
    ErrorCode.VALIDATION_ERROR: 400,
}


class BiologyTutorError(Exception):
    """Custom exception for Biology Tutorial Workflow."""
    
    def __init__(
        self,
        error_code: ErrorCode,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize error.
        
        Args:
            error_code: Error code enum value
            message: Optional custom message (uses default if not provided)
            details: Optional additional error details
        """
        self.error_code = error_code.value
        self.message = message or ERROR_MESSAGES.get(error_code, "未知错误")
        self.details = details
        self.status_code = ERROR_STATUS_CODES.get(error_code, 500)
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON response."""
        result = {
            "error_code": self.error_code,
            "message": self.message
        }
        if self.details:
            result["details"] = self.details
        return result


class ErrorResponse:
    """Error response builder."""
    
    @staticmethod
    def session_not_found(session_id: str) -> BiologyTutorError:
        """Create session not found error."""
        return BiologyTutorError(
            ErrorCode.SESSION_NOT_FOUND,
            details={"session_id": session_id}
        )
    
    @staticmethod
    def image_required() -> BiologyTutorError:
        """Create image required error."""
        return BiologyTutorError(ErrorCode.IMAGE_REQUIRED)
    
    @staticmethod
    def invalid_image_format(mime_type: str) -> BiologyTutorError:
        """Create invalid image format error."""
        return BiologyTutorError(
            ErrorCode.INVALID_IMAGE_FORMAT,
            details={"received_type": mime_type}
        )
    
    @staticmethod
    def vision_failed(error: str) -> BiologyTutorError:
        """Create vision extraction failed error."""
        return BiologyTutorError(
            ErrorCode.VISION_FAILED,
            details={"error": error}
        )
    
    @staticmethod
    def model_timeout(task: str) -> BiologyTutorError:
        """Create model timeout error."""
        return BiologyTutorError(
            ErrorCode.MODEL_TIMEOUT,
            details={"task": task}
        )
    
    @staticmethod
    def internal_error(error: str) -> BiologyTutorError:
        """Create internal error."""
        return BiologyTutorError(
            ErrorCode.INTERNAL_ERROR,
            details={"error": error}
        )
    
    @staticmethod
    def auth_failed(model_type: str, error: str) -> BiologyTutorError:
        """Create authentication failed error."""
        return BiologyTutorError(
            ErrorCode.MODEL_AUTH_FAILED,
            message=f"API 鉴权失败 ({model_type}): {error}，请检查 API Key 是否正确~",
            details={"model_type": model_type, "error": error}
        )
    
    @staticmethod
    def invalid_api_key(model_type: str) -> BiologyTutorError:
        """Create invalid API key error."""
        return BiologyTutorError(
            ErrorCode.INVALID_API_KEY,
            message=f"{model_type} 的 API Key 无效或已过期，请在设置中更新~",
            details={"model_type": model_type}
        )
