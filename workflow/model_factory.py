"""Model factory for creating LangChain chat models.

This module provides a factory pattern for creating chat models from
different providers (doubao, openai, zhipu) based on configuration.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from config.settings import ModelConfig, VisionModelConfig


class ModelFactory:
    """Factory for creating LangChain chat models from configuration."""

    @staticmethod
    def create(config: ModelConfig) -> BaseChatModel:
        """Create a chat model from configuration.
        
        Args:
            config: Model configuration with provider, model_name, api_key, etc.
            
        Returns:
            LangChain BaseChatModel instance
            
        Raises:
            ValueError: If provider is not supported
        """
        provider = config.provider.lower()
        
        if provider == "doubao":
            return ModelFactory._create_doubao_model(config)
        elif provider == "openai":
            return ModelFactory._create_openai_model(config)
        elif provider == "zhipu":
            return ModelFactory._create_zhipu_model(config)
        else:
            raise ValueError(f"Unsupported model provider: {provider}")

    @staticmethod
    def _create_doubao_model(config: ModelConfig) -> BaseChatModel:
        """Create a Doubao/Volcengine model.
        
        Uses OpenAI-compatible API for Doubao models.
        """
        from langchain_openai import ChatOpenAI
        import logging
        logger = logging.getLogger(__name__)
        
        api_base = config.api_base or "https://ark.cn-beijing.volces.com/api/v3"
        logger.info(f"🤖 [ModelFactory] 创建豆包模型:")
        logger.info(f"   Model: {config.model_name}")
        logger.info(f"   API Base: {api_base}")
        logger.info(f"   API Key: {config.api_key[:15]}..." if config.api_key else "   API Key: None")
        logger.info(f"   Temperature: {config.temperature}")
        logger.info(f"   Max Tokens: {config.max_tokens}")
        
        return ChatOpenAI(
            model=config.model_name,
            openai_api_key=config.api_key,
            openai_api_base=api_base,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    @staticmethod
    def _create_openai_model(config: ModelConfig) -> BaseChatModel:
        """Create an OpenAI model."""
        from langchain_openai import ChatOpenAI
        
        return ChatOpenAI(
            model=config.model_name,
            openai_api_key=config.api_key,
            openai_api_base=config.api_base,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    @staticmethod
    def _create_zhipu_model(config: ModelConfig) -> BaseChatModel:
        """Create a Zhipu AI model.
        
        Uses OpenAI-compatible API for Zhipu models.
        """
        from langchain_openai import ChatOpenAI
        
        return ChatOpenAI(
            model=config.model_name,
            openai_api_key=config.api_key,
            openai_api_base=config.api_base or "https://open.bigmodel.cn/api/paas/v4",
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    @staticmethod
    def create_vision_model(config: VisionModelConfig) -> BaseChatModel:
        """Create a vision-capable model.
        
        Vision models support multimodal input (text + images).
        
        Args:
            config: Vision model configuration
            
        Returns:
            LangChain BaseChatModel with vision capabilities
        """
        # Most vision models use the same API as chat models
        # The difference is in how we format the input (with images)
        return ModelFactory.create(config)
