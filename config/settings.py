"""Configuration management module for Biology Tutorial Workflow.

This module provides configuration loading from settings.yaml with support for:
- Environment variable substitution (${VAR_NAME} syntax)
- Default values for optional fields
- Validation using Pydantic
"""
import os
import re
from typing import Optional, Any
from pydantic import BaseModel, Field, field_validator
import yaml


class ModelConfig(BaseModel):
    """Base configuration for AI models."""
    provider: str = Field(..., description="Model provider (doubao, openai, zhipu)")
    model_name: str = Field(..., description="Model name/identifier")
    api_key: str = Field(..., description="API key for the model")
    api_base: Optional[str] = Field(None, description="Custom API base URL")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(4096, gt=0, description="Maximum tokens in response")

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ModelConfig):
            return False
        return (
            self.provider == other.provider
            and self.model_name == other.model_name
            and self.api_key == other.api_key
            and self.api_base == other.api_base
            and abs(self.temperature - other.temperature) < 0.001
            and self.max_tokens == other.max_tokens
        )


class VisionModelConfig(ModelConfig):
    """Configuration for vision/multimodal models."""
    provider: str = Field("doubao", description="Vision model provider")
    model_name: str = Field("doubao-1.6-vision", description="Vision model name")
    temperature: float = Field(0.3, ge=0.0, le=2.0, description="Lower temp for extraction")
    max_tokens: int = Field(2048, gt=0, description="Max tokens for vision output")


class Settings(BaseModel):
    """Global application settings."""
    vision_model: VisionModelConfig = Field(
        default_factory=VisionModelConfig,
        description="Vision model configuration"
    )
    deep_thinking_model: ModelConfig = Field(
        ..., description="Deep thinking model configuration"
    )
    quick_model: ModelConfig = Field(
        ..., description="Quick response model configuration"
    )
    persona_prompt: str = Field(
        default="你是一位温柔的大姐姐，擅长辅导高三学生的生物学习。",
        description="AI persona prompt"
    )
    redis_url: Optional[str] = Field(None, description="Redis URL for session storage")

    @classmethod
    def _substitute_env_vars(cls, value: Any) -> Any:
        """Recursively substitute environment variables in config values.
        
        Supports ${VAR_NAME} syntax for environment variable substitution.
        """
        if isinstance(value, str):
            pattern = r'\$\{([^}]+)\}'
            matches = re.findall(pattern, value)
            for var_name in matches:
                env_value = os.environ.get(var_name, "")
                value = value.replace(f"${{{var_name}}}", env_value)
            return value
        elif isinstance(value, dict):
            return {k: cls._substitute_env_vars(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [cls._substitute_env_vars(item) for item in value]
        return value

    @classmethod
    def from_yaml(cls, path: str = "settings.yaml") -> "Settings":
        """Load settings from a YAML file.
        
        Args:
            path: Path to the settings.yaml file
            
        Returns:
            Settings instance with loaded configuration
            
        Raises:
            FileNotFoundError: If the settings file doesn't exist
            ValueError: If the YAML is malformed or validation fails
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML syntax in configuration file: {e}")
        
        if data is None:
            raise ValueError("Configuration file is empty")
        
        if not isinstance(data, dict):
            raise ValueError("Configuration file must contain a YAML mapping")
        
        # Substitute environment variables
        data = cls._substitute_env_vars(data)
        
        try:
            return cls(**data)
        except Exception as e:
            raise ValueError(f"Invalid configuration: {e}")

    def to_yaml(self) -> str:
        """Serialize settings to YAML string."""
        return yaml.dump(self.model_dump(), default_flow_style=False, allow_unicode=True)
