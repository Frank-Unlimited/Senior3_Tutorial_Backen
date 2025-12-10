"""Retry mechanism for model calls using tenacity.

This module provides retry decorators and utilities for handling
transient failures in AI model calls.
"""
import asyncio
import logging
from typing import Callable, Any, TypeVar, Awaitable
from functools import wraps

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    RetryError
)

from .errors import BiologyTutorError, ErrorCode


logger = logging.getLogger(__name__)

T = TypeVar('T')


# Common exceptions that should trigger retry
RETRYABLE_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    asyncio.TimeoutError,
)


def create_retry_decorator(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
    multiplier: float = 2.0
):
    """Create a retry decorator with configurable parameters.
    
    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time between retries (seconds)
        max_wait: Maximum wait time between retries (seconds)
        multiplier: Exponential backoff multiplier
        
    Returns:
        Configured retry decorator
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=multiplier, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )


# Default retry decorator for model calls
model_retry = create_retry_decorator(
    max_attempts=3,
    min_wait=2.0,
    max_wait=10.0
)


async def call_with_retry(
    func: Callable[..., Awaitable[T]],
    *args,
    max_attempts: int = 3,
    task_name: str = "unknown",
    **kwargs
) -> T:
    """Call an async function with retry logic.
    
    Args:
        func: Async function to call
        *args: Positional arguments for the function
        max_attempts: Maximum retry attempts
        task_name: Name of the task for error reporting
        **kwargs: Keyword arguments for the function
        
    Returns:
        Result of the function call
        
    Raises:
        BiologyTutorError: If all retries fail
    """
    last_error = None
    
    for attempt in range(max_attempts):
        try:
            return await func(*args, **kwargs)
        except RETRYABLE_EXCEPTIONS as e:
            last_error = e
            if attempt < max_attempts - 1:
                wait_time = min(2 ** attempt, 10)  # Exponential backoff
                logger.warning(
                    f"Attempt {attempt + 1}/{max_attempts} failed for {task_name}: {e}. "
                    f"Retrying in {wait_time}s..."
                )
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"All {max_attempts} attempts failed for {task_name}: {e}")
        except Exception as e:
            # Non-retryable exception
            logger.error(f"Non-retryable error in {task_name}: {e}")
            raise BiologyTutorError(
                ErrorCode.INTERNAL_ERROR,
                details={"task": task_name, "error": str(e)}
            )
    
    # All retries exhausted
    raise BiologyTutorError(
        ErrorCode.MODEL_TIMEOUT,
        message=f"任务 {task_name} 重试多次后仍然失败，请稍后再试~",
        details={"task": task_name, "last_error": str(last_error)}
    )


def with_timeout(timeout_seconds: float = 60.0):
    """Decorator to add timeout to async functions.
    
    Args:
        timeout_seconds: Timeout in seconds
        
    Returns:
        Decorated function with timeout
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                raise BiologyTutorError(
                    ErrorCode.MODEL_TIMEOUT,
                    details={"timeout": timeout_seconds}
                )
        return wrapper
    return decorator


class RetryableModelCall:
    """Context manager for retryable model calls with logging."""
    
    def __init__(
        self,
        task_name: str,
        max_attempts: int = 3,
        timeout: float = 60.0
    ):
        """Initialize retryable call context.
        
        Args:
            task_name: Name of the task for logging
            max_attempts: Maximum retry attempts
            timeout: Timeout per attempt in seconds
        """
        self.task_name = task_name
        self.max_attempts = max_attempts
        self.timeout = timeout
        self.attempt = 0
    
    async def __aenter__(self):
        self.attempt += 1
        logger.info(f"Starting {self.task_name} (attempt {self.attempt}/{self.max_attempts})")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            logger.info(f"Completed {self.task_name} successfully")
        else:
            logger.warning(f"Failed {self.task_name}: {exc_val}")
        return False  # Don't suppress exceptions
