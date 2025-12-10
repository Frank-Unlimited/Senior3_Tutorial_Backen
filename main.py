"""FastAPI application entry point for Biology Tutorial Workflow.

This module initializes the FastAPI application with all dependencies
and configures CORS, routes, and global exception handling.
"""
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, Request

# Load environment variables from .env file
load_dotenv()
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config.settings import Settings
from session.manager import SessionManager
from sse.publisher import SSEPublisher
from workflow.biology_tutor import BiologyTutorWorkflow
from api.routes import router
from utils.errors import BiologyTutorError


# Global instances
settings: Settings = None
session_manager: SessionManager = None
sse_publisher: SSEPublisher = None
workflow: BiologyTutorWorkflow = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown."""
    global settings, session_manager, sse_publisher, workflow
    
    # Startup
    print("🚀 Starting Biology Tutorial Workflow...")
    
    # Load settings
    settings_path = os.environ.get("SETTINGS_PATH", "settings.yaml")
    try:
        settings = Settings.from_yaml(settings_path)
        print(f"✅ Loaded settings from {settings_path}")
    except FileNotFoundError:
        print(f"⚠️ Settings file not found: {settings_path}, using defaults")
        # Create minimal settings for development
        from config.settings import ModelConfig
        settings = Settings(
            deep_thinking_model=ModelConfig(
                provider="doubao",
                model_name="doubao-pro-32k",
                api_key=os.environ.get("DOUBAO_API_KEY", ""),
            ),
            quick_model=ModelConfig(
                provider="doubao",
                model_name="doubao-lite-4k",
                api_key=os.environ.get("DOUBAO_API_KEY", ""),
            )
        )
    
    # Initialize components
    session_manager = SessionManager(redis_url=settings.redis_url)
    sse_publisher = SSEPublisher()
    workflow = BiologyTutorWorkflow(settings, session_manager, sse_publisher)
    
    # Store in app state for access in routes
    app.state.settings = settings
    app.state.session_manager = session_manager
    app.state.sse_publisher = sse_publisher
    app.state.workflow = workflow
    
    print("✅ All components initialized")
    print("🎓 Biology Tutorial Workflow is ready!")
    
    yield
    
    # Shutdown
    print("👋 Shutting down Biology Tutorial Workflow...")


# Create FastAPI app
app = FastAPI(
    title="Biology Tutorial Workflow",
    description="基于 LangChain 的高三生物错题辅导系统",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler
@app.exception_handler(BiologyTutorError)
async def biology_tutor_error_handler(request: Request, exc: BiologyTutorError):
    """Handle custom application errors."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error_code": exc.error_code,
            "message": exc.message,
            "details": exc.details
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors."""
    return JSONResponse(
        status_code=500,
        content={
            "error_code": "INTERNAL_ERROR",
            "message": "服务器内部错误，请稍后重试",
            "details": str(exc) if os.environ.get("DEBUG") else None
        }
    )


# Include API routes
app.include_router(router)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "biology-tutorial-workflow"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
