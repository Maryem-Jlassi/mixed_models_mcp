"""
Configuration settings for the CV Processing API and Unified Host.

This file contains all the configuration settings for the application.
For sensitive information like API keys, use environment variables.
"""
import os
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""
    
    # API Configuration
    API_TITLE: str = "CV Processing API"
    API_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # MCP Client/Tool timeout to prevent indefinite hangs
    MCP_TOOL_TIMEOUT: int = 90  # seconds
    MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017/talentmind")

    # LLM Configuration
    # Lower default for faster first tokens with local HF host; can be overridden via env
    LLM_MAX_TOKENS: int = 128
    # Orchestration mode
    LLM_ORCHESTRATION_ENABLED: bool = False
    # Default local model (used by Ollama or as fallback)
    LLM_MODEL: str = "llama3.2:latest"
    # Groq LLM Configuration (used by Groq-based orchestrator). Read from env/.env only.
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    GROQ_TIMEOUT: int = int(os.getenv("GROQ_TIMEOUT", "60"))
    # Base URL for Groq's OpenAI-compatible API
    GROQ_API_URL: str = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1")
    
    # MCP Server Scripts
    JOB_MCP_SERVER_SCRIPT: str = "server.py"
    CV_MCP_SERVER_SCRIPT: str = "cv_ocr_server.py"
    
    # Auth / JWT Configuration
    JWT_SECRET: str = os.getenv("JWT_SECRET", "change-me-please")
    JWT_ALGO: str = os.getenv("JWT_ALGO", "HS256")
    JWT_EXPIRES_MIN: int = int(os.getenv("JWT_EXPIRES_MIN", "60"))
    
    # Pydantic settings config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding='utf-8',
        case_sensitive=True,
        extra='ignore'  # Ignore extra fields in .env
    )


# Create settings instance
settings = Settings()

# Do not create any directories on disk; storage is dynamic (MongoDB/GridFS)

# Legacy config object for backwards compatibility
class Config:
    """Legacy config object for backwards compatibility"""
    LOG_LEVEL = settings.LOG_LEVEL
    MCP_TOOL_TIMEOUT = settings.MCP_TOOL_TIMEOUT
    LLM_MAX_TOKENS = settings.LLM_MAX_TOKENS
    LLM_ORCHESTRATION_ENABLED = settings.LLM_ORCHESTRATION_ENABLED
    LLM_MODEL = settings.LLM_MODEL

config = Config()
