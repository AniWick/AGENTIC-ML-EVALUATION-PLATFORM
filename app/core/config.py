import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from typing import Optional, Union, Any

class Settings(BaseSettings):
    APP_NAME: str = "Agentic ML Evaluation Platform"
    APP_ENV: str = "development"
    DEBUG: Any = True
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    # Ollama LLM Configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "llama3.1"
    FALLBACK_LLM_MODEL: str = "phi3"
    LLM_TIMEOUT_SECONDS: int = 60

    # Agent Configuration
    MAX_AGENT_ITERATIONS: int = 12

    # Storage Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    STORAGE_DIR: str = "storage"
    CHROMA_PERSIST_DIR: str = "storage/chromadb"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="allow"
    )

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug(cls, v: Any) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() in ("1", "true", "yes", "debug", "dev", "development")
        return bool(v)

    @property
    def storage_path(self) -> Path:
        p = self.BASE_DIR / self.STORAGE_DIR
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def chroma_path(self) -> Path:
        p = self.BASE_DIR / self.CHROMA_PERSIST_DIR
        p.mkdir(parents=True, exist_ok=True)
        return p

settings = Settings()

