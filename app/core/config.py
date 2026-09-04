from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from typing import Any


class Settings(BaseSettings):
    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    APP_NAME: str = "Agentic ML Evaluation Platform"
    APP_ENV: str = "development"
    DEBUG: Any = True
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    # ------------------------------------------------------------------
    # LLM Configuration
    #
    # Keep the old OLLAMA_* names because the existing codebase imports
    # these settings. We are simply pointing them at the Qwen HTTP server.
    # ------------------------------------------------------------------
    OLLAMA_BASE_URL: str = "http://127.0.0.1:8001/v1"
    LLM_MODEL: str = "qwen"
    FALLBACK_LLM_MODEL: str = "qwen"
    LLM_TIMEOUT_SECONDS: int = 120

    # ------------------------------------------------------------------
    # Qwen Configuration (Molab Flask + ngrok)
    #
    # The Molab notebook runs a plain Flask server (NOT OpenAI-compat).
    # Endpoint: POST /generate  Body: {"prompt": str, "max_new_tokens": int, "temperature": float}
    # Response: {"response": str}
    #
    # Override QWEN_BASE_URL in .env so you never need to edit this file
    # when the ngrok tunnel URL changes between Molab sessions.
    # ------------------------------------------------------------------
    QWEN_BASE_URL: str = "https://lair-likewise-apostle.ngrok-free.dev"
    QWEN_MODEL: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    QWEN_TIMEOUT_SECONDS: int = 120

    # ------------------------------------------------------------------
    # Agent Configuration
    # ------------------------------------------------------------------
    MAX_AGENT_ITERATIONS: int = 12

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent

    STORAGE_DIR: str = "storage"
    CHROMA_PERSIST_DIR: str = "storage/chromadb"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="allow",
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug(cls, v: Any) -> bool:
        if isinstance(v, bool):
            return v

        if isinstance(v, str):
            return v.lower() in (
                "1",
                "true",
                "yes",
                "debug",
                "dev",
                "development",
            )

        return bool(v)

    # ------------------------------------------------------------------
    # Storage helpers
    # ------------------------------------------------------------------
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
