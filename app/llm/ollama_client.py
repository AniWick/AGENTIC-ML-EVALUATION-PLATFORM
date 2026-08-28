import json
import httpx
from typing import Dict, Any, Optional, List
from app.core.config import settings

class OllamaClient:
    """Client for querying local Ollama LLM runtime with fast connectivity check and fallback."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None
    ):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or settings.LLM_MODEL
        self.fallback_model = settings.FALLBACK_LLM_MODEL
        self.timeout = timeout or settings.LLM_TIMEOUT_SECONDS

    def is_available(self) -> bool:
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=1.0)
            return r.status_code == 200
        except Exception:
            return False

    def generate(self, prompt: str, system: Optional[str] = None, temperature: float = 0.2) -> str:
        """Sends a completion request to Ollama if service is active."""
        if not self.is_available():
            return ""

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature}
        }
        if system:
            payload["system"] = system

        try:
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=httpx.Timeout(10.0, connect=1.0)
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception:
            return ""

    def generate_json(self, prompt: str, system: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Sends structured request expecting JSON response."""
        if not self.is_available():
            return None

        payload = {
            "model": self.model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.1}
        }
        if system:
            payload["system"] = system

        try:
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=httpx.Timeout(10.0, connect=1.0)
            )
            response.raise_for_status()
            res_text = response.json().get("response", "")
            return json.loads(res_text)
        except Exception:
            return None

ollama_client = OllamaClient()
