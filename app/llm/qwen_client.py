"""
HTTP client for the Qwen 30B model running on Molab.

The Molab notebook serves a plain Flask API (NOT OpenAI-compatible):

    POST /generate
    Body : {"prompt": str, "max_new_tokens": int, "temperature": float}
    Reply: {"response": str}

Override QWEN_BASE_URL in .env when the ngrok URL changes between sessions.
"""
import json
import re
from typing import Any, Dict, Optional

import httpx

from app.core.config import settings


class QwenClient:
    """
    Client for Qwen 30B running on Molab through a plain Flask HTTP API.

    Qwen is used ONLY for:
      - Agent action decisions   (planner.py)
      - Chat intent classification (chat.py)
      - Conversational / contextual answers (chat.py)
      - Report narrative generation (generator.py)

    All numerical metrics are computed by deterministic Python tools.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        self.base_url = (base_url or settings.QWEN_BASE_URL).rstrip("/")
        self.timeout = timeout or settings.QWEN_TIMEOUT_SECONDS

    # ------------------------------------------------------------------
    # Availability check
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if the Molab Flask server is reachable."""
        try:
            # GET /generate returns 405 (Method Not Allowed) when the
            # server is up — that still means it is reachable.
            r = httpx.get(f"{self.base_url}/generate", timeout=5.0)
            return r.status_code in (200, 405)
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, prompt: str, system: Optional[str]) -> str:
        """Combine system instructions + user prompt into one string."""
        if system:
            return f"{system.strip()}\n\n{prompt.strip()}"
        return prompt.strip()

    def _clean_response(self, text: str) -> str:
        """
        Strip Qwen3 chain-of-thought <think>...</think> blocks.

        Qwen3 may emit internal reasoning inside <think> tags before
        producing the final answer.  We only forward the final answer.
        """
        if not text:
            return ""
        text = re.sub(
            r"<think>.*?</think>",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        return text.strip()

    # ------------------------------------------------------------------
    # Public API — text generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.2,
        max_new_tokens: int = 1024,
    ) -> str:
        """
        POST to /generate and return the model response as plain text.
        Returns "" on any failure so callers can fall back gracefully.
        """
        payload = {
            "prompt": self._build_prompt(prompt, system),
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }

        try:
            response = httpx.post(
                f"{self.base_url}/generate",
                json=payload,
                timeout=httpx.Timeout(self.timeout, connect=10.0),
            )
            response.raise_for_status()
            raw = response.json().get("response", "")
            return self._clean_response(raw)

        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Public API — JSON generation
    # ------------------------------------------------------------------

    def generate_json(
        self,
        prompt: str,
        system: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Ask Qwen for a JSON object and parse the result.

        Uses temperature=0 for maximum determinism.
        Falls back to regex extraction if the model wraps JSON in
        markdown fences or adds surrounding prose.
        """
        json_instruction = (
            "Return ONLY valid JSON.\n"
            "Do not include Markdown fences.\n"
            "Do not include any explanation outside the JSON object.\n"
        )
        combined_system = (
            f"{system.strip()}\n\n{json_instruction}" if system else json_instruction
        )

        text = self.generate(
            prompt=prompt,
            system=combined_system,
            temperature=0.0,
            max_new_tokens=512,
        )

        if not text:
            return None

        # Direct parse
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

        # Fallback: extract first {...} block
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None

        try:
            result = json.loads(match.group(0))
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            return None

        return None


qwen_client = QwenClient()



