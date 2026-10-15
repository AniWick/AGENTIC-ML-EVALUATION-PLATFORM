"""
HTTP client for Qwen 3.8 27B running on the Groq API.
Uses Groq's OpenAI-compatible Chat Completions endpoint.
"""
import json
import os
import re
from collections import Counter
from typing import Any, Dict, Optional

import httpx

from app.core.config import settings


def _looks_degenerate(
    text: str,
    ngram: int = 6,
    min_repeats: int = 3,
) -> bool:
    """Return True if `text` contains a repetitive loop pattern."""
    if not text:
        return False
    tokens = text.split()
    if len(tokens) < ngram * min_repeats:
        return False
    grams = [tuple(tokens[i:i + ngram]) for i in range(len(tokens) - ngram)]
    if not grams:
        return False
    return max(Counter(grams).values()) >= min_repeats


class QwenClient:
    """
    Client for Qwen 3.8 27B via Groq API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        self.api_key = (
            api_key
            or getattr(settings, "GROQ_API_KEY", None)
            or os.environ.get("GROQ_API_KEY", "")
        )
        self.base_url = (
            base_url
            or getattr(settings, "GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        ).rstrip("/")
        self.model_name = (
            model_name
            or getattr(settings, "GROQ_MODEL", "openai/gpt-oss-120b")
        )
        self.timeout = timeout or getattr(settings, "QWEN_TIMEOUT_SECONDS", 60)

    def is_available(self) -> bool:
        """Return True if the Groq API key is configured."""
        return bool(self.api_key)

    def _clean_response(self, text: str) -> str:
        """Strip reasoning or thought tags safely."""
        if not text:
            return ""
        if "</think>" in text:
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        else:
            text = re.sub(r"<think>", "", text, flags=re.IGNORECASE)
        return text.strip()

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.2,
        max_new_tokens: int = 4096,
        repetition_penalty: float = 1.15,
    ) -> str:
        """POST to Groq Chat Completions API using Qwen 27B."""
        if not self.api_key:
            return ""

        messages = []
        if system:
            messages.append({"role": "system", "content": system.strip()})
        messages.append({"role": "user", "content": prompt.strip()})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_new_tokens,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            with httpx.Client(timeout=httpx.Timeout(self.timeout, connect=10.0)) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()
                raw = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                cleaned = self._clean_response(raw)

                if _looks_degenerate(cleaned):
                    return ""

                return cleaned

        except Exception as exc:
            print(f"[Groq Qwen Client Error]: {exc}")
            return ""

    def generate_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_new_tokens: int = 4096,
    ) -> Optional[Dict[str, Any]]:
        """Ask Qwen 27B for a JSON object and parse the result reliably."""
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
            max_new_tokens=max_new_tokens,
        )

        if not text:
            return None

        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

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