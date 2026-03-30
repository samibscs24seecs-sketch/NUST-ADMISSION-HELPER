"""
Ollama LLM Connector
Async connector for local Ollama server with streaming support.
Default model: phi3:mini (configurable via MODEL env var)
"""

import json
import logging
import os
from pathlib import Path
import sys
from typing import AsyncGenerator

import httpx

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger("OllamaLLM")

OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL   = os.getenv("MODEL", "llama3.2:3b")
TEMPERATURE     = float(os.getenv("TEMPERATURE", "0.0"))
NUM_CTX         = int(os.getenv("NUM_CTX", "4096"))   # llama3.2 handles 4096 efficiently on CPU
NUM_PREDICT     = int(os.getenv("NUM_PREDICT", "1500"))
REQUEST_TIMEOUT = 180.0  # seconds — extra headroom for heavy CPU on 8 GB RAM


class OllamaLLM:
    """
    Async Ollama connector.
    Provides streaming and non-streaming inference via local Ollama.
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = DEFAULT_MODEL,
        temperature: float = TEMPERATURE,
        num_ctx: int = NUM_CTX,
        num_predict: int = NUM_PREDICT,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.num_predict = num_predict

    async def _detect_model(self) -> str:
        """
        Verify llama3.2:3b is available. Returns matched model name if found.
        No fallback to heavier models — this system targets 8 GB RAM / CPU-only.
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                if resp.status_code == 200:
                    available = [m["name"] for m in resp.json().get("models", [])]
                    match = next((m for m in available if "llama3.2:3b" in m), None)
                    if match:
                        return match
        except Exception:
            pass
        return self.model

    async def check_ollama_health(self) -> dict:
        """
        Check if Ollama is running and get available models.
        Returns {"status": "running"/"offline", "model": str, "models_available": [...]}
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                if resp.status_code == 200:
                    data = resp.json()
                    models_available = [m["name"] for m in data.get("models", [])]

                    # Update model if current not available but others are
                    if models_available and self.model not in models_available:
                        self.model = await self._detect_model()

                    return {
                        "status": "running",
                        "model": self.model,
                        "models_available": models_available,
                    }
                else:
                    return {
                        "status": "offline",
                        "model": self.model,
                        "models_available": [],
                    }
        except (httpx.ConnectError, httpx.TimeoutException):
            return {
                "status": "offline",
                "model": self.model,
                "models_available": [],
            }
        except Exception as e:
            logger.error("Unexpected error checking Ollama health: %s", e)
            return {
                "status": "offline",
                "model": self.model,
                "models_available": [],
            }

    def _build_payload(self, messages: list[dict], stream: bool = True) -> dict:
        """Build the request payload for Ollama /api/chat."""
        return {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
                "repeat_penalty": 1.3,   # penalise repeated phrases — stops looping
                "repeat_last_n": 128,    # look back 128 tokens — catches header-level repetition
                "stop": ["<|eot_id|>", "<|end_of_text|>", "</s>"],
            },
            "keep_alive": "10m",  # keep model loaded for 10 min — prevents drop during CPU spikes
        }

    async def stream_response(self, messages: list[dict]) -> AsyncGenerator[str, None]:
        """
        Stream tokens from Ollama.
        Yields individual text tokens as strings.
        """
        payload = self._build_payload(messages, stream=True)

        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/chat",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        logger.error(
                            "Ollama returned %d: %s",
                            response.status_code,
                            error_text.decode(),
                        )
                        yield f"[Error: Ollama returned status {response.status_code}]"
                        return

                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                            token = data.get("message", {}).get("content", "")
                            if token:
                                yield token
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue

        except httpx.ConnectError:
            logger.error("Cannot connect to Ollama at %s", self.base_url)
            yield "[Ollama is not running. Please start Ollama and try again.]"
        except httpx.TimeoutException:
            logger.error("Ollama request timed out after %.0fs", REQUEST_TIMEOUT)
            yield "[Response timed out. The model may be loading — please try again.]"
        except Exception as e:
            logger.error("Unexpected streaming error: %s", e)
            yield f"[Error during streaming: {str(e)}]"

    async def get_full_response(self, messages: list[dict]) -> str:
        """
        Get a complete non-streaming response from Ollama.
        Returns the full response text.
        """
        payload = self._build_payload(messages, stream=False)

        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                resp = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("message", {}).get("content", "")
                else:
                    logger.error("Ollama error %d: %s", resp.status_code, resp.text)
                    return f"[Error: Ollama returned status {resp.status_code}]"

        except httpx.ConnectError:
            return "[Ollama is not running. Please start Ollama and try again.]"
        except httpx.TimeoutException:
            return "[Response timed out. Please try again.]"
        except Exception as e:
            logger.error("Unexpected error in get_full_response: %s", e)
            return f"[Error: {str(e)}]"

    async def is_model_available(self) -> bool:
        """Check if the configured model is available in Ollama."""
        health = await self.check_ollama_health()
        return (
            health["status"] == "running"
            and self.model in health.get("models_available", [])
        )


# Singleton
llm = OllamaLLM()
