"""GLM5.1 auditor client for simulated expert audit.

Uses Doubao Token Plan API to call GLM-5.1 model.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional


class GLM51Client:
    """Client for calling GLM-5.1 via Doubao Token Plan API."""

    def __init__(self, api_key: str = "", base_url: str = "", model_id: str = "glm-5.1"):
        self.api_key = api_key or os.environ.get("DOUBAO_API_KEY", "")
        self.base_url = base_url or "https://ark.cn-beijing.volces.com/api/coding/v3"
        self.model_id = model_id

    def chat(self, messages: list[dict], max_tokens: int = 2048, temperature: float = 0.3) -> Any:
        """Call GLM-5.1 with messages."""
        import openai

        client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        response = client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return response.choices[0].message.content

    def test_connection(self) -> dict:
        """Test connection to GLM-5.1."""
        start = time.time()
        try:
            response = self.chat(
                messages=[{"role": "user", "content": "请用一句话介绍碳纤维。"}],
                max_tokens=100,
                temperature=0.3,
            )
            latency = time.time() - start
            return {
                "status": "ok",
                "model_id": self.model_id,
                "latency_seconds": round(latency, 2),
                "response_length": len(response),
                "response_preview": response[:100],
            }
        except Exception as e:
            return {
                "status": "error",
                "model_id": self.model_id,
                "error": str(e)[:200],
                "latency_seconds": round(time.time() - start, 2),
            }


def load_api_key_from_file(api_config_path: str) -> tuple[str, str]:
    """Load API key and base URL from config file."""
    api_key = ""
    base_url = ""

    try:
        with open(api_config_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("DOUBAO_API_KEY:"):
                    api_key = line.split(":", 1)[1].strip()
                elif line.startswith("DOUBAO_OpenAI_URL:"):
                    raw_url = line.split(":", 1)[1].strip()
                    # Ensure /v3 suffix for OpenAI compatibility
                    if not raw_url.endswith("/v3"):
                        base_url = raw_url + "/v3"
                    else:
                        base_url = raw_url
    except FileNotFoundError:
        pass

    return api_key, base_url
