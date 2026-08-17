"""OpenAI 兼容客户端（requests 实现，无需 openai 包）。"""

from __future__ import annotations

import json
from typing import Iterator

import requests


class AcademicAgentClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")

    def health(self) -> dict:
        return requests.get(f"{self.base_url}/health", timeout=10).json()

    def retrieve(self, query: str, top_k: int = 5) -> dict:
        resp = requests.post(
            f"{self.base_url}/v1/retrieve",
            json={"query": query, "top_k": top_k},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 1024,
        stream: bool = False,
    ):
        body = {
            "model": "academic-agent",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions", json=body, timeout=120, stream=stream
        )
        resp.raise_for_status()
        if not stream:
            return resp.json()

        def iter_sse() -> Iterator[str]:
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                yield line[len("data: ") :]

        return iter_sse()
