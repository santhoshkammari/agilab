# aspy/lm.py
import asyncio
import os

import aiohttp
import requests
from typing import Any, Dict, List, Union

class LM:
    def __init__(self, model: str="", api_base="http://localhost:11434", api_key: str = "-"):
        self.provider, self.model = model.split(":", 1) if model else ("vllm","")
        self.api_base = api_base
        self.api_key = api_key

    def __call__(self, messages, **params):
        if self.provider == "vllm":
            # Smart detection of input type
            if self._is_batch(messages):
                return asyncio.run(self._batch_async(messages, **params))
            else:
                return asyncio.run(self._single_async(messages, **params))
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _is_batch(self, messages) -> bool:
        """Intelligently detect if input is batch or single conversation"""
        # Case 1: List of conversations (each is a list of messages)
        if isinstance(messages, list) and len(messages) > 0:
            if isinstance(messages[0], list):
                return True  # [[msg1, msg2], [msg3, msg4]] - batch
            elif isinstance(messages[0], dict) and 'role' in messages[0]:
                return False  # [{"role": "user", "content": "hi"}] - single

        # Case 2: String input (convert to single message)
        if isinstance(messages, str):
            return False

        return False

    async def _single_async(self, messages, **params):
        """Handle single conversation asynchronously"""
        # Handle string input
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        async with aiohttp.ClientSession() as session:
            body = {"model": self.model, "messages": messages, **params}
            async with session.post(f"{self.api_base}/v1/chat/completions", json=body) as resp:
                return await resp.json()

    async def _batch_async(self, messages_batch, **params):
        """Handle batch of conversations asynchronously"""
        tasks = [self._single_async(msgs, **params) for msgs in messages_batch]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def _call_vllm(self, messages, **params):
        """Legacy sync method - deprecated, kept for compatibility"""
        url = f"{self.api_base}/v1/chat/completions"
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        print(messages)

        body = {
            "model": self.model,
            "messages": messages,
            **params
        }

        resp = requests.post(url, json=body)
        resp.raise_for_status()
        data = resp.json()
        return data
