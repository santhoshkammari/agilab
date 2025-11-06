# aspy/lm_threadpooled.py - ThreadPool version for true batch processing
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import aiohttp
import requests
from typing import Any, Dict, List, Union

class LM:
    def __init__(self, model: str="", api_base="http://localhost:11434", api_key: str = "-", max_workers: int = 8):
        self.provider, self.model = model.split(":", 1) if model else ("vllm","")
        self.api_base = api_base
        self.api_key = api_key
        self.max_workers = max_workers

    def __call__(self, messages, **params):
        if self.provider == "vllm":
            # Smart detection of input type
            if self._is_batch(messages):
                return self._batch_threadpool(messages, **params)
            else:
                return self._single_sync(messages, **params)
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

    def _single_sync(self, messages, **params):
        """Handle single conversation synchronously"""
        # Handle string input
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        url = f"{self.api_base}/v1/chat/completions"
        body = {"model": self.model, "messages": messages, **params}

        resp = requests.post(url, json=body)
        resp.raise_for_status()
        return resp.json()

    def _batch_threadpool(self, messages_batch, **params):
        """Handle batch of conversations using ThreadPoolExecutor for true concurrency"""
        print(f"🚀 Processing batch of {len(messages_batch)} requests with {self.max_workers} workers")
        start_time = time.time()

        def make_request(messages):
            """Make a single API request - runs in thread"""
            try:
                return self._single_sync(messages, **params)
            except Exception as e:
                return e

        # Use ThreadPoolExecutor for true concurrent requests
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all requests concurrently
            future_to_messages = {
                executor.submit(make_request, msgs): i
                for i, msgs in enumerate(messages_batch)
            }

            # Collect results in order
            results = [None] * len(messages_batch)
            completed = 0

            for future in as_completed(future_to_messages):
                idx = future_to_messages[future]
                results[idx] = future.result()
                completed += 1

                # Progress indicator
                if completed % max(1, len(messages_batch) // 10) == 0 or completed == len(messages_batch):
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    print(f"  ⚡ {completed}/{len(messages_batch)} completed ({rate:.1f} req/s)")

        total_time = time.time() - start_time
        avg_rate = len(messages_batch) / total_time if total_time > 0 else 0
        print(f"✅ Batch completed in {total_time:.2f}s (avg {avg_rate:.1f} req/s)")

        return results

    # Keep async methods for backward compatibility
    async def _single_async(self, messages, **params):
        """Handle single conversation asynchronously (legacy)"""
        # Handle string input
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        async with aiohttp.ClientSession() as session:
            body = {"model": self.model, "messages": messages, **params}
            async with session.post(f"{self.api_base}/v1/chat/completions", json=body) as resp:
                return await resp.json()

    async def _batch_async(self, messages_batch, **params):
        """Handle batch of conversations asynchronously (legacy)"""
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
