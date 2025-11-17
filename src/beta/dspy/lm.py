import asyncio
import aiohttp
import requests
import json

class LM:
    def __init__(self, model: str="", api_base="http://localhost:8000", api_key: str = "-"):
        self.provider, self.model = model.split(":", 1) if ":" in model else ("vllm","")
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
          
    async def stream(self, messages, tools=None, **params):
      """Streaming interface for LLM"""
      
      # Handle string input
      if isinstance(messages, str):
          messages = [{"role": "user", "content": messages}]
      
      async with aiohttp.ClientSession() as session:
          # Build request body
          body = {
              "model": self.model,
              "messages": messages,
              "stream": True,
              **params
          }
          
          # Add tools if provided
          if tools:
              body["tools"] = tools
          
          # Stream response
          async with session.post(
              f"{self.api_base}/v1/chat/completions",
              json=body
          ) as resp:
              async for line in resp.content:
                  line = line.decode().strip()
                  
                  # Skip empty lines and done marker
                  if not line or line == "data: [DONE]":
                      continue
                  
                  # Parse SSE format
                  if line.startswith("data: "):
                      data = json.loads(line[6:])
                      yield data
                      

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
                data = await resp.json()
                if resp.status >= 400:
                    print(f"DEBUG: Error response: {data}")
                    resp.raise_for_status()
                # Debug: print response structure if it doesn't have expected format
                if "choices" not in data:
                    print(f"DEBUG: Unexpected response structure: {data}")
                return data

    async def _batch_async(self, messages_batch, **params):
        """Handle batch of conversations asynchronously"""
        tasks = [self._single_async(msgs, **params) for msgs in messages_batch]
        return await asyncio.gather(*tasks, return_exceptions=True)


"""
{
  "messages": [
    {
      "content": "string",
      "role": "developer",
      "name": "string"
    },
    {
      "content": "string",
      "role": "system",
      "name": "string"
    },
    {
      "content": "string",
      "role": "user",
      "name": "string"
    },
    {
      "role": "assistant",
      "audio": {
        "id": "string"
      },
      "content": "string",
      "function_call": {
        "arguments": "string",
        "name": "string"
      },
      "name": "string",
      "refusal": "string",
      "tool_calls": [
        {
          "id": "string",
          "function": {
            "arguments": "string",
            "name": "string"
          },
          "type": "function"
        },
        {
          "id": "string",
          "custom": {
            "input": "string",
            "name": "string"
          },
          "type": "custom"
        }
      ]
    },
    {
      "content": "string",
      "role": "tool",
      "tool_call_id": "string"
    },
    {
      "content": "string",
      "name": "string",
      "role": "function"
    },
    {
      "role": "string",
      "content": "string",
      "name": "string",
      "tool_call_id": "string",
      "tool_calls": [
        {
          "id": "string",
          "function": {
            "arguments": "string",
            "name": "string"
          },
          "type": "function"
        }
      ]
    },
    {
      "author": {
        "role": "user",
        "name": "string"
      },
      "content": [
        {}
      ],
      "channel": "string",
      "recipient": "string",
      "content_type": "string"
    }
  ],
  "model": "string",
  "frequency_penalty": 0,
  "logit_bias": {
    "additionalProp1": 0,
    "additionalProp2": 0,
    "additionalProp3": 0
  },
  "logprobs": false,
  "top_logprobs": 0,
  "max_completion_tokens": 0,
  "n": 1,
  "presence_penalty": 0,
  "response_format": {
    "type": "text",
    "json_schema": {
      "name": "string",
      "description": "string",
      "schema": {
        "additionalProp1": {}
      },
      "strict": true,
      "additionalProp1": {}
    },
    "additionalProp1": {}
  },
  "seed": -9223372036854776000,
  "stop": [],
  "stream": false,
  "stream_options": {
    "include_usage": true,
    "continuous_usage_stats": false,
    "additionalProp1": {}
  },
  "temperature": 0,
  "top_p": 0,
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "string",
        "description": "string",
        "parameters": {
          "additionalProp1": {}
        },
        "additionalProp1": {}
      },
      "additionalProp1": {}
    }
  ],
  "tool_choice": "none",
  "reasoning_effort": "low",
  "include_reasoning": true,
  "parallel_tool_calls": false,
  "user": "string",
  "best_of": 0,
  "use_beam_search": false,
  "top_k": 0,
  "min_p": 0,
  "repetition_penalty": 0,
  "length_penalty": 1,
  "stop_token_ids": [],
  "include_stop_str_in_output": false,
  "ignore_eos": false,
  "min_tokens": 0,
  "skip_special_tokens": true,
  "spaces_between_special_tokens": true,
  "truncate_prompt_tokens": -1,
  "prompt_logprobs": 0,
  "allowed_token_ids": [
    0
  ],
  "bad_words": [
    "string"
  ],
  "echo": false,
  "add_generation_prompt": true,
  "continue_final_message": false,
  "add_special_tokens": false,
  "documents": [
    {
      "additionalProp1": "string",
      "additionalProp2": "string",
      "additionalProp3": "string"
    }
  ],
  "chat_template": "string",
  "chat_template_kwargs": {
    "additionalProp1": {}
  },
  "mm_processor_kwargs": {
    "additionalProp1": {}
  },
  "guided_json": "string",
  "guided_regex": "string",
  "guided_choice": [
    "string"
  ],
  "guided_grammar": "string",
  "structural_tag": "string",
  "guided_decoding_backend": "string",
  "guided_whitespace_pattern": "string",
  "priority": 0,
  "request_id": "string",
  "logits_processors": [
    "string",
    {
      "qualname": "string",
      "args": [
        "string"
      ],
      "kwargs": {
        "additionalProp1": {}
      }
    }
  ],
  "return_tokens_as_token_ids": true,
  "return_token_ids": true,
  "cache_salt": "string",
  "kv_transfer_params": {
    "additionalProp1": {}
  },
  "vllm_xargs": {
    "additionalProp1": "string",
    "additionalProp2": "string",
    "additionalProp3": "string"
  },
  "additionalProp1": {}
}
"""