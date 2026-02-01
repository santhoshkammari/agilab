"""
Anthropic Messages API compatible server for vLLM backend.
Mimics LM Studio's /v1/messages endpoint.
Routes requests to local vLLM server at 192.168.170.76:8000

Usage:
    python api.py --port 1234

Then set environment variables for Claude Code:
    export ANTHROPIC_BASE_URL=http://localhost:1234
    export ANTHROPIC_AUTH_TOKEN=lmstudio
    claude --model qwen
"""

import os
import json
import uuid
from typing import Optional, List, Dict, Any
import httpx
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://192.168.170.76:8000")
VLLM_TIMEOUT = float(os.environ.get("VLLM_TIMEOUT", "300"))
NO_THINK_FLAG = False  # Will be set from command line args

app = FastAPI(title="Anthropic Messages API (vLLM Backend)")


# Request/Response Models
class ContentBlock(BaseModel):
    type: str
    text: Optional[str] = None
    # For tool_use blocks
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[Dict[str, Any]] = None


class Message(BaseModel):
    role: str
    content: Any  # Can be string or list


class MessagesRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: int = 1024
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    system: Optional[Any] = None
    stream: Optional[bool] = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None  # can be dict or string


class UsageInfo(BaseModel):
    input_tokens: int
    output_tokens: int


class MessageResponse(BaseModel):
    id: str
    type: str = "message"
    role: str = "assistant"
    content: List[ContentBlock]
    model: str
    stop_reason: str
    stop_sequence: Optional[str] = None
    usage: UsageInfo


# Utility functions
async def get_available_models() -> List[str]:
    """Fetch available models from vLLM."""
    try:
        async with httpx.AsyncClient(timeout=VLLM_TIMEOUT) as client:
            response = await client.get(f"{VLLM_BASE_URL}/v1/models")
            response.raise_for_status()
            data = response.json()
            return [model["id"] for model in data.get("data", [])]
    except Exception as e:
        logger.warning(f"Could not fetch models: {e}")
        return []


async def normalize_model_name(model: str) -> str:
    """
    Normalize model name to match vLLM available models.
    If exact match not found, use the first available model.
    """
    available = await get_available_models()

    if not available:
        return model

    # Check for exact match
    if model in available:
        return model

    # Check for partial match (case-insensitive)
    model_lower = model.lower()
    for avail_model in available:
        if model_lower in avail_model.lower() or avail_model.lower().split("/")[-1] in model_lower:
            return avail_model

    # Return first available model
    logger.info(f"Model '{model}' not found. Using '{available[0]}'")
    return available[0]


def process_content(content: Any) -> str:
    """
    Process content blocks and convert to text format for vLLM.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "image":
                    text_parts.append("[Image content]")
        return "\n".join(text_parts)

    return str(content)


def normalize_system_message(system: Any) -> Optional[str]:
    """
    Normalize system parameter to string format.
    Handles string, list of TextBlockParam objects (with optional cache_control).

    TextBlockParam structure:
    {
        "type": "text",
        "text": "actual text content",
        "cache_control": {"type": "ephemeral"}  # optional
    }
    """
    if system is None:
        return None

    if isinstance(system, str):
        return system

    if isinstance(system, list):
        text_parts = []
        for block in system:
            if isinstance(block, dict):
                # Extract text from TextBlockParam (ignores cache_control)
                if "text" in block:
                    text = block.get("text", "")
                    if text:
                        text_parts.append(text)
        if text_parts:
            return "\n".join(text_parts)
        return None

    if isinstance(system, dict):
        # Handle single TextBlockParam dict
        if "text" in system:
            return system.get("text", "")

    return str(system) if system else None


def convert_anthropic_tools_to_openai(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert Anthropic tool format to OpenAI format.

    Anthropic: {name, description, input_schema, ...}
    OpenAI: {type: "function", function: {name, description, parameters}}
    """
    openai_tools = []
    for tool in tools:
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool.get("name"),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {})
            }
        }
        openai_tools.append(openai_tool)
    return openai_tools


def convert_tool_choice_to_openai(tool_choice: Any) -> Any:
    """
    Convert Anthropic tool_choice to OpenAI format.

    Anthropic "auto" -> OpenAI "auto"
    Anthropic "any" -> OpenAI "required"
    Anthropic {type: "tool", name: "X"} -> OpenAI {type: "function", function: {name: "X"}}
    Anthropic "none" / None -> OpenAI "none"
    """
    if tool_choice is None:
        return "auto"

    if isinstance(tool_choice, str):
        if tool_choice == "auto":
            return "auto"
        elif tool_choice == "any":
            return "required"
        elif tool_choice == "none":
            return "none"
        return "auto"

    if isinstance(tool_choice, dict):
        choice_type = tool_choice.get("type")
        if choice_type == "auto":
            return "auto"
        elif choice_type == "any":
            return "required"
        elif choice_type == "tool":
            tool_name = tool_choice.get("name")
            return {
                "type": "function",
                "function": {"name": tool_name}
            }

    return "auto"


def convert_openai_tool_calls_to_anthropic(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert OpenAI tool_calls to Anthropic tool_use content blocks.

    OpenAI: [{id, type: "function", function: {name, arguments}}]
    Anthropic: [{type: "tool_use", id, name, input}]
    """
    anthropic_blocks = []
    for tool_call in tool_calls:
        func = tool_call.get("function", {})
        arguments_str = func.get("arguments", "{}")

        # Parse JSON arguments
        try:
            arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
        except json.JSONDecodeError:
            arguments = {}

        anthropic_block = {
            "type": "tool_use",
            "id": tool_call.get("id", ""),
            "name": func.get("name", ""),
            "input": arguments
        }
        anthropic_blocks.append(anthropic_block)

    return anthropic_blocks


# API Endpoints
@app.get("/v1/models")
async def list_models():
    """List available models from vLLM."""
    try:
        async with httpx.AsyncClient(timeout=VLLM_TIMEOUT) as client:
            response = await client.get(f"{VLLM_BASE_URL}/v1/models")
            response.raise_for_status()
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"vLLM error: {str(e)}")


@app.post("/v1/messages")
async def create_message(
    request: MessagesRequest,
    authorization: Optional[str] = Header(None),
):
    """
    Anthropic Messages API compatible endpoint.
    Routes to vLLM backend.
    """
    try:
        # Normalize model name
        model = await normalize_model_name(request.model)
        logger.info(f"Processing message with model: {model}")

        # Normalize system message
        system_message = normalize_system_message(request.system)

        # Process messages
        processed_messages = []
        for msg in request.messages:
            role = msg.role
            content = msg.content

            # Check for tool_use blocks in assistant messages (need to reconstruct)
            if role == "assistant" and isinstance(content, list):
                # Check if this is a tool_use response
                tool_calls = []
                text_content = ""

                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "tool_use":
                            # Convert to OpenAI tool_call format
                            tool_calls.append({
                                "id": block.get("id"),
                                "type": "function",
                                "function": {
                                    "name": block.get("name"),
                                    "arguments": json.dumps(block.get("input", {}))
                                }
                            })
                        elif block.get("type") == "text":
                            text_content += block.get("text", "")

                if tool_calls:
                    processed_messages.append({
                        "role": "assistant",
                        "content": text_content or None,
                        "tool_calls": tool_calls
                    })
                else:
                    processed_content = process_content(content)
                    processed_messages.append({"role": role, "content": processed_content})

            # Check for tool_result blocks in user messages
            elif role == "user" and isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get("type")
                        if block_type == "tool_result":
                            # Add as separate tool message
                            tool_content = block.get("content", "")
                            if isinstance(tool_content, list):
                                # Extract text from content blocks
                                tool_text_parts = []
                                for content_block in tool_content:
                                    if isinstance(content_block, dict) and content_block.get("type") == "text":
                                        tool_text_parts.append(content_block.get("text", ""))
                                tool_content = "\n".join(tool_text_parts)

                            processed_messages.append({
                                "role": "tool",
                                "tool_call_id": block.get("tool_use_id"),
                                "content": str(tool_content)
                            })
                        elif block_type == "text":
                            text_parts.append(block.get("text", ""))

                if text_parts:
                    processed_messages.append({
                        "role": "user",
                        "content": "\n".join(text_parts)
                    })
            else:
                processed_content = process_content(content)
                processed_messages.append({"role": role, "content": processed_content})

        # Prepare vLLM request payload
        payload = {
            "model": model,
            "messages": processed_messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature or 0.7,
        }

        # Add system message if provided
        if system_message:
            content = system_message
            if NO_THINK_FLAG:
                content += "\n\n /no_think"
            payload["messages"] = [
                {"role": "system", "content": content}
            ] + processed_messages

        # Add tools if provided
        if request.tools:
            payload["tools"] = convert_anthropic_tools_to_openai(request.tools)

            # Add tool_choice if specified
            if request.tool_choice:
                payload["tool_choice"] = convert_tool_choice_to_openai(request.tool_choice)
            else:
                payload["tool_choice"] = "auto"

        # Add optional parameters
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.top_k is not None:
            payload["top_k"] = request.top_k

        # Handle streaming vs non-streaming
        if request.stream:
            return await stream_message(payload, model)
        else:
            return await get_message(payload, model)

    except Exception as e:
        logger.error(f"Error processing message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def get_message(payload: Dict[str, Any], model: str) -> MessageResponse:
    """Handle non-streaming message request."""
    try:
        async with httpx.AsyncClient(timeout=VLLM_TIMEOUT) as client:
            response = await client.post(
                f"{VLLM_BASE_URL}/v1/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            vllm_response = response.json()

            # Extract response content
            choice = vllm_response.get("choices", [{}])[0]
            message = choice.get("message", {})
            message_content = message.get("content", "")
            finish_reason = choice.get("finish_reason", "stop")
            tool_calls = message.get("tool_calls")

            # Build content blocks
            content_blocks = []

            # Add text content if present
            if message_content:
                content_blocks.append(ContentBlock(type="text", text=message_content))

            # Add tool_use blocks if tool_calls present
            if tool_calls:
                anthropic_tools = convert_openai_tool_calls_to_anthropic(tool_calls)
                for tool_block in anthropic_tools:
                    content_blocks.append(ContentBlock(**tool_block))

            # If no content blocks, add empty text block
            if not content_blocks:
                content_blocks.append(ContentBlock(type="text", text=""))

            # Map vLLM finish reasons to Anthropic finish reasons
            stop_reason_map = {
                "stop": "end_turn",
                "length": "max_tokens",
                "eos_token": "end_turn",
                "tool_calls": "tool_use",
            }
            stop_reason = stop_reason_map.get(finish_reason, finish_reason)

            # Build Anthropic-style response
            return MessageResponse(
                id=f"msg_{uuid.uuid4().hex[:24]}",
                type="message",
                role="assistant",
                content=content_blocks,
                model=model,
                stop_reason=stop_reason,
                stop_sequence=None,
                usage=UsageInfo(
                    input_tokens=vllm_response.get("usage", {}).get(
                        "prompt_tokens", 0
                    ),
                    output_tokens=vllm_response.get("usage", {}).get(
                        "completion_tokens", 0
                    ),
                ),
            )

    except httpx.HTTPError as e:
        logger.error(f"vLLM API error: {e}")
        raise HTTPException(status_code=502, detail=f"vLLM error: {str(e)}")


async def stream_message(payload: Dict[str, Any], model: str):
    """Handle streaming message request."""
    payload["stream"] = True

    async def event_generator():
        try:
            async with httpx.AsyncClient(timeout=VLLM_TIMEOUT) as client:
                async with client.stream(
                    "POST",
                    f"{VLLM_BASE_URL}/v1/chat/completions",
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    message_id = f"msg_{uuid.uuid4().hex[:24]}"
                    sent_start = False
                    tool_call_accumulator = {}  # {index: {id, name, arguments}}
                    content_block_index = 0
                    text_block_started = False

                    async for line in response.aiter_lines():
                        if not line or line.startswith(":"):
                            continue

                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                # Send content_block_stop for any accumulated tool calls
                                for idx in tool_call_accumulator.keys():
                                    yield f"data: {json.dumps({'type': 'content_block_stop', 'index': idx})}\n\n"
                                # Send message_stop event
                                yield f"data: {json.dumps({'type': 'message_stop'})}\n\n"
                                break

                            try:
                                chunk = json.loads(data)
                                choice = chunk.get("choices", [{}])[0]
                                delta = choice.get("delta", {})
                                finish_reason = choice.get("finish_reason")

                                # Send message_start on first delta
                                if not sent_start and (delta.get("content") or delta.get("tool_calls")):
                                    yield f"data: {json.dumps({'type': 'message_start', 'message': {'id': message_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': model}})}\n\n"
                                    sent_start = True

                                # Handle text content delta
                                if "content" in delta and delta["content"]:
                                    if not text_block_started:
                                        yield f"data: {json.dumps({'type': 'content_block_start', 'index': content_block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                                        text_block_started = True
                                    yield f"data: {json.dumps({'type': 'content_block_delta', 'index': content_block_index, 'delta': {'type': 'text_delta', 'text': delta['content']}})}\n\n"

                                # Handle tool_calls delta
                                if "tool_calls" in delta:
                                    for tool_call_delta in delta["tool_calls"]:
                                        idx = tool_call_delta.get("index", 0)

                                        # Initialize accumulator for this tool call
                                        if idx not in tool_call_accumulator:
                                            tool_call_id = tool_call_delta.get("id", f"tool_{idx}")
                                            tool_call_accumulator[idx] = {
                                                "id": tool_call_id,
                                                "name": "",
                                                "arguments": ""
                                            }
                                            # Send content_block_start for tool_use
                                            yield f"data: {json.dumps({'type': 'content_block_start', 'index': idx, 'content_block': {'type': 'tool_use', 'id': tool_call_id, 'name': ''}})}\n\n"

                                        # Accumulate function name and arguments
                                        func_delta = tool_call_delta.get("function", {})
                                        if "name" in func_delta:
                                            tool_call_accumulator[idx]["name"] = func_delta["name"]
                                        if "arguments" in func_delta:
                                            tool_call_accumulator[idx]["arguments"] += func_delta["arguments"]
                                            # Send input_json_delta
                                            yield f"data: {json.dumps({'type': 'content_block_delta', 'index': idx, 'delta': {'type': 'input_json_delta', 'partial_json': func_delta['arguments']}})}\n\n"

                                # Handle finish_reason
                                if finish_reason:
                                    # Send content_block_stop for text if started
                                    if text_block_started:
                                        yield f"data: {json.dumps({'type': 'content_block_stop', 'index': content_block_index})}\n\n"

                            except json.JSONDecodeError:
                                continue

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.get(f"{VLLM_BASE_URL}/health")
        return {"status": "ok", "vllm": "connected"}
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "vllm": f"disconnected: {str(e)}"},
        )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Anthropic Messages API (vLLM Backend)",
        "version": "2.0.0",
        "features": [
            "Messages API",
            "Streaming responses",
            "Tool calling (Anthropic ↔ OpenAI format)",
            "System prompts",
            "Token counting"
        ],
        "endpoints": {
            "models": "/v1/models",
            "messages": "/v1/messages",
            "health": "/health",
        },
    }


def main():
    """Run the server."""
    parser = argparse.ArgumentParser(
        description="Anthropic Messages API server for vLLM"
    )
    parser.add_argument(
        "--port", type=int, default=1234, help="Port to run server on (default: 1234)"
    )
    parser.add_argument(
        "--no-think", action="store_true", help="Add /no_think to system prompts"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--vllm-url",
        default=VLLM_BASE_URL,
        help=f"vLLM server URL (default: {VLLM_BASE_URL})",
    )

    args = parser.parse_args()

    # Update vLLM base URL if provided
    globals()["VLLM_BASE_URL"] = args.vllm_url

    # Store no_think flag globally
    globals()["NO_THINK_FLAG"] = args.no_think

    logger.info(f"Starting Anthropic Messages API server on {args.host}:{args.port}")
    logger.info(f"vLLM backend: {args.vllm_url}")
    logger.info(f"No-think mode: {'enabled' if args.no_think else 'disabled'}")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
