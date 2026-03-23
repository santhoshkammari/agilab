"""Simple agent loop for interacting with LLMs via vLLM endpoint."""

import json
import os
import sys
import urllib.request
from tools import TOOL_SCHEMAS, TOOLS_BY_NAME

# ── constants ─────────────────────────────────────────────────────────────────

VLLM_BASE_URL = "http://192.168.170.76:8000"
VLLM_API_KEY  = "dummy"
VLLM_MODEL    = "/home/ng6309/datascience/santhosh/models/qwen3.5-9b"
MAX_TURNS     = 20
MAX_TOKENS    = 10000
CWD           = os.getcwd()

# ── ANSI ──────────────────────────────────────────────────────────────────────

BOLD    = "\033[1m"
DIM     = "\033[2m"
CYAN    = "\033[36m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
MAGENTA = "\033[35m"
RESET   = "\033[0m"

# ── HTTP SSE streaming ────────────────────────────────────────────────────────

def _stream_chat(messages: list, enable_thinking: bool):
    """Stream from vLLM /v1/chat/completions. Yields raw SSE data strings."""
    system = f"You are a helpful AI coding assistant. The user's current working directory is: {CWD}"
    payload = {
        "model": VLLM_MODEL,
        "messages": [{"role": "system", "content": system}] + messages,
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "tools": TOOL_SCHEMAS,
        "tool_choice": "auto",
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{VLLM_BASE_URL}/v1/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {VLLM_API_KEY}",
        },
    )
    with urllib.request.urlopen(req) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if line.startswith("data: "):
                yield line[6:]

# ── tool execution ────────────────────────────────────────────────────────────

def run_tool(name: str, args: dict) -> str:
    tool = TOOLS_BY_NAME.get(name)
    if not tool:
        return f"Unknown tool: {name}"
    result = tool["handler"](args)
    return "\n".join(p.get("text", "") for p in result.get("content", []))

# ── agent turn ────────────────────────────────────────────────────────────────

def agent_turn(messages: list, enable_thinking: bool, show_thinking: bool) -> list:
    """One full agentic turn with tool loop. Returns updated messages."""

    for _ in range(MAX_TURNS):
        text           = ""
        think_buf      = ""
        tool_calls_raw = {}
        finish_reason  = None
        in_think       = False
        interrupted    = False

        try:
            for chunk in _stream_chat(messages, enable_thinking):
                if chunk == "[DONE]":
                    break
                try:
                    obj = json.loads(chunk)
                except Exception:
                    continue

                choice = obj.get("choices", [{}])[0]
                finish_reason = choice.get("finish_reason") or finish_reason
                delta = choice.get("delta", {})

                # ── thinking tokens (reasoning_content field) ──
                reasoning = delta.get("reasoning_content") or ""
                if reasoning:
                    think_buf += reasoning
                    if show_thinking:
                        print(f"{DIM}{reasoning}{RESET}", end="", flush=True)
                    elif not in_think:
                        print(f"{DIM}Thinking...{RESET}", flush=True)
                        in_think = True

                # ── text content (may contain inline <think> tags) ──
                content = delta.get("content") or ""
                if content:
                    if "<think>" in content or in_think:
                        parts = content.split("<think>", 1)
                        if len(parts) == 2:
                            print(parts[0], end="", flush=True)
                            text += parts[0]
                            rest = parts[1]
                            if show_thinking:
                                print(f"{DIM}{rest}{RESET}", end="", flush=True)
                            elif not in_think:
                                print(f"{DIM}Thinking...{RESET}", flush=True)
                            in_think = True
                            think_buf += rest
                        elif "</think>" in content:
                            end_parts = content.split("</think>", 1)
                            think_buf += end_parts[0]
                            if show_thinking:
                                print(f"{DIM}{end_parts[0]}{RESET}", end="", flush=True)
                            in_think = False
                            after = end_parts[1]
                            text += after
                            print(after, end="", flush=True)
                        else:
                            if in_think:
                                think_buf += content
                                if show_thinking:
                                    print(f"{DIM}{content}{RESET}", end="", flush=True)
                            else:
                                text += content
                                print(content, end="", flush=True)
                    else:
                        text += content
                        print(content, end="", flush=True)

                # ── tool calls ──
                for tc in delta.get("tool_calls", []):
                    idx = tc.get("index", 0)
                    if idx not in tool_calls_raw:
                        tool_calls_raw[idx] = {"id": tc.get("id", ""), "name": "", "args": ""}
                    if tc.get("id"):
                        tool_calls_raw[idx]["id"] = tc["id"]
                    fn = tc.get("function", {})
                    if fn.get("name"):
                        tool_calls_raw[idx]["name"] = fn["name"]
                        print(f"\n{YELLOW}⚙ {fn['name']}{RESET}", end=" ", flush=True)
                    if fn.get("arguments"):
                        tool_calls_raw[idx]["args"] += fn["arguments"]
                        print(f"{DIM}{fn['arguments']}{RESET}", end="", flush=True)

        except KeyboardInterrupt:
            print(f"\n{DIM}interrupted{RESET}")
            interrupted = True

        print()

        # Always append assistant message (even partial) to keep history consistent
        asst_msg = {"role": "assistant", "content": text or "(interrupted)"}
        messages.append(asst_msg)

        if interrupted:
            break

        # Update assistant message with tool_calls if any
        if tool_calls_raw:
            messages[-1]["tool_calls"] = [
                {"id": v["id"], "type": "function",
                 "function": {"name": v["name"], "arguments": v["args"]}}
                for v in tool_calls_raw.values()
            ]

        if finish_reason == "tool_calls" and tool_calls_raw:
            for v in tool_calls_raw.values():
                try:
                    args = json.loads(v["args"])
                except Exception:
                    args = {}
                print(f"{DIM}  args: {v['args'][:200]}{RESET}")
                output = run_tool(v["name"], args)
                if v["name"] == "Read":
                    preview = output[:300] + ("…" if len(output) > 300 else "")
                    print(f"{GREEN}  ✓ {v['name']}{RESET}: {DIM}{preview}{RESET}\n")
                else:
                    print(f"{GREEN}  ✓ {v['name']}{RESET}:\n{output}\n")
                messages.append({
                    "role": "tool",
                    "tool_call_id": v["id"],
                    "content": output,
                })
            continue

        break

    return messages

# ── REPL ──────────────────────────────────────────────────────────────────────

def print_help(enable_thinking, show_thinking):
    e = f"{MAGENTA}ON{RESET}" if enable_thinking else f"{DIM}OFF{RESET}"
    s = f"{MAGENTA}ON{RESET}" if show_thinking  else f"{DIM}OFF{RESET}"
    print(f"""
{BOLD}Commands:{RESET}
  {CYAN}/think{RESET}   — toggle model thinking [{e}]
  {CYAN}/show{RESET}    — toggle display of thinking [{s}]
  {CYAN}/clear{RESET}   — clear conversation history
  {CYAN}/help{RESET}    — show this help
  {CYAN}exit{RESET}     — quit
  {DIM}Ctrl+C{RESET}   — interrupt generation
""")

def repl():
    history: list  = []
    enable_thinking = False
    show_thinking   = False

    print(f"{BOLD}{CYAN}Agent ready{RESET} — type your prompt, or /help")
    print(f"Model: {DIM}{VLLM_MODEL}{RESET}")
    print(f"cwd:   {DIM}{CWD}{RESET}\n")

    while True:
        indicators = ""
        if enable_thinking: indicators += f"{MAGENTA}[think]{RESET} "
        if show_thinking:   indicators += f"{DIM}[show]{RESET} "
        try:
            user_input = input(f"{BOLD}{GREEN}>{RESET} {indicators}").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            sys.exit(0)

        if not user_input:
            continue

        if user_input == "/think":
            enable_thinking = not enable_thinking
            s = f"{MAGENTA}ON{RESET}" if enable_thinking else f"{DIM}OFF{RESET}"
            print(f"  model thinking {s}\n")
            continue

        if user_input == "/show":
            show_thinking = not show_thinking
            s = f"{MAGENTA}ON{RESET}" if show_thinking else f"{DIM}OFF{RESET}"
            print(f"  show thinking {s}\n")
            continue

        if user_input == "/clear":
            history = []
            print(f"  {DIM}history cleared{RESET}\n")
            continue

        if user_input == "/help":
            print_help(enable_thinking, show_thinking)
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Bye!")
            sys.exit(0)

        history.append({"role": "user", "content": user_input})
        print()
        try:
            history = agent_turn(history, enable_thinking, show_thinking)
        except Exception as e:
            print(f"{YELLOW}error: {e}{RESET}\n")
            continue
        print()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        agent_turn([{"role": "user", "content": prompt}], False, False)
    else:
        repl()
