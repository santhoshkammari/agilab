"""
Minimal dspy-style agent API with sync/async support.
Simple, Pythonic, no fancy wrappers.
"""
import json
import asyncio
import inspect
from typing import Callable, Optional
from dataclasses import dataclass
from contextvars import ContextVar
from transformers.utils import get_json_schema
import aiohttp

# Global context for default LM
_default_lm: ContextVar[Optional['LM']] = ContextVar('default_lm', default=None)


@dataclass
class AgentResult:
    """Rich result object from agent calls."""
    text: str
    history: list[dict]
    iterations: int = 0
    tool_calls_count: int = 0


class LM:
    """Language model client."""

    def __init__(
        self,
        model: str,
        api_base: str = "http://localhost:8000",
        api_key: str = "-",
        timeout: Optional[aiohttp.ClientTimeout] = None,
        **defaults
    ):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.timeout = timeout or aiohttp.ClientTimeout(
            total=None,
            connect=10,
            sock_read=None
        )
        self.defaults = defaults  # temperature, seed, max_tokens, etc.

    def __enter__(self):
        """Context manager entry - set as default LM."""
        self._token = _default_lm.set(self)
        return self

    def __exit__(self, *args):
        """Context manager exit - restore previous LM."""
        _default_lm.reset(self._token)

    @staticmethod
    def configure(lm: 'LM'):
        """Configure default LM globally (dspy-style)."""
        _default_lm.set(lm)

    @staticmethod
    def get_default() -> Optional['LM']:
        """Get the configured default LM."""
        return _default_lm.get()

    async def stream(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        **params
    ):
        """Stream LLM response."""
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            body = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                **self.defaults,  # Apply LM defaults first
                **params          # Override with call-specific params
            }

            if tools:
                body["tools"] = tools

            async with session.post(
                f"{self.api_base}/v1/chat/completions",
                json=body
            ) as resp:
                resp.raise_for_status()

                async for line in resp.content:
                    line = line.decode().strip()

                    if not line or line == "data: [DONE]":
                        continue

                    if line.startswith("data: "):
                        yield json.loads(line[6:])

    async def complete(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        **params
    ) -> dict:
        """Non-streaming completion."""
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            body = {
                "model": self.model,
                "messages": messages,
                **self.defaults,  # Apply LM defaults first
                **params          # Override with call-specific params
            }

            if tools:
                body["tools"] = tools

            async with session.post(
                f"{self.api_base}/v1/chat/completions",
                json=body
            ) as resp:
                data = await resp.json()
                if resp.status >= 400:
                    raise RuntimeError(f"LLM error: {data}")
                return data


class Agent:
    """Agent orchestrator with sync/async support."""

    def __init__(
        self,
        lm: Optional[LM] = None,
        tools: Optional[list[Callable]] = None,
        max_iterations: int = 10,
        **defaults
    ):
        self.lm = lm  # Can be None, will use default
        self._tools = {t.__name__: t for t in (tools or [])}
        self.max_iterations = max_iterations
        self.defaults = defaults  # Agent-level overrides (temperature, etc.)
        self._history: list[dict] = []

    @property
    def tool_schemas(self) -> list[dict]:
        """Get tool schemas for API calls."""
        return [get_json_schema(t) for t in self._tools.values()]

    def _get_lm(self) -> LM:
        """Get LM (from instance or default context)."""
        if self.lm:
            return self.lm
        default = LM.get_default()
        if not default:
            raise RuntimeError("No LM configured. Use LM.configure() or pass lm to Agent()")
        return default

    def add_message(self, role: str, content: str) -> None:
        """Add message to history."""
        self._history.append({"role": role, "content": content})

    def add_tool(self, tool: Callable) -> None:
        """Register a new tool."""
        self._tools[tool.__name__] = tool

    async def _execute_tool(self, name: str, arguments: str, tool_id: str) -> dict:
        """Execute a tool asynchronously."""
        try:
            tool_fn = self._tools.get(name)
            if not tool_fn:
                return {
                    "tool_call_id": tool_id,
                    "output": f"Tool '{name}' not found",
                    "is_error": True
                }

            args = json.loads(arguments) if arguments else {}

            if asyncio.iscoroutinefunction(tool_fn):
                output = await tool_fn(**args)
            else:
                output = tool_fn(**args)

            return {
                "tool_call_id": tool_id,
                "output": str(output),
                "is_error": False
            }

        except Exception as e:
            return {
                "tool_call_id": tool_id,
                "output": f"Error: {str(e)}",
                "is_error": True
            }

    async def _stream_step(self, messages: list[dict], **kwargs):
        """Execute one generation step."""
        tool_buffer = {}
        tool_futures = {}
        last_tool_id = None
        msg = {"role": "assistant", "content": "", "tool_calls": []}
        lm = self._get_lm()

        # Merge params: agent defaults + call-specific kwargs
        params = {**self.defaults, **kwargs}

        async for chunk in lm.stream(messages, self.tool_schemas, **params):
            delta = chunk['choices'][0]['delta']

            if 'content' in delta and delta['content']:
                msg["content"] += delta['content']
                yield {"type": "content", "content": delta['content']}

            elif 'tool_calls' in delta:
                tc = delta['tool_calls'][0]

                if 'id' in tc:
                    tool_id = tc['id']
                    tool_buffer[tool_id] = {
                        "id": tool_id,
                        "type": "function",
                        "function": {
                            "name": tc['function']['name'],
                            "arguments": tc['function'].get('arguments', '')
                        }
                    }
                    last_tool_id = tool_id
                elif tc.get('function', {}).get('arguments'):
                    if last_tool_id in tool_buffer:
                        tool_buffer[last_tool_id]['function']['arguments'] += tc['function']['arguments']

        if tool_buffer:
            msg["tool_calls"] = list(tool_buffer.values())
            for tool_id, tc in tool_buffer.items():
                tool_futures[tool_id] = asyncio.create_task(
                    self._execute_tool(tc['function']['name'], tc['function']['arguments'], tool_id)
                )

        # Clean up empty fields
        if not msg["content"]:
            msg.pop("content", None)
        if not msg["tool_calls"]:
            msg.pop("tool_calls", None)

        yield {"type": "step_complete", "message": msg, "tool_futures": tool_futures}

    async def run(self, input=None, system: str = "", **kwargs):
        """Run agent loop, yielding events."""
        # Prepare messages
        if input is not None:
            if isinstance(input, list) and system:
                raise ValueError("Cannot use both message list and system prompt")

            if isinstance(input, str):
                if system:
                    self.add_message("system", system)
                self.add_message("user", input)
            elif isinstance(input, list):
                for msg in input:
                    if isinstance(msg, dict):
                        self._history.append({"role": msg['role'], "content": msg.get('content', '')})
                    else:
                        self._history.append(msg)
            else:
                raise ValueError("Input must be str or list of messages")

        history = self._history.copy()
        iterations = 0
        total_tools = 0

        for _ in range(self.max_iterations):
            iterations += 1
            assistant_msg = None
            tool_futures = {}

            async for event in self._stream_step(history, **kwargs):
                if event['type'] == 'step_complete':
                    assistant_msg = event['message']
                    tool_futures = event['tool_futures']
                else:
                    yield event

            self._history.append(assistant_msg)
            history.append(assistant_msg)

            if assistant_msg.get("tool_calls"):
                total_tools += len(assistant_msg["tool_calls"])
                results = await asyncio.gather(*[tool_futures[tc['id']] for tc in assistant_msg["tool_calls"]])

                for result in results:
                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": result["tool_call_id"],
                        "content": result["output"]
                    }
                    self._history.append(tool_msg)
                    history.append(tool_msg)
                    yield {"type": "tool_result", "output": result["output"]}
            else:
                break

        yield {
            "type": "complete",
            "response": self._history[-1].get("content", "") if self._history else "",
            "iterations": iterations,
            "tool_calls_count": total_tools
        }

    async def _arun(self, input, system: str = "", return_result: bool = False, **kwargs):
        """Async runner that returns final response."""
        response = ""
        iterations = 0
        tool_calls_count = 0

        async for event in self.run(input, system, **kwargs):
            if event['type'] == 'complete':
                response = event['response']
                iterations = event.get('iterations', 0)
                tool_calls_count = event.get('tool_calls_count', 0)

        if return_result:
            return AgentResult(
                text=response,
                history=self._history.copy(),
                iterations=iterations,
                tool_calls_count=tool_calls_count
            )
        return response

    def __call__(self, input, system: str = "", return_result: bool = False, **kwargs):
        """Call agent (sync or async based on context).

        Args:
            input: Either a string prompt or list of message dicts [{"role": "user", "content": "..."}]
            system: System prompt (only if input is string, not list)
            return_result: If True, return AgentResult object with metadata, else return string
            **kwargs: Additional LLM parameters (temperature, seed, etc.)

        Returns:
            str or AgentResult (sync) or coroutine (async)
        """
        # Check if we're in async context
        try:
            asyncio.get_running_loop()
            # Already in async context - return coroutine
            return self._arun(input, system, return_result, **kwargs)
        except RuntimeError:
            # Not in async context - run sync
            return asyncio.run(self._arun(input, system, return_result, **kwargs))

    @property
    def history(self) -> list[dict]:
        """Get conversation history."""
        return self._history

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._history.clear()


class Eval:
    """Minimal evaluation (dspy-style) with batching and caching."""

    def __init__(
        self,
        metric: Callable,
        dataset: list[dict],
        save_path: Optional[str] = None,
        parallel: int = 1,
        batch_size: int = 10,
        show_progress: bool = True
    ):
        """
        Args:
            metric: Function (example, prediction) -> score (0-1 or True/False)
            dataset: List of dicts with input/output pairs
            save_path: Path to save results (enables auto-caching if provided)
            parallel: Number of parallel threads (default 1 = sequential)
            batch_size: Save results every N examples
            show_progress: Show tqdm progress bar
        """
        self.metric = metric
        self.dataset = dataset
        self.save_path = save_path
        self.parallel = parallel
        self.batch_size = batch_size
        self.show_progress = show_progress

    def _load_cached_results(self):
        """Load existing results if cache enabled."""
        if not self.save_path:
            return []

        try:
            with open(self.save_path, 'r') as f:
                data = json.load(f)
                return data.get('results', [])
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_results(self, all_results):
        """Save results to JSON file."""
        if not self.save_path:
            return

        scores = [r['score'] for r in all_results]
        total = len(scores)
        correct = sum(scores)
        avg_score = correct / total if total > 0 else 0.0

        data = {
            'score': round(avg_score * 100, 2),
            'correct': int(correct),
            'total': total,
            'results': all_results
        }

        with open(self.save_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _create_cache_key(self, example):
        """Create unique key for example."""
        return json.dumps(example.get('input', example), sort_keys=True)

    def __call__(self, agent: Agent) -> dict:
        """Evaluate agent on dataset with batching and caching."""
        from concurrent.futures import ThreadPoolExecutor
        from tqdm import tqdm

        # Load cached results
        cached_results = self._load_cached_results()
        cached_keys = {self._create_cache_key(r['example']): r for r in cached_results}

        def eval_one(example):
            # Check cache
            cache_key = self._create_cache_key(example)
            if self.save_path and cache_key in cached_keys:
                return cached_keys[cache_key]

            try:
                agent.clear_history()
                prediction = agent(example['input'])
                score = self.metric(example, prediction)
                return {'example': example, 'prediction': prediction, 'score': float(score)}
            except Exception as e:
                return {'example': example, 'prediction': None, 'score': 0.0, 'error': str(e)}

        # Filter out cached examples
        if self.save_path:
            to_eval = [ex for ex in self.dataset if self._create_cache_key(ex) not in cached_keys]
        else:
            to_eval = self.dataset

        all_results = list(cached_results)

        if not to_eval:
            # All cached
            scores = [r['score'] for r in all_results]
            return {
                'score': round(sum(scores) / len(scores) * 100, 2) if scores else 0.0,
                'correct': int(sum(scores)),
                'total': len(scores),
                'results': all_results
            }

        # Process in batches
        iterator = tqdm(range(0, len(to_eval), self.batch_size), disable=not self.show_progress, desc="Batches")

        for batch_start in iterator:
            batch = to_eval[batch_start:batch_start + self.batch_size]

            if self.parallel == 1:
                # Sequential
                batch_results = [eval_one(ex) for ex in batch]
            else:
                # Parallel
                with ThreadPoolExecutor(max_workers=self.parallel) as executor:
                    batch_results = list(executor.map(eval_one, batch))

            all_results.extend(batch_results)

            # Save after each batch
            self._save_results(all_results)

        # Final stats
        scores = [r['score'] for r in all_results]
        total = len(scores)
        correct = sum(scores)
        avg_score = correct / total if total > 0 else 0.0

        return {
            'score': round(avg_score * 100, 2),
            'correct': int(correct),
            'total': total,
            'results': all_results
        }


if __name__ == "__main__":
    """Minimal usage examples."""

    # Example 1: Basic usage with parameter inheritance
    print("Example 1: Parameter inheritance")
    lm = LM(
        model="Qwen/Qwen3-4B-Instruct-2507",
        api_base="http://192.168.170.76:8000",
        temperature=0.7,  # LM-level default
        seed=42
    )
    agent = Agent(lm, temperature=0.0)  # Agent overrides to 0.0
    result = agent("What is 2+2?", temperature=0.9)  # Call overrides to 0.9
    print(f"Result: {result[:100]}...")

    # Example 2: Rich result object
    print("\nExample 2: Rich result object")
    result_obj = agent("Hello!", return_result=True)
    print(f"Text: {result_obj.text[:50]}...")
    print(f"Iterations: {result_obj.iterations}")
    print(f"Tool calls: {result_obj.tool_calls_count}")
    print(f"History length: {len(result_obj.history)}")

    # Example 3: Simplified Eval
    print("\nExample 3: Simplified Eval")
    def exact_match(example, prediction):
        return prediction.strip() == example['output'].strip()

    dataset = [
        {'input': 'What is 2+2?', 'output': '4'},
        {'input': 'What is 3+3?', 'output': '6'}
    ]

    # Simple usage - auto-enables cache
    evaluator = Eval(
        metric=exact_match,
        dataset=dataset,
        save_path="/tmp/eval_simple.json"  # Auto cache + save
    )
    result = evaluator(agent)
    print(f"Score: {result['score']}% ({result['correct']}/{result['total']})")

    # Advanced usage
    evaluator2 = Eval(
        metric=exact_match,
        dataset=dataset,
        save_path="/tmp/eval_parallel.json",
        parallel=4,      # 4 threads
        batch_size=20    # Save every 20
    )

    print("\n✅ All examples completed!")
