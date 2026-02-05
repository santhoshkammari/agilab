"""
Minimal dspy-style agent API with sync/async support.
Simple, Pythonic, no fancy wrappers.
"""
import json
import asyncio
import inspect
from typing import Callable, Optional
from dataclasses import dataclass, field
from contextvars import ContextVar
from transformers.utils import get_json_schema
import aiohttp

# Global context for default LM
_default_lm: ContextVar[Optional['LM']] = ContextVar('default_lm', default=None)


@dataclass
class Message:
    """Represents a chat message."""
    role: str
    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to API message format."""
        msg = {"role": self.role}

        if self.content:
            msg["content"] = self.content

        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls

        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id

        if self.name:
            msg["name"] = self.name

        return msg


@dataclass
class ToolResult:
    """Result from tool execution."""
    tool_call_id: str
    output: str
    is_error: bool = False

    def to_message(self) -> Message:
        """Convert to message format."""
        return Message(
            role="tool",
            content=self.output,
            tool_call_id=self.tool_call_id
        )


class ToolRegistry:
    """Manages tool functions and their schemas."""

    def __init__(self):
        self._tools: dict[str, Callable] = {}
        self._schemas: list[dict] = []

    def register(self, func: Callable) -> None:
        """Register a tool function."""
        name = func.__name__
        self._tools[name] = func
        schema = get_json_schema(func) if callable(func) else func
        self._schemas.append(schema)

    def register_many(self, tools: list[Callable]) -> None:
        """Register multiple tools."""
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> Optional[Callable]:
        """Get tool by name."""
        return self._tools.get(name)

    @property
    def schemas(self) -> list[dict]:
        """Get all tool schemas."""
        return self._schemas

    async def execute(self, name: str, arguments: str, tool_id: str) -> ToolResult:
        """Execute a tool asynchronously."""
        try:
            tool_fn = self.get(name)
            if not tool_fn:
                return ToolResult(
                    tool_call_id=tool_id,
                    output=f"Tool '{name}' not found",
                    is_error=True
                )

            args = json.loads(arguments) if arguments else {}

            if asyncio.iscoroutinefunction(tool_fn):
                output = await tool_fn(**args)
            else:
                output = tool_fn(**args)

            return ToolResult(
                tool_call_id=tool_id,
                output=str(output),
                is_error=False
            )

        except Exception as e:
            return ToolResult(
                tool_call_id=tool_id,
                output=f"Error: {str(e)}",
                is_error=True
            )


class LM:
    """Language model client."""

    def __init__(
        self,
        model: str,
        api_base: str = "http://localhost:8000",
        api_key: str = "-",
        timeout: Optional[aiohttp.ClientTimeout] = None
    ):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.timeout = timeout or aiohttp.ClientTimeout(
            total=None,
            connect=10,
            sock_read=None
        )

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
                **params
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
                **params
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
        max_iterations: int = 10
    ):
        self.lm = lm  # Can be None, will use default
        self.tools = ToolRegistry()
        self.max_iterations = max_iterations

        if tools:
            self.tools.register_many(tools)

        self._history: list[Message] = []

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
        self._history.append(Message(role=role, content=content))

    def _prepare_messages(self, input, system: str = ""):
        """Prepare messages from various input formats."""
        if isinstance(input, list) and system:
            raise ValueError("Cannot use both message list and system prompt")

        if isinstance(input, str):
            # String input - add as user message
            if system:
                self.add_message("system", system)
            self.add_message("user", input)
        elif isinstance(input, list):
            # List of messages - add all
            for msg in input:
                if isinstance(msg, dict):
                    self._history.append(Message(role=msg['role'], content=msg.get('content', '')))
                else:
                    self._history.append(msg)
        else:
            raise ValueError("Input must be str or list of messages")

    def add_tool(self, tool: Callable) -> None:
        """Register a new tool."""
        self.tools.register(tool)

    async def _stream_step(self, messages: list[dict], **kwargs):
        """Execute one generation step."""
        tool_buffer = {}
        tool_futures = {}
        last_tool_id = None
        msg = Message(role="assistant")
        lm = self._get_lm()

        async for chunk in lm.stream(messages, self.tools.schemas, **kwargs):
            delta = chunk['choices'][0]['delta']

            if 'content' in delta and delta['content']:
                msg.content += delta['content']
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
            msg.tool_calls = list(tool_buffer.values())
            for tool_id, tc in tool_buffer.items():
                tool_futures[tool_id] = asyncio.create_task(
                    self.tools.execute(tc['function']['name'], tc['function']['arguments'], tool_id)
                )

        yield {"type": "step_complete", "message": msg, "tool_futures": tool_futures}

    async def run(self, input=None, system: str = "", **kwargs):
        """Run agent loop, yielding events."""
        if input is not None:
            self._prepare_messages(input, system)

        history = [msg.to_dict() for msg in self._history]

        for _ in range(self.max_iterations):
            assistant_msg = None
            tool_futures = {}

            async for event in self._stream_step(history, **kwargs):
                if event['type'] == 'step_complete':
                    assistant_msg = event['message']
                    tool_futures = event['tool_futures']
                else:
                    yield event

            self._history.append(assistant_msg)
            history.append(assistant_msg.to_dict())

            if assistant_msg.tool_calls:
                results = await asyncio.gather(*[tool_futures[tc['id']] for tc in assistant_msg.tool_calls])

                for result in results:
                    tool_msg = result.to_message()
                    self._history.append(tool_msg)
                    history.append(tool_msg.to_dict())
                    yield {"type": "tool_result", "output": result.output}
            else:
                break

        yield {"type": "complete", "response": self._history[-1].content if self._history else ""}

    async def _arun(self, input, system: str = "", **kwargs) -> str:
        """Async runner that returns final response."""
        response = ""
        async for event in self.run(input, system, **kwargs):
            if event['type'] == 'complete':
                response = event['response']
        return response

    def __call__(self, input, system: str = "", **kwargs):
        """Call agent (sync or async based on context).

        Args:
            input: Either a string prompt or list of message dicts [{"role": "user", "content": "..."}]
            system: System prompt (only if input is string, not list)
            **kwargs: Additional LLM parameters

        Returns:
            str (sync) or coroutine (async)
        """
        # Check if we're in async context
        try:
            asyncio.get_running_loop()
            # Already in async context - return coroutine
            return self._arun(input, system, **kwargs)
        except RuntimeError:
            # Not in async context - run sync
            return asyncio.run(self._arun(input, system, **kwargs))

    @property
    def history(self) -> list[Message]:
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
        num_threads: int = 1,
        show_progress: bool = True,
        batch_size: int = 10,
        save_json_path: Optional[str] = None,
        cache: bool = True
    ):
        """
        Args:
            metric: Function (example, prediction) -> score (0-1 or True/False)
            dataset: List of dicts with input/output pairs
            num_threads: Parallel threads (default 1 = sequential)
            show_progress: Show tqdm progress bar
            batch_size: Save results every N examples
            save_json_path: Path to save results incrementally (batched saves)
            cache: If True, skip already evaluated examples (requires save_json_path)
        """
        self.metric = metric
        self.dataset = dataset
        self.num_threads = num_threads
        self.show_progress = show_progress
        self.batch_size = batch_size
        self.save_json_path = save_json_path
        self.cache = cache

    def _load_cached_results(self):
        """Load existing results if cache enabled."""
        if not self.cache or not self.save_json_path:
            return []

        try:
            with open(self.save_json_path, 'r') as f:
                data = json.load(f)
                return data.get('results', [])
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_results(self, all_results):
        """Save results to JSON file."""
        if not self.save_json_path:
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

        with open(self.save_json_path, 'w') as f:
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
            if self.cache and cache_key in cached_keys:
                return cached_keys[cache_key]

            try:
                agent.clear_history()
                prediction = agent(example['input'])
                score = self.metric(example, prediction)
                return {'example': example, 'prediction': prediction, 'score': float(score)}
            except Exception as e:
                return {'example': example, 'prediction': None, 'score': 0.0, 'error': str(e)}

        # Filter out cached examples
        if self.cache:
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

            if self.num_threads == 1:
                # Sequential
                batch_results = [eval_one(ex) for ex in batch]
            else:
                # Parallel
                with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
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

    # Example 1: Basic usage
    print("Example 1: Basic LM and Agent")
    lm = LM(model="Qwen/Qwen3-4B-Instruct-2507", api_base="http://192.168.170.76:8000")
    agent = Agent(lm)
    result = agent("What is 2+2?")
    print(f"Result: {result[:100]}...")

    # Example 2: Global configure (dspy-style)
    print("\nExample 2: Global configure")
    LM.configure(lm)
    agent2 = Agent()  # No lm needed
    result = agent2("Hello!")
    print(f"Result: {result[:50]}...")

    # Example 3: Context manager
    print("\nExample 3: Context manager")
    with lm:
        agent3 = Agent()
        result = agent3("Hi")
        print(f"Result: {result[:50]}...")

    # Example 4: System prompt
    print("\nExample 4: System prompt")
    result = agent("Count to 3", system="Be brief")
    print(f"Result: {result[:50]}...")

    # Example 5: Message list
    print("\nExample 5: Message list")
    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hi"}
    ]
    result = agent(messages)
    print(f"Result: {result[:50]}...")

    # Example 6: Async usage
    print("\nExample 6: Async usage")
    import asyncio
    async def async_example():
        result = await agent("Hello async")
        return result
    result = asyncio.run(async_example())
    print(f"Result: {result[:50]}...")

    # Example 7: Tools (commented - vLLM may not support tools)
    print("\nExample 7: Agent with tools (skipped - server may not support)")
    # def add(a: int, b: int) -> int:
    #     """Add two numbers.
    #     Args:
    #         a: First number
    #         b: Second number
    #     """
    #     return a + b
    # agent_tools = Agent(lm, tools=[add])
    # result = agent_tools("What is 5+3?")

    # Example 8: Evaluation
    print("\nExample 8: Evaluation")
    def exact_match(example, prediction):
        return prediction.strip() == example['output'].strip()

    dataset = [
        {'input': 'What is 2+2?', 'output': '4'},
        {'input': 'What is 3+3?', 'output': '6'}
    ]

    evaluator = Eval(
        metric=exact_match,
        dataset=dataset,
        num_threads=2,
        batch_size=1,
        save_json_path="/tmp/eval_results_demo.json",
        cache=True,
        show_progress=False
    )
    result = evaluator(agent)
    print(f"Score: {result['score']}% ({result['correct']}/{result['total']})")

    print("\n✅ All examples completed!")
