"""
Minimal dspy-style agent API with sync/async support.
Simple, Pythonic, no fancy wrappers.
"""
import json
import asyncio
import inspect
from typing import Callable, Optional, Union
from dataclasses import dataclass
from contextvars import ContextVar
from transformers.utils import get_json_schema
import aiohttp

# DSPy imports for Signature support
try:
    from dspy import Signature, InputField, OutputField
    from dspy.adapters.chat_adapter import ChatAdapter
    from dspy.signatures.signature import ensure_signature
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    Signature = None
    InputField = None
    OutputField = None

# Global context for default LM
_default_lm: ContextVar[Optional['LM']] = ContextVar('default_lm', default=None)


@dataclass
class AgentResult:
    """Rich result object from agent calls."""
    text: str
    history: list[dict]
    iterations: int = 0
    tool_calls_count: int = 0


class Prediction:
    """DSPy-style prediction result with field access.

    Allows accessing output fields by name:
        pred = Predict("query -> answer")
        result = pred(query="What is 2+2?")
        print(result.answer)  # Access by field name
        print(str(result))     # Or convert to string
    """
    def __init__(self, text: str, signature_obj=None):
        self._text = text
        self._signature_obj = signature_obj

        # If we have a signature, parse output fields from text
        if signature_obj:
            self._parse_fields()
        else:
            # No signature, just store as generic text
            self.text = text

    def _parse_fields(self):
        """Parse output fields from DSPy-formatted text."""
        # DSPy formats outputs like:
        # [[ ## field_name ## ]]
        # value
        # [[ ## completed ## ]]

        import re

        # Extract field values
        output_fields = {}

        # Get output field names from signature
        if hasattr(self._signature_obj, 'output_fields'):
            for field_name in self._signature_obj.output_fields.keys():
                # Look for pattern: [[ ## field_name ## ]]\nvalue
                pattern = rf'\[\[\s*##\s*{field_name}\s*##\s*\]\]\s*\n?(.*?)(?:\[\[|$)'
                match = re.search(pattern, self._text, re.DOTALL)
                if match:
                    value = match.group(1).strip()
                    # Remove trailing [[ ## completed ## ]] if present
                    value = re.sub(r'\s*\[\[\s*##\s*completed\s*##\s*\]\]\s*$', '', value)
                    output_fields[field_name] = value
                    setattr(self, field_name, value)
                else:
                    # Field not found in DSPy format - use cleaned text as fallback
                    # This handles cases where LLM doesn't follow exact format
                    cleaned = re.sub(r'\[\[\s*##\s*\w+\s*##\s*\]\]\s*', '', self._text)
                    cleaned = re.sub(r'\s*\[\[\s*##\s*completed\s*##\s*\]\]\s*', '', cleaned)
                    setattr(self, field_name, cleaned.strip())
                    output_fields[field_name] = cleaned.strip()

        # If no fields were parsed, use the whole text
        if not output_fields:
            # Try to extract anything between first field marker and completed
            pattern = r'\[\[\s*##\s*\w+\s*##\s*\]\]\s*\n?(.*?)\[\[\s*##\s*completed\s*##\s*\]\]'
            match = re.search(pattern, self._text, re.DOTALL)
            if match:
                self.text = match.group(1).strip()
            else:
                self.text = self._text.strip()
        else:
            # Use the first output field as default text
            self.text = list(output_fields.values())[0] if output_fields else self._text

    def __str__(self):
        """Return text representation."""
        return self.text

    def __repr__(self):
        """Return representation."""
        fields = [k for k in dir(self) if not k.startswith('_') and k != 'text']
        if fields:
            field_str = ', '.join(f"{k}={getattr(self, k)!r}" for k in fields if not callable(getattr(self, k)))
            return f"Prediction({field_str})"
        return f"Prediction(text={self.text!r})"


class LM:
    """Language model client."""

    def __init__(
        self,
        model: str="",
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


# Global configure (dspy-style)
def configure(lm: LM):
    """Configure default LM globally."""
    LM.configure(lm)


class Predict:
    """Predict orchestrator with sync/async support (dspy-style).

    Flexible predictor supporting multiple modes:

    Mode 1: DSPy Signature
        pred = Predict("query -> answer")
        pred = Predict(CustomSignature)

    Mode 2: Signature + System
        pred = Predict("query -> answer", system="Be concise")

    Mode 3: Raw System Prompt
        pred = Predict(system="You are a classifier")

    Mode 4: With Tools
        pred = Predict("query -> answer", tools=[search, calculate])
    """

    def __init__(
        self,
        signature: Optional[Union[str, type]] = None,  # DSPy-style: "query -> answer" or Signature class
        system: Optional[str] = None,           # Additional system prompt
        instructions: Optional[str] = None,     # Override signature instructions
        lm: Optional[LM] = None,
        tools: Optional[list[Callable]] = None,
        postprocess: Optional[Callable] = None,  # Post-processing function
        max_iterations: int = 10,
        **defaults
    ):
        self.signature = signature
        self.system = system
        self.instructions = instructions
        self.lm = lm  # Can be None, will use default
        self._tools = {t.__name__: t for t in (tools or [])}
        self.postprocess = postprocess
        self.max_iterations = max_iterations
        self.defaults = defaults  # Predict-level overrides (temperature, etc.)
        self._history: list[dict] = []

        # Initialize DSPy adapter if signature provided
        self._adapter = None
        self._signature_obj = None
        self._kwarg_remap = {}  # user kwarg name -> internal DSPy field name
        if signature and DSPY_AVAILABLE:
            # DSPy reserves some field names (e.g. "instructions") on Signature.
            # Remap them internally so users can still use natural names.
            _RESERVED = {"instructions": "inst"}
            if isinstance(signature, str):
                import re
                for reserved, alias in _RESERVED.items():
                    if re.search(rf'\b{reserved}\b', signature):
                        signature = re.sub(rf'\b{reserved}\b', alias, signature)
                        self._kwarg_remap[reserved] = alias
            self._signature_obj = ensure_signature(signature)
            self._adapter = ChatAdapter()

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
            raise RuntimeError("No LM configured. Use LM.configure() or pass lm to Predict()")
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

    async def run(self, input=None, system: str = "", history: Optional[list[dict]] = None, **kwargs):
        """Run agent loop, yielding events.

        Args:
            input: String input or list of messages
            system: Additional system prompt (overrides default)
            history: Conversation history to inject [{"role": "user", "content": "..."}, ...]
            **kwargs: Signature fields (query="...", context="...") or LLM params
        """
        # Prepare messages based on mode
        messages = []

        if self._adapter and self._signature_obj:
            # Mode 1: DSPy Signature path
            # Use adapter to format signature → messages
            try:
                # Separate signature fields from LLM params
                signature_fields = {}
                llm_params = {}
                for key, value in kwargs.items():
                    if key in ['temperature', 'seed', 'max_tokens', 'top_p', 'top_k']:
                        llm_params[key] = value
                    else:
                        # Remap reserved names (e.g. instructions -> inst)
                        mapped = self._kwarg_remap.get(key, key)
                        signature_fields[mapped] = value

                # Format with DSPy adapter
                formatted = self._adapter.format(self._signature_obj, [], signature_fields)
                messages = formatted  # [{"role": "system", ...}, {"role": "user", ...}]

                # Override with custom instructions if provided
                if self.instructions and messages:
                    messages[0]["content"] = messages[0]["content"].replace(
                        self._signature_obj.instructions or "",
                        self.instructions
                    )

                # Add extra system prompt if provided
                if self.system and messages:
                    messages[0]["content"] = self.system + "\n\n" + messages[0]["content"]
                elif system and messages:
                    messages[0]["content"] = system + "\n\n" + messages[0]["content"]

                # Inject conversation history between system and user
                if history and len(messages) >= 2:
                    system_msg = messages[0]
                    user_msg = messages[-1]
                    messages = [system_msg] + history + [user_msg]

            except Exception as e:
                # Fallback to simple signature string
                sig_str = str(self.signature)
                system_prompt = self.system or system or f"Follow this task signature: {sig_str}"
                messages.append({"role": "system", "content": system_prompt})

                if history:
                    messages.extend(history)

                # Add user input
                if signature_fields:
                    user_content = "\n".join(f"{k}: {v}" for k, v in signature_fields.items())
                    messages.append({"role": "user", "content": user_content})
                elif input:
                    messages.append({"role": "user", "content": str(input)})

        else:
            # Mode 2: Raw system prompt path
            if self.system or system:
                messages.append({"role": "system", "content": self.system or system})

            if history:
                messages.extend(history)

            # Add current input
            if input:
                if isinstance(input, str):
                    messages.append({"role": "user", "content": input})
                elif isinstance(input, list):
                    messages.extend(input)
            elif kwargs.get('query'):
                # Support query kwarg for raw mode
                messages.append({"role": "user", "content": str(kwargs['query'])})

        # Update internal history
        self._history = messages.copy()

        # Separate LLM params from signature fields
        llm_params = {}
        if self._signature_obj:
            for key in ['temperature', 'seed', 'max_tokens', 'top_p', 'top_k']:
                if key in kwargs:
                    llm_params[key] = kwargs[key]
        else:
            llm_params = kwargs

        iterations = 0
        total_tools = 0
        current_messages = messages.copy()

        for _ in range(self.max_iterations):
            iterations += 1
            assistant_msg = None
            tool_futures = {}

            async for event in self._stream_step(current_messages, **llm_params):
                if event['type'] == 'step_complete':
                    assistant_msg = event['message']
                    tool_futures = event['tool_futures']
                else:
                    yield event

            self._history.append(assistant_msg)
            current_messages.append(assistant_msg)

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
                    current_messages.append(tool_msg)
                    yield {"type": "tool_result", "output": result["output"]}
            else:
                break

        yield {
            "type": "complete",
            "response": self._history[-1].get("content", "") if self._history else "",
            "iterations": iterations,
            "tool_calls_count": total_tools
        }

    async def _arun(self, input=None, system: str = "", history: Optional[list[dict]] = None, return_result: bool = False, **kwargs):
        """Async runner that returns final response as Prediction object."""
        response = ""
        iterations = 0
        tool_calls_count = 0

        async for event in self.run(input, system, history, **kwargs):
            if event['type'] == 'complete':
                response = event['response']
                iterations = event.get('iterations', 0)
                tool_calls_count = event.get('tool_calls_count', 0)

        # Create Prediction object (DSPy-style)
        prediction = Prediction(response, self._signature_obj)

        # Apply post-processing if provided
        if self.postprocess:
            try:
                processed = self.postprocess(prediction)

                # Handle different return types from postprocess
                if isinstance(processed, str):
                    prediction = Prediction(processed, None)
                elif isinstance(processed, Prediction):
                    prediction = processed
                elif hasattr(processed, 'text'):
                    prediction = Prediction(processed.text, None)
                else:
                    prediction = Prediction(str(processed), None)
            except Exception as e:
                print(f"Warning: postprocess failed: {e}")

        if return_result:
            return AgentResult(
                text=str(prediction),
                history=self._history.copy(),
                iterations=iterations,
                tool_calls_count=tool_calls_count
            )

        return prediction

    def __call__(self, input=None, system: str = "", history: Optional[list[dict]] = None, return_result: bool = False, **kwargs):
        """Call predict (sync or async based on context).

        Args:
            input: Either a string prompt or list of message dicts (for raw mode)
            system: System prompt (overrides default)
            history: Conversation history [{"role": "user", "content": "..."}, ...]
            return_result: If True, return AgentResult object with metadata, else return string
            **kwargs: Signature fields (query="...", context="...") or LLM parameters (temperature, seed, etc.)

        Returns:
            str or AgentResult (sync) or coroutine (async)

        Examples:
            # Signature mode
            pred = Predict("query -> answer")
            result = pred(query="What is 2+2?")

            # With history
            result = pred(query="What's my name?", history=[{"role": "user", "content": "I'm Alice"}])

            # Raw system mode
            pred = Predict(system="You are a classifier")
            result = pred(input="Classify this text")
        """
        # Check if we're in async context
        try:
            asyncio.get_running_loop()
            # Already in async context - return coroutine
            return self._arun(input, system, history, return_result, **kwargs)
        except RuntimeError:
            # Not in async context - run sync
            return asyncio.run(self._arun(input, system, history, return_result, **kwargs))

    @property
    def history(self) -> list[dict]:
        """Get conversation history."""
        return self._history

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._history.clear()


class Module:
    """Base class for building LLM pipelines (DSPy-style).

    Enables composable multi-step pipelines with automatic history tracking.

    Example:
        class RAGPipeline(Module):
            def __init__(self):
                super().__init__()
                self.classify = Predict("query -> classification")
                self.answer = Predict("context, query -> answer")

            def forward(self, query):
                classification = self.classify(query=query)
                context = self.retrieve(classification.text)
                answer = self.answer(context=context, query=query)
                return answer.text
    """

    def __init__(self):
        self._compiled = False
        self.call_history = []  # Track all LLM calls

    def __call__(self, *args, **kwargs):
        """Routes to forward() with automatic history tracking."""
        result = self.forward(*args, **kwargs)
        return result

    def forward(self, **kwargs):
        """Override this in subclasses to define pipeline logic."""
        raise NotImplementedError("Subclasses must implement forward()")

    def named_predictors(self):
        """Get all Predict instances as (name, predictor) tuples."""
        predictors = []
        for name in dir(self):
            if not name.startswith('_'):
                attr = getattr(self, name)
                if isinstance(attr, Predict):
                    predictors.append((name, attr))
        return predictors

    def inspect_history(self, predictor_name: Optional[str] = None):
        """Print history of LLM calls.

        Args:
            predictor_name: If provided, show only calls from that predictor
        """
        if predictor_name:
            predictor = getattr(self, predictor_name, None)
            if predictor and isinstance(predictor, Predict):
                print(f"\n=== History for {predictor_name} ===")
                for i, msg in enumerate(predictor.history):
                    print(f"{i+1}. [{msg['role']}]: {msg.get('content', '')[:100]}...")
        else:
            # Show all predictors
            for name, pred in self.named_predictors():
                if pred.history:
                    print(f"\n=== {name} ===")
                    for i, msg in enumerate(pred.history):
                        print(f"{i+1}. [{msg['role']}]: {msg.get('content', '')[:100]}...")

    def reset(self):
        """Clear history from all predictors."""
        for _, pred in self.named_predictors():
            pred.clear_history()
        self.call_history.clear()


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

    def __call__(self, predict: Predict) -> dict:
        """Evaluate predict on dataset with batching and caching."""
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
                predict.clear_history()
                prediction = predict(example['input'])
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


# Re-export DSPy essentials for convenience
if DSPY_AVAILABLE:
    __all__ = [
        'LM', 'Predict', 'Prediction', 'Module', 'Eval', 'AgentResult', 'configure',
        'Signature', 'InputField', 'OutputField'
    ]
else:
    __all__ = [
        'LM', 'Predict', 'Prediction', 'Module', 'Eval', 'AgentResult', 'configure'
    ]


if __name__ == "__main__":
    """Usage examples showcasing signature, system prompts, history, and Module pipelines."""

    # Configure LM once
    print("=== Configuring LM ===")
    lm = LM(
        model="Qwen/Qwen3-4B-Instruct-2507",
        api_base="http://192.168.170.76:8000",
        temperature=0.1
    )
    configure(lm)  # Set global default

    # Example 1: Simple Predict with signature
    print("\n=== Example 1: Simple Predict with signature ===")
    pred = Predict("query -> answer")
    result = pred(query="What is 2+2?")
    print(f"Result: {result}")

    # Example 2: Signature + System Prompt
    print("\n=== Example 2: Signature + System Prompt ===")
    classifier = Predict(
        "query -> classification",
        system="You are an expert query classifier. Return only: SQL or VECTOR"
    )
    result = classifier(query="Show me sales data from last month")
    print(f"Classification: {result}")

    # Example 3: With Conversation History
    print("\n=== Example 3: With Conversation History ===")
    history = [
        {"role": "user", "content": "My name is Alice"},
        {"role": "assistant", "content": "Hello Alice! Nice to meet you."}
    ]
    pred = Predict("query -> answer")
    result = pred(query="What's my name?", history=history)
    print(f"Result: {result}")

    # Example 4: With Post-processing
    print("\n=== Example 4: With Post-processing ===")
    def to_uppercase(pred):
        return pred.text.upper()

    pred_upper = Predict(
        "query -> answer",
        postprocess=to_uppercase
    )
    result = pred_upper(query="Say hello")
    print(f"Result: {result}")

    # Example 5: Raw System Prompt (no signature)
    print("\n=== Example 5: Raw System Prompt ===")
    raw_pred = Predict(system="You are a helpful assistant. Be concise.")
    result = raw_pred(input="What is Python?")
    print(f"Result: {result[:100]}...")

    # Example 6: Module-based RAG Pipeline
    print("\n=== Example 6: Module-based RAG Pipeline ===")

    class SimpleRAG(Module):
        def __init__(self):
            super().__init__()

            # Define pipeline steps
            self.classify = Predict(
                "query -> classification",
                system="Classify as SQL or VECTOR"
            )

            self.refine = Predict(
                "query, classification -> refined_query",
                system="Refine the query for better retrieval"
            )

            self.answer = Predict(
                "context, query -> answer",
                system="Answer based on the provided context"
            )

        def forward(self, query):
            # Step 1: Classify
            c = self.classify(query=query)
            print(f"[Classify] {c}")

            # Step 2: Refine
            r = self.refine(query=query, classification=c)
            print(f"[Refine] {r}")

            # Step 3: Mock retrieval (not LLM)
            if "sql" in c.lower():
                context = "Sales data: Q4 2023 revenue was $1.2M"
            else:
                context = "Customer reviews are generally positive"

            # Step 4: Generate answer
            ans = self.answer(context=context, query=r)
            print(f"[Answer] {ans}")

            return ans

    pipeline = SimpleRAG()
    answer = pipeline(query="Show me sales from Q4")
    print(f"\nFinal Answer: {answer}")

    # Inspect pipeline history
    print("\n=== Pipeline History ===")
    pipeline.inspect_history("classify")

    # Example 7: Eval
    print("\n=== Example 7: Eval ===")
    def exact_match(example, prediction):
        return "4" in prediction.strip()

    dataset = [
        {'input': 'What is 2+2?', 'output': '4'},
    ]

    evaluator = Eval(
        metric=exact_match,
        dataset=dataset,
        save_path="/tmp/eval_simple.json"
    )
    pred_eval = Predict("query -> answer")
    result = evaluator(pred_eval)
    print(f"Score: {result['score']}% ({result['correct']}/{result['total']})")

    print("\n✅ All examples completed!")
