"""
Minimal evaluation framework - parallel to LM.stream() and LM.batch()

Supports two evaluation modes:
1. Direct module evaluation: eval_example(module_fn, example, metric)
2. Agent-based evaluation: eval_example(module_fn, example, metric, use_step=True, lm=..., tools=...)
"""

import asyncio
from dataclasses import dataclass, field
from typing import Callable, List, AsyncIterator, Any, Optional
import tqdm.asyncio
import tqdm


@dataclass
class EvalResult:
    """Result for ONE example evaluation"""
    example: Any
    prediction: Any
    score: float
    metadata: dict = field(default_factory=dict)

    def __repr__(self):
        return f"EvalResult(score={self.score:.2f}, metadata={self.metadata})"


async def eval_example(
    module_fn: Callable,
    example: Any,
    metric: Callable,
    use_step: bool = False,
    lm: Any = None,
    tools: List[Callable] = None,
    **kwargs
) -> EvalResult:
    """
    Evaluate ONE example.

    Two modes:
    1. Direct: Just call module_fn with example inputs
    2. Agent (use_step=True): Use agent.step() for tool-based evaluation

    Args:
        module_fn: Callable(**example.inputs()) → prediction
        example: Single Example to evaluate
        metric: Callable(example, prediction) → float [0-1]
        use_step: If True, use agent.step() instead of direct call
        lm: LM instance (required if use_step=True)
        tools: List of tools (optional for step mode)
        **kwargs: Extra params passed to module

    Returns:
        EvalResult with prediction and score
    """
    try:
        # Get inputs from example
        if hasattr(example, 'inputs'):
            inputs = example.inputs()
        elif isinstance(example, dict):
            inputs = {k: v for k, v in example.items() if k not in ['answer', 'output', 'target', 'label']}
        else:
            inputs = {}

        # Agent-based evaluation mode
        if use_step:
            if not lm:
                raise ValueError("lm is required for use_step=True")

            from .agent import step

            # Build message from inputs
            user_content = " ".join([f"{k}: {v}" for k, v in inputs.items()])
            history = [{"role": "user", "content": user_content}]

            # Single step evaluation
            result = await step(lm=lm, history=history, tools=tools)

            # Extract content as prediction
            pred = result.message.get("content", "")

            # If there are tool calls, wait for results and include them
            if result.tool_calls:
                try:
                    tool_results = await result.tool_results()
                    # Combine content and tool results
                    tool_outputs = [f"{tr.tool_call_id}: {tr.output}" for tr in tool_results]
                    if pred:
                        pred = f"{pred}\n" + "\n".join(tool_outputs)
                    else:
                        pred = "\n".join(tool_outputs)
                except Exception as tool_err:
                    # If tool execution fails, just use content
                    pass
        else:
            # Direct module evaluation
            pred = module_fn(**inputs, **kwargs)

            # Handle async functions
            if asyncio.iscoroutine(pred):
                pred = await pred

        # Calculate metric
        score = metric(example, pred)

        return EvalResult(
            example=example,
            prediction=pred,
            score=score,
            metadata={}
        )
    except Exception as e:
        return EvalResult(
            example=example,
            prediction=None,
            score=0.0,
            metadata={"error": str(e)}
        )


async def eval_stream(
    module_fn: Callable,
    examples: List[Any],
    metric: Callable,
    use_step: bool = False,
    lm: Any = None,
    tools: List[Callable] = None,
    **kwargs
) -> AsyncIterator[EvalResult]:
    """
    Stream evaluation results one-by-one.

    Parallel to LM.stream() - yields results as they complete.
    Useful for real-time monitoring and early stopping.

    Args:
        module_fn: Callable(**example.inputs()) → prediction
        examples: List of Examples
        metric: Callable(example, prediction) → float
        use_step: If True, use agent.step()
        lm: LM instance (required if use_step=True)
        tools: List of tools (optional for step mode)
        **kwargs: Extra params passed to module

    Yields:
        EvalResult for each example

    Usage:
        async for result in eval_stream(module, examples, metric):
            print(f"Score: {result.score}")
            if result.score < 0.5:
                break  # Early stop
    """
    for example in examples:
        result = await eval_example(
            module_fn, example, metric,
            use_step=use_step, lm=lm, tools=tools,
            **kwargs
        )
        yield result


async def eval_batch(
    module_fn: Callable,
    examples: List[Any],
    metric: Callable,
    batch_size: int = 4,
    parallel: bool = False,
    progress: bool = True,
    use_step: bool = False,
    lm: Any = None,
    tools: List[Callable] = None,
    **kwargs
) -> dict:
    """
    Batch evaluate examples.

    Parallel to LM.batch() - processes multiple examples together.
    Can be sequential or concurrent.

    Args:
        module_fn: Callable(**example.inputs()) → prediction
        examples: List of Examples
        metric: Callable(example, prediction) → float
        batch_size: Size for concurrent batches (if parallel=True)
        parallel: If True, evaluate batch_size examples concurrently
        progress: Show progress bar
        use_step: If True, use agent.step()
        lm: LM instance (required if use_step=True)
        tools: List of tools (optional for step mode)
        **kwargs: Extra params passed to module

    Returns:
        {
            "score": float (0-100%),
            "passed": int (count where score > 0.5),
            "total": int,
            "results": List[EvalResult]
        }

    Usage:
        result = await eval_batch(
            module_fn=my_module,
            examples=dev_set,
            metric=exact_match,
            batch_size=8,
            parallel=True
        )
        print(f"Score: {result['score']:.1f}%")
    """
    results = []
    scores = []

    if parallel:
        # Process in concurrent batches
        async def process_batch(batch):
            tasks = [
                eval_example(module_fn, ex, metric,
                           use_step=use_step, lm=lm, tools=tools,
                           **kwargs)
                for ex in batch
            ]
            return await asyncio.gather(*tasks)

        # Create batches
        batches = [
            examples[i:i + batch_size]
            for i in range(0, len(examples), batch_size)
        ]

        # Process all batches concurrently
        pbar = tqdm.asyncio.tqdm(batches, desc="Evaluating") if progress else batches
        async for batch_results in pbar:
            batch_results = await process_batch(batch_results)
            results.extend(batch_results)
            scores.extend([r.score for r in batch_results])
    else:
        # Sequential processing
        iterator = tqdm.tqdm(examples, desc="Evaluating") if progress else examples
        for example in iterator:
            result = await eval_example(module_fn, example, metric,
                                      use_step=use_step, lm=lm, tools=tools,
                                      **kwargs)
            results.append(result)
            scores.append(result.score)

    final_score = (sum(scores) / len(scores) * 100) if scores else 0.0
    passed = sum(1 for s in scores if s > 0.5)

    return {
        "score": final_score,
        "passed": passed,
        "total": len(examples),
        "results": results
    }
