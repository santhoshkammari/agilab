"""
Minimal evaluation framework - parallel to LM.stream() and LM.batch()
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
    **kwargs
) -> EvalResult:
    """
    Evaluate ONE example.

    Args:
        module_fn: Callable(**example.inputs()) → prediction
        example: Single Example to evaluate
        metric: Callable(example, prediction) → float [0-1]
        **kwargs: Extra params (temperature, top_p, etc) passed to module

    Returns:
        EvalResult with prediction and score
    """
    try:
        # Get inputs from example
        if hasattr(example, 'inputs'):
            inputs = example.inputs()
        elif isinstance(example, dict):
            # If example is dict, use it directly
            inputs = {k: v for k, v in example.items() if k not in ['answer', 'output', 'target', 'label']}
        else:
            inputs = {}

        # Call module with example inputs
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
        result = await eval_example(module_fn, example, metric, **kwargs)
        yield result


async def eval_batch(
    module_fn: Callable,
    examples: List[Any],
    metric: Callable,
    batch_size: int = 4,
    parallel: bool = False,
    progress: bool = True,
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
                eval_example(module_fn, ex, metric, **kwargs)
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
            result = await eval_example(module_fn, example, metric, **kwargs)
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
