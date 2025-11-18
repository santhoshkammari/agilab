"""
Evaluation framework demo using agent.step() instead of lm.stream()

Shows how to evaluate agent-based modules with tool calling support.
"""

import asyncio
import sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])

from eval import eval_example, eval_batch
from example import Example
from lm import LM


# Tool definitions
def calculator(expression: str) -> str:
    """
    Simple calculator tool for math expressions

    Args:
        expression: Math expression like "2+2" or "5*5"

    Returns:
        Result of the calculation
    """
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def search(query: str) -> str:
    """
    Mock search tool

    Args:
        query: Search query

    Returns:
        Search results
    """
    return f"Results for '{query}': [mock data]"


# Dataset
dev_examples = [
    Example(question="What is 2+2", answer="4"),
    Example(question="What is 5*5", answer="25"),
    Example(question="What is 10-3", answer="7"),
    Example(question="What is 100/10", answer="10.0"),
]


# Metric
def exact_match(example, prediction):
    """Check if prediction matches answer"""
    expected = str(getattr(example, 'answer', '')).strip()
    # Extract numbers from prediction
    import re
    numbers = re.findall(r'\d+(?:\.\d+)?', str(prediction))
    if numbers:
        predicted = numbers[0]
        return 1.0 if predicted == expected else 0.0
    return 0.0


async def demo_step_based_eval():
    """Demo: Evaluate using agent.step() with tools"""
    print("\n" + "="*60)
    print("AGENT-BASED EVALUATION DEMO (Using step())")
    print("="*60)
    print(f"Module: Agent with calculator and search tools")
    print(f"Dataset: {len(dev_examples)} examples")
    print(f"Metric: exact_match")

    lm = LM()
    tools = [calculator, search]

    # Evaluate with step-based approach
    result = await eval_batch(
        module_fn=lambda **inputs: None,  # Not used with use_step=True
        examples=dev_examples[:2],  # Just 2 examples for demo
        metric=exact_match,
        batch_size=2,
        parallel=False,
        progress=True,
        use_step=True,
        lm=lm,
        tools=tools
    )

    print(f"\nScore: {result['score']:.1f}%")
    print(f"Passed: {result['passed']}/{result['total']}")
    print("\nDetailed Results:")
    for r in result['results']:
        status = "✓" if r.score > 0.5 else "✗"
        pred_short = str(r.prediction)[:40] + "..." if len(str(r.prediction)) > 40 else r.prediction
        print(f"  {status} Q: {r.example.question:20} | Expected: {r.example.answer:5} | Score: {r.score:.2f}")


async def demo_single_step_eval():
    """Demo: Single example with step-based evaluation"""
    print("\n" + "="*60)
    print("SINGLE EXAMPLE EVALUATION (Using step())")
    print("="*60)

    lm = LM()
    tools = [calculator, search]

    result = await eval_example(
        module_fn=lambda **inputs: None,  # Not used with use_step=True
        example=dev_examples[0],
        metric=exact_match,
        use_step=True,
        lm=lm,
        tools=tools
    )

    print(f"Example: {dev_examples[0].question}")
    print(f"Expected: {dev_examples[0].answer}")
    print(f"Predicted: {result.prediction[:100]}..." if len(str(result.prediction)) > 100 else f"Predicted: {result.prediction}")
    print(f"Score: {result.score}")


async def demo_stream_step_eval():
    """Demo: Streaming evaluation with step()"""
    print("\n" + "="*60)
    print("STREAMING EVALUATION (Using step())")
    print("="*60)

    from eval import eval_stream

    lm = LM()
    tools = [calculator]

    print("Evaluating examples in streaming mode:\n")
    count = 0
    async for result in eval_stream(
        module_fn=lambda **inputs: None,
        examples=dev_examples[:3],
        metric=exact_match,
        use_step=True,
        lm=lm,
        tools=tools
    ):
        count += 1
        status = "✓" if result.score > 0.5 else "✗"
        print(f"  {status} Example {count}: {result.example.question:25} | Score: {result.score:.2f}")


async def main():
    """Run all step-based demos"""
    print("\n" + "="*60)
    print("EVALUATION WITH agent.step()")
    print("="*60)

    await demo_single_step_eval()
    await demo_stream_step_eval()
    await demo_step_based_eval()

    print("\n" + "="*60)
    print("DEMOS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
