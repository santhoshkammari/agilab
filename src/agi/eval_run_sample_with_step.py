"""
Evaluation framework demo using agent.step() and agent.agent()

Shows how to evaluate using conversation histories with tool calling support.
"""

import asyncio
import sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])

from eval import eval_example, eval_batch, eval_stream
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


# Dataset - now as histories and targets
dev_data = [
    {
        "history": [{"role": "user", "content": "What is 2+2?"}],
        "target": "4"
    },
    {
        "history": [{"role": "user", "content": "What is 5*5?"}],
        "target": "25"
    },
    {
        "history": [{"role": "user", "content": "What is 10-3?"}],
        "target": "7"
    },
    {
        "history": [{"role": "user", "content": "What is 100/10?"}],
        "target": "10.0"
    },
]


# Metric
def exact_match(target, prediction):
    """Check if prediction matches target"""
    expected = str(target).strip()
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
    print(f"Dataset: {len(dev_data)} examples")
    print(f"Metric: exact_match")

    lm = LM()
    tools = [calculator, search]

    # Extract histories and targets
    histories = [d["history"] for d in dev_data[:2]]
    targets = [d["target"] for d in dev_data[:2]]

    # Evaluate with step-based approach
    result = await eval_batch(
        histories=histories,
        targets=targets,
        metric=exact_match,
        lm=lm,
        tools=tools,
        batch_size=2,
        parallel=False,
        progress=True,
        use_agent=False  # Single step mode
    )

    print(f"\nScore: {result['score']:.1f}%")
    print(f"Passed: {result['passed']}/{result['total']}")
    print("\nDetailed Results:")
    for i, r in enumerate(result['results']):
        status = "✓" if r.score > 0.5 else "✗"
        question = r.history[0]["content"]
        target = targets[i]
        pred_short = str(r.prediction)[:40] + "..." if len(str(r.prediction)) > 40 else r.prediction
        print(f"  {status} Q: {question:30} | Expected: {target:5} | Score: {r.score:.2f}")


async def demo_single_step_eval():
    """Demo: Single example with step-based evaluation"""
    print("\n" + "="*60)
    print("SINGLE EXAMPLE EVALUATION (Using step())")
    print("="*60)

    lm = LM()
    tools = [calculator, search]

    history = dev_data[0]["history"]
    target = dev_data[0]["target"]

    result = await eval_example(
        history=history,
        target=target,
        metric=exact_match,
        lm=lm,
        tools=tools,
        use_agent=False
    )

    print(f"Question: {history[0]['content']}")
    print(f"Expected: {target}")
    print(f"Predicted: {result.prediction[:100]}..." if len(str(result.prediction)) > 100 else f"Predicted: {result.prediction}")
    print(f"Score: {result.score}")
    print(f"Metadata: {result.metadata}")


async def demo_stream_step_eval():
    """Demo: Streaming evaluation with step()"""
    print("\n" + "="*60)
    print("STREAMING EVALUATION (Using step())")
    print("="*60)

    lm = LM()
    tools = [calculator]

    # Extract histories and targets
    histories = [d["history"] for d in dev_data[:3]]
    targets = [d["target"] for d in dev_data[:3]]

    print("Evaluating examples in streaming mode:\n")
    count = 0
    async for result in eval_stream(
        histories=histories,
        targets=targets,
        metric=exact_match,
        lm=lm,
        tools=tools,
        use_agent=False
    ):
        count += 1
        status = "✓" if result.score > 0.5 else "✗"
        question = result.history[0]["content"]
        print(f"  {status} Example {count}: {question:35} | Score: {result.score:.2f}")


async def demo_agent_loop_eval():
    """Demo: Evaluate using full agent loop with multiple iterations"""
    print("\n" + "="*60)
    print("AGENT LOOP EVALUATION (Using agent())")
    print("="*60)

    lm = LM()
    tools = [calculator, search]

    # More complex history that might need multiple steps
    history = [{"role": "user", "content": "What is 2+2? Then multiply that by 5."}]
    target = "20"

    result = await eval_example(
        history=history,
        target=target,
        metric=exact_match,
        lm=lm,
        tools=tools,
        use_agent=True,  # Use full agent loop
        max_iterations=5
    )

    print(f"Question: {history[0]['content']}")
    print(f"Expected: {target}")
    print(f"Predicted: {result.prediction}")
    print(f"Score: {result.score}")
    print(f"Metadata: {result.metadata}")


async def main():
    """Run all demos"""
    print("\n" + "="*60)
    print("EVALUATION FRAMEWORK DEMOS")
    print("="*60)

    await demo_single_step_eval()
    await demo_stream_step_eval()
    await demo_step_based_eval()
    await demo_agent_loop_eval()

    print("\n" + "="*60)
    print("DEMOS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
