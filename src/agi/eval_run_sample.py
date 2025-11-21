"""
Sample evaluation run demonstrating eval framework with real LM
"""

import asyncio
import sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])

from eval import eval_example, eval_stream, eval_batch
from example import Example
from lm import LM


# Real LM-based QA module
class LMQAModule:
    def __init__(self):
        self.lm = LM()

    async def forward(self, question: str) -> str:
        """Process question through LM"""
        try:
            messages = [
                {
                    "role": "user",
                    "content": f"Answer with ONLY the number. Q: {question}"
                }
            ]

            # Use stream to get response
            response_text = ""
            async for chunk in self.lm.stream(messages):
                if "choices" in chunk and chunk["choices"]:
                    delta = chunk["choices"][0].get("delta", {})
                    if "content" in delta:
                        response_text += delta["content"]

            # Extract just the final number (after </think> tag if present)
            if "</think>" in response_text:
                response_text = response_text.split("</think>")[-1]

            # Extract first number from response
            import re
            match = re.search(r'\d+', response_text)
            if match:
                return match.group(0)

            return response_text.strip() if response_text else "unknown"
        except Exception as e:
            print(f"Error calling LM: {e}")
            return "error"


# Create sample module
module = LMQAModule()


# Metric function
def exact_match(example, prediction):
    """Check if prediction matches expected answer"""
    expected = str(getattr(example, 'answer', '')).strip()
    predicted = str(prediction).strip()
    return 1.0 if expected == predicted else 0.0


def contains_match(example, prediction):
    """Check if prediction contains expected answer"""
    expected = str(getattr(example, 'answer', '')).strip().lower()
    predicted = str(prediction).strip().lower()
    return 1.0 if expected in predicted or predicted in expected else 0.0


# Create sample dataset
dev_examples = [
    Example(question="What is 2+2", answer="4"),
    Example(question="What is 5+5", answer="10"),
    Example(question="What is 3*3", answer="9"),
    Example(question="What is 10-2", answer="8"),
    Example(question="What is 100/10", answer="10"),
    Example(question="What is 7+7", answer="14"),  # This will fail
    Example(question="What is 9*9", answer="81"),  # This will fail
]


async def demo_single_eval():
    """Demo 1: Evaluate single example"""
    print("\n" + "="*60)
    print("DEMO 1: Single Example Evaluation")
    print("="*60)

    result = await eval_example(
        module_fn=module.forward,
        example=dev_examples[0],
        metric=exact_match
    )
    print(f"Example: {dev_examples[0].question}")
    print(f"Expected: {dev_examples[0].answer}")
    print(f"Predicted: {result.prediction}")
    print(f"Score: {result.score}")


async def demo_stream_eval():
    """Demo 2: Stream evaluation with early stopping"""
    print("\n" + "="*60)
    print("DEMO 2: Streaming Evaluation (Real-time Results)")
    print("="*60)

    results = []
    passed = 0
    async for result in eval_stream(
        module_fn=module.forward,
        examples=dev_examples[:4],
        metric=exact_match
    ):
        results.append(result)
        if result.score > 0.5:
            passed += 1
        print(f"  Q: {result.example.question:25} | Pred: {result.prediction:10} | Score: {result.score:.2f}")

    avg_score = (sum([r.score for r in results]) / len(results) * 100) if results else 0.0
    print(f"\nRunning Average: {avg_score:.1f}% ({passed}/{len(results)})")


async def demo_batch_sequential():
    """Demo 3: Sequential batch evaluation"""
    print("\n" + "="*60)
    print("DEMO 3: Sequential Batch Evaluation")
    print("="*60)

    result = await eval_batch(
        module_fn=module.forward,
        examples=dev_examples,
        metric=exact_match,
        batch_size=4,
        parallel=False,
        progress=True
    )

    print(f"\nFinal Score: {result['score']:.1f}%")
    print(f"Passed: {result['passed']}/{result['total']}")
    print("\nDetailed Results:")
    for r in result['results']:
        status = "✓" if r.score > 0.5 else "✗"
        print(f"  {status} {r.example.question:25} → {r.prediction:10} (score: {r.score:.2f})")


async def demo_batch_parallel():
    """Demo 4: Parallel batch evaluation"""
    print("\n" + "="*60)
    print("DEMO 4: Parallel Batch Evaluation")
    print("="*60)

    result = await eval_batch(
        module_fn=module.forward,
        examples=dev_examples,
        metric=exact_match,
        batch_size=2,
        parallel=True,
        progress=True
    )

    print(f"\nFinal Score: {result['score']:.1f}%")
    print(f"Passed: {result['passed']}/{result['total']}")
    print("\nDetailed Results:")
    for r in result['results']:
        status = "✓" if r.score > 0.5 else "✗"
        print(f"  {status} {r.example.question:25} → {r.prediction:10} (score: {r.score:.2f})")


async def demo_multiple_metrics():
    """Demo 5: Compare multiple metrics"""
    print("\n" + "="*60)
    print("DEMO 5: Multiple Metrics Comparison")
    print("="*60)

    result_exact = await eval_batch(
        module_fn=module.forward,
        examples=dev_examples,
        metric=exact_match,
        parallel=False,
        progress=False
    )

    result_contains = await eval_batch(
        module_fn=module.forward,
        examples=dev_examples,
        metric=contains_match,
        parallel=False,
        progress=False
    )

    print(f"Exact Match:     {result_exact['score']:.1f}% ({result_exact['passed']}/{result_exact['total']})")
    print(f"Contains Match:  {result_contains['score']:.1f}% ({result_contains['passed']}/{result_contains['total']})")


async def main():
    """Run all demos"""
    print("\n" + "="*60)
    print("EVALUATION FRAMEWORK DEMO")
    print("="*60)
    print(f"Module: SimpleQAModule")
    print(f"Dataset: {len(dev_examples)} examples")
    print(f"Metrics: exact_match, contains_match")

    await demo_single_eval()
    await demo_stream_eval()
    await demo_batch_sequential()
    await demo_batch_parallel()
    await demo_multiple_metrics()

    print("\n" + "="*60)
    print("ALL DEMOS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
