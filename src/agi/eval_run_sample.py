"""
Sample evaluation run demonstrating eval framework with real LM
Uses history-based evaluation with agent.step()
"""

import asyncio
import sys
import re
sys.path.insert(0, __file__.rsplit('/', 1)[0])

from eval import eval_example, eval_stream, eval_batch
from lm import LM


# Metric functions
def exact_match(target, prediction):
    """Check if prediction matches expected answer"""
    expected = str(target).strip()
    # Extract first number from prediction
    match = re.search(r'\d+(?:\.\d+)?', str(prediction))
    if match:
        predicted = match.group(0)
        return 1.0 if expected == predicted else 0.0
    return 0.0


def contains_match(target, prediction):
    """Check if prediction contains expected answer"""
    expected = str(target).strip().lower()
    predicted = str(prediction).strip().lower()
    return 1.0 if expected in predicted or predicted in expected else 0.0


# Create sample dataset with histories and targets
dev_data = [
    {
        "history": [{"role": "user", "content": "Answer with ONLY the number. Q: What is 2+2"}],
        "target": "4"
    },
    {
        "history": [{"role": "user", "content": "Answer with ONLY the number. Q: What is 5+5"}],
        "target": "10"
    },
    {
        "history": [{"role": "user", "content": "Answer with ONLY the number. Q: What is 3*3"}],
        "target": "9"
    },
    {
        "history": [{"role": "user", "content": "Answer with ONLY the number. Q: What is 10-2"}],
        "target": "8"
    },
    {
        "history": [{"role": "user", "content": "Answer with ONLY the number. Q: What is 100/10"}],
        "target": "10"
    },
    {
        "history": [{"role": "user", "content": "Answer with ONLY the number. Q: What is 7+7"}],
        "target": "14"
    },
    {
        "history": [{"role": "user", "content": "Answer with ONLY the number. Q: What is 9*9"}],
        "target": "81"
    },
]


async def demo_single_eval():
    """Demo 1: Evaluate single example"""
    print("\n" + "="*60)
    print("DEMO 1: Single Example Evaluation")
    print("="*60)

    lm = LM()
    history = dev_data[0]["history"]
    target = dev_data[0]["target"]

    result = await eval_example(
        history=history,
        target=target,
        metric=exact_match,
        lm=lm,
        use_agent=False
    )

    print(f"Question: {history[0]['content']}")
    print(f"Expected: {target}")
    print(f"Predicted: {result.prediction}")
    print(f"Score: {result.score}")
    print(f"Metadata: {result.metadata}")


async def demo_stream_eval():
    """Demo 2: Stream evaluation with early stopping"""
    print("\n" + "="*60)
    print("DEMO 2: Streaming Evaluation (Real-time Results)")
    print("="*60)

    lm = LM()
    histories = [d["history"] for d in dev_data[:4]]
    targets = [d["target"] for d in dev_data[:4]]

    results = []
    passed = 0
    async for result in eval_stream(
        histories=histories,
        targets=targets,
        metric=exact_match,
        lm=lm,
        use_agent=False
    ):
        results.append(result)
        if result.score > 0.5:
            passed += 1
        question = result.history[0]["content"][:40] + "..." if len(result.history[0]["content"]) > 40 else result.history[0]["content"]
        pred_str = str(result.prediction)[:10]
        print(f"  Q: {question:43} | Pred: {pred_str:10} | Score: {result.score:.2f}")

    avg_score = (sum([r.score for r in results]) / len(results) * 100) if results else 0.0
    print(f"\nRunning Average: {avg_score:.1f}% ({passed}/{len(results)})")


async def demo_batch_sequential():
    """Demo 3: Sequential batch evaluation"""
    print("\n" + "="*60)
    print("DEMO 3: Sequential Batch Evaluation")
    print("="*60)

    lm = LM()
    histories = [d["history"] for d in dev_data]
    targets = [d["target"] for d in dev_data]

    result = await eval_batch(
        histories=histories,
        targets=targets,
        metric=exact_match,
        lm=lm,
        batch_size=4,
        parallel=False,
        progress=True,
        use_agent=False
    )

    print(f"\nFinal Score: {result['score']:.1f}%")
    print(f"Passed: {result['passed']}/{result['total']}")
    print("\nDetailed Results:")
    for i, r in enumerate(result['results']):
        status = "✓" if r.score > 0.5 else "✗"
        question = r.history[0]["content"].split("Q: ")[-1] if "Q: " in r.history[0]["content"] else r.history[0]["content"]
        pred_str = str(r.prediction)[:10]
        print(f"  {status} {question:25} → {pred_str:10} (score: {r.score:.2f})")


async def demo_batch_parallel():
    """Demo 4: Parallel batch evaluation"""
    print("\n" + "="*60)
    print("DEMO 4: Parallel Batch Evaluation")
    print("="*60)

    lm = LM()
    histories = [d["history"] for d in dev_data]
    targets = [d["target"] for d in dev_data]

    result = await eval_batch(
        histories=histories,
        targets=targets,
        metric=exact_match,
        lm=lm,
        batch_size=2,
        parallel=True,
        progress=True,
        use_agent=False
    )

    print(f"\nFinal Score: {result['score']:.1f}%")
    print(f"Passed: {result['passed']}/{result['total']}")
    print("\nDetailed Results:")
    for i, r in enumerate(result['results']):
        status = "✓" if r.score > 0.5 else "✗"
        question = r.history[0]["content"].split("Q: ")[-1] if "Q: " in r.history[0]["content"] else r.history[0]["content"]
        pred_str = str(r.prediction)[:10]
        print(f"  {status} {question:25} → {pred_str:10} (score: {r.score:.2f})")


async def demo_multiple_metrics():
    """Demo 5: Compare multiple metrics"""
    print("\n" + "="*60)
    print("DEMO 5: Multiple Metrics Comparison")
    print("="*60)

    lm = LM()
    histories = [d["history"] for d in dev_data]
    targets = [d["target"] for d in dev_data]

    result_exact = await eval_batch(
        histories=histories,
        targets=targets,
        metric=exact_match,
        lm=lm,
        parallel=False,
        progress=False,
        use_agent=False
    )

    result_contains = await eval_batch(
        histories=histories,
        targets=targets,
        metric=contains_match,
        lm=lm,
        parallel=False,
        progress=False,
        use_agent=False
    )

    print(f"Exact Match:     {result_exact['score']:.1f}% ({result_exact['passed']}/{result_exact['total']})")
    print(f"Contains Match:  {result_contains['score']:.1f}% ({result_contains['passed']}/{result_contains['total']})")


async def main():
    """Run all demos"""
    print("\n" + "="*60)
    print("EVALUATION FRAMEWORK DEMO")
    print("="*60)
    print(f"Evaluation Mode: History-based with agent.step()")
    print(f"Dataset: {len(dev_data)} examples")
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
