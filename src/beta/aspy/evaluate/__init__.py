"""
aspy.Evaluate - Simplified evaluation module inspired by dspy.Evaluate
"""
import json
from typing import Callable, List, Any, Optional
import tqdm
from ..predict.predict import Module, Prediction
from .example import Example


class EvaluationResult:
    """Container for evaluation results."""

    def __init__(self, score: float, results: List[tuple]):
        self.score = score
        self.results = results

    def __repr__(self):
        return f"EvaluationResult(score={self.score:.2f}%, results=<{len(self.results)} items>)"


class Evaluate:
    """
    Simple evaluation class for aspy modules.

    Usage:
        evaluator = aspy.Evaluate(devset=examples, metric=accuracy_metric)
        result = evaluator(module)
    """

    def __init__(
        self,
        devset: List[Example],
        metric: Callable,
        display_progress: bool = True,
        save_as_json: Optional[str] = None,
        batch_size: int = 4,
        batched: bool = True
    ):
        """
        Args:
            devset: List of Examples to evaluate on
            metric: Function that takes (example, prediction) and returns a score
            display_progress: Whether to show tqdm progress bar
            save_as_json: Optional path to save results as JSON
            batch_size: Size of batches for evaluation (default: 4)
            batched: Whether to use batch processing (default: True)
        """
        self.devset = devset
        self.metric = metric
        self.display_progress = display_progress
        self.save_as_json = save_as_json
        self.batch_size = batch_size
        self.batched = batched

    def __call__(self, module: Module) -> EvaluationResult:
        """
        Evaluate a module on the devset.

        Args:
            module: The aspy Module to evaluate

        Returns:
            EvaluationResult with score and detailed results
        """
        if self.batched and hasattr(module, 'batch_forward'):
            return self._evaluate_batched(module)
        else:
            return self._evaluate_sequential(module)

    def _evaluate_sequential(self, module: Module) -> EvaluationResult:
        """Sequential evaluation (original implementation)."""
        results = []
        total_score = 0.0

        # Create progress bar
        iterator = tqdm.tqdm(self.devset, desc="Evaluating", disable=not self.display_progress)

        for example in iterator:
            try:
                # Get prediction from module
                inputs = example.inputs()
                prediction = module(**inputs)

                # Calculate metric score
                score = self.metric(example, prediction)
                total_score += score

                # Store result
                results.append((example, prediction, score))

                # Update progress bar with current average
                current_avg = total_score / len(results)
                iterator.set_postfix(avg_score=f"{current_avg:.3f}")

            except Exception as e:
                # Handle errors gracefully
                print(f"Error evaluating example: {e}")
                results.append((example, Prediction(error=str(e)), 0.0))

        # Calculate final score as percentage
        final_score = (total_score / len(self.devset)) * 100 if self.devset else 0.0

        print(f"\nEvaluation complete: {final_score:.2f}% ({total_score:.1f}/{len(self.devset)})")

        # Save results if requested
        if self.save_as_json:
            self._save_results_json(results, final_score)

        return EvaluationResult(score=final_score, results=results)

    def _evaluate_batched(self, module: Module) -> EvaluationResult:
        """Batched evaluation for improved performance."""
        results = []
        total_score = 0.0

        # Create batches
        batches = [
            self.devset[i:i + self.batch_size]
            for i in range(0, len(self.devset), self.batch_size)
        ]

        # Create progress bar for batches
        iterator = tqdm.tqdm(batches, desc="Evaluating (batched)", disable=not self.display_progress)

        for batch in iterator:
            try:
                # Prepare batch inputs
                batch_inputs = [example.inputs() for example in batch]

                # Get batch predictions
                predictions = module.batch_forward(batch_inputs)

                # Process batch results
                batch_scores = []
                for example, prediction in zip(batch, predictions):
                    try:
                        score = self.metric(example, prediction)
                        total_score += score
                        batch_scores.append(score)
                        results.append((example, prediction, score))
                    except Exception as e:
                        print(f"Error calculating metric for example: {e}")
                        results.append((example, prediction, 0.0))
                        batch_scores.append(0.0)

                # Update progress bar with current average
                current_avg = total_score / len(results) if results else 0.0
                iterator.set_postfix(avg_score=f"{current_avg:.3f}", batch_avg=f"{sum(batch_scores)/len(batch_scores):.3f}")

            except Exception as e:
                # Handle batch errors gracefully
                print(f"Error evaluating batch: {e}")
                for example in batch:
                    results.append((example, Prediction(error=str(e)), 0.0))

        # Calculate final score as percentage
        final_score = (total_score / len(self.devset)) * 100 if self.devset else 0.0

        print(f"\nEvaluation complete: {final_score:.2f}% ({total_score:.1f}/{len(self.devset)}) [Batched: batch_size={self.batch_size}]")

        # Save results if requested
        if self.save_as_json:
            self._save_results_json(results, final_score)

        return EvaluationResult(score=final_score, results=results)

    def _save_results_json(self, results: List[tuple], score: float):
        """Save evaluation results to JSON file."""
        output_data = {
            "score": score,
            "total_examples": len(results),
            "results": []
        }

        for example, prediction, score in results:
            result_item = {
                "example": example.model_dump() if hasattr(example, 'model_dump') else str(example),
                "prediction": prediction.__dict__ if hasattr(prediction, '__dict__') else str(prediction),
                "score": score
            }
            output_data["results"].append(result_item)

        with open(self.save_as_json, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"Results saved to {self.save_as_json}")


# Common metric functions
def exact_match(example: Example, prediction: Prediction) -> float:
    """Simple exact match metric."""
    expected = getattr(example, 'answer', getattr(example, 'output', ''))
    predicted = getattr(prediction, 'answer', getattr(prediction, 'output', ''))
    return 1.0 if str(expected).strip() == str(predicted).strip() else 0.0


def contains_match(example: Example, prediction: Prediction) -> float:
    """Check if prediction contains the expected answer."""
    expected = str(getattr(example, 'answer', getattr(example, 'output', ''))).strip().lower()
    predicted = str(getattr(prediction, 'answer', getattr(prediction, 'output', ''))).strip().lower()
    return 1.0 if expected in predicted else 0.0