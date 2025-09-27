"""
MIPROv2 - Multi-prompt Instruction Proposal and Refinement Optimizer
Simplified version inspired by dspy.MIPROv2 for aspy
"""
import json
import random
from typing import List, Callable, Optional, Dict, Any
import tqdm
from ..predict.predict import Module, Predict, Prediction
from ..evaluate import Evaluate, Example


class OptimizationResult:
    """Container for optimization results."""

    def __init__(self, optimized_module: Module, score: float, history: List[Dict]):
        self.optimized_module = optimized_module
        self.score = score
        self.history = history

    def __repr__(self):
        return f"OptimizationResult(score={self.score:.2f}%, history={len(self.history)} steps)"


class MIPROv2:
    """
    Simplified MIPROv2 optimizer for aspy.

    Optimizes natural language instructions in signatures to improve performance.
    """

    def __init__(
        self,
        metric: Callable,
        num_candidates: int = 8,
        num_trials: int = 6,
        verbose: bool = True,
        seed: int = 42
    ):
        """
        Args:
            metric: Function that takes (example, prediction) and returns a score
            num_candidates: Number of instruction candidates to generate per trial
            num_trials: Number of optimization rounds
            verbose: Whether to show progress
            seed: Random seed for reproducibility
        """
        self.metric = metric
        self.num_candidates = num_candidates
        self.num_trials = num_trials
        self.verbose = verbose
        self.seed = seed
        random.seed(seed)

    def compile(
        self,
        module: Module,
        trainset: List[Example],
        valset: List[Example]
    ) -> OptimizationResult:
        """
        Optimize the module's instructions using MIPROv2 approach.

        Args:
            module: The aspy module to optimize
            trainset: Training examples for generating candidates
            valset: Validation examples for evaluation

        Returns:
            OptimizationResult with optimized module and performance history
        """
        if self.verbose:
            print("🔄 Starting MIPROv2 optimization...")

        best_module = module
        best_score = self._evaluate_module(best_module, valset)
        history = [{"trial": 0, "score": best_score, "instruction": "original"}]

        if self.verbose:
            print(f"📊 Baseline score: {best_score:.2f}%")

        # Optimization loop
        for trial in range(1, self.num_trials + 1):
            if self.verbose:
                print(f"\n🧪 Trial {trial}/{self.num_trials}: Generating instruction candidates...")

            # Generate instruction candidates
            candidates = self._generate_instruction_candidates(module, trainset)

            # Evaluate all candidates
            best_trial_score = best_score
            best_trial_module = best_module
            best_trial_instruction = "no improvement"

            iterator = tqdm.tqdm(candidates, desc=f"Trial {trial}", disable=not self.verbose)

            for i, instruction in enumerate(iterator):
                # Create module with new instruction
                candidate_module = self._create_module_with_instruction(module, instruction)

                # Evaluate candidate
                score = self._evaluate_module(candidate_module, valset)

                # Update progress
                iterator.set_postfix(best=f"{best_trial_score:.1f}%", current=f"{score:.1f}%")

                # Track best candidate
                if score > best_trial_score:
                    best_trial_score = score
                    best_trial_module = candidate_module
                    best_trial_instruction = instruction[:50] + "..." if len(instruction) > 50 else instruction

            # Update overall best
            if best_trial_score > best_score:
                best_score = best_trial_score
                best_module = best_trial_module
                if self.verbose:
                    print(f"✅ New best score: {best_score:.2f}% (improved by {best_trial_score - history[-1]['score']:.2f}%)")
            else:
                if self.verbose:
                    print(f"❌ No improvement in trial {trial}")

            history.append({
                "trial": trial,
                "score": best_trial_score,
                "instruction": best_trial_instruction
            })

        if self.verbose:
            print(f"\n🎉 Optimization complete! Final score: {best_score:.2f}%")

        return OptimizationResult(
            optimized_module=best_module,
            score=best_score,
            history=history
        )

    def _evaluate_module(self, module: Module, valset: List[Example]) -> float:
        """Evaluate module performance on validation set."""
        evaluator = Evaluate(
            devset=valset,
            metric=self.metric,
            display_progress=False
        )
        result = evaluator(module)
        return result.score

    def _generate_instruction_candidates(self, module: Module, trainset: List[Example]) -> List[str]:
        """Generate instruction candidates using the trainset for context."""

        # Sample some training examples for context
        sample_examples = random.sample(trainset, min(3, len(trainset)))

        # Create instruction proposer
        proposer = Predict("examples, current_task -> improved_instruction")

        candidates = []

        # Create context from examples
        examples_text = "\n".join([
            f"Input: {ex.inputs()}\nExpected: {getattr(ex, 'answer', getattr(ex, 'output', 'N/A'))}"
            for ex in sample_examples
        ])

        # Generate multiple candidates with different approaches
        base_prompts = [
            "Look at these examples and generate a clearer, more specific instruction for this task.",
            "Based on these examples, create a more detailed instruction that guides the model better.",
            "Analyze these examples and propose an improved instruction with better clarity and specificity.",
            "Given these examples, write a more effective instruction that will improve accuracy.",
            "Study these examples and create a more precise instruction for better performance.",
        ]

        for i in range(self.num_candidates):
            base_prompt = base_prompts[i % len(base_prompts)]

            try:
                # Generate candidate instruction
                result = proposer(
                    examples=examples_text,
                    current_task=base_prompt
                )

                instruction = getattr(result, 'improved_instruction', f"Generated instruction {i+1}")
                candidates.append(instruction)

            except Exception as e:
                # Fallback instruction
                candidates.append(f"Analyze the input carefully and provide an accurate response for this task (candidate {i+1}).")

        return candidates

    def _create_module_with_instruction(self, original_module: Module, new_instruction: str) -> Module:
        """Create a new module with modified instruction."""

        if hasattr(original_module, 'signature'):
            # Get current signature string
            original_sig = original_module.signature

            # Create new signature with instruction
            if hasattr(original_sig, 'instructions'):
                # Update existing signature
                original_sig.instructions = new_instruction
                new_module = type(original_module)(original_sig)
            else:
                # Create new module with instruction in signature
                new_module = type(original_module)(original_sig)
                if hasattr(new_module, 'signature'):
                    new_module.signature.instructions = new_instruction
        else:
            # Fallback: clone the original module
            new_module = original_module

        return new_module