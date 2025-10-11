"""
GEPA - Genetic Evolution of Prompts and Architectures
Simplified version inspired by dspy.GEPA for aspy
"""
import json
import random
from typing import List, Callable, Optional, Dict, Any, Tuple
import tqdm
from ..predict.predict import Module, Predict, Prediction
from ..lm.lm import LM
from ..evaluate import Evaluate, Example


class GEPAResult:
    """Container for GEPA optimization results."""

    def __init__(
        self,
        best_module: Module,
        best_score: float,
        candidates: List[Dict],
        generations: List[Dict]
    ):
        self.best_module = best_module
        self.best_score = best_score
        self.candidates = candidates
        self.generations = generations

    def __repr__(self):
        return f"GEPAResult(score={self.best_score:.2f}%, generations={len(self.generations)})"


class GEPA:
    """
    GEPA optimizer for aspy - aligned with official research paper.

    Uses Genetic-Pareto evolutionary approach with reflective prompt mutation
    and Pareto-based candidate selection for sample-efficient optimization.
    """

    def __init__(
        self,
        metric: Callable,
        feedback_function: Optional[Callable] = None,
        reflection_lm: Optional[Any] = None,
        minibatch_size: int = 3,
        pareto_set_size: int = 300,
        budget: int = 1000,
        verbose: bool = True,
        seed: int = 42
    ):
        """
        Args:
            metric: Function that takes (example, prediction) and returns a score
            feedback_function: Function that returns feedback traces along with scores
            reflection_lm: Optional separate LM for reflective analysis (cost optimization)
            minibatch_size: Size of minibatch for mutation evaluation
            pareto_set_size: Size of Pareto validation set
            budget: Total rollout budget
            verbose: Whether to show progress
            seed: Random seed for reproducibility
        """
        self.metric = metric
        self.feedback_function = feedback_function or self._default_feedback
        self.reflection_lm = reflection_lm
        self.minibatch_size = minibatch_size
        self.pareto_set_size = pareto_set_size
        self.budget = budget
        self.verbose = verbose
        self.seed = seed
        self.rollouts_used = 0
        random.seed(seed)

    def compile(
        self,
        module: Module,
        trainset: List[Example],
        valset: Optional[List[Example]] = None
    ) -> GEPAResult:
        """
        Optimize the module using GEPA algorithm from paper.

        Args:
            module: The aspy module to optimize
            trainset: Training examples (will be split into Dfeedback/Dpareto)
            valset: Optional validation set (if None, uses part of trainset)

        Returns:
            GEPAResult with optimized module and evolution history
        """
        if self.verbose:
            print("🧬 Starting GEPA optimization...")

        # Split dataset according to paper (Dfeedback, Dpareto)
        dfeedback, dpareto = self._split_dataset(trainset, valset)

        # Initialize candidates and tracking structures
        candidates = [module]  # P in paper
        parents = [None]       # A in paper
        scores_matrix = []     # S in paper
        all_candidates = []

        # Evaluate initial candidate on Dpareto
        initial_scores = []
        for example in dpareto:
            score = self._evaluate_single(module, example)
            initial_scores.append(score)
            self.rollouts_used += 1

        scores_matrix.append(initial_scores)

        if self.verbose:
            avg_score = sum(initial_scores) / len(initial_scores) if initial_scores else 0
            print(f"🏆 Initial score: {avg_score:.2f}%")

        # Main optimization loop - Algorithm 1 from paper
        iteration = 0
        while self.rollouts_used < self.budget:
            iteration += 1
            if self.verbose:
                print(f"\n🔄 Iteration {iteration} (rollouts: {self.rollouts_used}/{self.budget})")

            # Step 7: Select candidate using Pareto-based selection
            k = self._select_candidate(candidates, scores_matrix)
            selected_module = candidates[k]

            # Step 8: Select module to optimize (round-robin for simplicity)
            j = self._select_module(selected_module)

            # Step 9: Sample minibatch from Dfeedback
            minibatch = random.sample(dfeedback, min(self.minibatch_size, len(dfeedback)))

            # Step 10-11: Gather feedback and update prompt
            new_module = self._reflective_prompt_mutation(selected_module, minibatch, j)

            # Step 12-13: Evaluate on minibatch
            old_scores = [self._evaluate_single(selected_module, ex) for ex in minibatch]
            new_scores = [self._evaluate_single(new_module, ex) for ex in minibatch]
            self.rollouts_used += len(minibatch) * 2

            avg_old = sum(old_scores) / len(old_scores) if old_scores else 0
            avg_new = sum(new_scores) / len(new_scores) if new_scores else 0

            # Step 14-18: If improved, add to candidates and evaluate on Dpareto
            if avg_new > avg_old:
                candidates.append(new_module)
                parents.append(k)

                # Evaluate on full Dpareto set
                pareto_scores = []
                for example in dpareto:
                    if self.rollouts_used >= self.budget:
                        break
                    score = self._evaluate_single(new_module, example)
                    pareto_scores.append(score)
                    self.rollouts_used += 1

                scores_matrix.append(pareto_scores)

                if self.verbose:
                    avg_pareto = sum(pareto_scores) / len(pareto_scores) if pareto_scores else 0
                    print(f"✅ Improved! New candidate score: {avg_pareto:.2f}%")
            else:
                if self.verbose:
                    print(f"❌ No improvement: {avg_old:.2f}% -> {avg_new:.2f}%")

            # Track all attempts
            all_candidates.append({
                "module": new_module,
                "parent": k,
                "iteration": iteration,
                "minibatch_improvement": avg_new > avg_old,
                "avg_score": avg_new
            })

        # Step 21: Return best candidate based on average score on Dpareto
        if not scores_matrix:
            best_module = module
            best_score = 0.0
        else:
            avg_scores = [sum(scores)/len(scores) for scores in scores_matrix if scores]
            if avg_scores:
                best_idx = max(range(len(avg_scores)), key=lambda i: avg_scores[i])
                best_module = candidates[best_idx]
                best_score = avg_scores[best_idx]
            else:
                best_module = module
                best_score = 0.0

        if self.verbose:
            print(f"\n🎉 Optimization complete! Best score: {best_score:.2f}% (rollouts used: {self.rollouts_used})")
            if self.reflection_lm:
                print(f"🤖 Used separate reflection LM for cost optimization")

        return GEPAResult(
            best_module=best_module,
            best_score=best_score,
            candidates=all_candidates,
            generations=[{"scores_matrix": scores_matrix, "candidates": candidates}]
        )

    def _split_dataset(self, trainset: List[Example], valset: Optional[List[Example]] = None) -> Tuple[List[Example], List[Example]]:
        """Split dataset into Dfeedback and Dpareto according to paper."""
        if valset is not None:
            # Use provided validation set as Dpareto, trainset as Dfeedback
            return trainset, valset[:self.pareto_set_size]
        else:
            # Split trainset: smaller part for Dpareto, rest for Dfeedback
            pareto_size = min(self.pareto_set_size, len(trainset) // 3)
            shuffled = trainset.copy()
            random.shuffle(shuffled)
            return shuffled[pareto_size:], shuffled[:pareto_size]

    def _select_candidate(self, candidates: List[Module], scores_matrix: List[List[float]]) -> int:
        """Algorithm 2: Pareto-based candidate selection from paper."""
        if len(candidates) == 1:
            return 0

        num_tasks = len(scores_matrix[0]) if scores_matrix else 0
        if num_tasks == 0:
            return 0

        # Step 3-6: Build instance-wise Pareto sets
        pareto_sets = []  # P*[i] for each task i
        for i in range(num_tasks):
            # Find max score for task i across all candidates
            max_score = max(scores[i] for scores in scores_matrix if len(scores) > i)
            # Find candidates that achieve this max score
            best_candidates = []
            for j, scores in enumerate(scores_matrix):
                if len(scores) > i and scores[i] == max_score:
                    best_candidates.append(j)
            pareto_sets.append(best_candidates)

        # Step 7: Get unique candidates in union of all Pareto sets
        union_candidates = set()
        for pareto_set in pareto_sets:
            union_candidates.update(pareto_set)
        union_candidates = list(union_candidates)

        if not union_candidates:
            return 0

        # Step 8-11: Remove dominated candidates
        non_dominated = []
        for candidate_idx in union_candidates:
            is_dominated = False
            candidate_scores = scores_matrix[candidate_idx]

            for other_idx in union_candidates:
                if other_idx == candidate_idx:
                    continue

                other_scores = scores_matrix[other_idx]
                # Check if other dominates candidate (better or equal on all, strictly better on at least one)
                if len(other_scores) >= len(candidate_scores):
                    dominates = True
                    strictly_better = False
                    for i in range(len(candidate_scores)):
                        if other_scores[i] < candidate_scores[i]:
                            dominates = False
                            break
                        elif other_scores[i] > candidate_scores[i]:
                            strictly_better = True

                    if dominates and strictly_better:
                        is_dominated = True
                        break

            if not is_dominated:
                non_dominated.append(candidate_idx)

        if not non_dominated:
            non_dominated = union_candidates

        # Step 12-14: Filter Pareto sets and calculate frequencies
        frequencies = {}
        for candidate_idx in non_dominated:
            freq = 0
            for pareto_set in pareto_sets:
                if candidate_idx in pareto_set:
                    freq += 1
            frequencies[candidate_idx] = freq

        # Step 14: Sample proportionally to frequency
        total_freq = sum(frequencies.values())
        if total_freq == 0:
            return random.choice(non_dominated)

        rand_val = random.random() * total_freq
        cumulative = 0
        for candidate_idx, freq in frequencies.items():
            cumulative += freq
            if rand_val <= cumulative:
                return candidate_idx

        return non_dominated[-1]  # Fallback

    def _select_module(self, module: Module) -> int:
        """Select which module to optimize. Simple round-robin for now."""
        # For single module systems, always return 0
        # For multi-module systems, could implement round-robin or other strategies
        return 0

    def _reflective_prompt_mutation(self, module: Module, minibatch: List[Example], module_idx: int) -> Module:
        """Perform reflective prompt mutation using feedback traces."""
        # Gather execution traces and feedback
        feedbacks = []
        traces = []

        for example in minibatch:
            try:
                # Execute module and gather trace
                inputs = example.inputs()
                prediction = module(**inputs)

                # Get feedback from feedback function
                feedback_result = self.feedback_function(example, prediction, self.metric)
                feedbacks.append(feedback_result.get('feedback', ''))
                traces.append(feedback_result.get('trace', str(prediction)))
            except Exception as e:
                feedbacks.append(f"Error: {str(e)}")
                traces.append("")

        # Use reflective mutation to update prompt
        return self._update_prompt_with_reflection(module, feedbacks, traces, module_idx)

    def _update_prompt_with_reflection(self, module: Module, feedbacks: List[str], traces: List[str], module_idx: int) -> Module:
        """Update module prompt based on reflective analysis of feedback and traces."""

        # Combine all feedback and traces
        combined_feedback = "\n".join(f"Example {i+1}: {fb}" for i, fb in enumerate(feedbacks) if fb)
        combined_traces = "\n".join(f"Trace {i+1}: {tr}" for i, tr in enumerate(traces) if tr)

        # Create reflection prompt
        reflection_prompt = f"""
        Analyze the following execution traces and feedback to improve the instruction:

        Current traces:
        {combined_traces}

        Feedback:
        {combined_feedback}

        Based on this analysis, propose an improved instruction that addresses the identified issues.
        Focus on being specific and actionable.
        """

        try:
            # Use reflection LM if available, otherwise use default Predict
            if self.reflection_lm:
                # Use the dedicated reflection LM
                reflector = Predict(
                    "analysis_context -> improved_instruction",
                    lm=self.reflection_lm
                )
            else:
                # Use default LM
                reflector = Predict("analysis_context -> improved_instruction")

            result = reflector(analysis_context=reflection_prompt)
            new_instruction = getattr(result, 'improved_instruction', "Analyze carefully and provide accurate responses.")

            # Create new module with updated instruction
            return self._create_module_with_instruction(module, new_instruction)

        except Exception as e:
            if self.verbose:
                print(f"⚠️ Reflection failed: {e}, using fallback")
            # Fallback to simple instruction variations
            fallback_instructions = [
                "Analyze the input carefully and provide precise, accurate responses.",
                "Think step by step and provide clear, well-reasoned answers.",
                "Consider all aspects of the input and provide comprehensive responses.",
                "Be specific and detailed in your analysis and response.",
                "Focus on accuracy and clarity when responding to this task."
            ]
            new_instruction = random.choice(fallback_instructions)
            return self._create_module_with_instruction(module, new_instruction)

    def _default_feedback(self, example: Example, prediction: Any, metric: Callable) -> Dict[str, str]:
        """Default feedback function that provides basic score and trace info."""
        try:
            score = metric(example, prediction)
            return {
                'feedback': f"Score: {score:.2f}",
                'trace': str(prediction)
            }
        except Exception as e:
            return {
                'feedback': f"Error in evaluation: {str(e)}",
                'trace': str(prediction)
            }

    def _evaluate_single(self, module: Module, example: Example) -> float:
        """Evaluate module on a single example."""
        try:
            # Call module with keyword arguments, not positional
            inputs = example.inputs()
            prediction = module(**inputs)
            score = self.metric(example, prediction)
            return score
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Evaluation failed: {e}")
            return 0.0

    def _create_module_with_instruction(self, original_module: Module, new_instruction: str) -> Module:
        """Create a new module with modified instruction."""
        try:
            # Preserve the original signature and just update instructions
            if hasattr(original_module, 'signature'):
                from ..signature.signature import Signature

                # Create new signature preserving all original properties
                new_sig = Signature(
                    original_module.signature.spec,
                    instructions=new_instruction,
                    custom_types=original_module.signature.custom_types
                )

                # Create new module with preserved signature
                new_module = type(original_module)(new_sig)
                return new_module
            else:
                # Create basic Predict with instruction
                new_module = Predict("input -> output", instructions=new_instruction)
                return new_module
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Module creation failed: {e}, using original")
            # Fallback: return copy of original
            return original_module