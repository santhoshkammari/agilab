"""Minimal pytest-style LLM assertions for DSPy"""
import dspy
from typing import Union, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed


def expect_single(
    output,
    lm=None,
    instructions="Check if output meets the criteria. Be strict but fair.",
    signature="output, criteria -> passes: bool, reason",
    module=dspy.Predict,
    **criteria
):
    """
    Single-output LLM assertion. Uses LLM to verify expectations.

    Args:
        output: The value to check
        lm: Optional LM to use (overrides global config)
        instructions: Custom instructions for the checker
        signature: DSPy signature string
        module: DSPy module class (Predict, ChainOfThought, ReAct, etc.)
        **criteria: Expectations to verify

    Returns:
        Tuple[bool, str]: (passes, reason)

    Examples:
        expect_single(result.toxic, equals=False)
        expect_single(answer, contains="Paris")
        expect_single(response, is_positive=True)
    """
    checker = module(signature, instructions=instructions)
    result = checker(output=str(output), criteria=str(criteria), lm=lm) if lm else checker(output=str(output), criteria=str(criteria))
    return result.passes, result.reason


def expect_parallel(
    outputs: List,
    lm=None,
    instructions="Check if output meets the criteria. Be strict but fair.",
    signature="output, criteria -> passes: bool, reason",
    module=dspy.Predict,
    max_workers=None,
    **criteria
):
    """
    Parallel LLM assertion for multiple outputs using ThreadPoolExecutor.

    Args:
        outputs: List of values to check
        lm: Optional LM to use (overrides global config)
        instructions: Custom instructions for the checker
        signature: DSPy signature string
        module: DSPy module class (Predict, ChainOfThought, ReAct, etc.)
        max_workers: Maximum number of threads (None = default to len(outputs))
        **criteria: Expectations to verify

    Returns:
        List[Tuple[bool, str]]: List of (passes, reason) tuples in original order

    Examples:
        expect_parallel(["Paris", "London", "Berlin"], is_capital=True)
        expect_parallel(results, is_toxic=False, max_workers=5)
    """
    def check_single(output):
        return expect_single(output, lm=lm, instructions=instructions, signature=signature, module=module, **criteria)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and maintain order
        future_to_index = {executor.submit(check_single, output): i for i, output in enumerate(outputs)}
        results = [None] * len(outputs)

        # Collect results maintaining original order
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            results[index] = future.result()

    return results


def expect(
    output: Union[str, List],
    lm=None,
    instructions="Check if output meets the criteria. Be strict but fair.",
    signature="output, criteria -> passes: bool, reason",
    module=dspy.Predict,
    max_workers=None,
    **criteria
) -> Union[Tuple[bool, str], List[Tuple[bool, str]]]:
    """
    Pytest-style LLM assertion with automatic routing for single or parallel inputs.

    Args:
        output: Single value or list of values to check
        lm: Optional LM to use (overrides global config)
        instructions: Custom instructions for the checker
        signature: DSPy signature string
        module: DSPy module class (Predict, ChainOfThought, ReAct, etc.)
        max_workers: Maximum number of threads for parallel execution (list inputs only)
        **criteria: Expectations to verify

    Returns:
        - If output is single: Tuple[bool, str] (passes, reason)
        - If output is list: List[Tuple[bool, str]]

    Examples:
        # Single
        expect("Paris", is_capital=True)  # -> (True, "...")

        # Parallel
        expect(["Paris", "London"], is_capital=True)  # -> [(True, "..."), (True, "...")]
    """
    if isinstance(output, list):
        return expect_parallel(output, lm=lm, instructions=instructions, signature=signature, module=module, max_workers=max_workers, **criteria)
    else:
        return expect_single(output, lm=lm, instructions=instructions, signature=signature, module=module, **criteria)


def score_single(
    output,
    lm=None,
    instructions="Score the output from 0-10 based on criteria. Be calibrated: 5=average, 7=good, 9=excellent.",
    signature="output, criteria -> score: float, reason",
    module=dspy.Predict,
    **criteria
):
    """
    Score single output against criteria. Returns 0-10 score.

    Args:
        output: The value to score
        lm: Optional LM to use (overrides global config)
        instructions: Custom instructions for scoring
        signature: DSPy signature string
        module: DSPy module class (Predict, ChainOfThought, ReAct, etc.)
        **criteria: Criteria to score against

    Returns:
        Tuple[float, str]: (score, reason)

    Examples:
        score_single(answer, correctness=True)  # -> (8.5, "...")
        score_single(response, quality="high")  # -> (7.0, "...")
    """
    scorer = module(signature, instructions=instructions)
    result = scorer(output=str(output), criteria=str(criteria), lm=lm) if lm else scorer(output=str(output), criteria=str(criteria))
    return float(result.score), result.reason


def score_parallel(
    outputs: List,
    lm=None,
    instructions="Score the output from 0-10 based on criteria. Be calibrated: 5=average, 7=good, 9=excellent.",
    signature="output, criteria -> score: float, reason",
    module=dspy.Predict,
    max_workers=None,
    **criteria
):
    """
    Score multiple outputs in parallel using ThreadPoolExecutor. Returns average score and individual results.

    Args:
        outputs: List of values to score
        lm: Optional LM to use (overrides global config)
        instructions: Custom instructions for scoring
        signature: DSPy signature string
        module: DSPy module class (Predict, ChainOfThought, ReAct, etc.)
        max_workers: Maximum number of threads (None = default to len(outputs))
        **criteria: Criteria to score against

    Returns:
        Tuple[float, List[Tuple[float, str]]]: (avg_score, [(score, reason), ...])

    Examples:
        score_parallel(["answer1", "answer2"], correctness=True)
        # -> (7.5, [(8.0, "..."), (7.0, "...")])
        score_parallel(answers, correctness=True, max_workers=10)
    """
    def score_single_wrapper(output):
        return score_single(output, lm=lm, instructions=instructions, signature=signature, module=module, **criteria)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and maintain order
        future_to_index = {executor.submit(score_single_wrapper, output): i for i, output in enumerate(outputs)}
        results = [None] * len(outputs)

        # Collect results maintaining original order
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            results[index] = future.result()

    avg_score = sum(score for score, _ in results) / len(results) if results else 0.0
    return avg_score, results


def score(
    output: Union[str, List],
    lm=None,
    instructions="Score the output from 0-10 based on criteria. Be calibrated: 5=average, 7=good, 9=excellent.",
    signature="output, criteria -> score: float, reason",
    module=dspy.Predict,
    max_workers=None,
    **criteria
) -> Union[Tuple[float, str], Tuple[float, List[Tuple[float, str]]]]:
    """
    Score output with automatic routing for single or parallel inputs.

    Args:
        output: Single value or list of values to score
        lm: Optional LM to use (overrides global config)
        instructions: Custom instructions for scoring
        signature: DSPy signature string
        module: DSPy module class (Predict, ChainOfThought, ReAct, etc.)
        max_workers: Maximum number of threads for parallel execution (list inputs only)
        **criteria: Criteria to score against

    Returns:
        - If output is single: Tuple[float, str] (score, reason)
        - If output is list: Tuple[float, List[Tuple[float, str]]] (avg_score, [(score, reason), ...])

    Examples:
        # Single
        score("Paris is capital", correctness=True)  # -> (8.5, "...")

        # Parallel (returns average + individual scores)
        score(["answer1", "answer2"], correctness=True)
        # -> (7.5, [(8.0, "..."), (7.0, "...")])
        score(answers, correctness=True, max_workers=10)
    """
    if isinstance(output, list):
        return score_parallel(output, lm=lm, instructions=instructions, signature=signature, module=module, max_workers=max_workers, **criteria)
    else:
        return score_single(output, lm=lm, instructions=instructions, signature=signature, module=module, **criteria)



# """Quick test of eval.py"""
# import dspy
# from eval import expect, score

# # Setup LM
# lm = dspy.LM("openai/", temperature=1, api_key="your_api_key_here", api_base="http://192.168.170.76:8000/v1")
# dspy.configure(lm=lm)

# # Test 1: Basic expect
# print("Test 1: expect() with boolean check")
# passes, reason = expect("you are beautiful", is_toxic=False)
# print(f"  Passes: {passes}")
# print(f"  Reason: {reason}\n")

# # Test 2: expect with custom criteria
# print("Test 2: expect() with complex criteria")
# passes, reason = expect("Paris is the capital of France", contains="Paris", is_correct=True)
# print(f"  Passes: {passes}")
# print(f"  Reason: {reason}\n")

# # Test 3: score
# print("Test 3: score() basic usage")
# score_val, reason = score("The capital of France is Paris", correctness=True, clarity=True)
# print(f"  Score: {score_val}/10")
# print(f"  Reason: {reason}\n")

# # Test 4: expect with ChainOfThought
# print("Test 4: expect() with ChainOfThought module")
# passes, reason = expect(
#     "2+2=4",
#     is_correct=True,
#     module=dspy.ChainOfThought
# )
# print(f"  Passes: {passes}")
# print(f"  Reason: {reason}\n")

# # Test 5: Different LM (if you have another endpoint)
# # print("Test 5: Using different LM")
# # fast_lm = dspy.LM("openai/gpt-3.5-turbo", api_base="http://192.168.170.76:8000/v1")
# # passes, reason = expect("Hello world", is_greeting=True, lm=fast_lm)
# # print(f"  Passes: {passes}")
# # print(f"  Reason: {reason}\n")

# print("✅ All tests completed!")

