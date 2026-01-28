"""Minimal pytest-style LLM assertions for DSPy"""
import dspy


def expect(
    output,
    lm=None,
    instructions="Check if output meets the criteria. Be strict but fair.",
    signature="output, criteria -> passes: bool, reason",
    module=dspy.Predict,
    **criteria
):
    """
    Pytest-style LLM assertion. Uses LLM to verify expectations.

    Args:
        output: The value to check
        lm: Optional LM to use (overrides global config)
        instructions: Custom instructions for the checker
        signature: DSPy signature string
        module: DSPy module class (Predict, ChainOfThought, ReAct, etc.)
        **criteria: Expectations to verify

    Examples:
        expect(result.toxic, equals=False)
        expect(answer, contains="Paris")
        expect(response, is_positive=True)
        expect(output, is_not="harmful", lm=cheap_lm)
        expect(answer, is_correct=True, module=dspy.ChainOfThought)
    """
    checker = module(signature, instructions=instructions)
    result = checker(output=str(output), criteria=str(criteria), lm=lm) if lm else checker(output=str(output), criteria=str(criteria))
    return result.passes, result.reason


def score(
    output,
    lm=None,
    instructions="Score the output from 0-10 based on criteria. Be calibrated: 5=average, 7=good, 9=excellent.",
    signature="output, criteria -> score: float, reason",
    module=dspy.Predict,
    **criteria
):
    """
    Score output against criteria. Returns 0-10 score.

    Args:
        output: The value to score
        lm: Optional LM to use (overrides global config)
        instructions: Custom instructions for scoring
        signature: DSPy signature string
        module: DSPy module class (Predict, ChainOfThought, ReAct, etc.)
        **criteria: Criteria to score against

    Examples:
        score(answer, correctness=True)  # -> 8.5
        score(response, quality="high", relevance=True)  # -> 7.0
        score(summary, conciseness=True, lm=fast_lm)
        score(answer, is_correct=True, module=dspy.ChainOfThought)
    """
    scorer = module(signature, instructions=instructions)
    result = scorer(output=str(output), criteria=str(criteria), lm=lm) if lm else scorer(output=str(output), criteria=str(criteria))
    return float(result.score), result.reason



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

