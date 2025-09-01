
import re
import ast
import math
import json
import numpy as np
from typing import List, Dict, Any, Optional, Callable


# ============================================================================
# 1. BASIC REWARD FUNCTIONS
# ============================================================================

def reward_sparse(success: bool, **kwargs) -> float:
    """Basic sparse reward: 1 for success, 0 for failure."""
    return 1.0 if success else 0.0


def reward_dense(progress: float, max_progress: float = 1.0, **kwargs) -> float:
    """Dense reward based on progress towards goal."""
    return max(0.0, min(1.0, progress / max_progress))


def reward_shaped(current_state: Any, goal_state: Any, distance_fn: Callable = None, **kwargs) -> float:
    """Potential-based reward shaping."""
    if distance_fn is None:
        distance_fn = lambda x, y: abs(x - y) if isinstance(x, (int, float)) else 0
    
    distance = distance_fn(current_state, goal_state)
    return max(0.0, 1.0 - distance)


# ============================================================================
# 2. REASONING MODEL SPECIFIC REWARDS
# ============================================================================

def reward_thinking_quality(completion: str, **kwargs) -> float:
    """Reward quality of reasoning and thinking process."""
    score = 0.0
    
    # Chain of thought indicators
    cot_indicators = ['let me think', 'first', 'then', 'next', 'therefore', 'because', 'since']
    cot_score = sum(1 for indicator in cot_indicators if indicator in completion.lower()) / len(cot_indicators)
    
    # Step structure
    step_indicators = completion.count('\n') + completion.count('. ') + completion.count('- ')
    step_score = min(1.0, step_indicators / 8.0)
    
    # Logical connectors
    logic_words = ['however', 'moreover', 'furthermore', 'consequently', 'thus', 'hence']
    logic_score = sum(1 for word in logic_words if word in completion.lower()) / len(logic_words)
    
    return (cot_score + step_score + logic_score) / 3.0


def reward_answer_quality(completion: str, ground_truth: str = None, **kwargs) -> float:
    """Reward answer correctness and quality."""
    if ground_truth is None:
        # Quality heuristics without ground truth
        length_score = min(1.0, len(completion) / 200)
        structure_score = min(1.0, (completion.count('.') + completion.count('!')) / 3)
        return (length_score + structure_score) / 2.0
    
    # With ground truth
    exact_match = 1.0 if ground_truth.strip().lower() in completion.lower() else 0.0
    
    # Fuzzy matching
    words_gt = set(ground_truth.lower().split())
    words_comp = set(completion.lower().split())
    overlap = len(words_gt.intersection(words_comp)) / max(len(words_gt), 1)
    
    return (exact_match + overlap) / 2.0


def reward_chain_of_thought(completion: str, **kwargs) -> float:
    """Reward coherent chain-of-thought reasoning."""
    lines = [line.strip() for line in completion.split('\n') if line.strip()]
    
    if len(lines) < 2:
        return 0.1
    
    # Sequential thinking indicators
    sequence_words = ['first', 'second', 'third', 'then', 'next', 'finally', 'lastly']
    sequence_score = sum(1 for line in lines for word in sequence_words if word in line.lower())
    sequence_score = min(1.0, sequence_score / 3.0)
    
    # Logical flow
    transition_words = ['therefore', 'thus', 'so', 'hence', 'because', 'since', 'given that']
    transition_score = sum(1 for line in lines for word in transition_words if word in line.lower())
    transition_score = min(1.0, transition_score / 2.0)
    
    # Step progression
    step_score = min(1.0, len(lines) / 5.0)
    
    return (sequence_score + transition_score + step_score) / 3.0


def reward_self_correction(completion: str, **kwargs) -> float:
    """Reward self-correction and error acknowledgment."""
    correction_indicators = [
        'wait', 'actually', 'correction', 'mistake', 'error', 'wrong', 
        'let me reconsider', 'on second thought', 'i was incorrect'
    ]
    
    corrections = sum(1 for indicator in correction_indicators if indicator in completion.lower())
    return min(1.0, corrections / 2.0)


def reward_confidence_calibration(completion: str, confidence: float = None, **kwargs) -> float:
    """Reward appropriate confidence expression."""
    uncertainty_words = ['might', 'maybe', 'possibly', 'probably', 'likely', 'uncertain', 'unsure']
    confidence_words = ['definitely', 'certainly', 'clearly', 'obviously', 'sure', 'confident']
    
    uncertainty_count = sum(1 for word in uncertainty_words if word in completion.lower())
    confidence_count = sum(1 for word in confidence_words if word in completion.lower())
    
    if confidence is not None:
        # Reward matching confidence level with language
        if confidence > 0.8 and confidence_count > uncertainty_count:
            return 1.0
        elif confidence < 0.5 and uncertainty_count > confidence_count:
            return 1.0
        else:
            return 0.5
    
    # Default: balanced uncertainty acknowledgment
    return min(1.0, uncertainty_count / 3.0)


# ============================================================================
# 3. CODE EXECUTION & PROGRAMMING REWARDS
# ============================================================================

def reward_code_correctness(code: str, test_cases: List[Dict] = None, **kwargs) -> float:
    """Reward syntactically correct and working code."""
    try:
        # Syntax check
        ast.parse(code)
        syntax_score = 1.0
    except:
        syntax_score = 0.0
    
    if test_cases is None:
        return syntax_score
    
    # Execution test
    passed_tests = 0
    for test in test_cases:
        try:
            local_vars = {}
            exec(code, local_vars)
            if 'function_name' in test:
                func = local_vars.get(test['function_name'])
                if func and callable(func):
                    result = func(*test.get('input', []))
                    if result == test.get('expected'):
                        passed_tests += 1
        except:
            continue
    
    test_score = passed_tests / max(len(test_cases), 1) if test_cases else 0
    return (syntax_score + test_score) / 2.0


def reward_code_quality(code: str, **kwargs) -> float:
    """Reward code quality and style."""
    # Basic quality metrics
    lines = code.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    
    # Documentation
    doc_score = min(1.0, code.count('"""') / 2.0)  # docstrings
    comment_score = min(1.0, code.count('#') / max(len(non_empty_lines), 1))
    
    # Structure
    function_score = min(1.0, code.count('def ') / 2.0)
    class_score = code.count('class ') * 0.2
    
    # Readability
    avg_line_length = sum(len(line) for line in non_empty_lines) / max(len(non_empty_lines), 1)
    readability_score = 1.0 if 40 <= avg_line_length <= 100 else 0.5
    
    return (doc_score + comment_score + function_score + class_score + readability_score) / 5.0


def reward_execution_success(code: str, expected_output: str = None, **kwargs) -> float:
    """Reward successful code execution."""
    try:
        import io
        import sys
        
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        
        exec(code)
        output = captured_output.getvalue()
        
        sys.stdout = old_stdout
        
        if expected_output is None:
            return 1.0  # Just successful execution
        
        return 1.0 if expected_output.strip() in output.strip() else 0.5
    except:
        return 0.0


# ============================================================================
# 4. SEARCH & INFORMATION RETRIEVAL REWARDS
# ============================================================================

def reward_search_relevance(query: str, results: List[str], **kwargs) -> float:
    """Reward relevant search results."""
    if not results:
        return 0.0
    
    query_words = set(query.lower().split())
    relevance_scores = []
    
    for result in results:
        result_words = set(result.lower().split())
        overlap = len(query_words.intersection(result_words))
        relevance = overlap / max(len(query_words), 1)
        relevance_scores.append(relevance)
    
    return sum(relevance_scores) / len(relevance_scores)


def reward_information_synthesis(sources: List[str], synthesis: str, **kwargs) -> float:
    """Reward quality information synthesis from multiple sources."""
    if not sources or not synthesis:
        return 0.0
    
    # Check if synthesis incorporates multiple sources
    source_words = set()
    for source in sources:
        source_words.update(source.lower().split())
    
    synthesis_words = set(synthesis.lower().split())
    coverage = len(source_words.intersection(synthesis_words)) / max(len(source_words), 1)
    
    # Synthesis quality indicators
    synthesis_indicators = ['according to', 'based on', 'research shows', 'studies indicate']
    citation_score = sum(1 for indicator in synthesis_indicators if indicator in synthesis.lower())
    citation_score = min(1.0, citation_score / 2.0)
    
    return (coverage + citation_score) / 2.0


def reward_fact_verification(claim: str, evidence: List[str], verification_result: bool, **kwargs) -> float:
    """Reward accurate fact verification."""
    if not evidence:
        return 0.5
    
    # Evidence quality
    evidence_strength = min(1.0, len(evidence) / 3.0)
    
    # Verification accuracy (if known)
    accuracy_score = 1.0 if verification_result else 0.0
    
    return (evidence_strength + accuracy_score) / 2.0


# ============================================================================
# 5. PROBLEM SOLVING & REASONING REWARDS
# ============================================================================

def reward_problem_decomposition(problem: str, decomposition: str, **kwargs) -> float:
    """Reward effective problem breakdown."""
    # Check for decomposition indicators
    decomp_indicators = ['step', 'part', 'component', 'subproblem', 'break down', 'divide']
    decomp_score = sum(1 for indicator in decomp_indicators if indicator in decomposition.lower())
    decomp_score = min(1.0, decomp_score / 3.0)
    
    # Structure indicators
    structure_score = min(1.0, (decomposition.count('\n') + decomposition.count('- ')) / 5.0)
    
    # Logical flow
    logic_words = ['first', 'then', 'next', 'finally', 'before', 'after']
    logic_score = sum(1 for word in logic_words if word in decomposition.lower()) / len(logic_words)
    
    return (decomp_score + structure_score + logic_score) / 3.0


def reward_logical_consistency(reasoning: str, **kwargs) -> float:
    """Reward logically consistent reasoning."""
    # Check for contradictions
    contradiction_pairs = [
        ('yes', 'no'), ('true', 'false'), ('correct', 'incorrect'),
        ('possible', 'impossible'), ('always', 'never')
    ]
    
    reasoning_lower = reasoning.lower()
    contradictions = 0
    for pair in contradiction_pairs:
        if pair[0] in reasoning_lower and pair[1] in reasoning_lower:
            contradictions += 1
    
    consistency_score = max(0.0, 1.0 - contradictions / 3.0)
    
    # Logical structure
    logical_indicators = ['if', 'then', 'therefore', 'thus', 'because', 'since']
    logic_score = min(1.0, sum(1 for word in logical_indicators if word in reasoning_lower) / 4.0)
    
    return (consistency_score + logic_score) / 2.0


def reward_solution_creativity(problem: str, solution: str, **kwargs) -> float:
    """Reward creative and novel solutions."""
    # Creativity indicators
    creative_words = ['novel', 'innovative', 'creative', 'unique', 'alternative', 'unconventional']
    creativity_score = sum(1 for word in creative_words if word in solution.lower()) / len(creative_words)
    
    # Multiple approaches
    approach_indicators = ['approach', 'method', 'way', 'technique', 'strategy']
    approach_count = sum(1 for word in approach_indicators if word in solution.lower())
    approach_score = min(1.0, approach_count / 3.0)
    
    # Complexity and depth
    complexity_score = min(1.0, len(solution.split()) / 100)
    
    return (creativity_score + approach_score + complexity_score) / 3.0


# ============================================================================
# 6. MULTI-OBJECTIVE & HIERARCHICAL REWARDS
# ============================================================================

def reward_pareto_multi_objective(objectives: Dict[str, float], weights: Dict[str, float] = None, **kwargs) -> float:
    """Combine multiple objectives with Pareto optimization."""
    if not objectives:
        return 0.0
    
    if weights is None:
        weights = {key: 1.0 for key in objectives.keys()}
    
    # Weighted sum
    total_weight = sum(weights.values())
    weighted_score = sum(objectives[key] * weights.get(key, 1.0) for key in objectives.keys())
    
    return weighted_score / total_weight


def reward_hierarchical_goal(goals: List[Dict], completion_status: List[bool], **kwargs) -> float:
    """Reward hierarchical goal completion with priority weighting."""
    if not goals or not completion_status:
        return 0.0
    
    total_score = 0.0
    total_weight = 0.0
    
    for i, (goal, completed) in enumerate(zip(goals, completion_status)):
        priority = goal.get('priority', 1.0)
        weight = goal.get('weight', 1.0)
        
        if completed:
            total_score += priority * weight
        
        total_weight += priority * weight
    
    return total_score / max(total_weight, 1.0)


# ============================================================================
# 7. LEARNING & ADAPTATION REWARDS
# ============================================================================

def reward_learning_progress(previous_performance: float, current_performance: float, **kwargs) -> float:
    """Reward improvement over time."""
    improvement = current_performance - previous_performance
    return max(0.0, min(1.0, improvement + 0.5))  # Normalize around 0.5


def reward_transfer_learning(source_task_performance: float, target_task_performance: float, **kwargs) -> float:
    """Reward successful knowledge transfer between tasks."""
    transfer_effectiveness = target_task_performance / max(source_task_performance, 0.1)
    return min(1.0, transfer_effectiveness)


def reward_meta_learning(learning_speed: float, adaptation_quality: float, **kwargs) -> float:
    """Reward meta-learning capabilities."""
    return (learning_speed + adaptation_quality) / 2.0


# ============================================================================
# 8. SAFETY & ALIGNMENT REWARDS
# ============================================================================

def reward_truthfulness(statement: str, fact_check_result: bool = None, **kwargs) -> float:
    """Reward truthful statements."""
    if fact_check_result is not None:
        return 1.0 if fact_check_result else 0.0
    
    # Heuristic indicators of truthfulness
    uncertainty_indicators = ['i think', 'i believe', 'might be', 'possibly', 'unsure']
    certainty_indicators = ['definitely', 'certainly', 'absolutely', 'without doubt']
    
    uncertainty_count = sum(1 for indicator in uncertainty_indicators if indicator in statement.lower())
    certainty_count = sum(1 for indicator in certainty_indicators if indicator in statement.lower())
    
    # Reward appropriate uncertainty
    return min(1.0, (uncertainty_count + 1) / (certainty_count + 2))


def reward_harmlessness(content: str, **kwargs) -> float:
    """Reward harmless content."""
    harmful_indicators = [
        'violence', 'harm', 'dangerous', 'illegal', 'unethical',
        'manipulation', 'deception', 'fraud', 'abuse'
    ]
    
    harmful_count = sum(1 for word in harmful_indicators if word in content.lower())
    return max(0.0, 1.0 - harmful_count / 5.0)


def reward_uncertainty_calibration(prediction: float, confidence: float, actual_result: float, **kwargs) -> float:
    """Reward well-calibrated uncertainty estimates."""
    prediction_error = abs(prediction - actual_result)
    calibration_error = abs(confidence - (1 - prediction_error))
    
    return max(0.0, 1.0 - calibration_error)


# ============================================================================
# 9. COMPOSITE & DYNAMIC REWARDS
# ============================================================================

def reward_adaptive(context: Dict, performance_history: List[float], **kwargs) -> float:
    """Dynamically adapt reward based on context and history."""
    if not performance_history:
        return 0.5
    
    recent_performance = np.mean(performance_history[-5:])
    trend = np.mean(np.diff(performance_history[-10:])) if len(performance_history) > 1 else 0
    
    # Adaptive weighting based on performance
    if recent_performance > 0.8:
        # High performance: emphasize consistency
        return min(1.0, recent_performance - abs(trend))
    else:
        # Low performance: emphasize improvement
        return max(0.0, recent_performance + trend)


def reward_curriculum(difficulty_level: int, performance: float, target_difficulty: int, **kwargs) -> float:
    """Reward appropriate to curriculum difficulty."""
    difficulty_bonus = 1.0 + (difficulty_level / target_difficulty) * 0.5
    return performance * difficulty_bonus


# ============================================================================
# 10. SUPERINTELLIGENT CAPABILITY REWARDS
# ============================================================================

def reward_reasoning_depth(reasoning: str, **kwargs) -> float:
    """Reward deep, sophisticated reasoning."""
    # Abstract thinking indicators
    abstract_words = ['concept', 'principle', 'theory', 'framework', 'paradigm', 'abstraction']
    abstract_score = sum(1 for word in abstract_words if word in reasoning.lower()) / len(abstract_words)
    
    # Causal reasoning
    causal_words = ['because', 'causes', 'results in', 'leads to', 'due to', 'consequently']
    causal_score = sum(1 for word in causal_words if word in reasoning.lower()) / len(causal_words)
    
    # Meta-reasoning
    meta_words = ['reasoning', 'thinking', 'analysis', 'consideration', 'evaluation']
    meta_score = sum(1 for word in meta_words if word in reasoning.lower()) / len(meta_words)
    
    return (abstract_score + causal_score + meta_score) / 3.0


def reward_cross_domain_transfer(source_domain: str, target_domain: str, solution: str, **kwargs) -> float:
    """Reward successful cross-domain knowledge transfer."""
    # Check for domain-bridging language
    bridge_words = ['similar to', 'like in', 'analogous', 'parallel', 'comparable', 'equivalent']
    bridge_score = sum(1 for phrase in bridge_words if phrase in solution.lower()) / len(bridge_words)
    
    # Domain integration
    source_concepts = source_domain.lower().split()
    target_concepts = target_domain.lower().split()
    solution_words = solution.lower().split()
    
    source_integration = len(set(source_concepts).intersection(set(solution_words))) / max(len(source_concepts), 1)
    target_integration = len(set(target_concepts).intersection(set(solution_words))) / max(len(target_concepts), 1)
    
    return (bridge_score + source_integration + target_integration) / 3.0


def reward_long_term_planning(plan: str, time_horizon: int = 10, **kwargs) -> float:
    """Reward sophisticated long-term planning."""
    # Temporal indicators
    temporal_words = ['future', 'long-term', 'eventually', 'ultimately', 'phase', 'stage']
    temporal_score = sum(1 for word in temporal_words if word in plan.lower()) / len(temporal_words)
    
    # Planning structure
    planning_words = ['goal', 'objective', 'milestone', 'timeline', 'strategy', 'roadmap']
    planning_score = sum(1 for word in planning_words if word in plan.lower()) / len(planning_words)
    
    # Complexity scaling
    complexity_score = min(1.0, len(plan.split()) / (time_horizon * 20))
    
    return (temporal_score + planning_score + complexity_score) / 3.0


# ============================================================================
# LEGACY FUNCTIONS (for backward compatibility)
# ============================================================================

def reward_exact_match(prompt: str, completion: str, answer: str, **kwargs) -> float:
    """Reward exact matches with ground truth."""
    return reward_answer_quality(completion, answer, **kwargs)


def reward_length(prompt: str, completion: str, target_length: int = 100, **kwargs) -> float:
    """Reward completions close to target length."""
    return max(0.0, 1.0 - abs(len(completion) - target_length) / target_length)


def reward_step_by_step(prompt: str, completion: str, **kwargs) -> float:
    """Reward step-by-step reasoning (more line breaks = better)."""
    return reward_chain_of_thought(completion, **kwargs)


def reward_format(prompt: str, completion: str, required_format: str = None, **kwargs) -> float:
    """Reward specific formatting patterns."""
    if required_format is None:
        return 1.0
    return 1.0 if re.search(required_format, completion) else 0.0

# ============================================================================
# MATH FUNCTIONS
# ============================================================================

"""
Answer checker API that uses sympy to simplify expressions and check for equality.

Call grade_answer(given_answer: str, ground_truth: str).
"""

import re

import sympy
from pylatexenc import latex2text
from sympy.parsing import sympy_parser


# Dan Hendrycks' code
def mathd_normalize_answer(answer: str | None) -> str | None:
    if answer is None:
        return None
    answer = answer.strip()
    try:
        # Remove enclosing `\text{}`.
        m = re.search("^\\\\text\{(?P<text>.+?)\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return _strip_string(answer)
    except Exception:
        return answer


def _strip_string(string):
    def _fix_fracs(string):
        substrs = string.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += "\\frac"
                if substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except Exception:
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        else:
                            new_str += "{" + a + "}" + b
        string = new_str
        return string

    def _fix_a_slash_b(string):
        if len(string.split("/")) != 2:
            return string
        a = string.split("/")[0]
        b = string.split("/")[1]
        try:
            a = int(a)
            b = int(b)
            assert string == "{}/{}".format(a, b)
            new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
            return new_string
        except Exception:
            return string

    def _remove_right_units(string):
        # "\\text{ " only ever occurs (at least in the val set) when describing units
        if "\\text{ " in string:
            splits = string.split("\\text{ ")
            assert len(splits) == 2
            return splits[0]
        else:
            return string

    def _fix_sqrt(string):
        if "\\sqrt" not in string:
            return string
        splits = string.split("\\sqrt")
        new_string = splits[0]
        for split in splits[1:]:
            if split[0] != "{":
                a = split[0]
                new_substr = "\\sqrt{" + a + "}" + split[1:]
            else:
                new_substr = "\\sqrt" + split
            new_string += new_substr
        return new_string

    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


# sympy might hang -- we don't care about trying to be lenient in these cases
BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = ["\^[0-9]+\^", "\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"


def _sympy_parse(expr: str):
    """Parses an expression with sympy."""
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        ),
    )


def _parse_latex(expr: str) -> str:
    """Attempts to parse latex to an expression sympy can read."""
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")  # Play nice with mixed numbers.
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)

    # Replace the specific characters that this parser uses.
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")

    return expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False


def _str_to_int(x: str) -> bool:
    x = x.replace(",", "")
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str):
    """
    Automatically make a mixed number evalable
    e.g. 7 3/4 => 7+3/4
    """
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)  ## implicit mults
    return step


def _strip_properly_formatted_commas(expr: str):
    # We want to be careful because we don't want to strip tuple commas
    p1 = re.compile("(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _normalize(expr: str) -> str:
    """Normalize answer expressions."""
    if expr is None:
        return None

    # Remove enclosing `\text{}`.
    m = re.search("^\\\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
    ]:
        expr = re.sub(f"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub("\^ *\\\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except Exception:
            pass

    # edge case with mixed numbers and negative signs
    expr = re.sub("- *", "-", expr)

    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")

    # if we somehow still have latex braces here, just drop them
    expr = expr.replace("{", "")
    expr = expr.replace("}", "")

    # don't be case sensitive for text answers
    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def count_unknown_letters_in_expr(expr: str):
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)


def should_allow_eval(expr: str):
    # we don't want to try parsing unknown text or functions of more than two variables
    if count_unknown_letters_in_expr(expr) > 2:
        return False

    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False

    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False

    return True


def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    are_equal = False
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            if simplified == 0:
                are_equal = True
    except Exception:
        pass
    return are_equal


def split_tuple(expr: str):
    """
    Split the elements in a tuple/interval, while handling well-formatted commas in large numbers
    """
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
    ):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except Exception:
        return None


def extract_boxed_answer(solution: str) -> str:
    """Extract the answer from inside a LaTeX \\boxed{} command"""
    solution = last_boxed_only_string(solution)
    solution = remove_boxed(solution)
    return solution


def grade_answer_sympy(given_answer: str, ground_truth: str) -> bool:
    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)

    if ground_truth_normalized is None:
        return False

    if ground_truth_normalized == given_normalized:
        return True

    if len(given_normalized) == 0:
        return False

    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)

    if len(ground_truth_elems) > 1 and (
        ground_truth_normalized[0] != given_normalized[0]
        or ground_truth_normalized[-1] != given_normalized[-1]
    ):
        is_correct = False
    elif len(ground_truth_elems) != len(given_elems):
        is_correct = False
    else:
        for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems):
            if _is_frac(ground_truth_elem) and _is_frac(given_elem):
                # if fractions aren't reduced, then shouldn't be marked as correct
                # so, we don't want to allow sympy.simplify in this case
                is_correct = ground_truth_elem == given_elem
            elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
                # if the ground truth answer is an integer, we require the given answer to be a strict match (no sympy.simplify)
                is_correct = False
            else:
                is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
            if not is_correct:
                break

    return is_correct


def grade_answer_mathd(given_answer: str, ground_truth: str) -> bool:
    ground_truth_normalized_mathd = mathd_normalize_answer(ground_truth)
    given_answer_normalized_mathd = mathd_normalize_answer(given_answer)

    # be at least as lenient as mathd
    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True
    return False


def extract_answer(passage: str) -> str:
    if "\\boxed" in passage:
        return extract_boxed_answer(passage)
    return None


def reward_math_grade(solution_str: str, ground_truth: str) -> bool:
    if not ground_truth:
        return False
    if "\\boxed" in ground_truth:
        ground_truth = extract_answer(ground_truth)
    given_answer = extract_answer(solution_str)
    if given_answer is None:
        return False
    return grade_answer_mathd(given_answer, ground_truth) or grade_answer_sympy(
        given_answer, ground_truth
    )