"""
Concrete GEPA Implementation Example
Shows the actual mechanics of how GEPA works
"""

# ============================================
# Helper classes
# ============================================

class Predictor:
    """Simple predictor class for demonstration"""
    def __init__(self, instruction):
        self.instruction = instruction

    def __call__(self, **kwargs):
        # In real implementation, this would call an LLM
        return {"result": "mock"}


# ============================================
# STEP 1: Your Program (Multi-step pipeline)
# ============================================

class SentimentAnalyzer:
    """A 2-step sentiment analysis pipeline"""

    def __init__(self):
        # Step 1: Extract key phrases
        self.extract_phrases = Predictor(
            instruction="Extract emotional phrases from the text."
        )

        # Step 2: Classify sentiment
        self.classify = Predictor(
            instruction="Based on the phrases, classify sentiment."
        )

    def forward(self, text):
        # Step 1: Extract
        phrases = self.extract_phrases(text=text)

        # Step 2: Classify
        sentiment = self.classify(phrases=phrases)

        return sentiment


# ============================================
# STEP 2: How DSPy Discovers Your Predictors
# ============================================

def named_parameters(module):
    """DSPy's way of finding all Predict components"""
    params = []

    # Find all attributes that are Predictors
    for name, value in module.__dict__.items():
        if isinstance(value, Predictor):
            params.append((name, value))

    return params

# Example:
analyzer = SentimentAnalyzer()
predictors = named_parameters(analyzer)
print("=" * 60)
print("DISCOVERED PREDICTORS:")
print("=" * 60)
for name, pred in predictors:
    print(f"  {name}: {pred.instruction}")

print()

# Output:
# extract_phrases: Extract emotional phrases from the text.
# classify: Based on the phrases, classify sentiment.


# ============================================
# STEP 3: Execution Tracing
# ============================================

def run_with_trace(module, **inputs):
    """Run module and capture trace of all predictor calls"""
    trace = []  # Will store (predictor, inputs, outputs) tuples

    # Monkey-patch predictors to record calls
    def traced_call(predictor, original_call):
        def wrapper(**kwargs):
            output = original_call(**kwargs)
            trace.append((predictor, kwargs, output))
            return output
        return wrapper

    # Actually run the module
    output = module.forward(**inputs)

    return output, trace


# Example execution:
print("=" * 60)
print("EXECUTION WITH TRACING:")
print("=" * 60)

text = "Not bad at all"
# Simulated execution:
trace = [
    ("extract_phrases", {"text": "Not bad at all"}, {"phrases": "bad"}),
    ("classify", {"phrases": "bad"}, {"sentiment": "negative"}),
]

print(f"Input: {text}")
print("\nTrace:")
for i, (predictor, inputs, outputs) in enumerate(trace, 1):
    print(f"  Step {i} ({predictor}):")
    print(f"    Input:  {inputs}")
    print(f"    Output: {outputs}")

print()


# ============================================
# STEP 4: GEPA's Feedback Collection
# ============================================

def create_feedback_for_predictor(pred_name, trace, example, prediction, gold_label):
    """GEPA creates predictor-specific feedback"""

    # Find this predictor's execution in the trace
    for predictor, inputs, outputs in trace:
        if predictor == pred_name:
            # This is where GEPA's magic happens!
            # It can give feedback specific to THIS predictor

            if pred_name == "extract_phrases":
                if outputs["phrases"] == "bad":
                    return {
                        "score": 0,
                        "feedback": "You extracted 'bad' but missed the negation word 'Not'. "
                                  "When extracting emotional phrases, always include negation "
                                  "words like 'not', 'no', 'never' as they reverse sentiment."
                    }

            elif pred_name == "classify":
                if outputs["sentiment"] == "negative" and gold_label == "positive":
                    return {
                        "score": 0,
                        "feedback": f"You classified as negative but the phrases were '{inputs['phrases']}'. "
                                  "The word 'bad' appeared without context. You should ask for "
                                  "more context or look for negation markers in the phrases."
                    }

    return {"score": 1, "feedback": "Good job!"}


print("=" * 60)
print("PREDICTOR-SPECIFIC FEEDBACK:")
print("=" * 60)

# GEPA collects feedback for EACH predictor
for pred_name in ["extract_phrases", "classify"]:
    feedback = create_feedback_for_predictor(
        pred_name, trace,
        example="Not bad at all",
        prediction="negative",
        gold_label="positive"
    )
    print(f"{pred_name}:")
    print(f"  Score: {feedback['score']}")
    print(f"  Feedback: {feedback['feedback']}")
    print()


# ============================================
# STEP 5: GEPA's Reflection Process
# ============================================

def gepa_reflection(predictor_name, current_instruction, failed_examples):
    """
    This is where GEPA uses a strong LLM to improve the instruction
    """

    # Build the reflection prompt
    prompt = f"""
Current Instruction for {predictor_name}:
"{current_instruction}"

Failed Examples with Feedback:
"""

    for ex in failed_examples:
        prompt += f"\nInput: {ex['input']}\n"
        prompt += f"Output: {ex['output']}\n"
        prompt += f"Feedback: {ex['feedback']}\n"

    prompt += """
Task: Write an improved instruction that addresses the issues in the feedback.
"""

    # Simulated LLM response
    if predictor_name == "extract_phrases":
        new_instruction = """Extract emotional phrases from the text.

IMPORTANT: Always include negation words (not, no, never, don't, can't, won't)
with the emotional words they modify. These change the sentiment!

Examples:
- "not bad" → extract "not bad" (not just "bad")
- "don't hate" → extract "don't hate" (not just "hate")
- "can't complain" → extract "can't complain" (not just "complain")

Strategy: Look for negation words within 2 words of emotional terms.
"""

    elif predictor_name == "classify":
        new_instruction = """Based on the phrases, classify sentiment.

CRITICAL: Check for negation in the phrases!
- If phrase contains "not/don't/can't" + negative word → POSITIVE
- If phrase contains "not/don't/can't" + positive word → NEGATIVE

Strategy:
1. Split phrases into individual components
2. For each component, check for negation words
3. Apply negation rules before final classification
"""

    return new_instruction


print("=" * 60)
print("GEPA GENERATES NEW INSTRUCTIONS:")
print("=" * 60)

# Simulate failed examples for extract_phrases
failed_extract = [
    {
        "input": {"text": "Not bad at all"},
        "output": {"phrases": "bad"},
        "feedback": "You extracted 'bad' but missed 'Not'. Include negation words!"
    }
]

new_extract_instruction = gepa_reflection(
    "extract_phrases",
    "Extract emotional phrases from the text.",
    failed_extract
)

print("NEW extract_phrases instruction:")
print(new_extract_instruction)
print()


# ============================================
# STEP 6: Building the Optimized Program
# ============================================

def build_optimized_program(module, new_instructions):
    """GEPA creates a new module with improved instructions"""

    # Deep copy the module
    new_module = SentimentAnalyzer()

    # Update each predictor's instruction
    for name, new_instruction in new_instructions.items():
        predictor = getattr(new_module, name)
        predictor.instruction = new_instruction

    return new_module


print("=" * 60)
print("BEFORE vs AFTER GEPA:")
print("=" * 60)

# Original module
original = SentimentAnalyzer()
print("BEFORE:")
for name, pred in named_parameters(original):
    print(f"  {name}: {pred.instruction}")

# Optimized module
optimized = build_optimized_program(original, {
    "extract_phrases": new_extract_instruction,
    "classify": "Based on the phrases, classify sentiment. Check for negations!"
})

print("\nAFTER GEPA:")
for name, pred in named_parameters(optimized):
    print(f"  {name}: {pred.instruction[:80]}...")


# ============================================
# STEP 7: The Complete GEPA Loop
# ============================================

print("\n" + "=" * 60)
print("COMPLETE GEPA OPTIMIZATION LOOP:")
print("=" * 60)

loop_explanation = """
for iteration in range(max_iterations):
    # 1. Sample a few training examples
    minibatch = random.sample(trainset, k=3)

    # 2. Run current program and capture traces
    for example in minibatch:
        prediction, trace = run_with_trace(current_program, **example.inputs())

        # 3. Collect predictor-specific feedback
        for predictor_name in predictor_names:
            feedback = metric(
                example, prediction, trace,
                pred_name=predictor_name,  # ← Key: ask for THIS predictor
                pred_trace=extract_trace_for(predictor_name, trace)
            )
            reflective_dataset[predictor_name].append({
                "input": extract_inputs_for(predictor_name, trace),
                "output": extract_outputs_for(predictor_name, trace),
                "feedback": feedback
            })

    # 4. Use reflection LM to generate new instructions
    new_instructions = {}
    for predictor_name in select_predictors_to_optimize():
        new_instructions[predictor_name] = reflection_lm(
            f"Current: {current_instructions[predictor_name]}\\n"
            f"Failed examples: {reflective_dataset[predictor_name]}\\n"
            f"Task: Write better instruction"
        )

    # 5. Build new program with improved instructions
    candidate_program = build_program(new_instructions)

    # 6. Evaluate on validation set (Pareto frontier tracking)
    score = evaluate(candidate_program, valset)

    # 7. Keep if it's on the Pareto frontier
    if is_pareto_optimal(score, all_scores):
        candidates.append(candidate_program)
        all_scores.append(score)

# 8. Return best candidate
return max(candidates, key=lambda c: evaluate(c, valset))
"""

print(loop_explanation)

print("\n" + "=" * 60)
print("KEY TAKEAWAYS:")
print("=" * 60)

takeaways = """
1. MODULARITY via named_parameters():
   - DSPy discovers all Predict components automatically
   - Each gets optimized independently

2. TRACING captures execution:
   - (predictor, inputs, outputs) for EACH step
   - Allows predictor-specific feedback

3. FEEDBACK is the secret sauce:
   - Not just "score=0"
   - But "You missed the negation word because..."
   - Feedback is specific to EACH predictor

4. REFLECTION LM analyzes patterns:
   - Sees multiple failures
   - Identifies common issues
   - Generates explicit rules

5. EVOLUTIONARY SEARCH:
   - Keeps Pareto frontier of good candidates
   - Combines complementary solutions (merge)
   - Explores diverse approaches

Result: Each predictor learns explicit rules from failures,
        encoded directly in its instruction!
"""

print(takeaways)
