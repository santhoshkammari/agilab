"""
Simple GEPA Example: Sentiment Classification
Shows how GEPA evolves prompts from failures
"""

# ============================================
# STEP 1: Define the Task
# ============================================

# Training examples
examples = [
    {"text": "I love this!", "label": "positive"},
    {"text": "This is terrible", "label": "negative"},
    {"text": "Not bad at all", "label": "positive"},  # Tricky: negation
    {"text": "I don't hate it", "label": "positive"},  # Tricky: double negative
    {"text": "Could be worse", "label": "negative"},   # Tricky: indirect
]

# ============================================
# STEP 2: Initial Program (Zero-shot)
# ============================================

# Simple predictor with generic instruction
initial_instruction = "Classify the sentiment as positive or negative."

# Let's simulate what happens:
print("=" * 60)
print("INITIAL PROGRAM (Zero-shot)")
print("=" * 60)
print(f"Instruction: {initial_instruction}\n")

# Simulated predictions
predictions = [
    ("I love this!", "positive", True),
    ("This is terrible", "negative", True),
    ("Not bad at all", "negative", False),      # FAILED: didn't understand negation
    ("I don't hate it", "negative", False),     # FAILED: missed double negative
    ("Could be worse", "positive", False),      # FAILED: indirect negativity
]

correct = sum(1 for _, _, is_correct in predictions if is_correct)
print(f"Performance: {correct}/{len(predictions)} = {correct/len(predictions)*100:.1f}%\n")

for text, pred, is_correct in predictions:
    status = "✓" if is_correct else "✗"
    print(f"{status} Text: '{text}' → Predicted: {pred}")

# ============================================
# STEP 3: GEPA Captures Failures with Feedback
# ============================================

print("\n" + "=" * 60)
print("GEPA STEP 1: Capture Failures with Rich Feedback")
print("=" * 60)

# GEPA creates a "reflective dataset" - failures with detailed feedback
reflective_dataset = """
Example 1:
Input: "Not bad at all"
Predicted: negative
Correct: positive
Feedback: The phrase contains negation "Not bad" which actually means good.
The model failed to recognize that negating a negative word creates positive sentiment.

Example 2:
Input: "I don't hate it"
Predicted: negative
Correct: positive
Feedback: This is a double negative. "don't hate" = like. The model saw "hate"
and "don't" separately but didn't understand that double negation reverses sentiment.

Example 3:
Input: "Could be worse"
Predicted: positive
Correct: negative
Feedback: This is an indirect way of expressing dissatisfaction. While not
explicitly negative, "could be worse" implies the current state is bad. The
model failed to recognize implied negativity.
"""

print(reflective_dataset)

# ============================================
# STEP 4: Reflection LM Analyzes Patterns
# ============================================

print("\n" + "=" * 60)
print("GEPA STEP 2: Reflection LM Analyzes Failures")
print("=" * 60)

# GEPA sends this to a strong LM (like GPT-4) for reflection
reflection_prompt = f"""
Current Instruction:
{initial_instruction}

Failed Examples with Feedback:
{reflective_dataset}

Task: Analyze the failures and identify what the instruction is missing.
"""

print("Reflection LM analyzes the pattern...")
print("\nIdentified Issues:")
analysis = """
1. NEGATION HANDLING: The model doesn't understand that "not bad" = good
2. DOUBLE NEGATIVES: Fails to recognize "don't hate" reverses to positive
3. INDIRECT EXPRESSIONS: Misses implied sentiment in phrases like "could be worse"
4. MISSING STRATEGY: No explicit guidance on handling negation words
"""
print(analysis)

# ============================================
# STEP 5: Reflection LM Generates New Instruction
# ============================================

print("\n" + "=" * 60)
print("GEPA STEP 3: Generate Improved Instruction")
print("=" * 60)

improved_instruction = """Classify the sentiment as positive or negative.

CRITICAL: Pay special attention to negation and indirect expressions:

1. NEGATION RULES:
   - "not bad/terrible/awful" → POSITIVE (negating negative = positive)
   - "not good/great/amazing" → NEGATIVE (negating positive = negative)

2. DOUBLE NEGATIVES:
   - "don't hate/dislike/mind" → POSITIVE (double negative = positive)
   - "can't love/like/enjoy" → NEGATIVE (can't + positive = negative)

3. INDIRECT NEGATIVITY:
   - "could be better/worse" → Evaluate the implication
   - "could be worse" = current state is bad → NEGATIVE
   - "could be better" = current state is okay but not great → NEGATIVE

4. STRATEGY:
   - First, identify all negation words (not, don't, can't, won't, etc.)
   - Count negations: even count = preserve sentiment, odd count = flip sentiment
   - Look for indirect expressions that imply sentiment without explicit words
   - When in doubt, consider the overall tone and implication

Final answer must be exactly "positive" or "negative" (lowercase).
"""

print("New Instruction:")
print(improved_instruction)

# ============================================
# STEP 6: Test Improved Program
# ============================================

print("\n" + "=" * 60)
print("OPTIMIZED PROGRAM (After GEPA)")
print("=" * 60)

# Now with improved instruction, predictions should be better
optimized_predictions = [
    ("I love this!", "positive", True),
    ("This is terrible", "negative", True),
    ("Not bad at all", "positive", True),      # FIXED: understood negation
    ("I don't hate it", "positive", True),     # FIXED: understood double negative
    ("Could be worse", "negative", True),      # FIXED: recognized indirect negativity
]

correct = sum(1 for _, _, is_correct in optimized_predictions if is_correct)
print(f"Performance: {correct}/{len(optimized_predictions)} = {correct/len(optimized_predictions)*100:.1f}%\n")

for text, pred, is_correct in optimized_predictions:
    status = "✓" if is_correct else "✗"
    print(f"{status} Text: '{text}' → Predicted: {pred}")

# ============================================
# STEP 7: What Actually Happened
# ============================================

print("\n" + "=" * 60)
print("WHAT GEPA DID")
print("=" * 60)

summary = """
Before GEPA:
  Instruction: "Classify the sentiment as positive or negative."
  Performance: 40% (2/5 correct)

After GEPA:
  Instruction: [Contains explicit negation rules, double negative handling,
                indirect expression patterns, step-by-step strategy]
  Performance: 100% (5/5 correct)

Key Insight:
  GEPA turned FAILURES into KNOWLEDGE by:
  1. Capturing what went wrong (feedback)
  2. Using an LLM to analyze patterns across failures
  3. Generating explicit rules to prevent those errors
  4. Encoding those rules directly in the instruction

This is like "precomputing reasoning" - instead of figuring out
"not bad = positive" every time, the instruction now contains that rule!

vs. Traditional RL:
  - RL would see: score=0, score=0, score=0 → adjust weights
  - GEPA sees: "failed because didn't understand negation" → add negation rule

Result: 35x fewer examples needed because each failure teaches more!
"""

print(summary)

# ============================================
# STEP 8: The Module-Level Magic
# ============================================

print("\n" + "=" * 60)
print("HOW THIS WORKS FOR MULTI-STEP MODULES")
print("=" * 60)

multi_step_example = """
Imagine a 3-step module:

class EmailResponder(dspy.Module):
    def __init__(self):
        self.classify = dspy.Predict("email -> category")
        self.extract = dspy.Predict("email -> key_points")
        self.respond = dspy.Predict("category, key_points -> response")

    def forward(self, email):
        category = self.classify(email=email)
        points = self.extract(email=email)
        response = self.respond(category=category, key_points=points)
        return response

GEPA can optimize EACH step independently:

1. named_parameters() finds all 3 Predict modules
2. Traces execution: (classify_input → classify_output),
                     (extract_input → extract_output),
                     (respond_input → respond_output)
3. If response is wrong, GEPA gets feedback for EACH step:
   - classify feedback: "You classified complaint as question"
   - extract feedback: "You missed the order number"
   - respond feedback: "Your tone was too formal"
4. Reflection LM generates improved instructions for EACH:
   - classify.signature.instructions → "Look for anger words to detect complaints..."
   - extract.signature.instructions → "Always extract order numbers from pattern..."
   - respond.signature.instructions → "Use friendly tone for customer service..."

Result: Each component gets smarter independently, but works together!
"""

print(multi_step_example)
