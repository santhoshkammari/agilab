"""
Demo script showing how to use aspy.Evaluate and .with_inputs()
"""

import aspy

# Setup
lm = aspy.LM(api_base="http://192.168.170.76:8000")
aspy.configure(lm=lm)

print("=== Basic Example Usage ===")
# Basic example - auto-detects inputs vs targets
basic_example = aspy.Example(question="What is 2+2?", answer="4")
print("Basic example:", basic_example)
print("Auto-detected inputs:", basic_example.inputs())
print()

print("=== .with_inputs() Usage ===")
# Complex example with multiple fields
complex_example = aspy.Example(
    question="What is the capital of France?",
    context="France is a country in Europe",
    hint="It's a famous city",
    difficulty="easy",
    answer="Paris"
)

print("Complex example:", complex_example)
print("All fields:", complex_example.model_dump())
print("Auto-detected inputs:", complex_example.inputs())
print()

# Use .with_inputs() to specify exactly which fields are inputs
example_with_inputs = complex_example.with_inputs("question", "context")
print("After .with_inputs('question', 'context'):", example_with_inputs)
print("Specified input fields only:", example_with_inputs.inputs())
print()

# Different input combinations
example_with_hint = complex_example.with_inputs("question", "hint")
print("With question + hint:", example_with_hint.inputs())

example_all_inputs = complex_example.with_inputs("question", "context", "hint", "difficulty")
print("With all as inputs:", example_all_inputs.inputs())
print()

print("=== Evaluation Demo 1: Simple Question-Answer ===")
# Simple examples with just question -> answer
simple_examples = [
    aspy.Example(question="What is 2+2?", answer="4"),
    aspy.Example(question="Capital of France?", answer="Paris"),
    aspy.Example(question="What is 10-3?", answer="7"),
]

print("Simple evaluation examples:")
for i, ex in enumerate(simple_examples):
    print(f"  {i+1}. {ex}")
    print(f"     Inputs: {ex.inputs()}")
print()

# Create a simple module to evaluate
simple_qa = aspy.Predict("question -> answer")

# Evaluate simple examples
simple_evaluator = aspy.Evaluate(
    devset=simple_examples,
    metric=aspy.exact_match,
    display_progress=True,
    save_as_json="simple_results.json"
)

print("Running simple evaluation...")
simple_result = simple_evaluator(simple_qa)
print(f"Simple evaluation score: {simple_result.score}%")
print()

print("=== Evaluation Demo 2: Question + Context ===")
# Examples with context that should be included in inputs
context_examples = [
    aspy.Example(question="What is the result?", context="5 + 3", answer="8").with_inputs("question", "context"),
    aspy.Example(question="What is the capital?", context="France is a European country", answer="Paris").with_inputs("question", "context"),
    aspy.Example(question="What color is it?", context="The sky on a clear day", answer="blue").with_inputs("question", "context"),
]

print("Context-aware evaluation examples:")
for i, ex in enumerate(context_examples):
    print(f"  {i+1}. {ex}")
    print(f"     Inputs: {ex.inputs()}")
print()

# Create a context-aware module to evaluate
context_qa = aspy.Predict("question, context -> answer")

# Evaluate context examples
context_evaluator = aspy.Evaluate(
    devset=context_examples,
    metric=aspy.exact_match,
    display_progress=True,
    save_as_json="context_results.json"
)

print("Running context-aware evaluation...")
context_result = context_evaluator(context_qa)
print(f"Context-aware evaluation score: {context_result.score}%")