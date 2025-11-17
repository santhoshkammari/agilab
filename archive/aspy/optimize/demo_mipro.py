"""
Demo of MIPROv2 optimization in aspy
"""

import aspy

# Setup
lm = aspy.LM(api_base="http://192.168.170.76:8000")
aspy.configure(lm=lm)

print("🚀 MIPROv2 Optimization Demo")
print("=" * 50)

# Create training dataset
print("📚 Creating training dataset...")
trainset = [
    aspy.Example(question="What is 5 + 3?", answer="8"),
    aspy.Example(question="What is 12 - 4?", answer="8"),
    aspy.Example(question="What is 2 * 6?", answer="12"),
    aspy.Example(question="What is 15 / 3?", answer="5"),
    aspy.Example(question="What is 10 + 7?", answer="17"),
    aspy.Example(question="What is 20 - 8?", answer="12"),
    aspy.Example(question="What is 4 * 3?", answer="12"),
    aspy.Example(question="What is 18 / 2?", answer="9"),
]

# Create validation dataset
print("🔍 Creating validation dataset...")
valset = [
    aspy.Example(question="What is 7 + 6?", answer="13"),
    aspy.Example(question="What is 15 - 9?", answer="6"),
    aspy.Example(question="What is 3 * 8?", answer="24"),
    aspy.Example(question="What is 21 / 7?", answer="3"),
    aspy.Example(question="What is 9 + 4?", answer="13"),
]

print(f"Training examples: {len(trainset)}")
print(f"Validation examples: {len(valset)}")
print()

# Create initial module
print("🏗️ Creating initial math QA module...")
math_qa = aspy.Predict("question -> answer")

# Test baseline performance
print("📊 Testing baseline performance...")
baseline_evaluator = aspy.Evaluate(
    devset=valset,
    metric=aspy.exact_match,
    display_progress=True
)
baseline_result = baseline_evaluator(math_qa)
print(f"Baseline score: {baseline_result.score:.2f}%")
print()

# Setup MIPROv2 optimizer
print("🔧 Setting up MIPROv2 optimizer...")
optimizer = aspy.MIPROv2(
    metric=aspy.exact_match,
    num_candidates=6,  # Generate 6 instruction candidates per trial
    num_trials=4,      # Run 4 optimization rounds
    verbose=True,
    seed=42
)

# Run optimization
print("🚀 Starting MIPROv2 optimization...")
result = optimizer.compile(
    module=math_qa,
    trainset=trainset,
    valset=valset
)

print("\n" + "=" * 50)
print("📈 OPTIMIZATION RESULTS")
print("=" * 50)

print(f"🏆 Final optimized score: {result.score:.2f}%")
print(f"📊 Improvement: +{result.score - baseline_result.score:.2f}%")
print()

print("📝 Optimization History:")
for step in result.history:
    trial = step['trial']
    score = step['score']
    instruction = step['instruction']
    if trial == 0:
        print(f"  Trial {trial} (baseline): {score:.2f}% - {instruction}")
    else:
        print(f"  Trial {trial}: {score:.2f}% - {instruction}")

print()

# Test the optimized module
print("🧪 Testing optimized module on some examples...")
test_examples = [
    "What is 8 + 5?",
    "What is 16 - 7?",
    "What is 6 * 4?"
]

for question in test_examples:
    result_pred = result.optimized_module(question=question)
    print(f"Q: {question}")
    print(f"A: {getattr(result_pred, 'answer', 'No answer field')}")
    print()

print("✅ MIPROv2 optimization demo complete!")
print("\n🎯 Key Benefits:")
print("- Automatically improves instruction clarity")
print("- Uses training data to generate better prompts")
print("- Finds instructions that work better for your specific task")
print("- Simple to use with any aspy module")