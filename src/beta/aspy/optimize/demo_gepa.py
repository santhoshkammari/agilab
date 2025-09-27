"""
Demo of GEPA evolutionary optimization in aspy
"""

import aspy

# Setup
lm = aspy.LM(api_base="http://192.168.170.76:8000")
aspy.configure(lm=lm)

print("🧬 GEPA Evolutionary Optimization Demo")
print("=" * 50)

# Create training dataset for evolution
print("📚 Creating training dataset...")
trainset = [
    aspy.Example(question="Classify sentiment: 'I love this movie!'", answer="positive"),
    aspy.Example(question="Classify sentiment: 'This is terrible.'", answer="negative"),
    aspy.Example(question="Classify sentiment: 'It was okay, nothing special.'", answer="neutral"),
    aspy.Example(question="Classify sentiment: 'Amazing performance!'", answer="positive"),
    aspy.Example(question="Classify sentiment: 'Worst experience ever.'", answer="negative"),
    aspy.Example(question="Classify sentiment: 'Pretty good overall.'", answer="positive"),
    aspy.Example(question="Classify sentiment: 'Not bad, could be better.'", answer="neutral"),
    aspy.Example(question="Classify sentiment: 'Absolutely fantastic!'", answer="positive"),
]

# Create validation dataset
print("🔍 Creating validation dataset...")
valset = [
    aspy.Example(question="Classify sentiment: 'Great job!'", answer="positive"),
    aspy.Example(question="Classify sentiment: 'Very disappointing.'", answer="negative"),
    aspy.Example(question="Classify sentiment: 'It was fine.'", answer="neutral"),
    aspy.Example(question="Classify sentiment: 'Horrible quality.'", answer="negative"),
    aspy.Example(question="Classify sentiment: 'Excellent work!'", answer="positive"),
]

print(f"Training examples: {len(trainset)}")
print(f"Validation examples: {len(valset)}")
print()

# Create initial module
print("🏗️ Creating initial sentiment classifier...")
sentiment_classifier = aspy.Predict("question -> answer")

# Test baseline performance
print("📊 Testing baseline performance...")
baseline_evaluator = aspy.Evaluate(
    devset=valset,
    metric=aspy.exact_match,
    display_progress=True
)
baseline_result = baseline_evaluator(sentiment_classifier)
print(f"Baseline score: {baseline_result.score:.2f}%")
print()

# Setup GEPA optimizer
print("🔧 Setting up GEPA evolutionary optimizer...")
gepa_optimizer = aspy.GEPA(
    metric=aspy.exact_match,
    population_size=5,     # 5 candidates per generation
    num_generations=3,     # 3 evolutionary generations
    mutation_rate=0.3,     # 30% chance of mutation
    verbose=True,
    seed=42
)

# Run evolutionary optimization
print("🧬 Starting GEPA evolutionary optimization...")
print("This will evolve instructions over multiple generations...")
result = gepa_optimizer.compile(
    module=sentiment_classifier,
    trainset=trainset,
    valset=valset
)

print("\n" + "=" * 50)
print("🧬 EVOLUTIONARY RESULTS")
print("=" * 50)

print(f"🏆 Best evolved score: {result.best_score:.2f}%")
print(f"📊 Improvement: +{result.best_score - baseline_result.score:.2f}%")
print()

print("🧬 Evolution History:")
for gen_data in result.generations:
    gen = gen_data['generation']
    best = gen_data['best_score']
    avg = gen_data['avg_score']
    print(f"  Generation {gen}: Best={best:.2f}%, Avg={avg:.2f}%")

print()

print("🏆 Top 3 evolved candidates:")
# Sort all candidates by score
sorted_candidates = sorted(result.candidates, key=lambda x: x['score'], reverse=True)
for i, candidate in enumerate(sorted_candidates[:3]):
    score = candidate['score']
    instruction = candidate['instruction']
    parent = candidate.get('parent', 'unknown')

    if len(instruction) > 80:
        instruction = instruction[:80] + "..."

    print(f"  #{i+1}: {score:.2f}% - {instruction}")
    print(f"       Parent: {parent}")
    print()

# Test the evolved module
print("🧪 Testing evolved module on new examples...")
test_examples = [
    "Classify sentiment: 'This is the best thing ever!'",
    "Classify sentiment: 'I hate this so much.'",
    "Classify sentiment: 'It was alright, nothing special.'"
]

for question in test_examples:
    result_pred = result.best_module(question=question)
    answer = getattr(result_pred, 'answer', 'No answer field')
    print(f"Q: {question}")
    print(f"A: {answer}")
    print()

print("✅ GEPA evolutionary optimization demo complete!")
print("\n🧬 Key Benefits of GEPA:")
print("- Uses evolutionary principles (mutation, selection, crossover)")
print("- Explores diverse instruction variations")
print("- Finds instructions that work well across generations")
print("- Can discover unexpected but effective approaches")
print("- Maintains population diversity for robust optimization")

print("\n🔬 Evolution Process:")
print("1. Initialize population with instruction variants")
print("2. Evaluate fitness (performance) of each candidate")
print("3. Select best performers for reproduction")
print("4. Create offspring through crossover and mutation")
print("5. Repeat for multiple generations")
print("6. Return the best evolved instruction")