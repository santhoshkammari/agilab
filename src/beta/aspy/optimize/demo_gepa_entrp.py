"""
Demo of GEPA evolutionary optimization in aspy
"""
from typing import Literal, List

from pydantic import BaseModel

import aspy
from aspy.lm.lm import LM
# from aspy.lm.lm_threadpooled import LM

# Setup with threadpool LM for batch processing
lm = LM(api_base="http://192.168.170.76:8000")
aspy.configure(lm=lm)

print("🧬 GEPA Evolutionary Optimization Demo")
print("=" * 50)

import requests
import dspy
import json
import random

def init_dataset():
    # Load from the url
    url = "https://raw.githubusercontent.com/meta-llama/llama-prompt-ops/refs/heads/main/use-cases/facility-support-analyzer/dataset.json"
    dataset = json.loads(requests.get(url).text)
    dspy_dataset = [
        dspy.Example({
            "message": d['fields']['input'],
            "answer": d['answer'],
        }).with_inputs("message")
        for d in dataset
    ]
    random.Random(0).shuffle(dspy_dataset)
    train_set = dspy_dataset[:int(len(dspy_dataset) * 0.33)]
    val_set = dspy_dataset[int(len(dspy_dataset) * 0.33):int(len(dspy_dataset) * 0.66)]
    test_set = dspy_dataset[int(len(dspy_dataset) * 0.66):]

    return train_set[:10], val_set[:10], test_set

trainset,valset,testset=init_dataset()

print(f"Training examples: {len(trainset)}")
print(f"Validation examples: {len(valset)}")
print()
print(trainset[0])

# Create initial module
print("🏗️ Creating initial sentiment classifier...")
class FacilitySupportAnalyze(BaseModel):
    urgency: Literal['low', 'medium', 'high']
    sentiment: Literal['positive', 'neutral', 'negative']
    categories: List[Literal["emergency_repair_services", "routine_maintenance_requests", "quality_and_safety_concerns", "specialized_cleaning_services", "general_inquiries", "sustainability_and_environmental_practices", "training_and_support_requests", "cleaning_services_scheduling", "customer_feedback_and_complaints", "facility_management_issues"]]

sentiment_classifier = aspy.Predict("message -> FacilitySupportAnalyze")


def score_urgency(gold_urgency, pred_urgency):
    """
    Compute score for the urgency module.
    """
    score = 1.0 if gold_urgency == pred_urgency else 0.0
    return score

def score_sentiment(gold_sentiment, pred_sentiment):
    """
    Compute score for the sentiment module.
    """
    score = 1.0 if gold_sentiment == pred_sentiment else 0.0
    return score

def score_categories(gold_categories, pred_categories):
    """
    Compute score for the categories module.
    Uses the same match/mismatch logic as category accuracy in the score.
    """
    correct = 0
    for k, v in gold_categories.items():
        if v and k in pred_categories:
            correct += 1
        elif not v and k not in pred_categories:
            correct += 1
    score = correct / len(gold_categories)
    return score

def metric(example, pred):
    """
    Computes a score based on agreement between prediction and gold standard for categories, sentiment, and urgency.
    Returns the score (float).
    """
    # Parse gold standard from example
    gold = json.loads(example['answer'])

    # Compute scores for all modules
    score_urgency_val = score_urgency(gold['urgency'], pred.urgency)
    score_sentiment_val = score_sentiment(gold['sentiment'], pred.sentiment)
    score_categories_val = score_categories(gold['categories'], pred.categories)

    # Overall score: average of the three accuracies
    total = (score_urgency_val + score_sentiment_val + score_categories_val) / 3

    return total

# Test baseline performance
print("📊 Testing baseline performance...")
baseline_evaluator = aspy.Evaluate(
    devset=valset,
    metric=metric,
    display_progress=True
)
baseline_result = baseline_evaluator(sentiment_classifier)
print(f"Baseline score: {baseline_result.score:.2f}%")
print()

# Setup GEPA optimizer
print("🔧 Setting up GEPA optimizer...")
gepa_optimizer = aspy.GEPA(
    metric=metric,
    minibatch_size=3,      # Size of minibatch for evaluation
    budget=50,             # Total rollout budget
    pareto_set_size=5,     # Use valset size for Pareto evaluation
    verbose=True,
    seed=42
)

# Run GEPA optimization
print("🧬 Starting GEPA optimization...")
print("This will use reflective prompt mutation and Pareto-based selection...")
result = gepa_optimizer.compile(
    module=sentiment_classifier,
    trainset=trainset,
    valset=valset  # Use same valset as baseline for fair comparison
)

print("\n" + "=" * 50)
print("🧬 GEPA OPTIMIZATION RESULTS")
print("=" * 50)

print(f"🏆 Best optimized score: {result.best_score:.2f}%")
print(f"📊 Improvement: +{result.best_score - baseline_result.score:.2f}%")
print(f"🎯 Rollouts used: {gepa_optimizer.rollouts_used}")
print()

print("🏆 Top 3 optimized candidates:")
# Sort all candidates by score
sorted_candidates = sorted(result.candidates, key=lambda x: x.get('avg_score', 0), reverse=True)
for i, candidate in enumerate(sorted_candidates[:3]):
    score = candidate.get('avg_score', 0)
    iteration = candidate.get('iteration', 'unknown')
    improved = candidate.get('minibatch_improvement', False)
    parent = candidate.get('parent', 'unknown')

    print(f"  #{i+1}: {score:.2f}% - Iteration {iteration} (Improved: {improved})")
    print(f"       Parent: {parent}")
    print()

# Test the optimized module
print("🧪 Testing optimized module on new examples...")
test_examples = [
    "Classify sentiment: 'This is the best thing ever!'",
    "Classify sentiment: 'I hate this so much.'",
    "Classify sentiment: 'It was alright, nothing special.'"
]

for message in test_examples:
    result_pred = result.best_module(message=message)
    answer = getattr(result_pred, 'answer', 'No answer field')
    print(f"Q: {message}")
    print(f"A: {answer}")
    print()

print("✅ GEPA optimization demo complete!")
print("\n🧬 Key Benefits of GEPA:")
print("- Uses reflective prompt mutation with execution traces")
print("- Pareto-based candidate selection prevents local optima")
print("- Sample-efficient optimization with minibatch evaluation")
print("- Can discover targeted instruction improvements")
print("- Maintains candidate diversity through Pareto frontiers")

print("\n🔬 GEPA Process (from research paper):")
print("1. Split dataset into feedback and Pareto evaluation sets")
print("2. Select candidate using Pareto-based sampling")
print("3. Gather execution traces and feedback on minibatch")
print("4. Use reflective mutation to improve instructions")
print("5. Evaluate improvement and add to candidate pool")
print("6. Repeat until budget exhausted")