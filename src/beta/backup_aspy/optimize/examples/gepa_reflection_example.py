#!/usr/bin/env python3
"""
Example showing GEPA with separate reflection LM, similar to DSPy GEPA.

This demonstrates cost optimization by using a cheaper model for reflection
while keeping the main generation model for actual task execution.
"""

import sys
sys.path.append('..')

from aspy.lm.lm import LM
from aspy.predict.predict import Predict
from aspy.evaluate import Example
from aspy.optimize.gepa import GEPA

# 1) Configure the base LM for generation
base_lm = LM(model="vllm:gpt-4o-mini")

# 2) Exact-match metric (GEPA expects 5 args according to paper)
def exact_match(gold, pred, trace, pred_name, pred_trace):
    return float(pred.a == gold.a)

# 3) Define the tiny program and a one-shot train example
program = Predict('q -> a', lm=base_lm)
trainset = [Example(q='2+2?', a='4').with_inputs('q')]

# 4) Instantiate GEPA with separate reflection LM
reflection_lm = LM(model="vllm:gpt-3.5-turbo")  # Cheaper model for reflection
gepa = GEPA(
    metric=exact_match,
    reflection_lm=reflection_lm,
    budget=100,
    minibatch_size=3,
    verbose=True
)

# 5) Compile (optimize) the program
optimized_program = gepa.compile(program, trainset=trainset)

# 6) Run the optimized program
print(f"Result: {optimized_program.best_module(q='2+2?').a}")  # -> '4'

print("\n🎯 Benefits of separate reflection LM:")
print("💰 Cost savings: Use cheaper model for analysis")
print("⚡ Performance: Specialized models for different tasks")
print("🔧 Flexibility: Easy to experiment with different reflection models")