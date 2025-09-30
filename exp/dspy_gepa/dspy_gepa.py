# dspyxgepa.py
# pip install dspy-ai gepa

import dspy as d
from typing import Optional

from dspy.teleprompt.gepa.gepa_utils import DSPyTrace, ScoreWithFeedback

# 1) Configure the base LM for generation
model = "qwen3:1.7b"
lm = d.LM(f"ollama_chat/{model}", api_base="http://192.168.170.76:11434", api_key="")
d.configure(lm=lm)
reflection_lm = lm
# 2) Create GEPA-compatible metric wrapper
from dspy.evaluate import answer_exact_match


def metric(
    gold: d.Example,
    pred:d.Prediction,
    trace: Optional[DSPyTrace] = None,
    pred_name: Optional[str] = None,
    pred_trace: Optional[DSPyTrace] = None,
) -> float | ScoreWithFeedback:
    """
    This function is called with the following arguments:
    - gold: The gold example.
    - pred: The predicted output.
    - trace: Optional. The trace of the program's execution.
    - pred_name: Optional. The name of the target predictor currently being optimized by GEPA, for which
        the feedback is being requested.
    - pred_trace: Optional. The trace of the target predictor's execution GEPA is seeking feedback for.

    Note the `pred_name` and `pred_trace` arguments. During optimization, GEPA will call the metric to obtain
    feedback for individual predictors being optimized. GEPA provides the name of the predictor in `pred_name`
    and the sub-trace (of the trace) corresponding to the predictor in `pred_trace`.
    If available at the predictor level, the metric should return {'score': float, 'feedback': str} corresponding
    to the predictor.
    If not available at the predictor level, the metric can also return a text feedback at the program level
    (using just the gold, pred and trace).
    If no feedback is returned, GEPA will use a simple text feedback consisting of just the score:
    f"This trajectory got a score of {score}."
    """
    return float(pred.a == gold.a)

# 3) Define the tiny program and a one-shot train example
program = d.Predict("q -> a")
trainset = [d.Example(q="2+2?", a="4").with_inputs("q")]

# 4) Instantiate GEPA and compile (optimize) the program
gepa = d.GEPA(
    metric=metric,
    reflection_lm=reflection_lm,
    auto="light",
)
m = gepa.compile(program, trainset=trainset)

# 5) Run the optimized program
result = m(q="2+2?")
print("Result:", result.a)

print("\n--- Optimized Program ---")
print(m)
print("\n--- Instructions ---")
for key, value in m.named_parameters():
    print(f"{key}: {value}")

