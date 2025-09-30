# dspyxgepa.py
# pip install dspy-ai gepa

import dspy as d
from typing import Optional

from dspy.teleprompt.gepa.gepa_utils import DSPyTrace, ScoreWithFeedback

# 1) Configure the base LM for generation
model = "qwen3:1.7b"
lm = d.LM(f"ollama_chat/{model}", api_base="http://192.168.170.76:11434", api_key="")
d.configure(lm=lm)
print(lm('what is 2+3?'))

