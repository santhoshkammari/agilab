import gepa
import os

os.environ["OPENAI_BASE_URL"] = "http://192.168.170.76:8000/v1"
os.environ["OPENAI_API_KEY"] = "sk-xxx"


# Load AIME dataset
trainset, valset, _ = gepa.examples.aime.init_dataset()

seed_prompt = {
    "system_prompt": "You are a helpful assistant. You are given a question and you need to answer it. The answer should be given at the end of your response in exactly the format '### <final answer>'"
}

# Let's run GEPA optimization process.
gepa_result = gepa.optimize(
    seed_candidate=seed_prompt,
    trainset=trainset,
    valset=valset,
    task_lm="openai/", # <-- This is the model being optimized
    max_metric_calls=150, # <-- Set a budget
    reflection_lm="openai/", # <-- Use a strong model to reflect on mistakes and propose better prompts
)

print("GEPA Optimized Prompt:", gepa_result.best_candidate['system_prompt'])
