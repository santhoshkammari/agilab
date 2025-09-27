"""
vllm serve Qwen/Qwen3-4B-Instruct-2507 --gpu-memory-utilization 0.4 --max-model-len 10k


#
# prompt created via XML tagging + dspy signature to prompt style. + followed by output + inputfields.
# example:
# <system_prompt>
# </system_prompt>
# <output_format>
# </output_format>
# <static_input_fields>
# <static_input_fields>
# <dynamic_user_level_input_fields>
# </dynamic_user_level_input_fields>
# '''


"""
from pydantic import BaseModel, Field

class Dice(BaseModel):
    answer:float
    joke:str = Field(description='just 2 or 3 words')

import aspy as a


lm = a.LM(api_base="http://192.168.170.76:8000")
a.configure(lm=lm)

print("=== Testing Predict with float output ===")
sig = a.Signature("question -> answer:float")
math = a.Predict(sig)
print(math._build_prompt(question='what is 2+3?'))
exit()

from tqdm import trange
for x in trange(100):
    result = math(question="Two dice are tossed. What is the probability that the sum equals two?")
print(result)

exit()

print("\n=== Testing ChainOfThought with Dice class ===")
sig2 = a.Signature("question -> Dice")
cot = a.ChainOfThought(sig2)
result2 = cot(question="Two dice are tossed. What is the probability that the sum equals two?", temperature=0.1, seed=42)
print(result2)

exit()

print("\n=== Testing Predict ===")
predictor = a.Predict("query -> two_lines")
result = predictor(query="What is the capital of France?")
print(result)

print("\n=== Testing Multi-stage Module ===")


class DraftArticle(a.Module):
    def __init__(self):
        super().__init__()
        self.build_outline = a.ChainOfThought("topic -> title, sections:[str]")
        self.draft_section = a.ChainOfThought("topic, section_heading -> content")

    def forward(self, topic):
        outline = self.build_outline(topic=topic)
        sections = []

        # Handle sections list
        if hasattr(outline, 'sections') and outline.sections:
            for heading in outline.sections[:2]:  # Limit to 2 sections for demo
                section = self.draft_section(topic=outline.title, section_heading=f"## {heading}")
                sections.append(section.content)

        return a.Prediction(title=outline.title, sections=sections)


# Zero configuration needed - everything just works!
draft_article = DraftArticle()
article = draft_article(topic="World Cup 2002")
print(article)
