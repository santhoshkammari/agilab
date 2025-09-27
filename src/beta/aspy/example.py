"""
vllm serve Qwen/Qwen3-4B-Instruct-2507 --gpu-memory-utilization 0.4 --max-model-len 10k
"""
from pydantic import BaseModel

import aspy as a


sig = a.Signature("question:str,age:int -> age:int,year:int,name:str")
lm = a.LM(api_base="http://192.168.170.76:8000")

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
math = a.ChainOfThought(sig)
math.set_lm(lm)
result = math(question="Two dice are tossed. What is the probability that the sum equals two?")
print(result)

print("\n=== Testing Predict ===")
predictor = a.Predict("query -> two_lines")
predictor.set_lm(lm)
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

draft_article = DraftArticle()
draft_article.set_lm(lm)
draft_article.build_outline.set_lm(lm)
draft_article.draft_section.set_lm(lm)

article = draft_article(topic="World Cup 2002")
print(article)


