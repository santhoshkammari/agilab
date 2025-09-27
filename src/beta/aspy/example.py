# """
# vllm serve Qwen/Qwen3-4B-Instruct-2507 --gpu-memory-utilization 0.4 --max-model-len 10k
#
#
# #
# # prompt created via XML tagging + dspy signature to prompt style. + followed by output + inputfields.
# # example:
# # <system_prompt>
# # </system_prompt>
# # <output_format>
# # </output_format>
# # <static_input_fields>
# # <static_input_fields>
# # <dynamic_user_level_input_fields>
# # </dynamic_user_level_input_fields>
# # '''
#
#
# """
# import aspy as a
# lm = a.LM(api_base="http://192.168.170.76:8000")
# a.configure(lm=lm)
# sig = a.Signature("question -> answer:float")
# math = a.Predict(sig)
# result = math(question="Two dice are tossed. What is the probability that the sum equals two?")
# print(result)
#
# print("\n=== Testing Predict ===")
# predictor = a.Predict("query -> two_lines")
# result = predictor(query="What is the capital of France?")
# print(result)
#
# print("\n=== Testing Multi-stage Module ===")
#
#
# class DraftArticle(a.Module):
#     def __init__(self):
#         super().__init__()
#         self.build_outline = a.ChainOfThought("topic -> title, two_sections:[str]")
#         self.draft_section = a.ChainOfThought("topic, section_heading -> content")
#
#     def forward(self, topic):
#         outline = self.build_outline(topic=topic)
#         sections = []
#
#         # Handle sections list
#         if hasattr(outline, 'two_sections') and outline.two_sections:
#             for heading in outline.two_sections:
#                 section = self.draft_section(topic=outline.title, section_heading=f"## {heading}")
#                 sections.append(section.content)
#
#         return a.Prediction(title=outline.title, sections=sections)
#
#
# # Zero configuration needed - everything just works!
# draft_article = DraftArticle()
# article = draft_article(topic="World Cup 2002")
# print(article)


import aspy

# Setup
lm = aspy.LM(api_base="http://192.168.170.76:8000")
aspy.configure(lm=lm)

# Create some test examples
examples = [
  aspy.Example(question="What is 2+2?", answer="4"),
  aspy.Example(question="Capital of France?", answer="Paris"),
]

# Create a module to evaluate
math_qa = aspy.Predict("question -> answer")

# Evaluate with progress bar and nice scoring
evaluator = aspy.Evaluate(
  devset=examples,
  metric=aspy.exact_match,
  display_progress=True,
  save_as_json="results.json"
)

result = evaluator(math_qa)
print(f"Final score: {result.score}%")