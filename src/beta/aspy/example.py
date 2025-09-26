import aspy as a

sig = a.Signature("question -> answer")
Input, Output = sig()

# Input model: one required str field "question"
print(Input.model_json_schema()["properties"])   # {'question': {'title': 'Question', 'type': 'string'}}

# Output model: one required str field "answer"
print(Output.model_json_schema()["properties"])  # {'answer': {'title': 'Answer', 'type': 'string'}}

sig2 = a.Signature("a:int, flags:[bool] -> ok:bool, meta:str")
I2, O2 = sig2()
print(I2.model_json_schema()["properties"])  # a:int, flags:list(bool)
print(O2.model_json_schema()["properties"])  # ok:bool, meta:dict(str, any) [optional]

#
# class Outline(a.Signature):
#     """Outline a thorough overview of a topic."""
#
#     topic: str = a.InputField()
#     title: str = a.OutputField()
#     sections: list[str] = a.OutputField() #or a.OutputField(pydanticBaseModelclassname) ex: a.OutputField(Person).
# '''
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

lm = a.LM(api_base="http://192.168.170.76:8000")

# Test basic ChainOfThought usage
print("\n=== Testing ChainOfThought ===")
math = a.ChainOfThought("question -> answer: float")
math.set_lm(lm)
result = math(question="Two dice are tossed. What is the probability that the sum equals two?")
print(result)

# Test basic Predict usage
print("\n=== Testing Predict ===")
predictor = a.Predict("query -> response")
predictor.set_lm(lm)
result = predictor(query="What is the capital of France?")
print(result)

# Test multi-stage module like the DraftArticle example
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


