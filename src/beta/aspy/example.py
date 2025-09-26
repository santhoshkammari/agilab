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
res = lm([[{"role":"user","content":"tell me a fact in 5 lines"}]]*12, temperature=2)
print(res)






