from src.llm import LLM

llm = LLM('http://192.168.170.76:8000')

response = llm('hai')
print(response)


from pydantic import BaseModel
class Person(BaseModel):
    name:str
    age:int

response = llm('hi',format=Person)
print(response)
