import time

from flowgen import LlamaCpp

def get_weather(city:str):
    return f'sunny in {city}'

llm = LlamaCpp(model="/home/ntlpt59/Downloads/LFM2-350M-F16.gguf")

st = time.perf_counter()
print(llm('what is weather in gurgaon?',tools=[get_weather]))
print(time.perf_counter() - st)
