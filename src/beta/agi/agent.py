import requests
from transformers.utils import get_json_schema

class LM:
    def __init__(self, api_base="http://localhost:11434", model: str = ""):
        self.model = model
        self.api_base = api_base

    def __call__(self,**kwargs):
        if 'tools' in kwargs:
            kwargs['tools'] = [get_json_schema(f) if callable(f) else f for f in kwargs['tools']]
        return self.call_llm(**kwargs)

    def call_llm(self, messages, **params):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        response = requests.post(url=f"{self.api_base}/v1/chat/completions",
                                json={"model": self.model, "messages": messages, **params},
                                stream=params.get('stream', False))
        if params.get('stream', False):
            return response
        return response.json()


def get_weather(location: str, unit: str = "celsius"):
    """Get the weather for a location.

    Args:
        location: The city name
        unit: Temperature unit (celsius or fahrenheit)
    """
    return f"Weather in {location}: 22°{unit[0].upper()}"

if __name__ == '__main__':
    # lm = LM(api_base='http://192.168.170.76:8000')
    # agent = Agent(tools=[get_weather])
    # print(lm(messages = 'what is weather in gurgaon /no_think',tools=[get_weather],stream=True))

    lm = LM(api_base='http://192.168.170.76:8000')
    response = lm(messages='what is weather in gurgaon', tools=[get_weather], stream=True)

    # Iterate over chunks
    for line in response.iter_lines():
        if line:
            print(line.decode('utf-8'))
