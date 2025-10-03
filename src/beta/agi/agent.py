import requests


class LM:
    def __init__(self, api_base="http://localhost:11434", model: str = ""):
        self.model = model
        self.api_base = api_base

    def __call__(self, messages, **params):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        return requests.post(url=f"{self.api_base}/v1/chat/completions",
                             json={"model": self.model, "messages": messages, **params}).json()

if __name__ == '__main__':
    lm = LM(api_base='http://192.168.170.76:8000')
    print(lm('what is 2+3 /no_think'))
