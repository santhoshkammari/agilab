import requests
import json
from dataclasses import dataclass

@dataclass
class Chunk:
    text:str
    raw:str

async def llm(messages, base_url="http://192.168.170.76:8000", model="", **kwargs):
    """Streamable LLM function using requests.post with stream=True"""
    url = f"{base_url}/v1/chat/completions"

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        **kwargs
    }

    response = requests.post(url, json=payload, stream=True)
    response.raise_for_status()

    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                data = line[6:]  # Remove 'data: ' prefix
                if data == '[DONE]':
                    break
                chunk = json.loads(data)
                yield chunk


class Agent:
    def __init__(self,tools=None):
        self.tools = tools or {}
        self.system, self.header, self.body = "", [], []
        self.llm = llm
    
    def _build(self, ids=None):
        msgs = [{"role": "system", "content": self.system + "\n".join(self.header)}]
        msgs.extend(sum(([self.body[i] for i in ids] if ids else self.body), []))
        return msgs

    
    async def stream(self, msg, **kwargs):
        self.body.append([{"role": "user", "content": msg}])
        
        for _ in range(5):  # max tool rounds
            response, tool_calls = "", None
            
            async for chunk in self.llm(self._build(), **kwargs):
                response += chunk.text
                if hasattr(chunk, 'tool_calls'):
                    tool_calls = chunk.tool_calls
                yield chunk
            
            self.body[-1].append({"role": "assistant", "content": response})
            if not tool_calls:
                break
            
            for tc in tool_calls:
                result = await self.tools[tc.name](**tc.args)
                self.body[-1].append({"role": "tool", "content": result})
    
    def sub_agent(self, system, ids):
        sub = Agent(self.llm, self.tools)
        sub.system = self.system + "\n" + system
        sub.header = self.header
        sub.body = [self.body[i] for i in ids]
        return sub

if __name__=="__main__":
    import asyncio

    async def main():
        async for chunk in llm([{"role":"user","content":"hi, what is 2+3?"}],n=4):
            print(chunk, flush=True)
        print()

    asyncio.run(main())
