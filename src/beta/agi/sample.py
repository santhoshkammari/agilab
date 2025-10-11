
from agent import LM,configure
import asyncio

lm = LM(api_base='http://192.168.170.76:8000')
messages_batch = [
        [{"role": "user", "content": "What is 2+2? /no_think"}],
        [{"role": "user", "content": "What is 3+3?/no_think"}],
        [{"role": "user", "content": "What is the capital of France?/no_think"}]
    ]

res = asyncio.run(lm.async_batch_call_llm(messages_batch))
print(res)