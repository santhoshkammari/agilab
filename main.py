from llm import Ollama
from  tools import tools
llm = Ollama(host = "http://192.168.170.76:11434")

MODEL_NAME = "qwen3_14b_q6k"

SYSTEM_PROMT = """
You are Claude, an AI assistant with expertise in programming and software development.
Your task is to assist with coding-related questions, debugging, refactoring, and explaining code.


Guidelines:
- Provide clear, concise, and accurate responses
- Include code examples where helpful
- Prioritize modern best practices
- If you're unsure, acknowledge limitations instead of guessing
- Focus on understanding the user's intent, even if the question is ambiguous

Orchestration:
- Compulsory create todos and implement step-by-step implementation and update todo on the go. 

"""

for x in llm(
    # prompt="create sum.py file, simple one",
    # prompt="who won ipl, search web",
    prompt="Crete 3 files, each for sum, add, subtract, add main.py as well , then finally make it to package. "
           "First write todos",
    system_prompt=SYSTEM_PROMT,
    tools=tools,
    model=MODEL_NAME,
    num_ctx=8_000,
    temperature=0
):
    if x.message.tool_calls:
        print(x.message.tool_calls)
    else:
        print(x.message.content,end="",flush=True)