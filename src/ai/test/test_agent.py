import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ai
from ai.agent import orchestrate

"""
(.venv) ntlpt24@NTLPT24:~/buildmode/agilab/src/ai/test$ python test_agent.py 
Traceback (most recent call last):
  File "/home/ntlpt24/buildmode/agilab/src/ai/test/test_agent.py", line 1, in <module>
    import ai
ModuleNotFoundError: No module named 'ai'
"""

import sys
sys.path.append("../")  # Add parent directory to path to import ai

print("=== Configuring LM ===")
lm = ai.LM(
    model="Qwen/Qwen3-4B-Instruct-2507",
    api_base="http://192.168.170.76:8000",
    temperature=0.1
)
ai.configure(lm)

print("=== Defining Tools ===")
def read_file(file_path: str) -> str:
    """
    Reads the content of a file and returns it as a string.
    Args:
        file_path: The path to the file to be read.
    Returns:
        str: The content of the file.
    """
    with open(file_path, 'r') as f:
        return f.read()

def write_file(file_path: str, content: str) -> None:
    """
    Writes the given content to a file.
    Args:
        file_path: The path to the file to be written.
        content: The content to write to the file.
    """
    with open(file_path, 'w') as f:
        f.write(content)

handoffs = orchestrate(
    "Please read the content of 'input.txt', convert it to uppercase, and write the result to 'output.txt'.",
    tools=[read_file, write_file],
    workers=5,
)
for h in handoffs:
    print(h.summary)