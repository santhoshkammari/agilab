"""
Panda Agent — scratchpad-driven stateless agent with ChromaDB brain.

Agent code owns the scratchpad: creates instructions, tracks progress, marks done.
LLM only gets 3 tools (search, store, query) and is told "execute this instruction".
"""
import sys
import os
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'mcp_tools'))

from ai import LM
from research.memory import init as init_memory
from research.agent import run_agent_cycle
from panda.tools import ALL_TOOLS, set_current_task, set_answer_path

_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), 'SCRATCHPAD_TEMPLATE.md')


class PandaAgent:

    def __init__(
        self,
        task: str = "",
        scratchpad_path: str = "SCRATCHPAD.md",
        chromadb_path: str = ".chromadb",
        api_base: str = "http://192.168.170.76:8000",
        model: str = "",
        api_key: str = "-",
        max_steps: int = 100,
        max_turns_per_step: int = 6,
        temperature: float = 0.7,
        verbose: bool = True,
    ):
        self.task = task
        self.scratchpad_path = os.path.abspath(scratchpad_path)
        self.chromadb_path = chromadb_path
        self.max_steps = max_steps
        self.max_turns_per_step = max_turns_per_step
        self.verbose = verbose

        self.answer_path = os.path.join(os.path.dirname(self.scratchpad_path), 'answer.md')

        init_memory(chromadb_path)
        set_current_task(task)
        set_answer_path(self.answer_path)

        self.lm = LM(
            model=model,
            api_base=api_base,
            api_key=api_key,
            temperature=temperature,
        )

        if not os.path.exists(self.scratchpad_path):
            self._init_scratchpad()

    def _init_scratchpad(self):
        with open(_TEMPLATE_PATH, 'r') as f:
            template = f.read()
        content = template.replace('{task}', self.task)
        with open(self.scratchpad_path, 'w') as f:
            f.write(content)
        if self.verbose:
            print(f"  [init] Scratchpad: {self.scratchpad_path}")

    def _read_scratchpad(self) -> str:
        with open(self.scratchpad_path, 'r') as f:
            return f.read()

    def _write_scratchpad(self, content: str):
        with open(self.scratchpad_path, 'w') as f:
            f.write(content)

    # ------------------------------------------------------------------
    # Instruction management (all done by agent code, NOT by LLM)
    # ------------------------------------------------------------------

    def _has_instructions(self) -> bool:
        content = self._read_scratchpad()
        for line in content.split('\n'):
            if line.strip() == '# INSTRUCTIONS':
                return True
        return False

    def _create_instructions(self):
        """Ask LLM to plan instructions, then append to scratchpad."""
        response = run_agent_cycle(
            lm=self.lm,
            system_prompt="You are a task planner. Output ONLY instruction lines. No explanations, no markdown, no extra text.",
            user_input=f"""Break this task into minimal sequential instructions.

TASK: {self.task}

Available actions: SEARCH, QUERY, STORE, ANSWER
- SEARCH: web search + auto-scrape + auto-store in memory. No separate STORE needed after SEARCH.
- QUERY: recall from ChromaDB (mind or memory)
- STORE: manually store insight to mind collection
- ANSWER: append summary/report to answer.md

FORMAT (one per line, nothing else):
- [PENDING] ACTION "details" (0/N)

IMPORTANT:
- If a task says "N times", use ONE line with counter (0/N) — NOT N separate lines
- SEARCH already stores results in memory, so don't add a separate STORE after SEARCH
- End with an ANSWER instruction to write the final output
- Keep it minimal: fewest instructions possible

Example for "Search 5 articles about AI and summarize":
- [PENDING] SEARCH "AI articles" (0/5)
- [PENDING] ANSWER "summary of AI articles from memory and mind" (0/1)""",
            tools=[],
            max_turns=1,
            verbose=self.verbose,
        )

        # Append to scratchpad
        content = self._read_scratchpad()
        content += f"\n\n# INSTRUCTIONS\n{response.strip()}\n"
        self._write_scratchpad(content)

        if self.verbose:
            print(f"  [plan] Instructions:\n{response.strip()}")

    def _get_next_instruction(self) -> str | None:
        """Find first [PENDING] or [IN_PROGRESS] instruction."""
        content = self._read_scratchpad()
        in_instructions = False
        for line in content.split('\n'):
            if line.strip() == '# INSTRUCTIONS':
                in_instructions = True
                continue
            if in_instructions and ('[PENDING]' in line or '[IN_PROGRESS]' in line):
                return line.strip()
        return None

    def _mark_done(self, instruction: str):
        """Mark an instruction as done and increment counter."""
        content = self._read_scratchpad()
        updated = instruction

        # Increment counter: (X/Y) -> (X+1/Y)
        m = re.search(r'\((\d+)/(\d+)\)', updated)
        if m:
            current, total = int(m.group(1)), int(m.group(2))
            new_current = current + 1
            updated = updated.replace(f'({current}/{total})', f'({new_current}/{total})')
            if new_current >= total:
                updated = re.sub(r'\[PENDING\]|\[IN_PROGRESS\]', '[DONE]', updated)
            else:
                updated = re.sub(r'\[PENDING\]', '[IN_PROGRESS]', updated)
        else:
            updated = re.sub(r'\[PENDING\]|\[IN_PROGRESS\]', '[DONE]', updated)

        content = content.replace(instruction, updated)
        self._write_scratchpad(content)
        return updated

    def _is_all_done(self) -> bool:
        return self._has_instructions() and self._get_next_instruction() is None

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def step(self, instruction: str) -> str:
        """Execute one instruction. Scratchpad = system prompt, instruction = user input."""
        scratchpad = self._read_scratchpad()

        response = run_agent_cycle(
            lm=self.lm,
            system_prompt=scratchpad,
            user_input=f"Execute this instruction using your tools:\n{instruction}",
            tools=ALL_TOOLS,
            max_turns=self.max_turns_per_step,
            verbose=self.verbose,
        )
        return response

    def run(self, max_steps: int = None) -> str:
        max_steps = max_steps or self.max_steps

        print(f"{'='*60}")
        print(f"PANDA AGENT")
        print(f"Task: {self.task}")
        print(f"Scratchpad: {self.scratchpad_path}")
        print(f"{'='*60}\n")

        # Step 0: create instructions if needed
        if not self._has_instructions():
            print("--- Planning instructions ---")
            self._create_instructions()

        for step_num in range(1, max_steps + 1):
            instruction = self._get_next_instruction()
            if instruction is None:
                print(f"\n{'='*60}")
                print(f"ALL DONE after {step_num - 1} steps")
                print(f"{'='*60}")
                return "ALL_DONE"

            print(f"\n--- Step {step_num}/{max_steps}: {instruction} ---")

            try:
                response = self.step(instruction)
            except Exception as e:
                print(f"  [error] {e}")
                continue

            print(f"  [response] {response[:200]}")

            # Agent code marks done (not the LLM)
            updated = self._mark_done(instruction)
            print(f"  [updated] {updated}")

        print(f"\n{'='*60}")
        print(f"MAX STEPS reached ({max_steps})")
        print(f"{'='*60}")
        return "MAX_STEPS_REACHED"

    def reset(self):
        from research.memory import _get_client
        client = _get_client()
        for name in ["mind", "memory"]:
            try:
                client.delete_collection(name)
            except Exception:
                pass
        for f in [self.scratchpad_path, self.answer_path]:
            if os.path.exists(f):
                os.remove(f)
        self._init_scratchpad()
        print("  [reset] Done.")
