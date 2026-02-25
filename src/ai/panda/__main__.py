"""Allow running as: python -m panda"""
import argparse
from panda.agent import PandaAgent

parser = argparse.ArgumentParser(description="Panda Agent")
parser.add_argument("task", nargs="?",
                    default="Search for 3 articles about autonomous AI agents and store insights")
parser.add_argument("--scratchpad", default="SCRATCHPAD.md")
parser.add_argument("--model", default="")
parser.add_argument("--api-base", default="http://192.168.170.76:8000")
parser.add_argument("--chromadb-path", default=".chromadb")
parser.add_argument("--max-steps", type=int, default=50)
parser.add_argument("--verbose", action="store_true", default=True)
parser.add_argument("--fresh", action="store_true")
args = parser.parse_args()

agent = PandaAgent(
    task=args.task,
    scratchpad_path=args.scratchpad,
    model=args.model,
    api_base=args.api_base,
    chromadb_path=args.chromadb_path,
    max_steps=args.max_steps,
    verbose=args.verbose,
)

if args.fresh:
    agent.reset()

agent.run()
