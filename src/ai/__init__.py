"""
Minimal dspy-style agent API with sync/async support.
Simple, Pythonic, no fancy wrappers.
"""
from .ai import LM, Predict, AgentResult, Eval, Module, Prediction
from .agent import (
    Task,
    TaskStatus,
    Handoff,
    AgentEvent,
    EventType,
    Scratchpad,
    TaskQueue,
    Planner,
    Worker,
    Orchestrator,
    run as agent_run,
    orchestrate,
)

__all__ = [
    'LM',
    'Predict',
    'Prediction',
    'Module',
    'AgentResult',
    'Eval',
    'configure',
    # Multi-agent system
    'Task',
    'TaskStatus',
    'Handoff',
    'AgentEvent',
    'EventType',
    'Scratchpad',
    'TaskQueue',
    'Planner',
    'Worker',
    'Orchestrator',
    'agent_run',
    'orchestrate',
]


def configure(lm: LM) -> None:
    """Configure default LM globally (dspy-style).

    Usage:
        import ai
        ai.configure(ai.LM(...))
    """
    LM.configure(lm)
