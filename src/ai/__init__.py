"""
Minimal dspy-style agent API with sync/async support.
Simple, Pythonic, no fancy wrappers.
"""
from .ai import LM, Predict, AgentResult, Eval

__all__ = [
    'LM',
    'Predict',
    'AgentResult',
    'Eval',
    'configure',
]


def configure(lm: LM) -> None:
    """Configure default LM globally (dspy-style).

    Usage:
        import ai
        ai.configure(ai.LM(...))
    """
    LM.configure(lm)
