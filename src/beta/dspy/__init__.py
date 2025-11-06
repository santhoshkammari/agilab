import contextvars

from .lm import LM
from .signature import Signature, register_type, get_registered_types
from .predict import Module, Predict, ChainOfThought, Prediction
from .evaluate import Evaluate, EvaluationResult, exact_match, contains_match
from .evaluate_example import Example
from .mipro import MIPROv2, OptimizationResult
from .gepa import GEPA, GEPAResult
from .agent import Agent, Event

# Global LM context
_current_lm = contextvars.ContextVar('lm', default=None)

def configure(lm=None, **kwargs):
    """Configure the framework with default LM and other settings."""
    if lm:
        _current_lm.set(lm)

def get_lm():
    """Get the current LM from context."""
    return _current_lm.get()

# Keep set_lm internal/private
def _set_lm(lm):
    """Internal: Set the default LM for all modules in this context."""
    _current_lm.set(lm)