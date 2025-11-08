"""
Example class for aspy evaluation
"""
from typing import Dict, Any


class Example:
    """
    Simple example class similar to dspy.Example but simplified.

    Usage:
        example = Example(question="What is 2+2?", answer="4")
        inputs = example.inputs()  # Returns dict with input fields

        # Or specify explicit input fields:
        example = Example(question="What is 2+2?", context="Math problem", answer="4")
        example_with_inputs = example.with_inputs("question", "context")
        inputs = example_with_inputs.inputs()  # Only returns question and context
    """

    def __init__(self, **kwargs):
        """Initialize example with keyword arguments."""
        self._input_keys = None
        for key, value in kwargs.items():
            setattr(self, key, value)

    def with_inputs(self, *keys):
        """
        Set which fields should be treated as inputs.

        Args:
            *keys: Field names to treat as inputs

        Returns:
            A new Example instance with input keys set
        """
        copied = self.copy()
        copied._input_keys = set(keys)
        return copied

    def inputs(self) -> Dict[str, Any]:
        """
        Return a dictionary of input fields.

        If with_inputs() was called, returns only those specified fields.
        Otherwise, excludes common target fields like 'answer', 'output', etc.
        """
        all_attrs = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

        if self._input_keys is not None:
            # Return only explicitly specified input keys
            return {k: v for k, v in all_attrs.items() if k in self._input_keys}
        else:
            # Default behavior: exclude common target fields
            target_fields = {'answer', 'output', 'target', 'label'}
            return {k: v for k, v in all_attrs.items() if k not in target_fields}

    def copy(self, **kwargs):
        """
        Create a copy of this Example with optional additional kwargs.

        Returns:
            A new Example instance with the same data
        """
        # Copy all non-private attributes
        data = self.model_dump()
        data.update(kwargs)

        new_example = Example(**data)
        new_example._input_keys = self._input_keys
        return new_example

    def model_dump(self) -> Dict[str, Any]:
        """Return all attributes as a dictionary for JSON serialization."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def __repr__(self):
        attrs = self.model_dump()
        attr_str = ", ".join([f"{k}={repr(v)}" for k, v in attrs.items()])
        input_keys_str = f" (input_keys={self._input_keys})" if self._input_keys else ""
        return f"Example({attr_str}){input_keys_str}"