from typing import Union
from ..signature.signature import Signature
from ..lm.lm import LM
import json


class Prediction:
    """Container for module predictions with dynamic attribute access."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        repr_attrs = ",".join([f"{k}={v}" for k,v in attrs.items()])
        return f"Prediction({repr_attrs})"


class Module:
    """Base class for aspy modules, similar to dspy.Module."""

    def __init__(self):
        self._lm = None

    def set_lm(self, lm: LM):
        """Set the language model for this module."""
        self._lm = lm

    def forward(self, *args, **kwargs):
        """Override this method in subclasses."""
        raise NotImplementedError("Subclasses must implement forward()")

    def __call__(self, *args, **kwargs):
        """Make the module callable."""
        return self.forward(*args, **kwargs)


class Predict(Module):
    """Basic prediction module that directly calls LM with signature."""

    def __init__(self, signature: Union[str, Signature], lm: LM = None):
        super().__init__()
        # Handle string->Signature conversion internally
        if isinstance(signature, str):
            self.signature = Signature(signature)
        else:
            self.signature = signature

        if lm:
            self.set_lm(lm)

    def _build_prompt(self, **inputs):
        """Build messages for chat-based LMs with XML structure."""
        Input, Output = self.signature()
        input_instance = Input(**inputs)

        # Get output schema for structured generation
        output_schema = Output.model_json_schema()

        # System message with output format
        system_content = f"""<system_prompt>
You are a helpful assistant that provides accurate responses.
</system_prompt>

<output_format>
Respond with valid JSON matching this schema:
{json.dumps(output_schema, indent=2)}
</output_format>"""

        # User message with inputs (cache-friendly - dynamic at end)
        user_parts = ["<static_input_fields>"]
        if self.signature.instructions:
            user_parts.append(f"Instructions: {self.signature.instructions}")
        user_parts.extend([
            "</static_input_fields>",
            "",
            "<dynamic_user_level_input_fields>"
        ])

        for field_name, value in input_instance.model_dump().items():
            user_parts.append(f"{field_name}: {value}")

        user_parts.append("</dynamic_user_level_input_fields>")

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": "\n".join(user_parts)}
        ]

    def _get_response_format(self):
        """Get response_format for structured output."""
        _, Prediction = self.signature()
        schema = Prediction.model_json_schema()

        return {
            "type": "json_schema",
            "json_schema": {
                "name": "Prediction",
                "schema": schema,
                "strict": True
            }
        }

    def forward(self, **inputs):
        """Forward pass through the language model."""
        # Try to get LM: explicit -> context -> error
        lm = self._lm
        if not lm:
            from . import get_lm
            lm = get_lm()

        if not lm:
            raise ValueError("No language model set. Use aspy.configure(lm=...) or pass lm in constructor.")

        messages = self._build_prompt(**inputs)
        response_format = self._get_response_format()

        # Call LM with structured output
        response = lm(messages, response_format=response_format)

        # Extract content from response
        content = response["choices"][0]["message"]["content"]

        # Parse JSON response
        try:
            output_data = json.loads(content)
        except json.JSONDecodeError:
            # Fallback: create output with raw content
            output_fields = list(self.signature.output_types.keys())
            if len(output_fields) == 1:
                output_data = {output_fields[0]: content}
            else:
                output_data = {field: content for field in output_fields}

        # Create Prediction instance and return as Prediction
        _, Prediction = self.signature()
        try:
            prediction_instance = Prediction(**output_data)
            return Prediction(**prediction_instance.model_dump())
        except Exception:
            # Fallback to raw prediction
            return Prediction(**output_data)


class ChainOfThought(Predict):
    """Chain of thought reasoning module."""

    def _build_prompt(self, **inputs):
        """Build messages for chain of thought reasoning."""
        messages = super()._build_prompt(**inputs)

        # Enhance system message for reasoning
        messages[0]["content"] += """

<reasoning_instructions>
Think step by step before providing your final answer. Show your reasoning process.
</reasoning_instructions>"""

        return messages