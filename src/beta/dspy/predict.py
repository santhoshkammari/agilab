from typing import Union
from .signature import Signature
from .lm import LM
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

    def __init__(self, signature: Union[str, Signature],instructions:str=None ,lm: LM = None, tools=None):
        super().__init__()
        # Handle string->Signature conversion internally
        if isinstance(signature, str):
            self.signature = Signature(signature,instructions=instructions)
        else:
            self.signature = signature

        if lm:
            self.set_lm(lm)

        self.tools = tools

    def _build_prompt(self, **inputs):
        """Build messages for chat-based LMs with XML structure."""
        Input, Output = self.signature()
        input_instance = Input(**inputs)

        # Different prompts for tool mode vs structured output mode
        if self.tools:
            # Tool mode: simpler prompt, let model decide to use tools
            system_content = """<system_prompt>
You are a helpful assistant that can use tools to answer questions.
When you need information that requires external data or computation, use the available tools.
</system_prompt>"""

            user_parts = []
            if self.signature.instructions:
                user_parts.append(f"Instructions: {self.signature.instructions}\n")

            for field_name, value in input_instance.model_dump().items():
                user_parts.append(f"{field_name}: {value}")

            return [
                {"role": "system", "content": system_content},
                {"role": "user", "content": "\n".join(user_parts)}
            ]
        else:
            # Structured output mode: strict schema enforcement
            output_schema = Output.model_json_schema()

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
            raise ValueError("No language model set. Use dspy.configure(lm=...) or pass lm in constructor.")

        messages = self._build_prompt(**inputs)

        # Decision: Tool mode OR Structured output mode (not both)
        if self.tools:
            # Tool calling mode - no strict response format
            params = {"tools": self.tools}
            response = lm(messages, **params)

            message = response["choices"][0]["message"]

            # Check if model called a tool
            if "tool_calls" in message and message["tool_calls"]:
                return Prediction(
                    tool_calls=message["tool_calls"],
                    content=message.get("content", "")
                )
            else:
                # No tool call - try to parse text response into signature output
                content = message.get("content", "")
                return self._parse_text_to_prediction(content)
        else:
            # Structured output mode - strict JSON schema
            response_format = self._get_response_format()
            params = {"response_format": response_format}
            response = lm(messages, **params)

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
            _, OutputModel = self.signature()
            try:
                prediction_instance = OutputModel(**output_data)
                return Prediction(**prediction_instance.model_dump())
            except Exception:
                # Fallback to raw prediction
                return Prediction(**output_data)

    def _parse_text_to_prediction(self, content: str):
        """Parse text content into structured prediction when tools don't get called."""
        # Try JSON first
        try:
            output_data = json.loads(content)
        except json.JSONDecodeError:
            # Fallback: create output with raw content
            output_fields = list(self.signature.output_types.keys())
            if len(output_fields) == 1:
                output_data = {output_fields[0]: content}
            else:
                output_data = {field: content for field in output_fields}

        _, OutputModel = self.signature()
        try:
            prediction_instance = OutputModel(**output_data)
            return Prediction(**prediction_instance.model_dump())
        except Exception:
            return Prediction(**output_data)

    def batch_forward(self, inputs_list):
        """Batch forward pass through the language model."""
        # Try to get LM: explicit -> context -> error
        lm = self._lm
        if not lm:
            from . import get_lm
            lm = get_lm()

        if not lm:
            raise ValueError("No language model set. Use dspy.configure(lm=...) or pass lm in constructor.")

        # Build batch of messages
        messages_batch = []
        for inputs in inputs_list:
            messages = self._build_prompt(**inputs)
            messages_batch.append(messages)

        response_format = self._get_response_format()

        # Call LM with batch of messages
        params = {"response_format": response_format}
        if self.tools:
            params["tools"] = self.tools
        responses = lm(messages_batch, **params)

        # Process batch responses
        predictions = []
        for response in responses:
            if isinstance(response, Exception):
                # Handle exceptions in batch
                predictions.append(Prediction(error=str(response)))
                continue

            try:
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

                # Create Prediction instance
                _, Prediction = self.signature()
                try:
                    prediction_instance = Prediction(**output_data)
                    predictions.append(Prediction(**prediction_instance.model_dump()))
                except Exception:
                    # Fallback to raw prediction
                    predictions.append(Prediction(**output_data))

            except Exception as e:
                predictions.append(Prediction(error=str(e)))

        return predictions


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