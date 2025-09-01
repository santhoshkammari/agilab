
class BaseTemplate:
    """Base template class for managing prompts."""

    def get_system_prompt(self, **kwargs) -> str:
        """Get system prompt. Should be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement get_system_prompt")

    def get_user_prompt(self, **kwargs) -> str:
        """Get user prompt. Should be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement get_user_prompt")