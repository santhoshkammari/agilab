"""
Configuration management for Claude chat application.
"""

class Config:
    """Application configuration settings."""
    
    def __init__(self):
        self.host = None  # None means localhost:11434
        self.model = "qwen3:0.6b"
        self.num_ctx = 2048
    
    def set_host(self, host: str) -> None:
        """Set the LLM host URL."""
        self.host = host if host != "localhost:11434" else None
    
    def set_model(self, model: str) -> None:
        """Set the LLM model."""
        self.model = model
    
    def get_host_display(self) -> str:
        """Get host for display purposes."""
        return self.host if self.host else "localhost:11434"

# Global config instance
config = Config()