"""
Configuration management for Claude chat application.
"""

class Config:
    """Application configuration settings."""
    
    def __init__(self):
        # Provider selection (ollama, vllm, oai)
        self.provider = "ollama"  # Default to ollama
        
        # Legacy settings (for backward compatibility)
        self.host = None  # None means localhost:11434
        self.model = "qwen3:4b"
        self.num_ctx = 2048
        self.thinking_enabled = False  # Default to thinking disabled
        
        # Provider-specific configurations
        self.providers = {
            "ollama": {
                "host": "localhost:11434",
                "model": "qwen3:4b",
                "num_ctx": 2048,
                "temperature": 0.7
            },
            "vllm": {
                "host": "http://localhost:8000",
                "model": "microsoft/DialoGPT-medium",
                "num_ctx": 4096,
                "temperature": 0.7
            },
            "oai": {
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "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",
                "model": "google/gemini-2.0-flash-exp:free",
                "num_ctx": 4096,
                "temperature": 0.7
            }
        }
    
    def set_provider(self, provider: str) -> None:
        """Set the LLM provider."""
        if provider not in self.providers:
            raise ValueError(f"Unknown provider: {provider}. Available: {list(self.providers.keys())}")
        self.provider = provider
        
        # Update legacy settings for backward compatibility
        provider_config = self.providers[provider]
        if provider == "ollama":
            self.host = provider_config["host"] if provider_config["host"] != "localhost:11434" else None
            self.model = provider_config["model"]
            self.num_ctx = provider_config["num_ctx"]
        elif provider == "vllm":
            self.host = provider_config["host"]
            self.model = provider_config["model"]
            self.num_ctx = provider_config["num_ctx"]
        elif provider == "oai":
            # For OAI, host is stored as base_url
            self.host = provider_config["base_url"]
            self.model = provider_config["model"]
            self.num_ctx = provider_config["num_ctx"]
    
    def get_provider_config(self, provider: str = None) -> dict:
        """Get configuration for specific provider."""
        provider = provider or self.provider
        return self.providers.get(provider, {})
    
    def get_current_config(self) -> dict:
        """Get current provider configuration."""
        return self.get_provider_config(self.provider)
    
    def set_provider_option(self, provider: str, key: str, value) -> None:
        """Set a specific option for a provider."""
        if provider not in self.providers:
            raise ValueError(f"Unknown provider: {provider}")
        self.providers[provider][key] = value
        
        # Update legacy settings if current provider
        if provider == self.provider:
            self.set_provider(provider)
    
    def set_host(self, host: str) -> None:
        """Set the LLM host URL (legacy compatibility)."""
        if self.provider == "ollama":
            self.providers["ollama"]["host"] = host
            self.host = host if host != "localhost:11434" else None
        elif self.provider == "vllm":
            self.providers["vllm"]["host"] = host
            self.host = host
        elif self.provider == "oai":
            self.providers["oai"]["base_url"] = host
            self.host = host
    
    def set_model(self, model: str) -> None:
        """Set the LLM model (legacy compatibility)."""
        self.providers[self.provider]["model"] = model
        self.model = model
    
    def get_host_display(self) -> str:
        """Get host for display purposes."""
        current_config = self.get_current_config()
        if self.provider == "ollama":
            host = current_config.get("host", "localhost:11434")
            return host
        elif self.provider == "vllm":
            return current_config.get("host", "localhost:8000")
        elif self.provider == "oai":
            return current_config.get("base_url", "https://api.openai.com/v1")
        return self.host if self.host else "localhost:11434"
    
    def get_provider_display(self) -> str:
        """Get current provider for display."""
        return f"{self.provider.upper()}"

# Global config instance
config = Config()
