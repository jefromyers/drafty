"""Settings management for Drafty using environment variables."""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


class Settings:
    """Centralized settings management."""
    
    def __init__(self, env_file: Optional[Path] = None):
        """Initialize settings from environment variables.
        
        Args:
            env_file: Path to .env file (defaults to .env in project root)
        """
        # Load .env file
        if env_file:
            load_dotenv(env_file)
        else:
            # Try to find .env in common locations
            for path in [
                Path.cwd() / ".env",
                Path.home() / ".drafty" / ".env",
                Path(__file__).parent.parent.parent / ".env",
            ]:
                if path.exists():
                    load_dotenv(path)
                    break
            else:
                # Load from environment without file
                load_dotenv()
    
    # LLM Provider Keys
    @property
    def openai_api_key(self) -> Optional[str]:
        return os.getenv("OPENAI_API_KEY")
    
    @property
    def anthropic_api_key(self) -> Optional[str]:
        return os.getenv("ANTHROPIC_API_KEY")
    
    @property
    def gemini_api_key(self) -> Optional[str]:
        return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    @property
    def gemini_model(self) -> str:
        """Get default Gemini model from environment."""
        return os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
    
    # Data4SEO Credentials
    @property
    def data4seo_username(self) -> Optional[str]:
        return os.getenv("DATA4SEO_USERNAME")
    
    @property
    def data4seo_password(self) -> Optional[str]:
        return os.getenv("DATA4SEO_PASSWORD")
    
    @property
    def has_data4seo(self) -> bool:
        """Check if Data4SEO credentials are available."""
        return bool(self.data4seo_username and self.data4seo_password)
    
    # Browserless Configuration
    @property
    def browserless_url(self) -> str:
        return os.getenv("BROWSERLESS_URL", "http://localhost:3000")
    
    # Ollama Configuration
    @property
    def ollama_host(self) -> str:
        return os.getenv("OLLAMA_HOST", "http://localhost:11434")
    
    # API Base URLs (for custom endpoints)
    @property
    def openai_base_url(self) -> Optional[str]:
        return os.getenv("OPENAI_BASE_URL")
    
    @property
    def anthropic_base_url(self) -> Optional[str]:
        return os.getenv("ANTHROPIC_BASE_URL")
    
    # Default Settings
    @property
    def default_llm_provider(self) -> str:
        return os.getenv("DEFAULT_LLM_PROVIDER", "openai")
    
    @property
    def default_llm_model(self) -> Optional[str]:
        return os.getenv("DEFAULT_LLM_MODEL")
    
    @property
    def default_temperature(self) -> float:
        temp = os.getenv("DEFAULT_TEMPERATURE", "0.7")
        try:
            return float(temp)
        except ValueError:
            return 0.7
    
    # spaCy Configuration
    @property
    def spacy_model(self) -> str:
        return os.getenv("SPACY_MODEL", "en_core_web_sm")
    
    def get_llm_config(self, provider: str) -> dict:
        """Get configuration for a specific LLM provider.
        
        Args:
            provider: Provider name (openai, anthropic, gemini, ollama)
        """
        config = {}
        
        if provider == "openai":
            config["api_key"] = self.openai_api_key
            if self.openai_base_url:
                config["base_url"] = self.openai_base_url
            if self.default_llm_model and "gpt" in self.default_llm_model:
                config["model"] = self.default_llm_model
        
        elif provider == "anthropic":
            config["api_key"] = self.anthropic_api_key
            if self.anthropic_base_url:
                config["base_url"] = self.anthropic_base_url
            if self.default_llm_model and "claude" in self.default_llm_model:
                config["model"] = self.default_llm_model
        
        elif provider == "gemini":
            config["api_key"] = self.gemini_api_key
            # Use GEMINI_MODEL env var or default_llm_model if it's a gemini model
            if self.default_llm_model and "gemini" in self.default_llm_model:
                config["model"] = self.default_llm_model
            else:
                config["model"] = self.gemini_model
        
        elif provider == "ollama":
            config["base_url"] = self.ollama_host
            if self.default_llm_model:
                config["model"] = self.default_llm_model
        
        config["temperature"] = self.default_temperature
        
        return config
    
    def validate(self) -> dict:
        """Validate settings and return status.
        
        Returns:
            Dictionary with validation results
        """
        status = {
            "openai": bool(self.openai_api_key),
            "anthropic": bool(self.anthropic_api_key),
            "gemini": bool(self.gemini_api_key),
            "data4seo": self.has_data4seo,
            "configured_providers": [],
            "warnings": [],
        }
        
        # Check which providers are configured
        if status["openai"]:
            status["configured_providers"].append("openai")
        if status["anthropic"]:
            status["configured_providers"].append("anthropic")
        if status["gemini"]:
            status["configured_providers"].append("gemini")
        
        # Add warnings
        if not status["configured_providers"]:
            status["warnings"].append("No LLM providers configured. Add API keys to .env file.")
        
        if not status["data4seo"]:
            status["warnings"].append("Data4SEO not configured. Web search will use fallback mode.")
        
        return status
    
    def print_status(self):
        """Print current configuration status."""
        from rich.console import Console
        from rich.table import Table
        
        console = Console()
        status = self.validate()
        
        table = Table(title="Drafty Configuration Status")
        table.add_column("Service", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details")
        
        # LLM Providers
        table.add_row(
            "OpenAI",
            "✓" if status["openai"] else "✗",
            "API key configured" if status["openai"] else "Missing API key"
        )
        table.add_row(
            "Anthropic",
            "✓" if status["anthropic"] else "✗",
            "API key configured" if status["anthropic"] else "Missing API key"
        )
        table.add_row(
            "Gemini",
            "✓" if status["gemini"] else "✗",
            "API key configured" if status["gemini"] else "Missing API key"
        )
        
        # Data4SEO
        table.add_row(
            "Data4SEO",
            "✓" if status["data4seo"] else "✗",
            "Credentials configured" if status["data4seo"] else "Missing credentials"
        )
        
        # Browserless
        table.add_row(
            "Browserless",
            "◐",
            f"URL: {self.browserless_url}"
        )
        
        console.print(table)
        
        # Print warnings
        if status["warnings"]:
            console.print("\n[yellow]Warnings:[/yellow]")
            for warning in status["warnings"]:
                console.print(f"  • {warning}")
        
        # Print default provider
        console.print(f"\n[cyan]Default Provider:[/cyan] {self.default_llm_provider}")
        if self.default_llm_model:
            console.print(f"[cyan]Default Model:[/cyan] {self.default_llm_model}")


# Global settings instance
settings = Settings()