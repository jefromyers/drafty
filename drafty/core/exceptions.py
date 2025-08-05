"""Custom exceptions for Drafty."""

from typing import Any, Dict, Optional


class DraftyError(Exception):
    """Base exception for Drafty errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


class ConfigurationError(DraftyError):
    """Error in configuration or settings."""
    pass


class ProviderError(DraftyError):
    """Error from LLM provider."""
    pass


class WorkspaceError(DraftyError):
    """Error related to workspace operations."""
    pass


class ResearchError(DraftyError):
    """Error during research operations."""
    pass


class GenerationError(DraftyError):
    """Error during content generation."""
    pass


class ExportError(DraftyError):
    """Error during export operations."""
    pass


class ValidationError(DraftyError):
    """Error in data validation."""
    pass


class APIError(DraftyError):
    """Error from external API calls."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, **kwargs):
        super().__init__(message, kwargs)
        self.status_code = status_code


class RateLimitError(APIError):
    """Rate limit exceeded error."""
    pass


class AuthenticationError(APIError):
    """Authentication failed error."""
    pass