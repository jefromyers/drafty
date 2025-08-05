"""Base LLM provider interface."""

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


class LLMMessage(BaseModel):
    """Represents a message in the conversation."""

    role: str  # "system", "user", "assistant"
    content: str
    name: Optional[str] = None
    extra: Dict[str, Any] = {}


class LLMResponse(BaseModel):
    """Represents a response from an LLM."""

    content: str
    model: str
    provider: str
    usage: Optional[Dict[str, int]] = None
    raw_response: Optional[Any] = None
    extra: Dict[str, Any] = {}

    def as_json(self) -> Optional[Dict[str, Any]]:
        """Parse content as JSON if possible."""
        try:
            return json.loads(self.content)
        except (json.JSONDecodeError, TypeError):
            return None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = config.get("model", "")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens")
        self.timeout = config.get("timeout", 60)
        self.json_mode = config.get("json_mode", False)

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name."""
        pass

    @abstractmethod
    async def generate(
        self,
        messages: List[LLMMessage],
        **kwargs,
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    async def generate_stream(
        self,
        messages: List[LLMMessage],
        **kwargs,
    ):
        """Generate a streaming response from the LLM."""
        pass

    def prepare_messages(
        self, prompt: str, system_prompt: Optional[str] = None, history: Optional[List[LLMMessage]] = None
    ) -> List[LLMMessage]:
        """Prepare messages for the LLM."""
        messages = []

        if system_prompt:
            messages.append(LLMMessage(role="system", content=system_prompt))

        if history:
            messages.extend(history)

        messages.append(LLMMessage(role="user", content=prompt))

        return messages

    def validate_response(self, response: LLMResponse) -> bool:
        """Validate the response from the LLM."""
        if not response.content:
            return False

        if self.json_mode:
            return response.as_json() is not None

        return True

    async def generate_with_retry(
        self,
        messages: List[LLMMessage],
        max_retries: int = 3,
        **kwargs,
    ) -> LLMResponse:
        """Generate with automatic retry on failure."""
        last_error = None

        for attempt in range(max_retries):
            try:
                response = await self.generate(messages, **kwargs)
                if self.validate_response(response):
                    return response
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    await self._handle_retry(attempt, e)

        raise last_error or Exception("Failed to generate valid response")

    async def _handle_retry(self, attempt: int, error: Exception) -> None:
        """Handle retry logic."""
        import asyncio

        wait_time = 2 ** attempt  # Exponential backoff
        await asyncio.sleep(wait_time)


class LLMProviderFactory:
    """Factory for creating LLM providers."""

    _providers: Dict[str, type[LLMProvider]] = {}

    @classmethod
    def register(cls, name: str, provider_class: type[LLMProvider]) -> None:
        """Register a new provider."""
        cls._providers[name] = provider_class

    @classmethod
    def create(cls, provider_name: str, config: Dict[str, Any]) -> LLMProvider:
        """Create a provider instance."""
        if provider_name not in cls._providers:
            raise ValueError(f"Unknown provider: {provider_name}")

        provider_class = cls._providers[provider_name]
        return provider_class(config)

    @classmethod
    def list_providers(cls) -> List[str]:
        """List available providers."""
        return list(cls._providers.keys())