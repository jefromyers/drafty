"""OpenAI LLM provider implementation with structured outputs support."""

import json
import os
from typing import Any, Dict, List, Optional, Union

import openai
from openai import AsyncOpenAI
from pydantic import BaseModel

from drafty.providers.base import LLMMessage, LLMProvider, LLMProviderFactory, LLMResponse


class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation with structured outputs and json_mode support."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided")

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=config.get("base_url"),
            timeout=self.timeout,
        )
        
        # Default to gpt-4o-mini if no model specified
        if not self.model:
            self.model = "gpt-4o-mini"

    @property
    def name(self) -> str:
        return "openai"

    def _convert_messages(self, messages: List[LLMMessage]) -> List[Dict[str, Any]]:
        """Convert LLMMessage objects to OpenAI format."""
        openai_messages = []
        for msg in messages:
            openai_msg = {"role": msg.role, "content": msg.content}
            if msg.name:
                openai_msg["name"] = msg.name
            openai_messages.append(openai_msg)
        return openai_messages

    def _prepare_response_format(
        self, 
        json_mode: bool = False,
        json_schema: Optional[Dict[str, Any]] = None,
        response_model: Optional[type[BaseModel]] = None
    ) -> Optional[Dict[str, Any]]:
        """Prepare the response_format parameter based on the mode."""
        if response_model:
            # Use Pydantic model for structured output
            schema = response_model.model_json_schema()
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__.lower(),
                    "schema": schema,
                    "strict": True
                }
            }
        elif json_schema:
            # Use provided JSON schema for structured output
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": json_schema.get("name", "response"),
                    "schema": json_schema.get("schema", json_schema),
                    "strict": json_schema.get("strict", True)
                }
            }
        elif json_mode:
            # Use basic JSON mode (less strict, but works with more models)
            return {"type": "json_object"}
        
        return None

    async def generate(
        self,
        messages: List[LLMMessage],
        **kwargs,
    ) -> LLMResponse:
        """Generate a response from OpenAI with support for structured outputs."""
        openai_messages = self._convert_messages(messages)

        # Prepare request parameters
        request_params = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": kwargs.get("temperature", self.temperature),
        }

        if self.max_tokens:
            request_params["max_tokens"] = self.max_tokens

        # Handle different output format modes
        response_format = self._prepare_response_format(
            json_mode=self.json_mode or kwargs.get("json_mode", False),
            json_schema=kwargs.get("json_schema"),
            response_model=kwargs.get("response_model")
        )
        
        if response_format:
            request_params["response_format"] = response_format
            
            # Ensure the prompt mentions JSON if using basic json_mode
            if response_format.get("type") == "json_object":
                if not any("json" in msg.content.lower() for msg in messages):
                    openai_messages[-1]["content"] += "\n\nRespond with valid JSON."

        # Add any additional kwargs
        for key in ["frequency_penalty", "presence_penalty", "top_p", "n", "stop", "seed"]:
            if key in kwargs:
                request_params[key] = kwargs[key]

        try:
            response = await self.client.chat.completions.create(**request_params)

            # Extract the response
            choice = response.choices[0]
            
            # Handle potential refusal in structured outputs
            if hasattr(choice.message, 'refusal') and choice.message.refusal:
                return LLMResponse(
                    content="",
                    model=response.model,
                    provider=self.name,
                    extra={
                        "refusal": choice.message.refusal,
                        "finish_reason": choice.finish_reason,
                    },
                )
            
            content = choice.message.content or ""
            
            # Handle parsed response if using Pydantic model
            parsed = None
            if hasattr(choice.message, 'parsed'):
                parsed = choice.message.parsed

            # Build usage info
            usage = None
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            result = LLMResponse(
                content=content,
                model=response.model,
                provider=self.name,
                usage=usage,
                raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
                extra={
                    "finish_reason": choice.finish_reason,
                    "id": response.id if hasattr(response, "id") else None,
                },
            )
            
            # Add parsed data if available
            if parsed:
                result.extra["parsed"] = parsed
            
            return result

        except openai.APIError as e:
            raise Exception(f"OpenAI API error: {e}")
        except Exception as e:
            raise Exception(f"Error generating response: {e}")

    async def generate_stream(
        self,
        messages: List[LLMMessage],
        **kwargs,
    ):
        """Generate a streaming response from OpenAI."""
        openai_messages = self._convert_messages(messages)

        request_params = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "stream": True,
        }

        if self.max_tokens:
            request_params["max_tokens"] = self.max_tokens

        # Handle response format for streaming
        response_format = self._prepare_response_format(
            json_mode=self.json_mode or kwargs.get("json_mode", False),
            json_schema=kwargs.get("json_schema"),
            response_model=kwargs.get("response_model")
        )
        
        if response_format:
            request_params["response_format"] = response_format

        try:
            stream = await self.client.chat.completions.create(**request_params)

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except openai.APIError as e:
            raise Exception(f"OpenAI API error: {e}")
        except Exception as e:
            raise Exception(f"Error generating stream: {e}")

    async def generate_with_schema(
        self,
        messages: List[LLMMessage],
        response_model: type[BaseModel],
        **kwargs
    ) -> Any:
        """Generate a response that conforms to a Pydantic model schema."""
        response = await self.generate(
            messages,
            response_model=response_model,
            **kwargs
        )
        
        # Check for refusal
        if response.extra.get("refusal"):
            raise Exception(f"Model refused: {response.extra['refusal']}")
        
        # Return the parsed model if available
        if "parsed" in response.extra:
            return response.extra["parsed"]
        
        # Otherwise try to parse the JSON content
        try:
            data = json.loads(response.content)
            return response_model(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise Exception(f"Failed to parse response as {response_model.__name__}: {e}")

    async def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        try:
            import tiktoken

            # Get the right encoding for the model
            try:
                encoding = tiktoken.encoding_for_model(self.model)
            except KeyError:
                # Fallback to cl100k_base for newer models
                encoding = tiktoken.get_encoding("cl100k_base")
            
            return len(encoding.encode(text))
        except ImportError:
            # Fallback to rough estimation
            return len(text) // 4

    async def list_models(self) -> List[str]:
        """List available models."""
        try:
            models = await self.client.models.list()
            return [model.id for model in models.data if "gpt" in model.id.lower()]
        except Exception:
            # Return common models if API call fails
            return [
                "gpt-4o-mini",
                "gpt-4o",
                "gpt-4-turbo",
                "gpt-4",
                "gpt-3.5-turbo",
            ]

    def supports_structured_output(self, model: Optional[str] = None) -> bool:
        """Check if the model supports structured outputs."""
        check_model = model or self.model
        # Models that support structured outputs with json_schema
        structured_models = ["gpt-4o-mini", "gpt-4o-2024-08-06", "gpt-4o-2024-11-20"]
        return any(sm in check_model for sm in structured_models)

    def supports_json_mode(self, model: Optional[str] = None) -> bool:
        """Check if the model supports JSON mode."""
        check_model = model or self.model
        # Most modern OpenAI models support JSON mode
        return "gpt" in check_model.lower()


# Register the provider
LLMProviderFactory.register("openai", OpenAIProvider)