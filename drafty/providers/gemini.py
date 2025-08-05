"""Google Gemini LLM provider implementation."""

import json
import os
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from google.generativeai.types import HarmBlockThreshold, HarmCategory

from drafty.providers.base import LLMMessage, LLMProvider, LLMProviderFactory, LLMResponse


class GeminiProvider(LLMProvider):
    """Google Gemini provider implementation."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        api_key = config.get("api_key") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key not provided")

        genai.configure(api_key=api_key)
        
        # Default to GEMINI_MODEL env var or gemini-2.0-flash-exp
        if not self.model:
            # Try to get from environment first
            default_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
            self.model = default_model
        
        # Configure safety settings (can be customized via config)
        self.safety_settings = config.get("safety_settings", {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        })

    @property
    def name(self) -> str:
        return "gemini"

    def _convert_messages(self, messages: List[LLMMessage]) -> tuple[str, List[Dict[str, str]]]:
        """Convert LLMMessage objects to Gemini format.
        
        Returns system prompt and conversation history separately.
        """
        system_prompt = ""
        history = []
        
        for msg in messages:
            if msg.role == "system":
                # Gemini handles system prompts separately
                system_prompt = msg.content
            else:
                # Convert role names: assistant -> model
                role = "model" if msg.role == "assistant" else "user"
                history.append({
                    "role": role,
                    "parts": [{"text": msg.content}]
                })
        
        return system_prompt, history

    def _prepare_generation_config(self, **kwargs) -> Dict[str, Any]:
        """Prepare generation configuration for Gemini."""
        config = {
            "temperature": kwargs.get("temperature", self.temperature),
        }
        
        if self.max_tokens:
            config["max_output_tokens"] = self.max_tokens
        
        # Add response type for JSON mode
        if self.json_mode or kwargs.get("json_mode"):
            config["response_mime_type"] = "application/json"
            
            # If a JSON schema is provided, use it
            if "json_schema" in kwargs:
                config["response_schema"] = kwargs["json_schema"]
        
        # Add other supported parameters
        for key in ["top_p", "top_k", "stop_sequences"]:
            if key in kwargs:
                config[key] = kwargs[key]
        
        return config

    async def generate(
        self,
        messages: List[LLMMessage],
        **kwargs,
    ) -> LLMResponse:
        """Generate a response from Gemini."""
        system_prompt, history = self._convert_messages(messages)
        
        # Create the model with system instruction if provided
        model_kwargs = {}
        if system_prompt:
            model_kwargs["system_instruction"] = system_prompt
        
        model = genai.GenerativeModel(
            model_name=self.model,
            safety_settings=self.safety_settings,
            **model_kwargs
        )
        
        # Prepare generation config
        generation_config = self._prepare_generation_config(**kwargs)
        
        try:
            # Start a chat with history
            chat = model.start_chat(history=history[:-1] if len(history) > 1 else [])
            
            # Send the last message
            last_message = history[-1]["parts"][0]["text"] if history else ""
            
            # Generate response
            response = await chat.send_message_async(
                last_message,
                generation_config=generation_config,
            )
            
            # Extract content
            content = response.text
            
            # Parse JSON if in JSON mode
            parsed = None
            if self.json_mode or kwargs.get("json_mode"):
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    pass
            
            # Build usage info if available
            usage = None
            if hasattr(response, "usage_metadata"):
                usage = {
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count,
                }
            
            result = LLMResponse(
                content=content,
                model=self.model,
                provider=self.name,
                usage=usage,
                extra={
                    "finish_reason": response.candidates[0].finish_reason.name if response.candidates else None,
                    "safety_ratings": [
                        {"category": rating.category.name, "probability": rating.probability.name}
                        for rating in response.candidates[0].safety_ratings
                    ] if response.candidates else [],
                },
            )
            
            if parsed:
                result.extra["parsed"] = parsed
            
            return result
            
        except Exception as e:
            raise Exception(f"Gemini API error: {e}")

    async def generate_stream(
        self,
        messages: List[LLMMessage],
        **kwargs,
    ):
        """Generate a streaming response from Gemini."""
        system_prompt, history = self._convert_messages(messages)
        
        # Create the model with system instruction if provided
        model_kwargs = {}
        if system_prompt:
            model_kwargs["system_instruction"] = system_prompt
        
        model = genai.GenerativeModel(
            model_name=self.model,
            safety_settings=self.safety_settings,
            **model_kwargs
        )
        
        # Prepare generation config
        generation_config = self._prepare_generation_config(**kwargs)
        
        try:
            # Start a chat with history
            chat = model.start_chat(history=history[:-1] if len(history) > 1 else [])
            
            # Send the last message with streaming
            last_message = history[-1]["parts"][0]["text"] if history else ""
            
            response = await chat.send_message_async(
                last_message,
                generation_config=generation_config,
                stream=True,
            )
            
            async for chunk in response:
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            raise Exception(f"Gemini streaming error: {e}")

    async def count_tokens(self, text: str) -> int:
        """Count tokens in text using Gemini's tokenizer."""
        try:
            model = genai.GenerativeModel(model_name=self.model)
            return model.count_tokens(text).total_tokens
        except Exception:
            # Fallback to rough estimation
            return len(text) // 4

    async def list_models(self) -> List[str]:
        """List available Gemini models."""
        try:
            models = []
            for model in genai.list_models():
                if "generateContent" in model.supported_generation_methods:
                    models.append(model.name.replace("models/", ""))
            return models
        except Exception:
            # Return common models if API call fails
            return [
                "gemini-2.0-flash-exp",
                "gemini-1.5-flash",
                "gemini-1.5-flash-8b",
                "gemini-1.5-pro",
                "gemini-1.0-pro",
            ]

    def supports_json_mode(self) -> bool:
        """Check if the model supports JSON mode."""
        # Gemini 1.5 models support JSON response type
        return "gemini-1.5" in self.model


# Register the provider
LLMProviderFactory.register("gemini", GeminiProvider)