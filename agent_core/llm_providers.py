"""
Multi-Provider LLM System - Abstract interface for different LLM providers.

Supports: Gemini, OpenAI, Anthropic, Ollama, and custom providers.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Generator
from dataclasses import dataclass
import time
import json


@dataclass
class LLMMessage:
    """Standard message format across all providers."""
    role: str  # 'system', 'user', 'assistant'
    content: str


@dataclass
class LLMResponse:
    """Standard response format across all providers."""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    All providers must implement this interface.
    """
    
    @abstractmethod
    def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of conversation messages
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Provider-specific parameters
        
        Returns:
            LLMResponse with generated content
        """
        pass
    
    @abstractmethod
    def generate_stream(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Stream response tokens as they arrive.
        
        Yields:
            Text chunks
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g., 'gemini', 'openai')."""
        pass
    
    @property
    @abstractmethod
    def available_models(self) -> List[str]:
        """List of available models for this provider."""
        pass
    
    def estimate_cost(self, usage: Dict[str, int]) -> float:
        """
        Estimate cost based on token usage.
        
        Args:
            usage: Dict with 'prompt_tokens' and 'completion_tokens'
        
        Returns:
            Estimated cost in USD
        """
        return 0.0  # Override in subclasses


# ============================================================================
# GEMINI PROVIDER
# ============================================================================

class GeminiProvider(LLMProvider):
    """Google Gemini provider implementation."""
    
    PRICING = {
        "gemini-2.0-flash-exp": {"input": 0.075, "output": 0.30},
        "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
        "gemini-1.5-flash": {"input": 0.075, "output": 0.30}
    }
    
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash-exp"):
        """Initialize Gemini provider."""
        try:
            from google import genai
            from google.genai import types
            self.genai = genai
            self.types = types
        except ImportError:
            raise ImportError("Install google-generativeai: pip install google-generativeai")
        
        self.client = genai.Client(api_key=api_key)
        self.model = model
    
    @property
    def name(self) -> str:
        return "gemini"
    
    @property
    def available_models(self) -> List[str]:
        return list(self.PRICING.keys())
    
    def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using Gemini."""
        # Convert to Gemini format
        gemini_messages = [
            self.types.Content(
                role=msg.role if msg.role != "assistant" else "model",
                parts=[self.types.Part(text=msg.content)]
            )
            for msg in messages
        ]
        
        config = self.types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens
        )
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=gemini_messages,
            config=config
        )
        
        # Extract usage
        usage = None
        if hasattr(response, 'usage_metadata'):
            meta = response.usage_metadata
            usage = {
                "prompt_tokens": meta.prompt_token_count,
                "completion_tokens": meta.candidates_token_count,
                "total_tokens": meta.prompt_token_count + meta.candidates_token_count
            }
        
        return LLMResponse(
            content=response.text,
            model=self.model,
            usage=usage
        )
    
    def generate_stream(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream response from Gemini."""
        gemini_messages = [
            self.types.Content(
                role=msg.role if msg.role != "assistant" else "model",
                parts=[self.types.Part(text=msg.content)]
            )
            for msg in messages
        ]
        
        config = self.types.GenerateContentConfig(temperature=temperature)
        
        for chunk in self.client.models.generate_content_stream(
            model=self.model,
            contents=gemini_messages,
            config=config
        ):
            if hasattr(chunk, 'text') and chunk.text:
                yield chunk.text
    
    def estimate_cost(self, usage: Dict[str, int]) -> float:
        """Estimate cost for Gemini."""
        pricing = self.PRICING.get(self.model, self.PRICING["gemini-2.0-flash-exp"])
        input_cost = (usage.get("prompt_tokens", 0) / 1_000_000) * pricing["input"]
        output_cost = (usage.get("completion_tokens", 0) / 1_000_000) * pricing["output"]
        return input_cost + output_cost


# ============================================================================
# OPENAI PROVIDER
# ============================================================================

class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation."""
    
    PRICING = {
        "gpt-4": {"input": 30.00, "output": 60.00},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "gpt-4o": {"input": 5.00, "output": 15.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60}
    }
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        """Initialize OpenAI provider."""
        try:
            import openai
            self.openai = openai
        except ImportError:
            raise ImportError("Install openai: pip install openai")
        
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    @property
    def name(self) -> str:
        return "openai"
    
    @property
    def available_models(self) -> List[str]:
        return list(self.PRICING.keys())
    
    def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using OpenAI."""
        # Convert to OpenAI format
        openai_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Extract usage
        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=self.model,
            usage=usage
        )
    
    def generate_stream(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream response from OpenAI."""
        openai_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            temperature=temperature,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    def estimate_cost(self, usage: Dict[str, int]) -> float:
        """Estimate cost for OpenAI."""
        pricing = self.PRICING.get(self.model, self.PRICING["gpt-4o-mini"])
        input_cost = (usage.get("prompt_tokens", 0) / 1_000_000) * pricing["input"]
        output_cost = (usage.get("completion_tokens", 0) / 1_000_000) * pricing["output"]
        return input_cost + output_cost


# ============================================================================
# ANTHROPIC PROVIDER
# ============================================================================

class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider implementation."""
    
    PRICING = {
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25}
    }
    
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        """Initialize Anthropic provider."""
        try:
            import anthropic
            self.anthropic = anthropic
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
    
    @property
    def name(self) -> str:
        return "anthropic"
    
    @property
    def available_models(self) -> List[str]:
        return list(self.PRICING.keys())
    
    def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using Claude."""
        # Separate system message from conversation
        system_msg = ""
        conversation = []
        
        for msg in messages:
            if msg.role == "system":
                system_msg = msg.content
            else:
                conversation.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        response = self.client.messages.create(
            model=self.model,
            system=system_msg if system_msg else None,
            messages=conversation,
            temperature=temperature,
            max_tokens=max_tokens or 4096
        )
        
        # Extract usage
        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            }
        
        return LLMResponse(
            content=response.content[0].text,
            model=self.model,
            usage=usage
        )
    
    def generate_stream(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream response from Claude."""
        system_msg = ""
        conversation = []
        
        for msg in messages:
            if msg.role == "system":
                system_msg = msg.content
            else:
                conversation.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        with self.client.messages.stream(
            model=self.model,
            system=system_msg if system_msg else None,
            messages=conversation,
            temperature=temperature,
            max_tokens=4096
        ) as stream:
            for text in stream.text_stream:
                yield text
    
    def estimate_cost(self, usage: Dict[str, int]) -> float:
        """Estimate cost for Anthropic."""
        pricing = self.PRICING.get(self.model, self.PRICING["claude-3-5-sonnet-20241022"])
        input_cost = (usage.get("prompt_tokens", 0) / 1_000_000) * pricing["input"]
        output_cost = (usage.get("completion_tokens", 0) / 1_000_000) * pricing["output"]
        return input_cost + output_cost


# ============================================================================
# OLLAMA PROVIDER (Local Models)
# ============================================================================

class OllamaProvider(LLMProvider):
    """Ollama provider for local LLM execution."""
    
    def __init__(self, model: str = "llama2", base_url: str = "http://localhost:11434"):
        """Initialize Ollama provider."""
        try:
            import requests
            self.requests = requests
        except ImportError:
            raise ImportError("Install requests: pip install requests")
        
        self.model = model
        self.base_url = base_url
    
    @property
    def name(self) -> str:
        return "ollama"
    
    @property
    def available_models(self) -> List[str]:
        try:
            response = self.requests.get(f"{self.base_url}/api/tags")
            models = response.json().get("models", [])
            return [m["name"] for m in models]
        except:
            return ["llama2", "mistral", "codellama"]
    
    def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using Ollama."""
        # Convert to Ollama format
        prompt = "\n".join([
            f"{msg.role}: {msg.content}" for msg in messages
        ])
        
        response = self.requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "temperature": temperature,
                "stream": False
            }
        )
        
        data = response.json()
        
        return LLMResponse(
            content=data.get("response", ""),
            model=self.model,
            usage=None  # Ollama doesn't provide token counts by default
        )
    
    def generate_stream(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream response from Ollama."""
        prompt = "\n".join([
            f"{msg.role}: {msg.content}" for msg in messages
        ])
        
        response = self.requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "temperature": temperature,
                "stream": True
            },
            stream=True
        )
        
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if "response" in data:
                    yield data["response"]
    
    def estimate_cost(self, usage: Dict[str, int]) -> float:
        """Ollama is free (local execution)."""
        return 0.0


# ============================================================================
# UNIFIED CLIENT
# ============================================================================

class UnifiedLLMClient:
    """
    Unified client that works with any provider.
    
    Usage:
        # Gemini
        client = UnifiedLLMClient("gemini", api_key="...")
        
        # OpenAI
        client = UnifiedLLMClient("openai", api_key="...")
        
        # Anthropic
        client = UnifiedLLMClient("anthropic", api_key="...")
        
        # Local Ollama
        client = UnifiedLLMClient("ollama", model="llama2")
    """
    
    PROVIDERS = {
        "gemini": GeminiProvider,
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "ollama": OllamaProvider
    }
    
    def __init__(
        self,
        provider: str="ollama",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_retries: int = 3,
        verbose: bool = False,
        **provider_kwargs
    ):
        """
        Initialize unified client.
        
        Args:
            provider: Provider name ('gemini', 'openai', 'anthropic', 'ollama')
            api_key: API key (not needed for Ollama)
            model: Model name (provider-specific)
            max_retries: Maximum retry attempts
            verbose: Enable detailed logging
            **provider_kwargs: Additional provider-specific arguments
        """
        if provider not in self.PROVIDERS:
            raise ValueError(
                f"Unknown provider: {provider}. "
                f"Available: {list(self.PROVIDERS.keys())}"
            )
        
        self.provider_name = provider
        self.max_retries = max_retries
        self.verbose = verbose
        
        # Initialize provider
        provider_class = self.PROVIDERS[provider]
        
        if provider == "ollama":
            self.provider = provider_class(model=model or "llama2", **provider_kwargs)
        else:
            if not api_key:
                raise ValueError(f"API key required for {provider}")
            
            if model:
                self.provider = provider_class(api_key=api_key, model=model)
            else:
                self.provider = provider_class(api_key=api_key)
        
        # Track usage
        self._session_stats = {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_requests": 0,
            "failed_requests": 0,
            "total_cost_usd": 0.0
        }
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate response (compatible with old LLMClient interface).
        
        Args:
            system_prompt: System instruction
            user_prompt: User message
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
        
        Returns:
            Generated text
        """
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt)
        ]
        
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                if self.verbose:
                    print(f"🔄 API request (attempt {attempt + 1}/{self.max_retries})")
                
                response = self.provider.generate(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # Track usage
                if response.usage:
                    self._session_stats["total_prompt_tokens"] += response.usage.get("prompt_tokens", 0)
                    self._session_stats["total_completion_tokens"] += response.usage.get("completion_tokens", 0)
                    cost = self.provider.estimate_cost(response.usage)
                    self._session_stats["total_cost_usd"] += cost
                
                self._session_stats["total_requests"] += 1
                
                if self.verbose:
                    print(f"✅ Request succeeded ({self.provider.name})")
                    if response.usage:
                        print(f"📊 Tokens: {response.usage.get('total_tokens', 0):,}")
                
                return response.content
            
            except Exception as e:
                last_error = e
                
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    if self.verbose:
                        print(f"⚠️  Request failed: {e}")
                        print(f"⏳ Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    self._session_stats["failed_requests"] += 1
        
        raise Exception(f"All {self.max_retries} attempts failed: {last_error}")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        return {
            **self._session_stats,
            "provider": self.provider.name,
            "model": self.provider.model if hasattr(self.provider, 'model') else "unknown"
        }
    
    def print_session_summary(self):
        """Print formatted session summary."""
        stats = self.get_session_stats()
        
        print("\n" + "="*60)
        print(f"📊 SESSION SUMMARY ({stats['provider']})")
        print("="*60)
        print(f"Model:              {stats['model']}")
        print(f"Total Requests:     {stats['total_requests']}")
        print(f"Failed Requests:    {stats['failed_requests']}")
        print(f"Prompt Tokens:      {stats['total_prompt_tokens']:,}")
        print(f"Completion Tokens:  {stats['total_completion_tokens']:,}")
        print(f"Estimated Cost:     ${stats['total_cost_usd']:.4f}")
        print("="*60 + "\n")
    
    # Keep these for backward compatibility
    def extract_json_from_response(self, text: str) -> dict:
        """Extract JSON from response (uses old llm_client logic)."""
        import re
        
        # Try code blocks
        patterns = [
            r'```json\s*(\{.*?\})\s*```',
            r'```\s*(\{.*?\})\s*```',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            if matches:
                try:
                    return json.loads(matches[0])
                except:
                    continue
        
        # Try naked JSON
        json_pattern = r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                return json.loads(match)
            except:
                continue
        
        raise ValueError("No valid JSON found in response")
    
    def extract_code_from_response(self, text: str, language: Optional[str] = None) -> str:
        """Extract code from response."""
        import re
        
        if language:
            pattern = rf'```{language}\s*(.*?)```'
        else:
            pattern = r'```(?:\w+)?\s*(.*?)```'
        
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        
        if matches:
            return matches[0].strip()
        
        return text.strip()


# Example usage
if __name__ == "__main__":
    import os
    
    # Example 1: Gemini
    print("Testing Gemini...")
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        client = UnifiedLLMClient("gemini", api_key=gemini_key, verbose=True)
        response = client.generate(
            system_prompt="You are helpful.",
            user_prompt="Say 'Hello from Gemini!'"
        )
        print(f"Response: {response}\n")
        client.print_session_summary()
    
    # Example 2: OpenAI (if you have key)
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        print("\nTesting OpenAI...")
        client = UnifiedLLMClient("openai", api_key=openai_key, model="gpt-4o-mini", verbose=True)
        response = client.generate(
            system_prompt="You are helpful.",
            user_prompt="Say 'Hello from GPT!'"
        )
        print(f"Response: {response}\n")
        client.print_session_summary()
    
    # Example 3: Compare providers
    print("\n" + "="*60)
    print("PROVIDER COMPARISON")
    print("="*60)
    for provider_name in UnifiedLLMClient.PROVIDERS.keys():
        print(f"\n{provider_name.upper()}:")
        try:
            # Mock initialization to show models
            if provider_name == "gemini":
                provider = GeminiProvider(api_key="dummy", model="gemini-2.0-flash-exp")
            elif provider_name == "openai":
                provider = OpenAIProvider(api_key="dummy", model="gpt-4o-mini")
            elif provider_name == "anthropic":
                provider = AnthropicProvider(api_key="dummy")
            else:
                provider = OllamaProvider()
            
            print(f"  Available models: {', '.join(provider.available_models[:3])}...")
        except:
            print(f"  (Install {provider_name} SDK to see models)")