# import os
# import json
# import re
# import time
# from typing import Optional, Dict, Any, Generator
# from dataclasses import dataclass, asdict
# from google import genai
# from google.genai import types

# class LLMClient:
#     def __init__(
#         self, 
#         api_key: str, 
#         model: str = "gemini-2.0-flash-exp",
#         max_retries: int = 3,
#         retry_delay: float = 2.0,
#         timeout: int = 60,
#         verbose: bool = False
#     ):
#         self.client = genai.Client(api_key=api_key)
#         self.model = model
#         self.max_retries = max_retries
#         self.retry_delay = retry_delay
#         self.verbose = verbose
#         self._token_stats = {"total_prompt": 0, "total_response": 0, "requests": 0}
    
#     def generate(
#         self, 
#         system_prompt: str, 
#         user_prompt: str,
#         temperature: float = 0.7,
#         max_tokens: Optional[int] = None
#     ) -> str:
#         """
#         Generate content with automatic retries and error handling.
        
#         Raises:
#             APIError: If all retries fail
#         """
#         messages = [
#             types.Content(role="system", parts=[types.Part(text=system_prompt)]),
#             types.Content(role="user", parts=[types.Part(text=user_prompt)]),
#         ]
        
#         config = types.GenerateContentConfig(
#             temperature=temperature,
#             max_output_tokens=max_tokens,
#         )
        
#         last_error = None
#         for attempt in range(self.max_retries):
#             try:
#                 resp = self.client.models.generate_content(
#                     model=self.model, 
#                     contents=messages,
#                     config=config
#                 )
                
#                 # Track usage
#                 self._track_usage(resp)
                
#                 # Validate response
#                 if not hasattr(resp, "text") or not resp.text:
#                     raise ValueError("Empty response from API")
                
#                 if self.verbose:
#                     print(f"✓ API call succeeded (attempt {attempt + 1})")
                
#                 return resp.text
                
#             except Exception as e:
#                 last_error = e
#                 error_msg = str(e).lower()
                
#                 # Don't retry on certain errors
#                 if any(x in error_msg for x in ["invalid", "authentication", "permission"]):
#                     raise APIError(f"Non-retryable error: {e}") from e
                
#                 if attempt < self.max_retries - 1:
#                     wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
#                     if self.verbose:
#                         print(f"⚠ API call failed (attempt {attempt + 1}): {e}")
#                         print(f"  Retrying in {wait_time}s...")
#                     time.sleep(wait_time)
#                 else:
#                     if self.verbose:
#                         print(f"✗ All {self.max_retries} attempts failed")
        
#         raise APIError(f"Failed after {self.max_retries} attempts: {last_error}") from last_error
    
#     def _track_usage(self, response):
#         """Track token usage from API response."""
#         try:
#             meta = response.usage_metadata
#             prompt_tokens = meta.prompt_token_count
#             response_tokens = meta.candidates_token_count
            
#             self._token_stats["total_prompt"] += prompt_tokens
#             self._token_stats["total_response"] += response_tokens
#             self._token_stats["requests"] += 1
            
#             if self.verbose:
#                 print(f"📊 Tokens - Prompt: {prompt_tokens:,} | Response: {response_tokens:,} | Total: {prompt_tokens + response_tokens:,}")
#                 cost = self._estimate_cost()
#                 print(f"💰 Estimated cost this session: ${cost:.4f}")
#         except Exception as e:
#             if self.verbose:
#                 print(f"⚠ Could not track usage: {e}")


#     def extract_code_from_response(self, text: str, prefer_json: bool = True) -> str:
#         """
#         Robust extraction of code blocks from LLM responses.
        
#         Args:
#             text: Raw LLM response
#             prefer_json: If True, prioritize JSON code blocks
        
#         Returns:
#             Extracted code content
#         """
#         # Pattern 1: Language-tagged code blocks
#         patterns = [
#             r'```(?:json|python|javascript|typescript|yaml|toml)\n(.*?)```',  # Specific languages
#             r'```\w*\n(.*?)```',  # Any language tag
#             r'```(.*?)```',  # No language tag
#         ]
        
#         for pattern in patterns:
#             matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
#             if matches:
#                 if prefer_json:
#                     # Try to find JSON block first
#                     for match in matches:
#                         if match.strip().startswith('{'):
#                             return match.strip()
#                 return matches[0].strip()
        
#         # Pattern 2: Inline JSON objects (no code blocks)
#         json_pattern = r'(\{[\s\S]*\})'
#         json_matches = re.findall(json_pattern, text)
#         if json_matches:
#             # Return the largest JSON-like structure
#             return max(json_matches, key=len).strip()
        
#         # Fallback: return original text
#         return text.strip()
    
#     def get_token_stats(self) -> dict:
#         """Return cumulative token usage statistics."""
#         return {
#             **self._token_stats,
#             "total_cost_usd": self._estimate_cost()
#         }
    
#     def _estimate_cost(self) -> float:
#         """Rough cost estimation for Gemini 2.0 Flash."""
#         # Gemini 2.0 Flash pricing (as of Dec 2024)
#         # Input: $0.075 per 1M tokens
#         # Output: $0.30 per 1M tokens
#         prompt_cost = (self._token_stats["total_prompt"] / 1_000_000) * 0.075
#         response_cost = (self._token_stats["total_response"] / 1_000_000) * 0.30
#         return prompt_cost + response_cost

#     def generate_stream(
#         self, 
#         system_prompt: str, 
#         user_prompt: str
#     ) -> Generator[str, None, None]:
#         """
#         Stream response tokens as they arrive.
        
#         Yields:
#             Text chunks from the model
#         """
#         messages = [
#             types.Content(role="system", parts=[types.Part(text=system_prompt)]),
#             types.Content(role="user", parts=[types.Part(text=user_prompt)]),
#         ]
        
#         try:
#             for chunk in self.client.models.generate_content_stream(
#                 model=self.model,
#                 contents=messages
#             ):
#                 if hasattr(chunk, 'text') and chunk.text:
#                     yield chunk.text
#         except Exception as e:
#             raise APIError(f"Streaming failed: {e}") from e

"""
Enhanced LLM Client with robust error handling, retries, and token tracking.
"""
import os
import json
import re
import time
from typing import Optional, Dict, Any, Generator
from dataclasses import dataclass, asdict
from google import genai
from google.genai import types


@dataclass
class TokenUsage:
    """Track token usage for cost estimation."""
    prompt_tokens: int
    response_tokens: int
    total_tokens: int
    
    def to_dict(self) -> dict:
        return asdict(self)


class APIError(Exception):
    """Custom exception for API-related errors."""
    pass


class LLMClient:
    """
    Production-ready LLM client with:
    - Automatic retries with exponential backoff
    - Token usage tracking
    - Cost estimation
    - Robust response parsing
    - Streaming support
    """
    
    # Gemini 2.0 Flash pricing (per 1M tokens)
    PRICING = {
        "input": 0.075,   # $0.075 per 1M input tokens
        "output": 0.30,   # $0.30 per 1M output tokens
    }
    
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash-exp",
        max_retries: int = 3,
        retry_delay: float = 2.0,
        timeout: int = 60,
        verbose: bool = False
    ):
        """
        Initialize LLM client.
        
        Args:
            api_key: Gemini API key
            model: Model identifier
            max_retries: Maximum retry attempts for failed requests
            retry_delay: Base delay between retries (uses exponential backoff)
            timeout: Request timeout in seconds
            verbose: Enable detailed logging
        """
        if not api_key:
            raise ValueError("API key is required")
        
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.verbose = verbose
        
        # Track cumulative usage
        self._session_stats = {
            "total_prompt_tokens": 0,
            "total_response_tokens": 0,
            "total_requests": 0,
            "failed_requests": 0,
            "total_cost_usd": 0.0
        }
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[list] = None
    ) -> str:
        """
        Generate content with automatic retries and error handling.
        
        Args:
            system_prompt: System instruction for the model
            user_prompt: User's request/question
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            stop_sequences: Optional list of sequences that stop generation
        
        Returns:
            Generated text response
        
        Raises:
            APIError: If all retries fail or non-retryable error occurs
        """
        messages = [
            types.Content(role="system", parts=[types.Part(text=system_prompt)]),
            types.Content(role="user", parts=[types.Part(text=user_prompt)]),
        ]
        
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            stop_sequences=stop_sequences or []
        )
        
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                if self.verbose:
                    print(f"🔄 API request attempt {attempt + 1}/{self.max_retries}")
                
                # Make API call
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=messages,
                    config=config
                )
                
                # Validate response
                if not hasattr(response, "text") or not response.text:
                    raise ValueError("Empty response from API")
                
                # Track usage
                usage = self._track_usage(response)
                
                if self.verbose:
                    print(f"✅ Request succeeded")
                    print(f"📊 Tokens: {usage.prompt_tokens:,} in / {usage.response_tokens:,} out")
                    print(f"💰 Cost: ${self._estimate_request_cost(usage):.6f}")
                
                self._session_stats["total_requests"] += 1
                
                return response.text
            
            except Exception as e:
                last_error = e
                error_msg = str(e).lower()
                
                # Categorize error type
                if any(keyword in error_msg for keyword in [
                    "invalid", "authentication", "permission", "api key", "quota"
                ]):
                    # Non-retryable errors
                    self._session_stats["failed_requests"] += 1
                    raise APIError(f"Non-retryable error: {e}") from e
                
                # Retryable errors (rate limit, timeout, temporary issues)
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    
                    if self.verbose:
                        print(f"⚠️  Request failed: {e}")
                        print(f"⏳ Retrying in {wait_time:.1f}s...")
                    
                    time.sleep(wait_time)
                else:
                    if self.verbose:
                        print(f"❌ All {self.max_retries} attempts failed")
                    self._session_stats["failed_requests"] += 1
        
        raise APIError(f"Failed after {self.max_retries} attempts: {last_error}") from last_error
    
    def generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7
    ) -> Generator[str, None, None]:
        """
        Stream response tokens as they arrive.
        
        Args:
            system_prompt: System instruction
            user_prompt: User request
            temperature: Sampling temperature
        
        Yields:
            Text chunks from the model
        
        Raises:
            APIError: If streaming fails
        """
        messages = [
            types.Content(role="system", parts=[types.Part(text=system_prompt)]),
            types.Content(role="user", parts=[types.Part(text=user_prompt)]),
        ]
        
        config = types.GenerateContentConfig(temperature=temperature)
        
        try:
            if self.verbose:
                print("🌊 Starting streaming response...")
            
            for chunk in self.client.models.generate_content_stream(
                model=self.model,
                contents=messages,
                config=config
            ):
                if hasattr(chunk, 'text') and chunk.text:
                    yield chunk.text
        
        except Exception as e:
            raise APIError(f"Streaming failed: {e}") from e
    
    def extract_json_from_response(self, text: str, prefer_largest: bool = True) -> dict:
        """
        Robust extraction of JSON from LLM responses.
        
        Handles various formats:
        - JSON in markdown code blocks (```json ... ```)
        - Naked JSON objects
        - Multiple JSON objects (returns first or largest)
        
        Args:
            text: Raw LLM response text
            prefer_largest: If multiple JSONs found, return largest
        
        Returns:
            Parsed JSON dictionary
        
        Raises:
            ValueError: If no valid JSON found
        """
        # Pattern 1: JSON in code blocks with language tag
        patterns = [
            r'```json\s*(\{.*?\})\s*```',
            r'```\s*(\{.*?\})\s*```',
        ]
        
        candidates = []
        
        # Try code block patterns
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                try:
                    parsed = json.loads(match)
                    candidates.append((match, parsed))
                except json.JSONDecodeError:
                    continue
        
        # Pattern 2: Naked JSON objects
        if not candidates:
            # Find all JSON-like structures
            json_pattern = r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
            matches = re.findall(json_pattern, text, re.DOTALL)
            
            for match in matches:
                try:
                    parsed = json.loads(match)
                    candidates.append((match, parsed))
                except json.JSONDecodeError:
                    continue
        
        if not candidates:
            raise ValueError("No valid JSON found in response")
        
        # Return largest JSON if multiple found
        if prefer_largest:
            return max(candidates, key=lambda x: len(x[0]))[1]
        else:
            return candidates[0][1]
    
    def extract_code_from_response(self, text: str, language: Optional[str] = None) -> str:
        """
        Extract code blocks from LLM response.
        
        Args:
            text: Raw response text
            language: Optional specific language to extract (e.g., 'python', 'json')
        
        Returns:
            Extracted code content
        """
        # Build pattern based on language filter
        if language:
            pattern = rf'```{language}\s*(.*?)```'
        else:
            pattern = r'```(?:\w+)?\s*(.*?)```'
        
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        
        if matches:
            return matches[0].strip()
        
        # Fallback: return original if no code blocks
        return text.strip()
    
    def _track_usage(self, response) -> TokenUsage:
        """Extract and track token usage from API response."""
        try:
            meta = response.usage_metadata
            usage = TokenUsage(
                prompt_tokens=meta.prompt_token_count,
                response_tokens=meta.candidates_token_count,
                total_tokens=meta.prompt_token_count + meta.candidates_token_count
            )
            
            # Update session stats
            self._session_stats["total_prompt_tokens"] += usage.prompt_tokens
            self._session_stats["total_response_tokens"] += usage.response_tokens
            self._session_stats["total_cost_usd"] += self._estimate_request_cost(usage)
            
            return usage
        
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Could not track usage: {e}")
            # Return zero usage if tracking fails
            return TokenUsage(0, 0, 0)
    
    def _estimate_request_cost(self, usage: TokenUsage) -> float:
        """Calculate cost for a single request."""
        input_cost = (usage.prompt_tokens / 1_000_000) * self.PRICING["input"]
        output_cost = (usage.response_tokens / 1_000_000) * self.PRICING["output"]
        return input_cost + output_cost
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get cumulative statistics for this session.
        
        Returns:
            Dict with token usage and cost information
        """
        return {
            **self._session_stats,
            "success_rate": (
                (self._session_stats["total_requests"] - self._session_stats["failed_requests"]) 
                / max(self._session_stats["total_requests"], 1)
            ) * 100
        }
    
    def print_session_summary(self):
        """Print a formatted summary of session usage."""
        stats = self.get_session_stats()
        
        print("\n" + "="*60)
        print("📊 SESSION SUMMARY")
        print("="*60)
        print(f"Total Requests:     {stats['total_requests']}")
        print(f"Failed Requests:    {stats['failed_requests']}")
        print(f"Success Rate:       {stats['success_rate']:.1f}%")
        print(f"Prompt Tokens:      {stats['total_prompt_tokens']:,}")
        print(f"Response Tokens:    {stats['total_response_tokens']:,}")
        print(f"Total Tokens:       {stats['total_prompt_tokens'] + stats['total_response_tokens']:,}")
        print(f"Estimated Cost:     ${stats['total_cost_usd']:.4f}")
        print("="*60 + "\n")
    
    def reset_stats(self):
        """Reset session statistics."""
        self._session_stats = {
            "total_prompt_tokens": 0,
            "total_response_tokens": 0,
            "total_requests": 0,
            "failed_requests": 0,
            "total_cost_usd": 0.0
        }


# Example usage and testing
if __name__ == "__main__":
    # Test the client
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY environment variable")
        exit(1)
    
    client = LLMClient(api_key=api_key, verbose=True)
    
    # Test 1: Basic generation
    print("Test 1: Basic generation")
    response = client.generate(
        system_prompt="You are a helpful coding assistant.",
        user_prompt="Write a Python function to calculate fibonacci numbers.",
        temperature=0.7
    )
    print(f"Response preview: {response[:200]}...")
    
    # Test 2: JSON extraction
    print("\n\nTest 2: JSON extraction")
    response = client.generate(
        system_prompt="Return only valid JSON.",
        user_prompt='Create a JSON object with keys: name, age, city',
        temperature=0.3
    )
    try:
        extracted = client.extract_json_from_response(response)
        print(f"Extracted JSON: {json.dumps(extracted, indent=2)}")
    except ValueError as e:
        print(f"Failed to extract JSON: {e}")
    
    # Print session summary
    client.print_session_summary()