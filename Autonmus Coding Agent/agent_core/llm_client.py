import os
import json
from google import genai
from google.genai import types

class LLMClient:
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash-001", verbose: bool=False):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.verbose = verbose

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Simple wrapper: send system + user prompts, return response text.
        """
        messages = [
            types.Content(role="system", parts=[types.Part(text=system_prompt)]),
            types.Content(role="user", parts=[types.Part(text=user_prompt)]),
        ]
        resp = self.client.models.generate_content(model=self.model, contents=messages)
        if self.verbose:
            try:
                meta = resp.usage_metadata
                print(f"prompt tokens: {meta.prompt_token_count}, response tokens: {meta.candidates_token_count}")
            except Exception:
                pass
        return getattr(resp, "text", "")

    def extract_code_from_response(self, text: str) -> str:
        """
        Heuristic: prefer JSON codeblock or triple-backtick code block.
        If model returns a JSON object with filename/content mapping, return it as a string for parsing.
        """
        # look for triple backticks
        if "```" in text:
            # return contents of the first code fence
            parts = text.split("```")
            if len(parts) >= 3:
                return parts[1].strip()
        return text.strip()
