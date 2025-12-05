import json
from pathlib import Path
from typing import Dict
from .llm_client import LLMClient
from .utils import gather_code_context

class Coder:
    def __init__(self, llm: LLMClient, context_kb: int = 40_000):
        self.llm = llm
        self.context_kb = context_kb  # approximate characters of context

    # this method is writen by claude
    def _extract_json_from_response(self, payload: str) -> dict:
        """Extract JSON from LLM response, handling markdown wrapping."""
        # Try direct parse first
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            pass
        
        # Try extracting from code blocks
        import re
        patterns = [
            r'```json\s*(\{.*?\})\s*```',
            r'```\s*(\{.*?\})\s*```',
            r'(\{.*\})'  # Last resort: find any JSON-like structure
        ]
        
        for pattern in patterns:
            match = re.search(pattern, payload, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue
        
        raise ValueError("Could not extract valid JSON from response")

    def propose_changes(self, task: str, plan, project_root: str) -> Dict[str, dict]:
        """
        Returns proposals: { filename: { 'content': '...', 'explanation': '...' } }
        The model should return a JSON mapping of files to full new content. We'll try to parse it.
        """
        # collect short code context
        context = gather_code_context(project_root, char_limit=self.context_kb)
        system = (
            "You are a Python engineer. Propose concrete file edits for the project. "
            "Return a JSON object mapping relative file paths to their complete new file content and a short explanation.\n\n"
            "Example:\n{\n  \"package/module.py\": {\"content\": \"<file contents>\", \"explanation\": \"what changed\"}\n}\n"
            "If a file should be created, include it. If a file should remain unchanged, you may omit it."
        )
        user = f"Task:\n{task}\n\nPlan:\n" + "\n".join([f"- {s}" for s in plan]) + "\n\nProject context (truncated):\n" + context
        resp = self.llm.generate(system, user)
        payload = self.llm.extract_code_from_response(resp)
        # claude code
        self._extract_json_from_response(payload=payload)
        # Try to parse JSON from payload
        # try:
        #     parsed = json.loads(payload)
        #     proposals = {}
        #     for fname, info in parsed.items():
        #         if isinstance(info, dict) and "content" in info:
        #             proposals[fname] = {"content": info["content"], "explanation": info.get("explanation","")}
        #         else:
        #             # If value is string treat as full content
        #             proposals[fname] = {"content": info if isinstance(info, str) else str(info), "explanation": ""}
        #     return proposals
        # except Exception:
        #     # fallback: try to parse simple single-file content
        #     return {"proposed_change.txt": {"content": payload, "explanation": "Raw model output (couldn't JSON-parse)"}}

    