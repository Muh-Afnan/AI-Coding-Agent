# agent_core/context_manager.py
from typing import List, Dict
import tiktoken

class ContextManager:
    def __init__(self, model: str = "gpt-4", max_tokens: int = 100_000):
        self.encoding = tiktoken.encoding_for_model(model)  # Use tiktoken
        self.max_tokens = max_tokens
    
    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))
    
    def smart_truncate(self, files: Dict[str, str], priority_files: List[str] = None) -> Dict[str, str]:
        """
        Include priority files fully, truncate others to fit budget.
        """
        priority_files = priority_files or []
        result = {}
        remaining_tokens = self.max_tokens
        
        # Add priority files first
        for fname in priority_files:
            if fname in files:
                content = files[fname]
                tokens = self.count_tokens(content)
                if tokens < remaining_tokens:
                    result[fname] = content
                    remaining_tokens -= tokens
        
        # Add other files with truncation
        for fname, content in files.items():
            if fname not in result:
                tokens = self.count_tokens(content)
                if tokens < remaining_tokens:
                    result[fname] = content
                    remaining_tokens -= tokens
                elif remaining_tokens > 100:
                    # Truncate to fit
                    truncated = self._truncate_to_tokens(content, remaining_tokens - 50)
                    result[fname] = truncated + "\n... [truncated]"
                    break
        
        return result
    
    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        tokens = self.encoding.encode(text)
        return self.encoding.decode(tokens[:max_tokens])