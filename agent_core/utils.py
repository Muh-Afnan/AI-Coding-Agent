import os
from pathlib import Path
from typing import List, Dict, Set, Optional
import tiktoken

class CodeContextGatherer:
    """Smart context gathering for LLM consumption."""
    
    # Files/dirs to always exclude
    EXCLUDE_PATTERNS = {
        '.venv', 'venv', '__pycache__', '.git', '.pytest_cache',
        'node_modules', '.next', 'dist', 'build', '.egg-info',
        'site-packages', '.tox', '.mypy_cache'
    }
    
    # File extensions to include
    INCLUDE_EXTENSIONS = {
        '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.go',
        '.rs', '.cpp', '.c', '.h', '.hpp', '.cs', '.rb',
        '.php', '.swift', '.kt', '.scala'
    }
    
    def __init__(self, project_root: str, model: str = "gpt-4"):
        self.project_root = Path(project_root).resolve()
        self.encoding = tiktoken.encoding_for_model(model)
    
    def gather_context(
        self,
        task: str,
        max_tokens: int = 40_000,
        relevant_files: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Intelligently gather project context for LLM.
        
        Args:
            task: The task description (used for relevance scoring)
            max_tokens: Maximum tokens to include
            relevant_files: Optional list of specific files to prioritize
        
        Returns:
            Dict mapping file paths to content
        """
        # Get all code files
        all_files = self._discover_files()
        
        # Score files by relevance
        scored_files = self._score_files(all_files, task, relevant_files)
        
        # Build context within token budget
        context = {}
        used_tokens = 0
        
        for file_path, score in scored_files:
            try:
                content = file_path.read_text(encoding='utf-8')
                
                # Count tokens for this file
                file_header = f"\n{'='*60}\nFILE: {file_path.relative_to(self.project_root)}\n{'='*60}\n"
                tokens_needed = self.count_tokens(file_header + content)
                
                if used_tokens + tokens_needed > max_tokens:
                    # Try to include truncated version
                    available_tokens = max_tokens - used_tokens - self.count_tokens(file_header)
                    if available_tokens > 500:  # Only include if we can fit meaningful content
                        truncated = self._truncate_to_tokens(content, available_tokens - 50)
                        context[str(file_path.relative_to(self.project_root))] = file_header + truncated + "\n... [truncated]"
                    break
                
                context[str(file_path.relative_to(self.project_root))] = file_header + content
                used_tokens += tokens_needed
                
            except Exception as e:
                print(f"⚠ Could not read {file_path}: {e}")
                continue
        
        return context
    
    def _discover_files(self) -> List[Path]:
        """Find all relevant code files in project."""
        files = []
        
        for file_path in self.project_root.rglob("*"):
            # Skip if in excluded directory
            if any(excluded in file_path.parts for excluded in self.EXCLUDE_PATTERNS):
                continue
            
            # Skip if not a code file
            if file_path.suffix not in self.INCLUDE_EXTENSIONS:
                continue
            
            # Skip if too large (>1MB probably not useful context)
            if file_path.stat().st_size > 1_000_000:
                continue
            
            files.append(file_path)
        
        return files
    
    def _score_files(
        self,
        files: List[Path],
        task: str,
        priority_files: Optional[List[str]] = None
    ) -> List[tuple[Path, float]]:
        """
        Score files by relevance to task.
        
        Returns:
            List of (file_path, score) tuples, sorted by score descending
        """
        scores = []
        task_lower = task.lower()
        priority_set = set(priority_files or [])
        
        for file_path in files:
            score = 0.0
            rel_path = str(file_path.relative_to(self.project_root))
            
            # Boost if explicitly mentioned
            if rel_path in priority_set:
                score += 100.0
            
            # Boost if file name appears in task
            if file_path.name.lower() in task_lower:
                score += 50.0
            
            # Boost if directory name appears in task
            for parent in file_path.parents:
                if parent != self.project_root and parent.name.lower() in task_lower:
                    score += 20.0
            
            # Penalize deeply nested files
            depth = len(file_path.relative_to(self.project_root).parts)
            score -= depth * 2
            
            # Boost main/init files
            if file_path.name in ['__init__.py', 'main.py', 'index.py', 'app.py']:
                score += 30.0
            
            # Boost test files if task mentions testing
            if 'test' in task_lower and 'test' in file_path.name.lower():
                score += 40.0
            
            # Penalize very small files (likely not important)
            if file_path.stat().st_size < 100:
                score -= 10.0
            
            scores.append((file_path, score))
        
        # Sort by score descending
        return sorted(scores, key=lambda x: x[1], reverse=True)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token budget."""
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        truncated_tokens = tokens[:max_tokens]
        return self.encoding.decode(truncated_tokens)

    def format_for_llm(self, context: Dict[str, str]) -> str:
        """Format context dict as string for LLM."""
        return "\n\n".join(context.values())

    def gather_code_context(project_root: str, char_limit: int = 40_000) -> str:
        """
        Legacy function for backwards compatibility.
        Prefer using CodeContextGatherer directly.
        """
        gatherer = CodeContextGatherer(project_root)
        # Convert char limit to rough token estimate (1 token ≈ 4 chars)
        token_limit = char_limit // 4
        context = gatherer.gather_context("", max_tokens=token_limit)
        return gatherer.format_for_llm(context)