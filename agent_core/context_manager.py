"""
Context Manager - Smart code context gathering with relevance ranking.
"""
import os
from pathlib import Path
from typing import List, Dict, Set, Optional
import re


class CodeContextGatherer:
    """
    Smart context gathering for LLM consumption with:
    - Intelligent file prioritization
    - Token-aware truncation
    - Semantic relevance scoring
    - Efficient caching
    """
    
    # Files/dirs to always exclude
    EXCLUDE_PATTERNS = {
        '.venv', 'venv', '__pycache__', '.git', '.pytest_cache',
        'node_modules', '.next', 'dist', 'build', '.egg-info',
        'site-packages', '.tox', '.mypy_cache', 'coverage',
        '.idea', '.vscode', 'target', 'bin', 'obj'
    }
    
    # File extensions to include
    INCLUDE_EXTENSIONS = {
        '.py', '.js', '.ts', '.jsx', '.tsx',
        '.java', '.go', '.rs', '.cpp', '.c', '.h', '.hpp',
        '.cs', '.rb', '.php', '.swift', '.kt', '.scala',
        '.yaml', '.yml', '.json', '.toml', '.md','.html','.css'
    }
    
    def __init__(self, project_root: str, verbose: bool = False):
        """
        Initialize context gatherer.
        
        Args:
            project_root: Root directory of project
            verbose: Enable detailed logging
        """
        self.project_root = Path(project_root).resolve()
        self.verbose = verbose
        self._file_cache: Dict[str, str] = {}
    
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
        if self.verbose:
            print(f"📚 Gathering context (max {max_tokens:,} tokens)")
        
        # Discover all code files
        all_files = self._discover_files()
        
        if self.verbose:
            print(f"   Found {len(all_files)} files")
        
        # Score files by relevance
        scored_files = self._score_files(all_files, task, relevant_files)
        
        # Build context within token budget
        context = self._build_context(scored_files, max_tokens)
        
        if self.verbose:
            print(f"   Included {len(context)} files in context")
        
        return context
    
    def _discover_files(self) -> List[Path]:
        """Find all relevant code files in project."""
        files = []
        
        try:
            for file_path in self.project_root.rglob("*"):
                # Skip if not a file
                if not file_path.is_file():
                    continue
                
                # Skip if in excluded directory
                if any(excluded in file_path.parts for excluded in self.EXCLUDE_PATTERNS):
                    continue
                
                # Skip if not a code file
                if file_path.suffix not in self.INCLUDE_EXTENSIONS:
                    continue
                
                # Skip if too large (>1MB probably not useful context)
                try:
                    if file_path.stat().st_size > 1_000_000:
                        continue
                except:
                    continue
                
                files.append(file_path)
        
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Error discovering files: {e}")
        
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
        
        # Extract key terms from task
        task_terms = self._extract_key_terms(task)
        
        for file_path in files:
            score = 0.0
            rel_path = str(file_path.relative_to(self.project_root))
            
            # Priority 1: Explicitly mentioned files
            if rel_path in priority_set or file_path.name in priority_set:
                score += 100.0
            
            # Priority 2: File name matches task terms
            file_name_lower = file_path.name.lower()
            for term in task_terms:
                if term in file_name_lower:
                    score += 50.0
            
            # Priority 3: Directory name matches task terms
            for parent in file_path.parents:
                if parent == self.project_root:
                    break
                parent_name = parent.name.lower()
                for term in task_terms:
                    if term in parent_name:
                        score += 20.0
            
            # Priority 4: Important file names
            important_names = {
                '__init__.py': 30,
                'main.py': 40,
                'app.py': 40,
                'index.py': 30,
                'index.js': 30,
                'config.py': 25,
                'settings.py': 25,
                'README.md': 20
            }
            score += important_names.get(file_path.name, 0)
            
            # Priority 5: Test files if task mentions testing
            if 'test' in task_lower and 'test' in file_name_lower:
                score += 40.0
            
            # Penalty: Deeply nested files (less likely to be relevant)
            depth = len(file_path.relative_to(self.project_root).parts)
            score -= depth * 2
            
            # Penalty: Very small files (might be empty or trivial)
            try:
                size = file_path.stat().st_size
                if size < 50:
                    score -= 15.0
                elif size < 200:
                    score -= 5.0
            except:
                pass
            
            scores.append((file_path, score))
        
        # Sort by score descending
        return sorted(scores, key=lambda x: x[1], reverse=True)
    
    def _extract_key_terms(self, text: str) -> Set[str]:
        """Extract key terms from text for matching."""
        # Remove common words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        
        # Extract words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter and return
        return {w for w in words if len(w) > 2 and w not in stop_words}
    
    def _build_context(
        self,
        scored_files: List[tuple[Path, float]],
        max_tokens: int
    ) -> Dict[str, str]:
        """Build context within token budget."""
        context = {}
        used_tokens = 0
        
        for file_path, score in scored_files:
            try:
                # Read file content (with caching)
                rel_path = str(file_path.relative_to(self.project_root))
                
                if rel_path in self._file_cache:
                    content = self._file_cache[rel_path]
                else:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    self._file_cache[rel_path] = content
                
                # Format with header
                file_header = f"\n{'='*60}\nFILE: {rel_path}\n{'='*60}\n"
                full_content = file_header + content
                
                # Estimate tokens (rough: 1 token ≈ 4 chars)
                tokens_needed = len(full_content) // 4
                
                if used_tokens + tokens_needed > max_tokens:
                    # Try to include truncated version
                    available_chars = (max_tokens - used_tokens) * 4 - len(file_header)
                    
                    if available_chars > 500:  # Only if meaningful content fits
                        truncated = content[:available_chars]
                        context[rel_path] = file_header + truncated + "\n... [truncated]"
                    
                    # Stop adding files
                    break
                
                context[rel_path] = full_content
                used_tokens += tokens_needed
            
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Could not read {file_path}: {e}")
                continue
        
        return context
    
    def format_for_llm(self, context: Dict[str, str]) -> str:
        """Format context dict as string for LLM."""
        return "\n\n".join(context.values())
    
    def clear_cache(self):
        """Clear file content cache."""
        self._file_cache.clear()
    
    def get_file_tree(self, max_depth: int = 3) -> str:
        """Get formatted file tree of project."""
        tree_lines = []
        
        def add_tree(path: Path, prefix: str = "", depth: int = 0):
            if depth > max_depth:
                return
            
            try:
                entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name))
                
                for i, entry in enumerate(entries):
                    # Skip excluded
                    if entry.name in self.EXCLUDE_PATTERNS:
                        continue
                    
                    is_last = i == len(entries) - 1
                    current_prefix = "└── " if is_last else "├── "
                    tree_lines.append(prefix + current_prefix + entry.name)
                    
                    if entry.is_dir():
                        next_prefix = prefix + ("    " if is_last else "│   ")
                        add_tree(entry, next_prefix, depth + 1)
            
            except PermissionError:
                pass
        
        tree_lines.append(self.project_root.name + "/")
        add_tree(self.project_root)
        
        return "\n".join(tree_lines)


# Example usage
if __name__ == "__main__":
    gatherer = CodeContextGatherer(".", verbose=True)
    
    print("\n" + "="*60)
    print("FILE TREE")
    print("="*60)
    print(gatherer.get_file_tree(max_depth=2))
    
    print("\n" + "="*60)
    print("GATHERING CONTEXT")
    print("="*60)
    
    task = "Add authentication to the API endpoints"
    context = gatherer.gather_context(task, max_tokens=10000)
    
    print(f"\nIncluded files:")
    for filepath in context.keys():
        print(f"  - {filepath}")
    
    print(f"\nTotal context length: {len(gatherer.format_for_llm(context))} chars")