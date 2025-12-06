"""
Enhanced Coder - Intelligent code generation with context awareness.
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
import re


@dataclass
class CodeProposal:
    """Proposal for code changes."""
    filepath: str
    content: str
    explanation: str
    change_type: str  # 'create', 'modify', 'delete'
    confidence: float = 0.8  # 0.0 to 1.0
    affected_lines: Optional[tuple] = None  # (start, end) for modifications
    
    def to_dict(self) -> dict:
        return {
            "filepath": self.filepath,
            "content": self.content,
            "explanation": self.explanation,
            "change_type": self.change_type,
            "confidence": self.confidence,
            "affected_lines": self.affected_lines
        }


class Coder:
    """
    Enhanced code generator with:
    - Smart context selection
    - Confidence scoring
    - Change type detection
    - Code validation
    """
    
    def __init__(
        self,
        llm,
        context_gatherer=None,
        max_context_tokens: int = 40000,
        verbose: bool = False
    ):
        self.llm = llm
        self.context_gatherer = context_gatherer
        self.max_context_tokens = max_context_tokens
        self.verbose = verbose
    
    def propose_changes(
        self,
        task: str,
        plan: Any,
        project_root: str,
        relevant_files: Optional[List[str]] = None
    ) -> Dict[str, CodeProposal]:
        """
        Generate code change proposals for a task.
        
        Args:
            task: Task description
            plan: ExecutionPlan or list of steps
            project_root: Project root directory
            relevant_files: Optional list of specific files to focus on
        
        Returns:
            Dict mapping filepath to CodeProposal
        """
        # Gather context
        context = self._gather_context(
            task,
            project_root,
            relevant_files
        )
        
        # Build prompt
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(task, plan, context)
        
        try:
            # Generate proposals
            response = self.llm.generate(
                system_prompt,
                user_prompt,
                temperature=0.4  # Balanced creativity and precision
            )
            
            # Parse response
            proposals = self._parse_proposals(response)
            
            if self.verbose:
                print(f"📝 Generated {len(proposals)} code proposals")
            
            return proposals
        
        except Exception as e:
            if self.verbose:
                print(f"❌ Code generation failed: {e}")
            
            # Return empty dict on failure
            return {}
    
    def _gather_context(
        self,
        task: str,
        project_root: str,
        relevant_files: Optional[List[str]] = None
    ) -> str:
        """Gather relevant code context."""
        if self.context_gatherer:
            # Use smart context gathering
            context_dict = self.context_gatherer.gather_context(
                task=task,
                max_tokens=self.max_context_tokens,
                relevant_files=relevant_files
            )
            return self.context_gatherer.format_for_llm(context_dict)
        else:
            # Fallback to simple gathering
            from agent_core.utils import gather_code_context
            return gather_code_context(
                project_root,
                char_limit=self.max_context_tokens * 4  # Rough conversion
            )
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for code generation."""
        return """You are an expert software engineer generating code changes.

Follow these principles:
1. **Minimal Changes**: Only modify what's necessary
2. **Code Quality**: Follow best practices and existing patterns
3. **Completeness**: Provide full file contents, not diffs
4. **Clarity**: Explain what and why you changed
5. **Safety**: Don't break existing functionality

Return a JSON object with this structure:
{
  "thought_process": "Your analysis of what needs to be done",
  "changes": {
    "path/to/file.py": {
      "content": "<complete file content>",
      "explanation": "What changed and why",
      "change_type": "create|modify|delete",
      "confidence": 0.9
    }
  }
}

CRITICAL RULES:
- Return ONLY valid JSON, no markdown code blocks
- Include COMPLETE file content, not diffs or snippets
- Don't include files that don't need changes
- Preserve existing code style and formatting
- Add type hints and docstrings where appropriate"""
    
    def _build_user_prompt(
        self,
        task: str,
        plan: Any,
        context: str
    ) -> str:
        """Build user prompt with task, plan, and context."""
        # Format plan
        if hasattr(plan, 'steps'):
            # ExecutionPlan object
            plan_text = "\n".join([
                f"{i+1}. {step.description}"
                for i, step in enumerate(plan.steps)
            ])
        elif isinstance(plan, list):
            # Simple list of strings
            plan_text = "\n".join([
                f"{i+1}. {step}"
                for i, step in enumerate(plan)
            ])
        else:
            plan_text = str(plan)
        
        return f"""# Task
{task}

# Execution Plan
{plan_text}

# Current Project Code
{context}

# Your Task
Generate the necessary code changes to accomplish this task. Return the JSON with your proposals."""
    
    def _parse_proposals(self, response: str) -> Dict[str, CodeProposal]:
        """Parse proposals from LLM response."""
        try:
            # Extract JSON
            data = self.llm.extract_json_from_response(response)
            
            proposals = {}
            changes = data.get("changes", {})
            
            for filepath, details in changes.items():
                # Handle both dict and string formats
                if isinstance(details, dict):
                    proposal = CodeProposal(
                        filepath=filepath,
                        content=details.get("content", ""),
                        explanation=details.get("explanation", ""),
                        change_type=details.get("change_type", "modify"),
                        confidence=details.get("confidence", 0.8)
                    )
                else:
                    # Treat as raw content
                    proposal = CodeProposal(
                        filepath=filepath,
                        content=str(details),
                        explanation="No explanation provided",
                        change_type="modify"
                    )
                
                proposals[filepath] = proposal
            
            return proposals
        
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Failed to parse proposals: {e}")
            
            # Try fallback parsing
            return self._fallback_parse(response)
    
    def _fallback_parse(self, response: str) -> Dict[str, CodeProposal]:
        """Fallback parsing for malformed responses."""
        # Look for file patterns in the response
        file_pattern = r'(?:File|Path|Filename):\s*([^\n]+)'
        code_pattern = r'```(?:python|javascript|java)?\n(.*?)```'
        
        files = re.findall(file_pattern, response, re.IGNORECASE)
        codes = re.findall(code_pattern, response, re.DOTALL)
        
        proposals = {}
        
        # Match files with code blocks
        for i, (filepath, code) in enumerate(zip(files, codes)):
            filepath = filepath.strip()
            proposals[filepath] = CodeProposal(
                filepath=filepath,
                content=code.strip(),
                explanation="Extracted from response",
                change_type="modify",
                confidence=0.5  # Low confidence for fallback
            )
        
        # If still nothing, create a single proposal
        if not proposals:
            proposals["proposed_change.txt"] = CodeProposal(
                filepath="proposed_change.txt",
                content=response,
                explanation="Raw model output (couldn't parse)",
                change_type="create",
                confidence=0.3
            )
        
        return proposals
    
    def validate_proposal(
        self,
        proposal: CodeProposal,
        file_manager=None
    ) -> tuple[bool, Optional[str]]:
        """
        Validate a code proposal before applying.
        
        Returns:
            (is_valid, error_message)
        """
        # Check for empty content
        if not proposal.content or not proposal.content.strip():
            return False, "Empty content"
        
        # Check for obvious syntax errors in Python files
        if proposal.filepath.endswith('.py'):
            try:
                compile(proposal.content, proposal.filepath, 'exec')
            except SyntaxError as e:
                return False, f"Python syntax error: {e}"
        
        # Check if creating a new file in a non-existent directory
        if proposal.change_type == "create" and file_manager:
            from pathlib import Path
            parent = Path(proposal.filepath).parent
            if parent != Path(".") and file_manager:
                # This is okay, FileManager will create directories
                pass
        
        # Check confidence threshold
        if proposal.confidence < 0.3:
            return False, f"Low confidence ({proposal.confidence:.2f})"
        
        return True, None
    
    def review_changes(
        self,
        proposals: Dict[str, CodeProposal],
        file_manager
    ) -> Dict[str, Any]:
        """
        Review proposed changes and provide summary.
        
        Returns:
            Dict with review information
        """
        review = {
            "total_files": len(proposals),
            "by_type": {"create": 0, "modify": 0, "delete": 0},
            "total_lines": 0,
            "avg_confidence": 0.0,
            "warnings": []
        }
        
        total_confidence = 0.0
        
        for filepath, proposal in proposals.items():
            # Count by type
            review["by_type"][proposal.change_type] += 1
            
            # Count lines
            review["total_lines"] += len(proposal.content.splitlines())
            
            # Track confidence
            total_confidence += proposal.confidence
            
            # Check for warnings
            is_valid, error = self.validate_proposal(proposal, file_manager)
            if not is_valid:
                review["warnings"].append(f"{filepath}: {error}")
            
            # Warn on large changes
            lines = len(proposal.content.splitlines())
            if lines > 500:
                review["warnings"].append(
                    f"{filepath}: Large file ({lines} lines)"
                )
        
        # Calculate average confidence
        if proposals:
            review["avg_confidence"] = total_confidence / len(proposals)
        
        return review
    
    def generate_diff_summary(
        self,
        proposals: Dict[str, CodeProposal],
        file_manager
    ) -> str:
        """Generate human-readable summary of changes."""
        summary = "# Proposed Changes\n\n"
        
        for filepath, proposal in proposals.items():
            summary += f"## {filepath}\n"
            summary += f"**Type**: {proposal.change_type}\n"
            summary += f"**Confidence**: {proposal.confidence:.0%}\n"
            summary += f"**Explanation**: {proposal.explanation}\n"
            
            # Show line count
            lines = len(proposal.content.splitlines())
            summary += f"**Lines**: {lines}\n"
            
            # Show diff for modifications
            if proposal.change_type == "modify" and file_manager:
                old_content = file_manager.read_file(filepath)
                if old_content:
                    old_lines = len(old_content.splitlines())
                    diff = lines - old_lines
                    summary += f"**Change**: {'+' if diff > 0 else ''}{diff} lines\n"
            
            summary += "\n"
        
        return summary


# Example usage
if __name__ == "__main__":
    from agent_core.llm_client import LLMClient
    from agent_core.file_manager import FileManager
    import os
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY to test")
        exit(1)
    
    llm = LLMClient(api_key=api_key, verbose=False)
    coder = Coder(llm, verbose=True)
    fm = FileManager(project_root=".")
    
    # Test code generation
    task = "Create a simple calculator class with add and subtract methods"
    plan = ["Design the Calculator class", "Implement methods", "Add docstrings"]
    
    proposals = coder.propose_changes(task, plan, ".")
    
    print("\n" + "="*60)
    print("GENERATED PROPOSALS")
    print("="*60)
    
    for filepath, proposal in proposals.items():
        print(f"\nFile: {filepath}")
        print(f"Type: {proposal.change_type}")
        print(f"Confidence: {proposal.confidence:.0%}")
        print(f"Explanation: {proposal.explanation}")
        print(f"Content preview: {proposal.content[:200]}...")
    
    # Review changes
    review = coder.review_changes(proposals, fm)
    print("\n" + "="*60)
    print("REVIEW SUMMARY")
    print("="*60)
    print(json.dumps(review, indent=2))