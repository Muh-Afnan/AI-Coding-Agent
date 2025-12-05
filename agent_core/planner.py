from typing import List
from .llm_client import LLMClient
from typing import List, Dict, Optional
from dataclasses import dataclass
import re

@dataclass
class PlanStep:
    """Single step in execution plan."""
    step_num: int
    description: str
    estimated_complexity: str  # 'low', 'medium', 'high'
    dependencies: List[int]  # Step numbers this depends on

class Planner:
    def __init__(self, llm: LLMClient):
        self.llm = llm
    
    def create_plan(self, task: str, context: Optional[str] = None) -> List[str]:
        """
        Create structured execution plan for task.
        
        Args:
            task: Natural language task description
            context: Optional project context
        
        Returns:
            List of actionable steps
        """
        system = """You are a software engineering planning assistant.

        Create a clear, actionable plan to accomplish the given task. Follow these rules:
        1. Break the task into 3-7 concrete steps
        2. Each step should be specific and testable
        3. Order steps by logical dependencies
        4. Focus on minimal changes to achieve the goal
        5. Consider error handling and edge cases

        Return ONLY the plan as a numbered list, one step per line.
        Example:
        1. Create new file api/endpoints.py with basic structure
        2. Add authentication middleware to validate requests
        3. Implement GET /users endpoint with pagination
        4. Add unit tests for the new endpoint
        5. Update API documentation"""

        user_prompt = f"Task: {task}"
        if context:
            user_prompt += f"\n\nProject Context:\n{context[:2000]}"  # Limit context
        
        response = self.llm.generate(system, user_prompt, temperature=0.3)  # Lower temp for planning
        
        return self._parse_plan(response)
    
    def _parse_plan(self, response: str) -> List[str]:
        """
        Robust parsing of plan from LLM response.
        
        Handles various formats:
        - "1. Step description"
        - "1) Step description"
        - "Step 1: Description"
        - "- Step description"
        """
        lines = [l.strip() for l in response.splitlines() if l.strip()]
        plan = []
        
        patterns = [
            r'^\d+[\.\)]\s*(.+)$',  # "1. " or "1) "
            r'^Step\s+\d+:\s*(.+)$',  # "Step 1: "
            r'^[-•]\s*(.+)$',  # "- " or "• "
        ]
        
        for line in lines:
            matched = False
            for pattern in patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    step = match.group(1).strip()
                    if step and len(step) > 10:  # Ignore very short steps
                        plan.append(step)
                        matched = True
                        break
            
            # If no pattern matched but looks like a step, include it
            if not matched and len(line) > 15 and not line.endswith(':'):
                plan.append(line)
        
        # Limit to reasonable number
        return plan[:8]
    
    def refine_plan_from_failure(
        self, 
        original_plan: List[str], 
        execution_result,
        iteration_num: int
    ) -> List[str]:
        """
        Update plan based on execution failure.
        
        Args:
            original_plan: The plan that was being executed
            execution_result: Result from failed execution
            iteration_num: Which iteration failed
        
        Returns:
            Revised plan
        """
        system = """You are a debugging assistant helping to fix code execution failures.

        Analyze the error and create a focused plan to fix the issue. Your plan should:
        1. Identify the root cause of the failure
        2. Propose specific fixes (not vague "debug the code")
        3. Include verification steps
        4. Be concise (3-5 steps max)

        Return ONLY the revised plan as a numbered list."""

        # Build context about the failure
        failure_context = f"""
        Original Plan:
        {self._format_plan(original_plan)}

        Failure at Iteration {iteration_num}:
        Return Code: {execution_result.returncode}

        Standard Output:
        {execution_result.stdout[:500]}

        Standard Error:
        {execution_result.stderr[:500]}
        """
        
        if execution_result.timed_out:
            failure_context += "\n⚠ Note: The command timed out."
        
        response = self.llm.generate(system, failure_context, temperature=0.3)
        revised_plan = self._parse_plan(response)
        
        # Ensure we have a valid plan
        if not revised_plan:
            # Fallback: create basic debugging plan
            revised_plan = [
                "Review the error message and identify the failing component",
                "Add logging or print statements to narrow down the issue",
                "Fix the identified bug",
                "Re-run tests to verify the fix"
            ]
        
        return revised_plan
    
    def _format_plan(self, plan: List[str]) -> str:
        """Format plan steps for display."""
        return "\n".join([f"{i+1}. {step}" for i, step in enumerate(plan)])
    
    def estimate_complexity(self, task: str) -> Dict[str, any]:
        """
        Estimate task complexity and required resources.
        
        Returns:
            Dict with complexity metrics
        """
        system = """You are a software project estimator.

        Analyze the given task and return a JSON object with:
        {
        "complexity": "low|medium|high",
        "estimated_steps": <number>,
        "estimated_time_minutes": <number>,
        "requires_new_files": true|false,
        "risk_level": "low|medium|high",
        "rationale": "brief explanation"
        }"""

        response = self.llm.generate(system, f"Task: {task}", temperature=0.2)
        
        try:
            import json
            # Try to extract JSON
            json_str = self.llm.extract_code_from_response(response)
            return json.loads(json_str)
        except Exception:
            # Fallback to medium complexity
            return {
                "complexity": "medium",
                "estimated_steps": 5,
                "estimated_time_minutes": 30,
                "requires_new_files": False,
                "risk_level": "medium",
                "rationale": "Could not parse complexity estimate"
            }