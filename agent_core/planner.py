"""
Enhanced Planner - Smart task decomposition and adaptive planning.
"""
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import re
import json


class StepStatus(Enum):
    """Status of a plan step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanStep:
    """Single step in an execution plan."""
    step_num: int
    description: str
    rationale: str = ""
    estimated_complexity: str = "medium"  # low, medium, high
    dependencies: List[int] = None
    status: StepStatus = StepStatus.PENDING
    result: Optional[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
    
    def to_dict(self) -> dict:
        data = asdict(self)
        data['status'] = self.status.value
        return data
    
    def mark_completed(self, result: str = ""):
        """Mark step as completed."""
        self.status = StepStatus.COMPLETED
        self.result = result
    
    def mark_failed(self, error: str = ""):
        """Mark step as failed."""
        self.status = StepStatus.FAILED
        self.result = error


@dataclass
class ExecutionPlan:
    """Complete execution plan with metadata."""
    task: str
    steps: List[PlanStep]
    strategy: str = "sequential"  # sequential, parallel, adaptive
    estimated_duration_minutes: int = 0
    risk_level: str = "medium"
    
    def to_dict(self) -> dict:
        data = asdict(self)
        data['steps'] = [step.to_dict() for step in self.steps]
        return data
    
    def get_pending_steps(self) -> List[PlanStep]:
        """Get all pending steps."""
        return [s for s in self.steps if s.status == StepStatus.PENDING]
    
    def get_completed_steps(self) -> List[PlanStep]:
        """Get all completed steps."""
        return [s for s in self.steps if s.status == StepStatus.COMPLETED]
    
    def is_complete(self) -> bool:
        """Check if all steps are completed."""
        return all(s.status == StepStatus.COMPLETED for s in self.steps)
    
    def has_failures(self) -> bool:
        """Check if any steps failed."""
        return any(s.status == StepStatus.FAILED for s in self.steps)


class Planner:
    """
    Enhanced task planner with:
    - Intelligent task decomposition
    - Complexity estimation
    - Adaptive replanning
    - Dependency tracking
    """
    
    def __init__(self, llm, verbose: bool = False):
        self.llm = llm
        self.verbose = verbose
    
    def create_plan(
        self,
        task: str,
        context: Optional[str] = None,
        max_steps: int = 8
    ) -> ExecutionPlan:
        """
        Create a structured execution plan for a task.
        
        Args:
            task: Natural language task description
            context: Optional project context
            max_steps: Maximum number of steps to generate
        
        Returns:
            ExecutionPlan with detailed steps
        """
        system_prompt = """You are an expert software engineering planner.

Your job is to break down tasks into clear, actionable steps. Follow these principles:

1. **Be Specific**: Each step should be concrete and testable
2. **Order Matters**: Steps should follow logical dependencies
3. **Minimal Changes**: Prefer small, incremental changes
4. **Error Handling**: Consider what could go wrong
5. **Testing**: Include verification steps

Return a JSON object with this structure:
{
  "strategy": "sequential|parallel|adaptive",
  "estimated_duration_minutes": <number>,
  "risk_level": "low|medium|high",
  "steps": [
    {
      "step_num": 1,
      "description": "Clear description of what to do",
      "rationale": "Why this step is needed",
      "estimated_complexity": "low|medium|high",
      "dependencies": [<list of step numbers this depends on>]
    }
  ]
}

Keep steps to {max_steps} or fewer."""

        user_prompt = f"""Task: {task}"""
        
        if context:
            user_prompt += f"\n\nProject Context:\n{context[:2000]}"
        
        user_prompt += f"\n\nCreate a detailed execution plan (max {max_steps} steps):"
        
        try:
            response = self.llm.generate(
                system_prompt,
                user_prompt,
                temperature=0.3  # Lower temperature for planning
            )
            
            # Extract and parse JSON
            plan_data = self.llm.extract_json_from_response(response)
            
            # Validate and create plan
            return self._parse_plan_data(task, plan_data, max_steps)
        
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Plan generation failed, creating fallback: {e}")
            
            # Fallback: create simple plan
            return self._create_fallback_plan(task)
    
    def _parse_plan_data(
        self,
        task: str,
        data: dict,
        max_steps: int
    ) -> ExecutionPlan:
        """Parse plan data from LLM response."""
        steps = []
        
        for step_data in data.get("steps", [])[:max_steps]:
            step = PlanStep(
                step_num=step_data.get("step_num", len(steps) + 1),
                description=step_data.get("description", ""),
                rationale=step_data.get("rationale", ""),
                estimated_complexity=step_data.get("estimated_complexity", "medium"),
                dependencies=step_data.get("dependencies", [])
            )
            steps.append(step)
        
        # Ensure we have at least one step
        if not steps:
            steps = [PlanStep(
                step_num=1,
                description=task,
                rationale="Direct execution of task"
            )]
        
        return ExecutionPlan(
            task=task,
            steps=steps,
            strategy=data.get("strategy", "sequential"),
            estimated_duration_minutes=data.get("estimated_duration_minutes", 0),
            risk_level=data.get("risk_level", "medium")
        )
    
    def _create_fallback_plan(self, task: str) -> ExecutionPlan:
        """Create a simple fallback plan."""
        return ExecutionPlan(
            task=task,
            steps=[
                PlanStep(
                    step_num=1,
                    description="Analyze the task requirements",
                    rationale="Understand what needs to be done"
                ),
                PlanStep(
                    step_num=2,
                    description=f"Implement: {task}",
                    rationale="Execute the main task",
                    dependencies=[1]
                ),
                PlanStep(
                    step_num=3,
                    description="Verify the implementation",
                    rationale="Ensure changes work correctly",
                    dependencies=[2]
                )
            ],
            strategy="sequential",
            risk_level="medium"
        )
    
    def refine_plan_from_failure(
        self,
        original_plan: ExecutionPlan,
        failed_step: PlanStep,
        error_details: Any
    ) -> ExecutionPlan:
        """
        Adapt the plan based on a failure.
        
        Args:
            original_plan: The original execution plan
            failed_step: The step that failed
            error_details: Details about the failure
        
        Returns:
            Revised ExecutionPlan
        """
        system_prompt = """You are a debugging expert helping to revise a failed plan.

Analyze the failure and create a focused plan to fix the issue. Your revised plan should:
1. Identify the root cause
2. Propose specific fixes (not vague "debug the code")
3. Include verification steps
4. Be concise (3-5 steps)

Return the same JSON structure as before with revised steps."""

        # Build context about failure
        failure_context = self._build_failure_context(
            original_plan,
            failed_step,
            error_details
        )
        
        user_prompt = f"""Original Task: {original_plan.task}

{failure_context}

Create a revised plan to fix this issue:"""
        
        try:
            response = self.llm.generate(
                system_prompt,
                user_prompt,
                temperature=0.2  # Very low temperature for debugging
            )
            
            plan_data = self.llm.extract_json_from_response(response)
            revised_plan = self._parse_plan_data(original_plan.task, plan_data, max_steps=6)
            
            if self.verbose:
                print("🔄 Created revised plan based on failure")
            
            return revised_plan
        
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Plan refinement failed: {e}")
            
            # Fallback: create debugging plan
            return self._create_debug_plan(original_plan.task, failed_step, error_details)
    
    def _build_failure_context(
        self,
        plan: ExecutionPlan,
        failed_step: PlanStep,
        error_details: Any
    ) -> str:
        """Build context description of the failure."""
        context = f"""
Execution Summary:
- Completed Steps: {len(plan.get_completed_steps())}/{len(plan.steps)}
- Failed Step: #{failed_step.step_num} - {failed_step.description}

Failure Details:
"""
        
        if hasattr(error_details, 'returncode'):
            # Command execution failure
            context += f"""
Return Code: {error_details.returncode}
Stdout: {error_details.stdout[:500]}
Stderr: {error_details.stderr[:500]}
"""
        elif isinstance(error_details, str):
            context += f"\nError: {error_details[:500]}"
        else:
            context += f"\nError: {str(error_details)[:500]}"
        
        # Add context of completed steps
        if plan.get_completed_steps():
            context += "\n\nPrevious Successful Steps:\n"
            for step in plan.get_completed_steps()[-3:]:  # Last 3
                context += f"  {step.step_num}. {step.description}\n"
        
        return context
    
    def _create_debug_plan(
        self,
        task: str,
        failed_step: PlanStep,
        error_details: Any
    ) -> ExecutionPlan:
        """Create a basic debugging plan as fallback."""
        return ExecutionPlan(
            task=f"Debug: {task}",
            steps=[
                PlanStep(
                    step_num=1,
                    description=f"Analyze error from step: {failed_step.description}",
                    rationale="Identify root cause of failure"
                ),
                PlanStep(
                    step_num=2,
                    description="Implement fix for the identified issue",
                    rationale="Address the root cause",
                    dependencies=[1]
                ),
                PlanStep(
                    step_num=3,
                    description="Re-run failed step to verify fix",
                    rationale="Confirm the issue is resolved",
                    dependencies=[2]
                )
            ],
            strategy="sequential",
            risk_level="high"
        )
    
    def estimate_task_complexity(self, task: str) -> Dict[str, Any]:
        """
        Estimate complexity and resource requirements for a task.
        
        Returns:
            Dict with complexity metrics
        """
        system_prompt = """You are a software project estimator.

Analyze the task and return a JSON object:
{
  "complexity": "low|medium|high",
  "estimated_steps": <number>,
  "estimated_time_minutes": <number>,
  "requires_new_files": true|false,
  "requires_external_deps": true|false,
  "risk_level": "low|medium|high",
  "challenges": ["challenge 1", "challenge 2"],
  "rationale": "brief explanation"
}"""

        user_prompt = f"Task: {task}\n\nAnalyze complexity:"
        
        try:
            response = self.llm.generate(
                system_prompt,
                user_prompt,
                temperature=0.2
            )
            
            return self.llm.extract_json_from_response(response)
        
        except Exception:
            # Fallback
            return {
                "complexity": "medium",
                "estimated_steps": 5,
                "estimated_time_minutes": 30,
                "requires_new_files": False,
                "requires_external_deps": False,
                "risk_level": "medium",
                "challenges": ["Unable to estimate"],
                "rationale": "Default estimate due to analysis failure"
            }
    
    def format_plan_for_display(self, plan: ExecutionPlan) -> str:
        """Format plan for human-readable display."""
        output = f"""
{'='*60}
EXECUTION PLAN
{'='*60}
Task: {plan.task}
Strategy: {plan.strategy}
Risk Level: {plan.risk_level}
Estimated Duration: {plan.estimated_duration_minutes} minutes
Total Steps: {len(plan.steps)}

{'─'*60}
STEPS
{'─'*60}
"""
        
        for step in plan.steps:
            status_emoji = {
                StepStatus.PENDING: "⏳",
                StepStatus.IN_PROGRESS: "🔄",
                StepStatus.COMPLETED: "✅",
                StepStatus.FAILED: "❌",
                StepStatus.SKIPPED: "⊘"
            }
            
            emoji = status_emoji.get(step.status, "⏳")
            output += f"\n{emoji} Step {step.step_num}: {step.description}\n"
            
            if step.rationale:
                output += f"   Rationale: {step.rationale}\n"
            
            if step.dependencies:
                output += f"   Depends on: {', '.join(map(str, step.dependencies))}\n"
            
            output += f"   Complexity: {step.estimated_complexity}\n"
            
            if step.result:
                output += f"   Result: {step.result[:100]}\n"
        
        output += f"\n{'='*60}\n"
        
        return output
    
    def get_next_executable_step(self, plan: ExecutionPlan) -> Optional[PlanStep]:
        """
        Get the next step that can be executed based on dependencies.
        
        Returns:
            Next executable PlanStep or None if no step is ready
        """
        for step in plan.steps:
            if step.status != StepStatus.PENDING:
                continue
            
            # Check if all dependencies are completed
            if not step.dependencies:
                return step
            
            deps_completed = all(
                plan.steps[dep - 1].status == StepStatus.COMPLETED
                for dep in step.dependencies
                if 0 < dep <= len(plan.steps)
            )
            
            if deps_completed:
                return step
        
        return None


# Example usage
if __name__ == "__main__":
    from agent_core.llm_client import LLMClient
    import os
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY to test")
        exit(1)
    
    llm = LLMClient(api_key=api_key, verbose=False)
    planner = Planner(llm, verbose=True)
    
    # Test plan creation
    print("\n" + "="*60)
    print("TESTING PLANNER")
    print("="*60)
    
    task = "Add user authentication with JWT tokens to the API"
    plan = planner.create_plan(task)
    
    print(planner.format_plan_for_display(plan))
    
    # Test complexity estimation
    print("\n" + "="*60)
    print("COMPLEXITY ESTIMATION")
    print("="*60)
    
    complexity = planner.estimate_task_complexity(task)
    print(json.dumps(complexity, indent=2))