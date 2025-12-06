"""
Agent Orchestrator - The brain that coordinates all components.

This is the "missing structure" that makes everything autonomous.
"""
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import time


class AgentState(Enum):
    """Current state of the agent."""
    IDLE = "idle"
    PLANNING = "planning"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    REFLECTING = "reflecting"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentAction:
    """An action the agent wants to take."""
    action_type: str  # 'read_file', 'write_file', 'run_command', 'generate_code', etc.
    parameters: Dict[str, Any]
    reasoning: str
    confidence: float = 0.8


@dataclass
class AgentObservation:
    """Result from an action."""
    action: AgentAction
    success: bool
    result: Any
    error: Optional[str] = None
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class AgentResult:
    """Final result from agent execution."""
    success: bool
    task: str
    iterations: int
    total_duration: float
    final_state: AgentState
    history: List[Dict[str, Any]]
    error: Optional[str] = None


class Agent:
    """
    Autonomous Agent Orchestrator.
    
    Implements the ReAct pattern (Reason + Act):
    1. Perceive: Understand current state
    2. Think: Reason about what to do
    3. Plan: Break down into steps
    4. Act: Execute using tools
    5. Observe: Check results
    6. Reflect: Learn from outcomes
    7. Repeat until task is complete
    """
    
    def __init__(
        self,
        llm,
        planner,
        coder,
        tools,
        memory,
        context_gatherer,
        project_root: str,
        max_iterations: int = 20,
        verbose: bool = False
    ):
        """
        Initialize the agent.
        
        Args:
            llm: LLM client for reasoning
            planner: Task planner
            coder: Code generator
            tools: Tool registry
            memory: Memory system
            context_gatherer: Context manager
            project_root: Project root directory
            max_iterations: Maximum reasoning cycles
            verbose: Enable detailed logging
        """
        self.llm = llm
        self.planner = planner
        self.coder = coder
        self.tools = tools
        self.memory = memory
        self.context_gatherer = context_gatherer
        self.project_root = project_root
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        self.state = AgentState.IDLE
        self.current_task_id: Optional[str] = None
        self.history: List[Dict[str, Any]] = []
    
    def run(self, task: str, auto_approve: bool = False) -> AgentResult:
        """
        Execute a task autonomously.
        
        Args:
            task: Natural language task description
            auto_approve: If True, apply changes without asking
        
        Returns:
            AgentResult with execution outcome
        """
        start_time = time.time()
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🤖 AGENT STARTING")
            print(f"{'='*60}")
            print(f"Task: {task}")
            print(f"Max iterations: {self.max_iterations}")
            print(f"{'='*60}\n")
        
        # Initialize
        self.current_task_id = self.memory.start_task(task)
        self.state = AgentState.PLANNING
        self.history = []
        
        try:
            # Phase 1: Create execution plan
            plan = self._planning_phase(task)
            
            if not plan or not plan.steps:
                return self._create_result(
                    success=False,
                    task=task,
                    start_time=start_time,
                    error="Failed to create execution plan"
                )
            
            # Phase 2: Execute plan with adaptive reasoning
            success = self._execution_phase(task, plan, auto_approve)
            
            # Finalize
            self.state = AgentState.COMPLETED if success else AgentState.FAILED
            self.memory.complete_task(self.current_task_id, success=success)
            
            return self._create_result(
                success=success,
                task=task,
                start_time=start_time
            )
        
        except KeyboardInterrupt:
            if self.verbose:
                print("\n⚠️  Agent interrupted by user")
            
            self.state = AgentState.FAILED
            self.memory.complete_task(self.current_task_id, success=False)
            
            return self._create_result(
                success=False,
                task=task,
                start_time=start_time,
                error="Interrupted by user"
            )
        
        except Exception as e:
            if self.verbose:
                print(f"\n❌ Agent crashed: {e}")
            
            self.state = AgentState.FAILED
            self.memory.complete_task(self.current_task_id, success=False)
            
            return self._create_result(
                success=False,
                task=task,
                start_time=start_time,
                error=str(e)
            )
    
    def _planning_phase(self, task: str):
        """Phase 1: Create execution plan."""
        if self.verbose:
            print("📋 Planning phase...")
        
        self.state = AgentState.PLANNING
        
        # Gather context for planning
        context = self.context_gatherer.gather_context(task, max_tokens=10000)
        context_str = self.context_gatherer.format_for_llm(context)
        
        # Create plan
        plan = self.planner.create_plan(task, context=context_str)
        
        if self.verbose:
            print(self.planner.format_plan_for_display(plan))
        
        self._record_event({
            'phase': 'planning',
            'plan': plan.to_dict(),
            'timestamp': time.time()
        })
        
        return plan
    
    def _execution_phase(self, task: str, plan, auto_approve: bool) -> bool:
        """Phase 2: Execute plan with autonomous reasoning."""
        iteration = 0
        
        while iteration < self.max_iterations:
            iteration += 1
            
            if self.verbose:
                print(f"\n{'─'*60}")
                print(f"Iteration {iteration}/{self.max_iterations}")
                print(f"{'─'*60}\n")
            
            # Get next step from plan
            next_step = self.planner.get_next_executable_step(plan)
            
            if not next_step:
                # Check if we're done
                if plan.is_complete():
                    if self.verbose:
                        print("✅ All plan steps completed!")
                    return True
                elif plan.has_failures():
                    if self.verbose:
                        print("❌ Plan has failed steps")
                    return False
                else:
                    if self.verbose:
                        print("⚠️  No more executable steps")
                    return False
            
            if self.verbose:
                print(f"📍 Executing: Step {next_step.step_num}")
                print(f"   {next_step.description}")
            
            # Mark step as in progress
            next_step.status = next_step.status.__class__.IN_PROGRESS
            
            # Think: Decide how to execute this step
            action = self._thinking_phase(task, plan, next_step)
            
            if not action:
                next_step.mark_failed("Failed to determine action")
                continue
            
            # Act: Execute the action
            observation = self._acting_phase(action)
            
            # Observe: Check results
            success = self._observing_phase(next_step, observation, auto_approve)
            
            # Update step status
            if success:
                next_step.mark_completed(str(observation.result)[:200])
            else:
                next_step.mark_failed(observation.error or "Unknown error")
                
                # Reflect: Should we replan?
                if self._should_replan(plan, next_step, observation):
                    if self.verbose:
                        print("🔄 Replanning due to failure...")
                    plan = self.planner.refine_plan_from_failure(
                        plan,
                        next_step,
                        observation
                    )
        
        if self.verbose:
            print(f"\n⚠️  Reached maximum iterations ({self.max_iterations})")
        
        return plan.is_complete()
    
    def _thinking_phase(self, task: str, plan, step) -> Optional[AgentAction]:
        """Phase 3: Reason about what action to take."""
        if self.verbose:
            print("🤔 Thinking phase...")
        
        self.state = AgentState.THINKING
        
        # Build reasoning prompt
        system_prompt = """You are an autonomous coding agent deciding what action to take.

Given a task, plan, and current step, decide the best action. Return JSON:
{
  "action_type": "generate_code|read_file|run_command|list_files",
  "parameters": {<action-specific params>},
  "reasoning": "Why this action",
  "confidence": 0.9
}

Action types:
- generate_code: Create/modify code files
- read_file: Read file contents
- run_command: Execute a command
- list_files: List project files"""

        # Get context about previous attempts
        context = self.memory.export_for_llm_context(self.current_task_id, max_iterations=3)
        
        user_prompt = f"""Task: {task}

Current Step: {step.description}

Previous Work:
{context}

Available tools: {', '.join(self.tools.list_tools())}

What action should I take?"""
        
        try:
            response = self.llm.generate(
                system_prompt,
                user_prompt,
                temperature=0.4
            )
            
            data = self.llm.extract_json_from_response(response)
            
            action = AgentAction(
                action_type=data.get('action_type', 'generate_code'),
                parameters=data.get('parameters', {}),
                reasoning=data.get('reasoning', ''),
                confidence=data.get('confidence', 0.8)
            )
            
            if self.verbose:
                print(f"💡 Action: {action.action_type}")
                print(f"   Reasoning: {action.reasoning}")
            
            return action
        
        except Exception as e:
            if self.verbose:
                print(f"❌ Thinking failed: {e}")
            
            # Fallback: default to code generation
            return AgentAction(
                action_type="generate_code",
                parameters={},
                reasoning="Fallback to code generation",
                confidence=0.5
            )
    
    def _acting_phase(self, action: AgentAction) -> AgentObservation:
        """Phase 4: Execute the chosen action."""
        if self.verbose:
            print("🔧 Acting phase...")
        
        self.state = AgentState.ACTING
        
        try:
            if action.action_type == "generate_code":
                # Use coder to generate code
                task = action.parameters.get('task', '')
                proposals = self.coder.propose_changes(
                    task=task,
                    plan=[],
                    project_root=self.project_root
                )
                
                return AgentObservation(
                    action=action,
                    success=len(proposals) > 0,
                    result=proposals
                )
            
            elif action.action_type in self.tools.list_tools():
                # Use tool from registry
                result = self.tools.execute(action.action_type, **action.parameters)
                
                return AgentObservation(
                    action=action,
                    success=result.is_success(),
                    result=result.output,
                    error=result.error
                )
            
            else:
                return AgentObservation(
                    action=action,
                    success=False,
                    result=None,
                    error=f"Unknown action type: {action.action_type}"
                )
        
        except Exception as e:
            return AgentObservation(
                action=action,
                success=False,
                result=None,
                error=str(e)
            )
    
    def _observing_phase(
        self,
        step,
        observation: AgentObservation,
        auto_approve: bool
    ) -> bool:
        """Phase 5: Check results and get approval if needed."""
        if self.verbose:
            print("👀 Observing phase...")
        
        self.state = AgentState.OBSERVING
        
        if not observation.success:
            if self.verbose:
                print(f"❌ Action failed: {observation.error}")
            return False
        
        # If code generation, need approval
        if observation.action.action_type == "generate_code":
            proposals = observation.result
            
            if auto_approve:
                if self.verbose:
                    print(f"✅ Auto-approving {len(proposals)} proposals")
                
                # Apply all proposals
                for filepath, proposal in proposals.items():
                    tool_result = self.tools.execute(
                        "write_file",
                        filepath=filepath,
                        content=proposal.content
                    )
                    
                    if not tool_result.is_success():
                        if self.verbose:
                            print(f"❌ Failed to write {filepath}")
                        return False
                
                return True
            else:
                # TODO: Interactive approval
                if self.verbose:
                    print("⚠️  Interactive approval not yet implemented")
                return False
        
        # Other actions succeed automatically
        return True
    
    def _should_replan(self, plan, failed_step, observation) -> bool:
        """Decide if we should create a new plan."""
        # Replan if high confidence action failed
        if observation.action.confidence > 0.7:
            return True
        
        # Replan if multiple consecutive failures
        recent_failures = sum(
            1 for step in plan.steps[-3:]
            if step.status == step.status.__class__.FAILED
        )
        if recent_failures >= 2:
            return True
        
        return False
    
    def _record_event(self, event: Dict[str, Any]):
        """Record event in history."""
        self.history.append(event)
    
    def _create_result(
        self,
        success: bool,
        task: str,
        start_time: float,
        error: Optional[str] = None
    ) -> AgentResult:
        """Create final result object."""
        return AgentResult(
            success=success,
            task=task,
            iterations=len(self.history),
            total_duration=time.time() - start_time,
            final_state=self.state,
            history=self.history,
            error=error
        )


# Example usage
if __name__ == "__main__":
    print("Agent Orchestrator - Run this through main.py")