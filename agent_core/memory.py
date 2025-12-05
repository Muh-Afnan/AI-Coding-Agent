"""
Enhanced persistent memory system for tracking agent tasks and iterations.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime
import hashlib


MEMORY_FILE = ".agent_memory.json"
SCHEMA_VERSION = "2.0"


@dataclass
class Iteration:
    """Single iteration of agent work."""
    timestamp: str
    iteration_num: int
    proposals: Dict[str, Dict[str, str]]  # filename -> {content, explanation}
    applied: bool
    test_results: Optional[Dict[str, Any]] = None
    tokens_used: Optional[Dict[str, int]] = None
    cost_usd: Optional[float] = None
    error: Optional[str] = None
    duration_seconds: Optional[float] = None
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return asdict(self)


@dataclass
class Task:
    """Complete task with all iterations."""
    task_id: str
    task_description: str
    status: str  # 'in_progress', 'completed', 'failed', 'cancelled'
    created_at: str
    updated_at: str
    iterations: List[Iteration] = field(default_factory=list)
    total_files_changed: int = 0
    total_tokens_used: int = 0
    total_cost_usd: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        data = asdict(self)
        # Convert Iteration dataclasses to dicts
        data['iterations'] = [it if isinstance(it, dict) else it.to_dict() for it in self.iterations]
        return data


class Memory:
    """
    Persistent memory for agent tasks and iterations.
    
    Features:
    - Task tracking with unique IDs
    - Iteration history with full context
    - Token usage and cost tracking
    - Export for LLM context
    - Query and analysis capabilities
    """
    
    def __init__(self, path: str, data: Dict[str, Any]):
        self.path = path
        self.data = data
        self._ensure_schema()
    
    def _ensure_schema(self):
        """Ensure data has correct structure."""
        if "version" not in self.data:
            self.data["version"] = SCHEMA_VERSION
        
        if "tasks" not in self.data:
            self.data["tasks"] = []
        
        if "metadata" not in self.data:
            self.data["metadata"] = {
                "total_tasks": 0,
                "total_iterations": 0,
                "total_tokens_used": 0,
                "total_cost_usd": 0.0,
                "last_updated": None
            }
    
    @classmethod
    def load(cls, project_root: str) -> 'Memory':
        """
        Load memory from disk or create new.
        
        Args:
            project_root: Path to project directory
        
        Returns:
            Memory instance
        """
        memory_path = Path(project_root) / MEMORY_FILE
        
        if memory_path.exists():
            try:
                data = json.loads(memory_path.read_text(encoding="utf-8"))
                
                # Migrate old schema if needed
                if data.get("version") != SCHEMA_VERSION:
                    data = cls._migrate_schema(data)
                
                return cls(str(memory_path), data)
            
            except Exception as e:
                print(f"⚠️  Failed to load memory, creating new: {e}")
                return cls(str(memory_path), {})
        else:
            return cls(str(memory_path), {})
    
    @classmethod
    def _migrate_schema(cls, old_data: dict) -> dict:
        """Migrate old schema to current version."""
        print(f"🔄 Migrating memory from version {old_data.get('version', '1.0')} to {SCHEMA_VERSION}")
        
        # Create new structure
        new_data = {
            "version": SCHEMA_VERSION,
            "tasks": [],
            "metadata": {
                "total_tasks": 0,
                "total_iterations": 0,
                "total_tokens_used": 0,
                "total_cost_usd": 0.0,
                "last_updated": datetime.now().isoformat()
            }
        }
        
        # Migrate old tasks if they exist
        if "tasks" in old_data and isinstance(old_data["tasks"], dict):
            for task_desc, iterations in old_data["tasks"].items():
                task_id = hashlib.md5(f"{task_desc}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]
                
                task = Task(
                    task_id=task_id,
                    task_description=task_desc,
                    status="completed",
                    created_at=datetime.now().isoformat(),
                    updated_at=datetime.now().isoformat(),
                    iterations=[]
                )
                
                new_data["tasks"].append(task.to_dict())
        
        return new_data
    
    def save(self):
        """Save memory to disk."""
        self.data["metadata"]["last_updated"] = datetime.now().isoformat()
        
        try:
            Path(self.path).write_text(
                json.dumps(self.data, indent=2, default=str),
                encoding="utf-8"
            )
        except Exception as e:
            print(f"❌ Failed to save memory: {e}")
    
    def start_task(self, task_description: str) -> str:
        """
        Start tracking a new task.
        
        Args:
            task_description: Human-readable task description
        
        Returns:
            task_id for referencing this task
        """
        task_id = self._generate_task_id(task_description)
        
        # Check if task already exists and is in progress
        existing = self.get_task(task_id)
        if existing and existing["status"] == "in_progress":
            print(f"ℹ️  Continuing existing task: {task_id}")
            return task_id
        
        task = Task(
            task_id=task_id,
            task_description=task_description,
            status="in_progress",
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            iterations=[]
        )
        
        self.data["tasks"].append(task.to_dict())
        self.data["metadata"]["total_tasks"] += 1
        self.save()
        
        print(f"✅ Started new task: {task_id}")
        return task_id
    
    def record_iteration(
        self,
        task_id: str,
        proposals: Dict[str, Dict[str, str]],
        applied: bool,
        test_results: Optional[Dict] = None,
        tokens_used: Optional[Dict[str, int]] = None,
        cost_usd: Optional[float] = None,
        error: Optional[str] = None,
        duration_seconds: Optional[float] = None
    ):
        """
        Record a single iteration of work on a task.
        
        Args:
            task_id: Task identifier
            proposals: Dict of filename -> {content, explanation}
            applied: Whether changes were applied
            test_results: Optional test execution results
            tokens_used: Optional token usage stats
            cost_usd: Optional cost of this iteration
            error: Optional error message if iteration failed
            duration_seconds: Time taken for iteration
        """
        task = self.get_task(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")
        
        iteration_num = len(task["iterations"]) + 1
        
        iteration = Iteration(
            timestamp=datetime.now().isoformat(),
            iteration_num=iteration_num,
            proposals=proposals,
            applied=applied,
            test_results=test_results,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            error=error,
            duration_seconds=duration_seconds
        )
        
        task["iterations"].append(iteration.to_dict())
        task["updated_at"] = datetime.now().isoformat()
        
        # Update aggregated stats
        if applied:
            task["total_files_changed"] += len(proposals)
        
        if tokens_used:
            total_tokens = sum(tokens_used.values())
            task["total_tokens_used"] += total_tokens
            self.data["metadata"]["total_tokens_used"] += total_tokens
        
        if cost_usd:
            task["total_cost_usd"] += cost_usd
            self.data["metadata"]["total_cost_usd"] += cost_usd
        
        self.data["metadata"]["total_iterations"] += 1
        self.save()
        
        print(f"📝 Recorded iteration {iteration_num} for task {task_id}")
    
    def complete_task(self, task_id: str, success: bool = True):
        """
        Mark a task as completed or failed.
        
        Args:
            task_id: Task identifier
            success: True if completed successfully, False if failed
        """
        task = self.get_task(task_id)
        if task:
            task["status"] = "completed" if success else "failed"
            task["updated_at"] = datetime.now().isoformat()
            self.save()
            
            status_emoji = "✅" if success else "❌"
            print(f"{status_emoji} Task {task_id} marked as {task['status']}")
    
    def cancel_task(self, task_id: str):
        """Mark task as cancelled."""
        task = self.get_task(task_id)
        if task:
            task["status"] = "cancelled"
            task["updated_at"] = datetime.now().isoformat()
            self.save()
            print(f"🚫 Task {task_id} cancelled")
    
    def get_task(self, task_id: str) -> Optional[Dict]:
        """
        Retrieve task by ID.
        
        Args:
            task_id: Task identifier
        
        Returns:
            Task dict or None if not found
        """
        for task in self.data["tasks"]:
            if task["task_id"] == task_id:
                return task
        return None
    
    def get_recent_tasks(self, n: int = 5, status: Optional[str] = None) -> List[Dict]:
        """
        Get n most recent tasks.
        
        Args:
            n: Number of tasks to return
            status: Optional filter by status
        
        Returns:
            List of task dicts
        """
        tasks = self.data["tasks"]
        
        if status:
            tasks = [t for t in tasks if t["status"] == status]
        
        sorted_tasks = sorted(tasks, key=lambda t: t["updated_at"], reverse=True)
        return sorted_tasks[:n]
    
    def get_all_tasks(self) -> List[Dict]:
        """Get all tasks."""
        return self.data["tasks"]
    
    def get_task_summary(self, task_id: str) -> str:
        """
        Generate human-readable summary of a task.
        
        Args:
            task_id: Task identifier
        
        Returns:
            Formatted summary string
        """
        task = self.get_task(task_id)
        if not task:
            return f"❌ Task {task_id} not found"
        
        summary = f"""
{'='*60}
TASK SUMMARY
{'='*60}
ID:              {task['task_id']}
Description:     {task['task_description']}
Status:          {task['status'].upper()}
Created:         {task['created_at']}
Last Updated:    {task['updated_at']}
Iterations:      {len(task['iterations'])}
Files Changed:   {task['total_files_changed']}
Tokens Used:     {task['total_tokens_used']:,}
Estimated Cost:  ${task['total_cost_usd']:.4f}
"""
        
        if task['iterations']:
            summary += f"\n{'─'*60}\nRECENT ITERATIONS\n{'─'*60}\n"
            
            for it in task['iterations'][-5:]:  # Last 5 iterations
                summary += f"\nIteration {it['iteration_num']} ({it['timestamp']})\n"
                summary += f"  Status:   {'✅ Applied' if it['applied'] else '❌ Not applied'}\n"
                summary += f"  Files:    {len(it['proposals'])} file(s)\n"
                
                for fname in it['proposals'].keys():
                    summary += f"    - {fname}\n"
                
                if it.get('tokens_used'):
                    total = sum(it['tokens_used'].values())
                    summary += f"  Tokens:   {total:,}\n"
                
                if it.get('error'):
                    summary += f"  Error:    {it['error'][:100]}\n"
        
        return summary
    
    def export_for_llm_context(self, task_id: str, max_iterations: int = 3) -> str:
        """
        Export task history in format suitable for LLM context.
        Useful for telling the LLM what was tried before.
        
        Args:
            task_id: Task identifier
            max_iterations: Maximum iterations to include
        
        Returns:
            Formatted context string
        """
        task = self.get_task(task_id)
        if not task:
            return ""
        
        context = f"# Previous Work on Task\n\n"
        context += f"**Task**: {task['task_description']}\n"
        context += f"**Status**: {task['status']}\n"
        context += f"**Iterations so far**: {len(task['iterations'])}\n\n"
        
        if not task['iterations']:
            return context + "No previous iterations.\n"
        
        context += "## Recent Attempts\n\n"
        recent_iterations = task['iterations'][-max_iterations:]
        
        for it in recent_iterations:
            context += f"### Iteration {it['iteration_num']}\n"
            context += f"- Timestamp: {it['timestamp']}\n"
            context += f"- Applied: {it['applied']}\n"
            context += f"- Files modified:\n"
            
            for fname, info in it['proposals'].items():
                explanation = info.get('explanation', 'No explanation')
                context += f"  - `{fname}`: {explanation}\n"
            
            if it.get('test_results'):
                passed = it['test_results'].get('passed', 'Unknown')
                context += f"- Test result: {passed}\n"
            
            if it.get('error'):
                context += f"- **Error**: {it['error']}\n"
            
            context += "\n"
        
        return context
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics for all tasks."""
        stats = {
            **self.data["metadata"],
            "tasks_by_status": {},
            "average_iterations_per_task": 0,
            "average_cost_per_task": 0
        }
        
        # Count by status
        for task in self.data["tasks"]:
            status = task["status"]
            stats["tasks_by_status"][status] = stats["tasks_by_status"].get(status, 0) + 1
        
        # Calculate averages
        total_tasks = len(self.data["tasks"])
        if total_tasks > 0:
            stats["average_iterations_per_task"] = stats["total_iterations"] / total_tasks
            stats["average_cost_per_task"] = stats["total_cost_usd"] / total_tasks
        
        return stats
    
    def print_statistics(self):
        """Print formatted statistics."""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("📊 MEMORY STATISTICS")
        print("="*60)
        print(f"Total Tasks:          {stats['total_tasks']}")
        print(f"Total Iterations:     {stats['total_iterations']}")
        print(f"Total Tokens Used:    {stats['total_tokens_used']:,}")
        print(f"Total Cost:           ${stats['total_cost_usd']:.4f}")
        print(f"Avg Iterations/Task:  {stats['average_iterations_per_task']:.1f}")
        print(f"Avg Cost/Task:        ${stats['average_cost_per_task']:.4f}")
        print(f"\nTasks by Status:")
        for status, count in stats['tasks_by_status'].items():
            print(f"  {status:12} {count}")
        print("="*60 + "\n")
    
    def _generate_task_id(self, task_description: str) -> str:
        """Generate unique task ID from description."""
        hash_input = f"{task_description}_{datetime.now().isoformat()}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def cleanup_old_tasks(self, days: int = 30, status: str = "completed"):
        """
        Remove old tasks from memory.
        
        Args:
            days: Remove tasks older than this many days
            status: Only remove tasks with this status
        """
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days)
        original_count = len(self.data["tasks"])
        
        self.data["tasks"] = [
            task for task in self.data["tasks"]
            if not (
                task["status"] == status and
                datetime.fromisoformat(task["updated_at"]) < cutoff_date
            )
        ]
        
        removed = original_count - len(self.data["tasks"])
        if removed > 0:
            self.save()
            print(f"🗑️  Removed {removed} old {status} task(s)")


# Example usage
if __name__ == "__main__":
    # Test memory system
    memory = Memory.load(".")
    
    # Start a new task
    task_id = memory.start_task("Implement user authentication system")
    
    # Record some iterations
    memory.record_iteration(
        task_id=task_id,
        proposals={
            "auth.py": {"content": "# auth code", "explanation": "Created auth module"},
            "models.py": {"content": "# user model", "explanation": "Added User model"}
        },
        applied=True,
        tokens_used={"prompt": 1000, "response": 500},
        cost_usd=0.001,
        duration_seconds=5.2
    )
    
    memory.record_iteration(
        task_id=task_id,
        proposals={
            "auth.py": {"content": "# fixed auth", "explanation": "Fixed validation bug"}
        },
        applied=True,
        tokens_used={"prompt": 800, "response": 400},
        cost_usd=0.0008
    )
    
    # Complete task
    memory.complete_task(task_id, success=True)
    
    # Print summary
    print(memory.get_task_summary(task_id))
    
    # Print statistics
    memory.print_statistics()