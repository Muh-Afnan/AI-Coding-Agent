#!/usr/bin/env python3
"""
Comprehensive integration test suite for refactored components.
"""
import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# Add agent_core to path
sys.path.insert(0, str(Path(__file__).parent))

from agent_core.llm_providers import UnifiedLLMClient as  LLMClient
from agent_core.memory import Memory
from dotenv import load_dotenv



class TestResult:
    """Track test results."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.errors = []
    
    def record_pass(self, test_name: str):
        self.passed += 1
        print(f"  ✅ {test_name}")
    
    def record_fail(self, test_name: str, error: str):
        self.failed += 1
        self.errors.append((test_name, error))
        print(f"  ❌ {test_name}: {error}")
    
    def record_skip(self, test_name: str, reason: str):
        self.skipped += 1
        print(f"  ⊘ {test_name} (skipped: {reason})")
    
    def print_summary(self):
        total = self.passed + self.failed + self.skipped
        print(f"\n{'='*60}")
        print(f"📊 TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total:   {total}")
        print(f"✅ Passed:  {self.passed}")
        print(f"❌ Failed:  {self.failed}")
        print(f"⊘ Skipped: {self.skipped}")
        
        if self.failed > 0:
            print(f"\n{'='*60}")
            print(f"FAILURES:")
            print(f"{'='*60}")
            for name, error in self.errors:
                print(f"\n{name}:")
                print(f"  {error}")
        
        print(f"{'='*60}\n")


def test_llm_basic_generation(client: LLMClient, result: TestResult):
    """Test basic LLM generation."""
    test_name = "Basic Generation"
    
    try:
        response = client.generate(
            system_prompt="You are a helpful assistant.",
            user_prompt="Say exactly: 'Hello, World!'",
            temperature=0.1
        )
        
        if len(response) > 0:
            result.record_pass(test_name)
        else:
            result.record_fail(test_name, "Empty response")
    
    except Exception as e:
        result.record_fail(test_name, str(e))


def test_llm_json_extraction(client: LLMClient, result: TestResult):
    """Test JSON extraction from responses."""
    test_name = "JSON Extraction"
    
    try:
        response = client.generate(
            system_prompt="Return only valid JSON, no other text or formatting.",
            user_prompt='Create JSON: {"name": "test", "value": 42}',
            temperature=0.1
        )
        
        extracted = client.extract_json_from_response(response)
        
        if isinstance(extracted, dict) and len(extracted) > 0:
            result.record_pass(test_name)
        else:
            result.record_fail(test_name, f"Invalid extraction: {type(extracted)}")
    
    except Exception as e:
        result.record_fail(test_name, str(e))


def test_llm_code_extraction(client: LLMClient, result: TestResult):
    """Test code extraction from responses."""
    test_name = "Code Extraction"
    
    try:
        response = client.generate(
            system_prompt="You are a code generator. Always wrap code in ```python blocks.",
            user_prompt="Write a function that returns 'hello'",
            temperature=0.3
        )
        
        code = client.extract_code_from_response(response, language="python")
        
        if len(code) > 0 and ("def " in code or "return" in code):
            result.record_pass(test_name)
        else:
            result.record_fail(test_name, "No valid code found")
    
    except Exception as e:
        result.record_fail(test_name, str(e))


def test_llm_error_handling(client: LLMClient, result: TestResult):
    """Test error handling with invalid inputs."""
    test_name = "Error Handling"
    
    try:
        # This should handle gracefully (empty prompt is unusual but valid)
        response = client.generate(
            system_prompt="You are helpful.",
            user_prompt="",
            temperature=0.5
        )
        
        # Should either get response or proper error
        result.record_pass(test_name)
    
    except APIError:
        # Expected error type
        result.record_pass(test_name)
    
    except Exception as e:
        result.record_fail(test_name, f"Unexpected error type: {type(e).__name__}")


def test_llm_session_stats(client: LLMClient, result: TestResult):
    """Test session statistics tracking."""
    test_name = "Session Statistics"
    
    try:
        stats = client.get_session_stats()
        
        required_keys = [
            "total_requests", "total_prompt_tokens", 
            "total_response_tokens", "total_cost_usd"
        ]
        
        if all(key in stats for key in required_keys):
            result.record_pass(test_name)
        else:
            result.record_fail(test_name, "Missing required keys in stats")
    
    except Exception as e:
        result.record_fail(test_name, str(e))


def test_memory_task_lifecycle(memory: Memory, result: TestResult):
    """Test complete task lifecycle."""
    test_name = "Task Lifecycle"
    
    try:
        # Start task
        task_id = memory.start_task("Test task for lifecycle")
        
        if not task_id:
            result.record_fail(test_name, "Failed to create task")
            return
        
        # Check task exists
        task = memory.get_task(task_id)
        if not task or task["status"] != "in_progress":
            result.record_fail(test_name, "Task not in correct initial state")
            return
        
        # Add iteration
        memory.record_iteration(
            task_id=task_id,
            proposals={"test.py": {"content": "# test", "explanation": "test"}},
            applied=True,
            tokens_used={"prompt": 100, "response": 50},
            cost_usd=0.001
        )
        
        # Complete task
        memory.complete_task(task_id, success=True)
        
        # Verify completion
        task = memory.get_task(task_id)
        if task["status"] == "completed" and len(task["iterations"]) == 1:
            result.record_pass(test_name)
        else:
            result.record_fail(test_name, "Task not properly completed")
    
    except Exception as e:
        result.record_fail(test_name, str(e))


def test_memory_multiple_iterations(memory: Memory, result: TestResult):
    """Test multiple iterations on same task."""
    test_name = "Multiple Iterations"
    
    try:
        task_id = memory.start_task("Multi-iteration test task")
        
        # Add 3 iterations
        for i in range(3):
            memory.record_iteration(
                task_id=task_id,
                proposals={f"file{i}.py": {"content": f"# code {i}", "explanation": f"Change {i}"}},
                applied=True,
                tokens_used={"prompt": 100 + i*10, "response": 50 + i*5}
            )
        
        task = memory.get_task(task_id)
        
        if len(task["iterations"]) == 3:
            result.record_pass(test_name)
        else:
            result.record_fail(test_name, f"Expected 3 iterations, got {len(task['iterations'])}")
    
    except Exception as e:
        result.record_fail(test_name, str(e))


def test_memory_query_functions(memory: Memory, result: TestResult):
    """Test memory query functions."""
    test_name = "Query Functions"
    
    try:
        # Create some tasks
        task_ids = []
        for i in range(3):
            tid = memory.start_task(f"Query test task {i}")
            task_ids.append(tid)
            
            if i < 2:
                memory.complete_task(tid, success=True)
        
        # Test recent tasks
        recent = memory.get_recent_tasks(n=2)
        if len(recent) != 2:
            result.record_fail(test_name, "get_recent_tasks returned wrong count")
            return
        
        # Test status filter
        completed = memory.get_recent_tasks(n=10, status="completed")
        if len(completed) < 2:
            result.record_fail(test_name, "Status filter not working")
            return
        
        # Test get_all_tasks
        all_tasks = memory.get_all_tasks()
        if len(all_tasks) < 3:
            result.record_fail(test_name, "get_all_tasks missing tasks")
            return
        
        result.record_pass(test_name)
    
    except Exception as e:
        result.record_fail(test_name, str(e))


def test_memory_statistics(memory: Memory, result: TestResult):
    """Test statistics generation."""
    test_name = "Statistics"
    
    try:
        stats = memory.get_statistics()
        
        required_keys = [
            "total_tasks", "total_iterations", 
            "total_tokens_used", "tasks_by_status"
        ]
        
        if all(key in stats for key in required_keys):
            result.record_pass(test_name)
        else:
            result.record_fail(test_name, "Missing required statistics")
    
    except Exception as e:
        result.record_fail(test_name, str(e))


def test_memory_llm_context_export(memory: Memory, result: TestResult):
    """Test LLM context export."""
    test_name = "LLM Context Export"
    
    try:
        # Create task with iterations
        task_id = memory.start_task("Context export test")
        
        memory.record_iteration(
            task_id=task_id,
            proposals={"file.py": {"content": "code", "explanation": "Added file"}},
            applied=True
        )
        
        memory.record_iteration(
            task_id=task_id,
            proposals={"file.py": {"content": "new code", "explanation": "Fixed bug"}},
            applied=True,
            error="Some error occurred"
        )
        
        # Export context
        context = memory.export_for_llm_context(task_id, max_iterations=2)
        
        if len(context) > 0 and "Previous Work" in context:
            result.record_pass(test_name)
        else:
            result.record_fail(test_name, "Invalid context export")
    
    except Exception as e:
        result.record_fail(test_name, str(e))


def test_memory_persistence(temp_dir: Path, result: TestResult):
    """Test that memory persists across sessions."""
    test_name = "Persistence"
    
    try:
        # Create memory and add data
        memory1 = Memory.load(str(temp_dir))
        task_id = memory1.start_task("Persistence test")
        memory1.record_iteration(
            task_id=task_id,
            proposals={"test.py": {"content": "test", "explanation": "test"}},
            applied=True
        )
        memory1.save()
        
        # Load in new instance
        memory2 = Memory.load(str(temp_dir))
        
        # Verify data persisted
        task = memory2.get_task(task_id)
        
        if task and len(task["iterations"]) == 1:
            result.record_pass(test_name)
        else:
            result.record_fail(test_name, "Data did not persist")
    
    except Exception as e:
        result.record_fail(test_name, str(e))


def run_llm_tests() -> TestResult:
    """Run all LLM client tests."""
    print("\n" + "="*60)
    print("🧪 LLM CLIENT TESTS")
    print("="*60)
    
    result = TestResult()
    cwd = os.getcwd()
    load_dotenv()
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        result.record_skip("All LLM tests", "GEMINI_API_KEY not set")
        return result
    
    try:
        client = LLMClient(api_key=api_key, verbose=False, max_retries=2)
        
        test_llm_basic_generation(client, result)
        test_llm_json_extraction(client, result)
        test_llm_code_extraction(client, result)
        test_llm_error_handling(client, result)
        test_llm_session_stats(client, result)
    
    except Exception as e:
        result.record_fail("LLM Client Setup", str(e))
    
    return result


def run_memory_tests() -> TestResult:
    """Run all memory system tests."""
    print("\n" + "="*60)
    print("🧪 MEMORY SYSTEM TESTS")
    print("="*60)
    
    result = TestResult()
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        memory = Memory.load(str(temp_dir))
        
        test_memory_task_lifecycle(memory, result)
        test_memory_multiple_iterations(memory, result)
        test_memory_query_functions(memory, result)
        test_memory_statistics(memory, result)
        test_memory_llm_context_export(memory, result)
        test_memory_persistence(temp_dir, result)
    
    except Exception as e:
        result.record_fail("Memory System Setup", str(e))
    
    finally:
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    
    return result


def main():
    """Run all test suites."""
    print("\n" + "="*60)
    print("🚀 COMPREHENSIVE TEST SUITE")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_results = []
    
    # Run LLM tests
    llm_result = run_llm_tests()
    all_results.append(("LLM Client", llm_result))
    
    # Run Memory tests
    memory_result = run_memory_tests()
    all_results.append(("Memory System", memory_result))
    
    # Print overall summary
    print("\n" + "="*60)
    print("📊 OVERALL RESULTS")
    print("="*60)
    
    total_passed = sum(r.passed for _, r in all_results)
    total_failed = sum(r.failed for _, r in all_results)
    total_skipped = sum(r.skipped for _, r in all_results)
    
    for name, result in all_results:
        status = "✅ PASSED" if result.failed == 0 else "❌ FAILED"
        print(f"\n{name}:")
        print(f"  {status}")
        print(f"  Passed:  {result.passed}")
        print(f"  Failed:  {result.failed}")
        print(f"  Skipped: {result.skipped}")
    
    print(f"\n{'='*60}")
    print(f"Total Tests: {total_passed + total_failed + total_skipped}")
    print(f"✅ Passed:  {total_passed}")
    print(f"❌ Failed:  {total_failed}")
    print(f"⊘ Skipped: {total_skipped}")
    print(f"{'='*60}")
    
    # Print detailed failures
    if total_failed > 0:
        print(f"\n{'='*60}")
        print("DETAILED FAILURES:")
        print(f"{'='*60}")
        for name, result in all_results:
            if result.errors:
                print(f"\n{name}:")
                for test_name, error in result.errors:
                    print(f"  • {test_name}")
                    print(f"    {error}")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Return exit code
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Test suite crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)