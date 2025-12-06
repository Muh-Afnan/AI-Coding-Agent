# import argparse
# import os
# import sys
# from agent_core.llm_client import LLMClient
# from agent_core.planner import Planner
# from agent_core.coder import Coder
# from agent_core.executor import Executor
# from agent_core.file_manager import FileManager
# from agent_core.memory import Memory

# def parse_args():
#     p = argparse.ArgumentParser(description="Semi-autonomous coding agent (cwd mode)")
#     p.add_argument("task", help="Natural language task for the agent (wrap in quotes)")
#     p.add_argument("--auto-approve", action="store_true", help="Apply proposed changes without asking")
#     p.add_argument("--run-cmd", default=None, help="Command to run after applying changes (e.g. pytest)")
#     p.add_argument("--max-iterations", type=int, default=5, help="Max refine iterations for this task")
#     p.add_argument("--verbose", action="store_true")
#     return p.parse_args()

# def main():
#     args = parse_args()
#     cwd = os.getcwd()
#     api_key = os.environ.get("GEMINI_API_KEY")
#     if not api_key:
#         print("GEMINI_API_KEY missing in environment")
#         sys.exit(1)

#     llm = LLMClient(api_key=api_key, verbose=args.verbose)
#     planner = Planner(llm)
#     coder = Coder(llm)
#     executor = Executor(cwd=cwd, verbose=args.verbose)
#     fm = FileManager(project_root=cwd)
#     memory = Memory.load(cwd)

#     # Plan
#     plan = planner.create_plan(args.task)
#     if args.verbose:
#         print("Plan:")
#         for i, step in enumerate(plan, 1):
#             print(f"  {i}. {step}")

#     iteration = 0
#     while iteration < args.max_iterations:
#         iteration += 1
#         if args.verbose:
#             print(f"\n--- Iteration {iteration} ---")

#         # propose changes
#         proposals = coder.propose_changes(task=args.task, plan=plan, project_root=cwd)
#         if not proposals:
#             print("No proposed changes returned by the model.")
#             break

#         # show diffs
#         applied_any = False
#         for fname, proposal in proposals.items():
#             old = fm.read_file(fname)
#             new = proposal["content"]
#             diff = fm.unified_diff(fname, old, new)
#             print(diff or f"[No diff] {fname} (no change)")

#         # approval
#         if not args.auto_approve:
#             user_input = input("\nApply all changes? (y/n/view <filename>/skip) ").strip().lower()
#             if user_input == "y":
#                 approve = True
#             elif user_input.startswith("view "):
#                 _, view_fname = user_input.split(maxsplit=1)
#                 print("\n" + proposals[view_fname]["content"])
#                 user_input = input("Apply all changes? (y/n) ").strip().lower()
#                 approve = (user_input == "y")
#             else:
#                 approve = False
#         else:
#             approve = True

#         if approve:
#             for fname, proposal in proposals.items():
#                 fm.write_file(fname, proposal["content"])
#                 applied_any = True
#                 print(f"Applied: {fname}")
#             memory.record_iteration(args.task, proposals)
#             memory.save()
#         else:
#             print("User declined changes. Exiting.")
#             break

#         # execute tests / run command if provided
#         if args.run_cmd:
#             print(f"\nRunning: {args.run_cmd}")
#             res = executor.run_command(args.run_cmd)
#             print("=== Command stdout/stderr ===")
#             print(res.stdout)
#             print(res.stderr)
#             if res.timed_out:
#                 print("Command timed out.")
#             # Feed results back to model for potential fix
#             if res.returncode != 0:
#                 # create a quick refinement from coder
#                 print("Execution returned non-zero. Asking model to propose fixes...")
#                 plan = planner.refine_plan_from_failure(plan, res)
#                 continue  # next iteration
#             else:
#                 print("Command succeeded. Task complete.")
#                 break
#         else:
#             # if no run_cmd, finish after apply
#             break

#     print("Done.")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
Semi-autonomous coding agent with enhanced error handling and tracking.
"""
import argparse
import os
import sys
import time
from pathlib import Path

# from agent_core.llm_client import LLMClient
from agent_core.llm_client import APIError

from agent_core.llm_providers import UnifiedLLMClient as LLMClient

from agent_core.planner import Planner
from agent_core.coder import Coder
from agent_core.executor import Executor
from agent_core.file_manager import FileManager
from agent_core.memory import Memory
from dotenv import load_dotenv



def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description="Semi-autonomous coding agent (cwd mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Add user authentication to the API"
  %(prog)s "Fix the bug in parser.py" --auto-approve
  %(prog)s "Implement tests for auth module" --run-cmd "pytest tests/"
  %(prog)s "Refactor database queries" --max-iterations 3 --verbose
        """
    )
    
    p.add_argument(
        "task",
        help="Natural language task for the agent (wrap in quotes)"
    )
    
    p.add_argument(
        "--auto-approve",
        action="store_true",
        help="Apply proposed changes without asking for confirmation"
    )
    
    p.add_argument(
        "--run-cmd",
        default=None,
        help="Command to run after applying changes (e.g., 'pytest', 'python main.py')"
    )
    
    p.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum refine iterations for this task (default: 5)"
    )
    
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output with detailed logging"
    )
    
    p.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM temperature for generation (0.0-1.0, default: 0.7)"
    )
    
    p.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum API retry attempts (default: 3)"
    )
    
    return p.parse_args()


def print_header(text: str, char: str = "="):
    """Print a formatted header."""
    print(f"\n{char * 60}")
    print(f"{text}")
    print(f"{char * 60}\n")


def show_diff(fm: FileManager, proposals: dict, verbose: bool = False):
    """Display diffs for all proposed changes."""
    print_header("PROPOSED CHANGES", "─")
    
    has_changes = False
    for fname, proposal in proposals.items():
        old_content = fm.read_file(fname)
        new_content = proposal["content"]
        
        # Check if file exists
        file_exists = bool(old_content)
        
        if not file_exists:
            print(f"📄 NEW FILE: {fname}")
            if verbose:
                print(f"   {proposal.get('explanation', 'No explanation provided')}")
        else:
            diff = fm.unified_diff(fname, old_content, new_content)
            if diff:
                print(f"📝 MODIFIED: {fname}")
                if verbose:
                    print(f"   {proposal.get('explanation', 'No explanation provided')}")
                print(diff)
                has_changes = True
            else:
                print(f"⚪ NO CHANGE: {fname}")
        
        print()
    
    return has_changes


def get_user_approval(proposals: dict, fm: FileManager) -> bool:
    """
    Get user approval for proposed changes.
    
    Returns:
        True if user approves, False otherwise
    """
    while True:
        user_input = input(
            "\n🤔 Apply these changes? (y/n/view <filename>/diff <filename>): "
        ).strip().lower()
        
        if user_input == "y" or user_input == "yes":
            return True
        
        elif user_input == "n" or user_input == "no":
            return False
        
        elif user_input.startswith("view "):
            # Show full file content
            _, fname = user_input.split(maxsplit=1)
            if fname in proposals:
                print(f"\n{'='*60}")
                print(f"FILE: {fname}")
                print(f"{'='*60}")
                print(proposals[fname]["content"])
                print(f"{'='*60}\n")
            else:
                print(f"❌ File '{fname}' not in proposals")
        
        elif user_input.startswith("diff "):
            # Show diff for specific file
            _, fname = user_input.split(maxsplit=1)
            if fname in proposals:
                old = fm.read_file(fname)
                new = proposals[fname]["content"]
                diff = fm.unified_diff(fname, old, new)
                print(f"\n{diff if diff else 'No changes'}\n")
            else:
                print(f"❌ File '{fname}' not in proposals")
        
        else:
            print("❌ Invalid input. Use: y, n, view <filename>, or diff <filename>")


def apply_changes(fm: FileManager, proposals: dict, verbose: bool = False) -> int:
    """
    Apply proposed changes to files.
    
    Returns:
        Number of files successfully written
    """
    applied_count = 0
    
    for fname, proposal in proposals.items():
        try:
            fm.write_file(fname, proposal["content"])
            applied_count += 1
            print(f"✅ Applied: {fname}")
        except Exception as e:
            print(f"❌ Failed to apply {fname}: {e}")
    
    return applied_count


def main():
    """Main execution function."""
    args = parse_args()
    
    print_header("🤖 SEMI-AUTONOMOUS CODING AGENT")
    
    # Validate environment
    cwd = os.getcwd()
    load_dotenv()

    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        print("❌ Error: GEMINI_API_KEY environment variable not set")
        print("\nTo fix this, run:")
        print("  export GEMINI_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    if args.verbose:
        print(f"📁 Working directory: {cwd}")
        print(f"🎯 Task: {args.task}")
        print(f"🔄 Max iterations: {args.max_iterations}")
        print(f"🌡️  Temperature: {args.temperature}")
    
    # Initialize components
    try:
        llm = LLMClient(
            api_key=api_key,
            verbose=args.verbose,
            max_retries=args.max_retries
        )
        planner = Planner(llm)
        coder = Coder(llm)
        executor = Executor(cwd=cwd, verbose=args.verbose)
        fm = FileManager(project_root=cwd)
        memory = Memory.load(cwd)
        
        if args.verbose:
            print("✅ All components initialized successfully\n")
    
    except Exception as e:
        print(f"❌ Failed to initialize components: {e}")
        sys.exit(1)
    
    # Start task tracking
    task_id = memory.start_task(args.task)
    start_time = time.time()
    
    # Create execution plan
    print_header("📋 CREATING PLAN")
    
    try:
        plan = planner.create_plan(args.task)
        
        if not plan:
            print("❌ Failed to create plan")
            memory.complete_task(task_id, success=False)
            sys.exit(1)
        
        print("Generated plan:")
        for i, step in enumerate(plan, 1):
            print(f"  {i}. {step}")
    
    except APIError as e:
        print(f"❌ API error while creating plan: {e}")
        memory.complete_task(task_id, success=False)
        sys.exit(1)
    
    # Main iteration loop
    iteration = 0
    task_completed = False
    
    while iteration < args.max_iterations:
        iteration += 1
        iteration_start = time.time()
        
        print_header(f"ITERATION {iteration}/{args.max_iterations}", "=")
        
        # Propose changes
        try:
            print("🤔 Proposing changes...")
            proposals = coder.propose_changes(
                task=args.task,
                plan=plan,
                project_root=cwd
            )
            
            if not proposals:
                print("⚠️  No changes proposed by the model")
                memory.record_iteration(
                    task_id=task_id,
                    proposals={},
                    applied=False,
                    error="No proposals returned",
                    duration_seconds=time.time() - iteration_start
                )
                break
        
        except APIError as e:
            print(f"❌ API error during proposal: {e}")
            memory.record_iteration(
                task_id=task_id,
                proposals={},
                applied=False,
                error=str(e),
                duration_seconds=time.time() - iteration_start
            )
            break
        
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            memory.record_iteration(
                task_id=task_id,
                proposals={},
                applied=False,
                error=str(e),
                duration_seconds=time.time() - iteration_start
            )
            break
        
        # Show diffs
        has_changes = show_diff(fm, proposals, verbose=args.verbose)
        
        # Get approval
        if not args.auto_approve:
            approve = get_user_approval(proposals, fm)
        else:
            approve = True
            print("✅ Auto-approve enabled, applying changes...")
        
        # Apply changes if approved
        if approve:
            applied_count = apply_changes(fm, proposals, verbose=args.verbose)
            
            # Get token stats for this iteration
            session_stats = llm.get_session_stats()
            
            # Record iteration in memory
            memory.record_iteration(
                task_id=task_id,
                proposals=proposals,
                applied=True,
                tokens_used={
                    "prompt": session_stats["total_prompt_tokens"],
                    "response": session_stats["total_response_tokens"]
                },
                cost_usd=session_stats["total_cost_usd"],
                duration_seconds=time.time() - iteration_start
            )
            
            print(f"\n✅ Applied changes to {applied_count} file(s)")
        else:
            print("❌ Changes rejected by user")
            memory.record_iteration(
                task_id=task_id,
                proposals=proposals,
                applied=False,
                error="User rejected changes",
                duration_seconds=time.time() - iteration_start
            )
            break
        
        # Execute test command if provided
        if args.run_cmd:
            print_header("🧪 RUNNING TESTS")
            print(f"Command: {args.run_cmd}")
            
            try:
                result = executor.run_command(args.run_cmd)
                
                print("\n--- Standard Output ---")
                print(result.stdout if result.stdout else "(empty)")
                
                if result.stderr:
                    print("\n--- Standard Error ---")
                    print(result.stderr)
                
                if result.timed_out:
                    print("\n⏱️  Command timed out")
                
                # Check if tests passed
                if result.returncode == 0:
                    print("\n✅ Tests passed!")
                    task_completed = True
                    
                    # Update memory with test results
                    task = memory.get_task(task_id)
                    if task and task["iterations"]:
                        last_iteration = task["iterations"][-1]
                        last_iteration["test_results"] = {
                            "passed": True,
                            "returncode": result.returncode,
                            "stdout": result.stdout[:500],
                            "stderr": result.stderr[:500]
                        }
                        memory.save()
                    
                    break
                else:
                    print(f"\n❌ Tests failed (exit code: {result.returncode})")
                    
                    # Update memory with failure
                    task = memory.get_task(task_id)
                    if task and task["iterations"]:
                        last_iteration = task["iterations"][-1]
                        last_iteration["test_results"] = {
                            "passed": False,
                            "returncode": result.returncode,
                            "stdout": result.stdout[:500],
                            "stderr": result.stderr[:500]
                        }
                        memory.save()
                    
                    # Refine plan based on failure
                    if iteration < args.max_iterations:
                        print("\n🔄 Refining plan based on failure...")
                        plan = planner.refine_plan_from_failure(plan, result)
                        
                        if args.verbose:
                            print("\nRevised plan:")
                            for i, step in enumerate(plan, 1):
                                print(f"  {i}. {step}")
                        
                        continue  # Next iteration
                    else:
                        print(f"\n⚠️  Max iterations ({args.max_iterations}) reached")
                        break
            
            except Exception as e:
                print(f"\n❌ Error running command: {e}")
                break
        else:
            # No test command, consider task done after applying changes
            task_completed = True
            break
    
    # Finalize task
    total_duration = time.time() - start_time
    
    if task_completed:
        memory.complete_task(task_id, success=True)
        print_header("✅ TASK COMPLETED SUCCESSFULLY", "=")
    else:
        memory.complete_task(task_id, success=False)
        print_header("⚠️  TASK INCOMPLETE", "=")
    
    # Print summaries
    print(f"⏱️  Total duration: {total_duration:.1f}s")
    print(f"🔄 Iterations: {iteration}")
    
    if args.verbose:
        print("\n" + memory.get_task_summary(task_id))
    
    llm.print_session_summary()
    
    print_header("🏁 DONE")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if os.environ.get("DEBUG"):
            import traceback
            traceback.print_exc()
        sys.exit(1)
