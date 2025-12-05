#!/usr/bin/env python3
import argparse
import os
import sys
from agent_core.llm_client import LLMClient
from agent_core.planner import Planner
from agent_core.coder import Coder
from agent_core.executor import Executor
from agent_core.file_manager import FileManager
from agent_core.memory import Memory

def parse_args():
    p = argparse.ArgumentParser(description="Semi-autonomous coding agent (cwd mode)")
    p.add_argument("task", help="Natural language task for the agent (wrap in quotes)")
    p.add_argument("--auto-approve", action="store_true", help="Apply proposed changes without asking")
    p.add_argument("--run-cmd", default=None, help="Command to run after applying changes (e.g. pytest)")
    p.add_argument("--max-iterations", type=int, default=5, help="Max refine iterations for this task")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    cwd = os.getcwd()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY missing in environment")
        sys.exit(1)

    llm = LLMClient(api_key=api_key, verbose=args.verbose)
    planner = Planner(llm)
    coder = Coder(llm)
    executor = Executor(cwd=cwd, verbose=args.verbose)
    fm = FileManager(project_root=cwd)
    memory = Memory.load(cwd)

    # Plan
    plan = planner.create_plan(args.task)
    if args.verbose:
        print("Plan:")
        for i, step in enumerate(plan, 1):
            print(f"  {i}. {step}")

    iteration = 0
    while iteration < args.max_iterations:
        iteration += 1
        if args.verbose:
            print(f"\n--- Iteration {iteration} ---")

        # propose changes
        proposals = coder.propose_changes(task=args.task, plan=plan, project_root=cwd)
        if not proposals:
            print("No proposed changes returned by the model.")
            break

        # show diffs
        applied_any = False
        for fname, proposal in proposals.items():
            old = fm.read_file(fname)
            new = proposal["content"]
            diff = fm.unified_diff(fname, old, new)
            print(diff or f"[No diff] {fname} (no change)")

        # approval
        if not args.auto_approve:
            user_input = input("\nApply all changes? (y/n/view <filename>/skip) ").strip().lower()
            if user_input == "y":
                approve = True
            elif user_input.startswith("view "):
                _, view_fname = user_input.split(maxsplit=1)
                print("\n" + proposals[view_fname]["content"])
                user_input = input("Apply all changes? (y/n) ").strip().lower()
                approve = (user_input == "y")
            else:
                approve = False
        else:
            approve = True

        if approve:
            for fname, proposal in proposals.items():
                fm.write_file(fname, proposal["content"])
                applied_any = True
                print(f"Applied: {fname}")
            memory.record_iteration(args.task, proposals)
            memory.save()
        else:
            print("User declined changes. Exiting.")
            break

        # execute tests / run command if provided
        if args.run_cmd:
            print(f"\nRunning: {args.run_cmd}")
            res = executor.run_command(args.run_cmd)
            print("=== Command stdout/stderr ===")
            print(res.stdout)
            print(res.stderr)
            if res.timed_out:
                print("Command timed out.")
            # Feed results back to model for potential fix
            if res.returncode != 0:
                # create a quick refinement from coder
                print("Execution returned non-zero. Asking model to propose fixes...")
                plan = planner.refine_plan_from_failure(plan, res)
                continue  # next iteration
            else:
                print("Command succeeded. Task complete.")
                break
        else:
            # if no run_cmd, finish after apply
            break

    print("Done.")

if __name__ == "__main__":
    main()
