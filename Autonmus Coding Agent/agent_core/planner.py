from typing import List
from .llm_client import LLMClient

class Planner:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def create_plan(self, task: str) -> List[str]:
        system = "You are a software engineering assistant that creates short, actionable plans (3-8 steps)."
        user = f"Create a concise plan of steps to accomplish this task:\n\n{task}\n\nReturn the plan as a numbered list."
        resp = self.llm.generate(system, user)
        # naive parse: split on newlines and strip numbers
        lines = [l.strip() for l in resp.splitlines() if l.strip()]
        plan = []
        for l in lines:
            # remove '1.' prefixes
            if l and (l[0].isdigit() and (l[1] in ['.',')'])):
                plan.append(l.split('.',1)[1].strip())
            else:
                plan.append(l)
        # limit to reasonable count
        return plan[:8]

    def refine_plan_from_failure(self, plan, execution_result):
        # Ask LLM to tweak plan based on failure (keeps simple)
        system = "You are a software engineering assistant that updates a plan when a step failed."
        user = f"Existing plan:\n{chr(10).join(['- '+p for p in plan])}\n\nExecution failure:\nReturn code: {execution_result.returncode}\nStdout:\n{execution_result.stdout}\nStderr:\n{execution_result.stderr}\n\nSuggest a revised plan (3-6 steps)."
        resp = self.llm.generate(system, user)
        lines = [l.strip() for l in resp.splitlines() if l.strip()]
        new_plan = []
        for l in lines:
            if l and (l[0].isdigit() and (l[1] in ['.',')'])):
                new_plan.append(l.split('.',1)[1].strip())
            else:
                new_plan.append(l)
        return new_plan[:6]
