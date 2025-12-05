import json
from pathlib import Path
from typing import Any, Dict

MEMORY_FILE = ".agent_memory.json"

class Memory:
    def __init__(self, path: str, data: Dict[str, Any]):
        self.path = path
        self.data = data

    @classmethod
    def load(cls, project_root: str):
        p = Path(project_root) / MEMORY_FILE
        if p.exists():
            try:
                return cls(str(p), json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                return cls(str(p), {})
        else:
            return cls(str(p), {})

    def save(self):
        Path(self.path).write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def record_iteration(self, task: str, proposals: Dict[str, Dict]):
        self.data.setdefault("tasks", {}).setdefault(task, []).append({
            "proposals": {k: {"summary": v.get("explanation","")} for k,v in proposals.items()}
        })
