import os
from pathlib import Path

def gather_code_context(project_root: str, char_limit: int = 40_000) -> str:
    """
    Walk the project and concatenate small snippets from key files to provide context to the model.
    We'll prioritize .py files and limit to approx char_limit characters.
    """
    root = Path(project_root)
    snippets = []
    total = 0
    # prefer source files near top-level
    for p in sorted(root.glob("**/*.py")):
        if ".venv" in str(p.parts) or "site-packages" in str(p.parts):
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        snippet = f"### FILE: {p.relative_to(root)}\n" + text[:2000] + "\n\n"
        if total + len(snippet) > char_limit:
            break
        snippets.append(snippet)
        total += len(snippet)
    return "\n".join(snippets)
