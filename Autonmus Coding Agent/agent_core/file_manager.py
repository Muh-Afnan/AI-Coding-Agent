import os
from pathlib import Path
from difflib import unified_diff

class FileManager:
    def __init__(self, project_root: str, backup_dir: str = ".agent_backups"):
        self.project_root = os.path.realpath(project_root)
        self.project_root = os.path.realpath(project_root)
        self.backup_dir = os.path.join(self.project_root, backup_dir)
        Path(self.backup_dir).mkdir(exist_ok=True)

    def _abs_path(self, rel_path: str) -> str:
        abs_p = os.path.realpath(os.path.join(self.project_root, rel_path))
        # safety: ensure within project root
        if not abs_p.startswith(self.project_root):
            raise PermissionError("Attempt to access path outside project root")
        return abs_p

    def read_file(self, rel_path: str) -> str:
        abs_p = self._abs_path(rel_path)
        if not Path(abs_p).exists():
            return ""
        with open(abs_p, "r", encoding="utf-8") as f:
            return f.read()

    def write_file(self, rel_path: str, content: str,create_backup: bool = True):
        abs_p = self._abs_path(rel_path)
        p = Path(abs_p)
        # Backup existing file
        if create_backup and p.exists():
            self._backup_file(rel_path)

        p.parent.mkdir(parents=True, exist_ok=True)
        with open(abs_p, "w", encoding="utf-8") as f:
            f.write(content)

    def _backup_file(self, rel_path: str):
        """Create timestamped backup of file."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{rel_path.replace('/', '_')}_{timestamp}"
        backup_path = os.path.join(self.backup_dir, backup_name)
        
        original = self._abs_path(rel_path)
        if Path(original).exists():
            import shutil
            shutil.copy2(original, backup_path)

    def unified_diff(self, rel_path: str, old: str, new: str) -> str:
        old_lines = old.splitlines(keepends=True)
        new_lines = new.splitlines(keepends=True)
        diff = unified_diff(old_lines, new_lines, fromfile=f"a/{rel_path}", tofile=f"b/{rel_path}")
        return ''.join(diff)
