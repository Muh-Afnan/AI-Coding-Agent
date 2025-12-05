# agent_core/project_index.py
import chromadb
from pathlib import Path
from typing import List, Dict

class ProjectIndex:
    """Vector database for semantic code search."""
    
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.client = chromadb.PersistentClient(path=f"{project_root}/.agent_cache")
        self.collection = self.client.get_or_create_collection("code_files")
    
    def index_project(self):
        """Index all Python files in project."""
        for py_file in Path(self.project_root).rglob("*.py"):
            rel_path = str(py_file.relative_to(self.project_root))
            try:
                content = py_file.read_text(encoding="utf-8")
                # Add to vector DB
                self.collection.upsert(
                    ids=[rel_path],
                    documents=[content],
                    metadatas=[{"path": rel_path, "size": len(content)}]
                )
            except Exception as e:
                print(f"Failed to index {rel_path}: {e}")
    
    def find_relevant_files(self, query: str, n_results: int = 5) -> List[str]:
        """Find files most relevant to query using semantic search."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results['ids'][0] if results['ids'] else []