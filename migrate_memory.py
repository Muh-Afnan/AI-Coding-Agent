#!/usr/bin/env python3
"""
Migration script to convert old memory format to new schema.

Usage:
    python migrate_memory.py [path_to_project]
    python migrate_memory.py --backup  # Create backup before migration
"""
import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime
import hashlib


OLD_MEMORY_FILE = ".agent_memory.json"
BACKUP_SUFFIX = ".backup"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Migrate old memory format to new schema"
    )
    parser.add_argument(
        "project_path",
        nargs="?",
        default=".",
        help="Path to project directory (default: current directory)"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup of original memory file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without making changes"
    )
    return parser.parse_args()


def load_old_memory(path: Path) -> dict:
    """Load old memory file."""
    memory_file = path / OLD_MEMORY_FILE
    
    if not memory_file.exists():
        raise FileNotFoundError(f"No memory file found at {memory_file}")
    
    try:
        with open(memory_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in memory file: {e}")


def detect_schema_version(data: dict) -> str:
    """Detect the schema version of the memory file."""
    if "version" in data:
        return data["version"]
    
    # Heuristic detection for old format
    if "tasks" in data and isinstance(data["tasks"], dict):
        return "1.0"
    
    return "unknown"


def migrate_v1_to_v2(old_data: dict) -> dict:
    """
    Migrate from schema v1.0 to v2.0.
    
    Old format:
    {
        "tasks": {
            "task_description": [
                {
                    "proposals": {...}
                }
            ]
        }
    }
    
    New format:
    {
        "version": "2.0",
        "tasks": [
            {
                "task_id": "...",
                "task_description": "...",
                "status": "...",
                "iterations": [...]
            }
        ],
        "metadata": {...}
    }
    """
    print("🔄 Migrating from v1.0 to v2.0...")
    
    new_data = {
        "version": "2.0",
        "tasks": [],
        "metadata": {
            "total_tasks": 0,
            "total_iterations": 0,
            "total_tokens_used": 0,
            "total_cost_usd": 0.0,
            "last_updated": datetime.now().isoformat()
        }
    }
    
    old_tasks = old_data.get("tasks", {})
    
    for task_description, old_iterations in old_tasks.items():
        # Generate task ID
        task_id = hashlib.md5(
            f"{task_description}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]
        
        # Convert iterations
        new_iterations = []
        for idx, old_iter in enumerate(old_iterations, 1):
            proposals = old_iter.get("proposals", {})
            
            # Convert old proposal format
            new_proposals = {}
            for fname, details in proposals.items():
                if isinstance(details, dict):
                    new_proposals[fname] = {
                        "content": details.get("content", ""),
                        "explanation": details.get("summary", details.get("explanation", ""))
                    }
                else:
                    # Very old format where details might be a string
                    new_proposals[fname] = {
                        "content": str(details),
                        "explanation": "Migrated from old format"
                    }
            
            new_iteration = {
                "timestamp": datetime.now().isoformat(),
                "iteration_num": idx,
                "proposals": new_proposals,
                "applied": True,  # Assume applied since it's in history
                "test_results": None,
                "tokens_used": None,
                "cost_usd": None,
                "error": None,
                "duration_seconds": None
            }
            
            new_iterations.append(new_iteration)
        
        # Create new task
        new_task = {
            "task_id": task_id,
            "task_description": task_description,
            "status": "completed",  # Old tasks are considered completed
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "iterations": new_iterations,
            "total_files_changed": sum(
                len(it["proposals"]) for it in new_iterations
            ),
            "total_tokens_used": 0,
            "total_cost_usd": 0.0,
            "metadata": {}
        }
        
        new_data["tasks"].append(new_task)
        new_data["metadata"]["total_tasks"] += 1
        new_data["metadata"]["total_iterations"] += len(new_iterations)
        
        print(f"  ✅ Migrated task: {task_description[:50]}...")
    
    return new_data


def create_backup(path: Path):
    """Create backup of memory file."""
    memory_file = path / OLD_MEMORY_FILE
    backup_file = path / f"{OLD_MEMORY_FILE}{BACKUP_SUFFIX}"
    
    shutil.copy2(memory_file, backup_file)
    print(f"📦 Created backup: {backup_file}")


def save_migrated_memory(path: Path, data: dict):
    """Save migrated memory to file."""
    memory_file = path / OLD_MEMORY_FILE
    
    with open(memory_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    
    print(f"💾 Saved migrated memory to: {memory_file}")


def print_migration_summary(old_data: dict, new_data: dict):
    """Print summary of migration."""
    print("\n" + "="*60)
    print("📊 MIGRATION SUMMARY")
    print("="*60)
    
    # Old stats
    old_tasks = old_data.get("tasks", {})
    old_task_count = len(old_tasks)
    old_iteration_count = sum(len(v) for v in old_tasks.values())
    
    print(f"\nOld Format (v{detect_schema_version(old_data)}):")
    print(f"  Tasks:      {old_task_count}")
    print(f"  Iterations: {old_iteration_count}")
    
    # New stats
    new_task_count = len(new_data["tasks"])
    new_iteration_count = new_data["metadata"]["total_iterations"]
    
    print(f"\nNew Format (v{new_data['version']}):")
    print(f"  Tasks:      {new_task_count}")
    print(f"  Iterations: {new_iteration_count}")
    
    print("\n" + "="*60)


def main():
    """Main migration function."""
    args = parse_args()
    project_path = Path(args.project_path).resolve()
    
    print("="*60)
    print("🔄 MEMORY MIGRATION TOOL")
    print("="*60)
    print(f"\nProject: {project_path}")
    
    # Load old memory
    try:
        old_data = load_old_memory(project_path)
        print(f"✅ Loaded memory file")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return 1
    except ValueError as e:
        print(f"❌ {e}")
        return 1
    
    # Detect version
    old_version = detect_schema_version(old_data)
    print(f"📋 Detected schema version: {old_version}")
    
    # Check if migration needed
    if old_version == "2.0":
        print("✅ Memory is already in latest format (v2.0)")
        return 0
    
    if old_version == "unknown":
        print("⚠️  Unknown schema version. Manual migration may be required.")
        return 1
    
    # Perform migration
    try:
        if old_version == "1.0":
            new_data = migrate_v1_to_v2(old_data)
        else:
            print(f"❌ No migration path from version {old_version}")
            return 1
        
        print("✅ Migration completed successfully")
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return 1
    
    # Print summary
    print_migration_summary(old_data, new_data)
    
    # Dry run check
    if args.dry_run:
        print("\n🔍 DRY RUN: No changes made")
        print("\nMigrated data preview:")
        print(json.dumps(new_data, indent=2)[:500] + "...")
        return 0
    
    # Create backup if requested
    if args.backup:
        try:
            create_backup(project_path)
        except Exception as e:
            print(f"⚠️  Failed to create backup: {e}")
            response = input("Continue without backup? (y/n): ").strip().lower()
            if response != 'y':
                print("❌ Migration cancelled")
                return 1
    
    # Save migrated data
    try:
        save_migrated_memory(project_path, new_data)
        print("\n✅ Migration completed successfully!")
        
        if args.backup:
            print(f"\n💡 Original memory backed up with {BACKUP_SUFFIX} extension")
        
        return 0
    
    except Exception as e:
        print(f"\n❌ Failed to save migrated data: {e}")
        return 1


if __name__ == "__main__":
    import sys
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Migration cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)