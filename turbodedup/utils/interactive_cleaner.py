#!/usr/bin/env python3
"""
Interactive Duplicate Cleaner - Human-in-the-loop file deletion
Safe, interactive duplicate file management with preview and confirmation

Features:
- Interactive selection of duplicates to delete
- Preview mode with detailed file information
- Smart recommendations (keep newest, largest, or in specific directories)
- Batch operations with confirmation
- Dry-run mode for safety
- Undo log for recovery
"""

import os
import shutil
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

class DuplicateGroup:
    """Represents a group of duplicate files"""
    
    def __init__(self, size: int, hash_val: str, paths: List[str]):
        self.size = size
        self.hash = hash_val
        self.paths = [Path(p) for p in paths]
        self.selected_for_deletion: Set[int] = set()
        self.protected_indices: Set[int] = set()
    
    @property
    def count(self) -> int:
        return len(self.paths)
    
    @property
    def wasted_space(self) -> int:
        return self.size * (self.count - 1)
    
    @property
    def potential_savings(self) -> int:
        return self.size * len(self.selected_for_deletion)
    
    def get_file_info(self, index: int) -> Dict:
        """Get detailed info about a specific file"""
        path = self.paths[index]
        try:
            stat = path.stat()
            return {
                'path': str(path),
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'created': datetime.fromtimestamp(stat.st_ctime),
                'exists': True,
                'readable': os.access(path, os.R_OK),
                'writable': os.access(path, os.W_OK),
                'is_protected': index in self.protected_indices
            }
        except (OSError, FileNotFoundError):
            return {
                'path': str(path),
                'size': 0,
                'modified': None,
                'created': None,
                'exists': False,
                'readable': False,
                'writable': False,
                'is_protected': False
            }
    
    def auto_select_duplicates(self, strategy: str = "keep_newest") -> None:
        """Auto-select files for deletion based on strategy"""
        if self.count <= 1:
            return
        
        file_infos = [(i, self.get_file_info(i)) for i in range(self.count)]
        # Filter out non-existent files
        valid_files = [(i, info) for i, info in file_infos if info['exists']]
        
        if len(valid_files) <= 1:
            return
        
        if strategy == "keep_newest":
            # Keep the file with the most recent modification time
            newest_idx = max(valid_files, key=lambda x: x[1]['modified'] or datetime.min)[0]
            self.selected_for_deletion = {i for i, _ in valid_files if i != newest_idx}
            
        elif strategy == "keep_oldest":
            # Keep the oldest file
            oldest_idx = min(valid_files, key=lambda x: x[1]['modified'] or datetime.max)[0]
            self.selected_for_deletion = {i for i, _ in valid_files if i != oldest_idx}
            
        elif strategy == "keep_first":
            # Keep the first file (by path sort order)
            sorted_files = sorted(valid_files, key=lambda x: x[1]['path'])
            keep_idx = sorted_files[0][0]
            self.selected_for_deletion = {i for i, _ in valid_files if i != keep_idx}
            
        elif strategy == "keep_in_priority_dirs":
            # Keep files in certain priority directories (like Documents, Desktop)
            priority_patterns = ['/Documents/', '/Desktop/', '/Pictures/', '/Videos/', 
                               'Documents\\', 'Desktop\\', 'Pictures\\', 'Videos\\',
                               'home/', 'Users/']
            
            priority_files = []
            other_files = []
            
            for i, info in valid_files:
                path_str = info['path']
                if any(pattern in path_str for pattern in priority_patterns):
                    priority_files.append((i, info))
                else:
                    other_files.append((i, info))
            
            if priority_files:
                # Keep the newest priority file, delete others
                keep_idx = max(priority_files, key=lambda x: x[1]['modified'] or datetime.min)[0]
                self.selected_for_deletion = {i for i, _ in valid_files if i != keep_idx}
            else:
                # Fallback to keep_newest if no priority files
                self.auto_select_duplicates("keep_newest")

class InteractiveCleaner:
    """Interactive duplicate file cleaner with human oversight"""
    
    def __init__(self, db_path: str, dry_run: bool = True, auto_apply_saved: bool = False):
        self.db_path = db_path
        self.dry_run = dry_run
        self.auto_apply_saved = auto_apply_saved
        self.deleted_files: List[Tuple[str, str]] = []  # (original_path, backup_path)
        self.undo_log_path = f"deletion_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        # Safety settings
        self.max_files_per_operation = 100
        self.protected_extensions = {'.exe', '.dll', '.sys', '.ini', '.cfg', '.conf'}
        self.protected_dirs = {'System32', 'Windows', 'Program Files', 'Program Files (x86)'}
        
        # Initialize decision tracking table
        self.init_decision_tracking()
    
    def init_decision_tracking(self) -> None:
        """Initialize the decision tracking table"""
        if not os.path.exists(self.db_path):
            return
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS duplicate_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    size INTEGER NOT NULL,
                    hash TEXT NOT NULL,
                    decision_type TEXT NOT NULL,  -- 'auto_newest', 'auto_oldest', 'auto_priority', 'manual', 'skip'
                    selected_paths TEXT,          -- JSON array of paths to delete (for manual decisions)
                    decision_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(size, hash)
                )
            """)
            
            # Create index for faster lookups
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_decision_size_hash 
                ON duplicate_decisions(size, hash)
            """)
            
            conn.commit()
    
    def save_decision(self, group: DuplicateGroup, decision_type: str, selected_paths: List[str] = None) -> None:
        """Save user decision for a duplicate group"""
        import json
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            selected_paths_json = json.dumps(selected_paths) if selected_paths else None
            
            cursor.execute("""
                INSERT OR REPLACE INTO duplicate_decisions 
                (size, hash, decision_type, selected_paths, decision_time)
                VALUES (?, ?, ?, ?, ?)
            """, (group.size, group.hash, decision_type, selected_paths_json, datetime.now().isoformat()))
            
            conn.commit()
    
    def get_saved_decision(self, group: DuplicateGroup) -> Optional[Dict]:
        """Get saved decision for a duplicate group"""
        import json
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT decision_type, selected_paths, decision_time
                FROM duplicate_decisions
                WHERE size = ? AND hash = ?
            """, (group.size, group.hash))
            
            result = cursor.fetchone()
            if result:
                decision_type, selected_paths_json, decision_time = result
                selected_paths = json.loads(selected_paths_json) if selected_paths_json else None
                return {
                    'decision_type': decision_type,
                    'selected_paths': selected_paths,
                    'decision_time': decision_time
                }
        
        return None
    
    def apply_saved_decision(self, group: DuplicateGroup, decision: Dict) -> bool:
        """Apply a previously saved decision to a duplicate group"""
        decision_type = decision['decision_type']
        
        if decision_type == 'skip':
            group.selected_for_deletion.clear()
            return True
        elif decision_type == 'auto_newest':
            group.auto_select_duplicates("keep_newest")
            return True
        elif decision_type == 'auto_oldest':
            group.auto_select_duplicates("keep_oldest")
            return True
        elif decision_type == 'auto_priority':
            group.auto_select_duplicates("keep_in_priority_dirs")
            return True
        elif decision_type == 'manual' and decision['selected_paths']:
            # Map saved paths to current indices
            selected_paths = set(decision['selected_paths'])
            group.selected_for_deletion.clear()
            
            for i, path in enumerate(group.paths):
                if str(path) in selected_paths:
                    group.selected_for_deletion.add(i)
            
            return True
        
        return False
    
    def clear_saved_decisions(self) -> int:
        """Clear all saved decisions"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM duplicate_decisions")
            deleted_count = cursor.rowcount
            conn.commit()
            return deleted_count
    
    def list_saved_decisions(self) -> None:
        """List all saved decisions"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT size, hash, decision_type, selected_paths, decision_time
                FROM duplicate_decisions
                ORDER BY decision_time DESC
            """)
            
            results = cursor.fetchall()
            
            if not results:
                print("No saved decisions found.")
                return
            
            print(f"\nSaved Decisions ({len(results)} total):")
            print("=" * 60)
            
            for size, hash_val, decision_type, selected_paths_json, decision_time in results:
                print(f"Size: {self.format_size(size)}")
                print(f"Hash: {hash_val[:16]}...")
                print(f"Decision: {decision_type}")
                print(f"Date: {decision_time[:19]}")
                
                if selected_paths_json:
                    import json
                    selected_paths = json.loads(selected_paths_json)
                    print(f"Selected paths: {len(selected_paths)} files")
                    for path in selected_paths[:3]:  # Show first 3 paths
                        print(f"  - {path}")
                    if len(selected_paths) > 3:
                        print(f"  ... and {len(selected_paths) - 3} more")
                
                print("-" * 40)
    
    def load_duplicates(self) -> List[DuplicateGroup]:
        """Load duplicate groups from database"""
        if not os.path.exists(self.db_path):
            print(f"Error: Database not found at {self.db_path}")
            return []
        
        groups = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT size, hash, COUNT(*) as cnt
                FROM files
                WHERE hash IS NOT NULL AND error IS NULL
                GROUP BY size, hash
                HAVING cnt > 1
                ORDER BY size DESC
            """)
            
            for size, hash_val, count in cursor.fetchall():
                cursor2 = conn.cursor()
                cursor2.execute(
                    "SELECT path FROM files WHERE hash=? AND error IS NULL ORDER BY path",
                    (hash_val,)
                )
                paths = [row[0] for row in cursor2]
                
                if len(paths) > 1:  # Verify we actually have duplicates
                    groups.append(DuplicateGroup(size, hash_val, paths))
        
        return groups
    
    def format_size(self, bytes_val: int) -> str:
        """Format bytes as human readable"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_val < 1024.0:
                return f"{bytes_val:.1f} {unit}"
            bytes_val /= 1024.0
        return f"{bytes_val:.1f} PB"
    
    def display_group_summary(self, group: DuplicateGroup, index: int) -> None:
        """Display summary of a duplicate group"""
        print(f"\n{'='*60}")
        print(f"Duplicate Group #{index + 1}")
        print(f"{'='*60}")
        print(f"Files: {group.count} copies")
        print(f"Size: {self.format_size(group.size)} each")
        print(f"Total wasted: {self.format_size(group.wasted_space)}")
        print(f"Hash: {group.hash[:16]}...")
        print()
        
        for i, path in enumerate(group.paths):
            info = group.get_file_info(i)
            status_indicators = []
            
            if not info['exists']:
                status_indicators.append("MISSING")
            if not info['writable']:
                status_indicators.append("READ-ONLY")
            if info['is_protected']:
                status_indicators.append("PROTECTED")
            if i in group.selected_for_deletion:
                status_indicators.append("SELECTED FOR DELETION")
            
            status = f" [{', '.join(status_indicators)}]" if status_indicators else ""
            
            print(f"  [{i+1}] {info['path']}{status}")
            if info['exists'] and info['modified']:
                print(f"      Modified: {info['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        if group.selected_for_deletion:
            savings = self.format_size(group.potential_savings)
            print(f"\nPotential savings: {savings}")
    
    def interactive_group_selection(self, group: DuplicateGroup) -> bool:
        """Interactive selection for a single group"""
        # Check for saved decision first
        saved_decision = self.get_saved_decision(group)
        if saved_decision:
            decision_time = saved_decision['decision_time']
            decision_type = saved_decision['decision_type']
            
            # Apply the saved decision
            if self.apply_saved_decision(group, saved_decision):
                print(f"\n✓ Applied saved decision: {decision_type} (from {decision_time[:19]})")
                self.display_group_summary(group, 0)
                
                if self.auto_apply_saved:
                    return True
                
                print(f"\nUsing saved decision. Press Enter to continue, or 'r' to revise:")
                choice = input().lower().strip()
                
                if choice != 'r':
                    return True
                # If 'r', fall through to interactive selection
        
        while True:
            self.display_group_summary(group, 0)
            
            print(f"\nOptions:")
            print(f"  a) Auto-select (keep newest)")
            print(f"  n) Auto-select (keep newest)")  
            print(f"  o) Auto-select (keep oldest)")
            print(f"  p) Auto-select (keep in priority dirs)")
            print(f"  m) Manual selection")
            print(f"  s) Skip this group")
            print(f"  q) Quit")
            
            choice = input("\nChoice: ").lower().strip()
            
            if choice in ['q', 'quit']:
                return False
            elif choice in ['s', 'skip']:
                group.selected_for_deletion.clear()
                self.save_decision(group, 'skip')
                return True
            elif choice in ['a', 'n', 'auto', 'newest']:
                group.auto_select_duplicates("keep_newest")
                self.save_decision(group, 'auto_newest')
                return True
            elif choice in ['o', 'oldest']:
                group.auto_select_duplicates("keep_oldest")
                self.save_decision(group, 'auto_oldest')
                return True
            elif choice in ['p', 'priority']:
                group.auto_select_duplicates("keep_in_priority_dirs")
                self.save_decision(group, 'auto_priority')
                return True
            elif choice in ['m', 'manual']:
                if self.manual_selection(group):
                    # Save manual selection
                    selected_paths = [str(group.paths[i]) for i in group.selected_for_deletion]
                    self.save_decision(group, 'manual', selected_paths)
                    return True
            else:
                print("Invalid choice. Please try again.")
    
    def manual_selection(self, group: DuplicateGroup) -> bool:
        """Manual file selection within a group"""
        group.selected_for_deletion.clear()
        
        while True:
            self.display_group_summary(group, 0)
            
            print(f"\nManual Selection:")
            print(f"  Enter file numbers to DELETE (e.g., 1,3,4)")
            print(f"  'all' to select all files")
            print(f"  'none' to clear selection")
            print(f"  'done' to finish selection")
            print(f"  'back' to return to auto options")
            
            choice = input("\nSelect files to DELETE: ").lower().strip()
            
            if choice == 'back':
                return False
            elif choice == 'done':
                if not group.selected_for_deletion:
                    print("No files selected for deletion.")
                    continue
                if len(group.selected_for_deletion) >= group.count:
                    print("Error: Cannot delete all copies. At least one must remain.")
                    continue
                return True
            elif choice == 'all':
                # Select all but the first one
                group.selected_for_deletion = set(range(1, group.count))
            elif choice == 'none':
                group.selected_for_deletion.clear()
            else:
                try:
                    # Parse comma-separated numbers
                    indices = []
                    for part in choice.split(','):
                        part = part.strip()
                        if '-' in part:
                            # Range like "2-4"
                            start, end = map(int, part.split('-'))
                            indices.extend(range(start-1, end))  # Convert to 0-based
                        else:
                            indices.append(int(part) - 1)  # Convert to 0-based
                    
                    # Validate indices
                    valid_indices = set()
                    for idx in indices:
                        if 0 <= idx < group.count:
                            valid_indices.add(idx)
                        else:
                            print(f"Warning: Index {idx+1} is out of range")
                    
                    if len(valid_indices) >= group.count:
                        print("Error: Cannot delete all copies. At least one must remain.")
                        continue
                    
                    group.selected_for_deletion = valid_indices
                    
                except ValueError:
                    print("Invalid input. Use numbers separated by commas (e.g., 1,3,4)")
    
    def preview_deletions(self, groups: List[DuplicateGroup]) -> None:
        """Preview all planned deletions"""
        total_files = sum(len(g.selected_for_deletion) for g in groups)
        total_savings = sum(g.potential_savings for g in groups)
        
        print(f"\n{'='*60}")
        print(f"DELETION PREVIEW")
        print(f"{'='*60}")
        print(f"Files to delete: {total_files}")
        print(f"Space to free: {self.format_size(total_savings)}")
        print(f"Mode: {'DRY RUN (no files will be deleted)' if self.dry_run else 'LIVE MODE'}")
        print()
        
        for i, group in enumerate(groups):
            if group.selected_for_deletion:
                print(f"Group #{i+1}: {len(group.selected_for_deletion)} files, "
                      f"{self.format_size(group.potential_savings)} savings")
                
                for idx in sorted(group.selected_for_deletion):
                    info = group.get_file_info(idx)
                    print(f"  DELETE: {info['path']}")
    
    def execute_deletions(self, groups: List[DuplicateGroup]) -> bool:
        """Execute the planned deletions"""
        total_files = sum(len(g.selected_for_deletion) for g in groups)
        
        if total_files == 0:
            print("No files selected for deletion.")
            return True
        
        if total_files > self.max_files_per_operation:
            print(f"Error: Too many files selected ({total_files}). "
                  f"Maximum allowed: {self.max_files_per_operation}")
            return False
        
        # Final confirmation
        if not self.dry_run:
            print(f"\n⚠️  WARNING: About to delete {total_files} files permanently!")
            confirm = input("Type 'DELETE' to confirm: ")
            if confirm != 'DELETE':
                print("Deletion cancelled.")
                return False
        
        # Create undo log
        with open(self.undo_log_path, 'w') as log:
            log.write(f"# Deletion Log - {datetime.now()}\n")
            log.write(f"# Mode: {'DRY RUN' if self.dry_run else 'LIVE'}\n\n")
        
        deleted_count = 0
        errors = []
        
        for group_idx, group in enumerate(groups):
            if not group.selected_for_deletion:
                continue
                
            print(f"\nProcessing group #{group_idx + 1}...")
            
            for file_idx in sorted(group.selected_for_deletion):
                path = group.paths[file_idx]
                
                try:
                    if self.dry_run:
                        print(f"  [DRY RUN] Would delete: {path}")
                        with open(self.undo_log_path, 'a') as log:
                            log.write(f"# WOULD DELETE: {path}\n")
                    else:
                        print(f"  Deleting: {path}")
                        
                        # Safety check
                        if not path.exists():
                            print(f"    Warning: File no longer exists")
                            continue
                        
                        # Create backup in recycle bin or temp location if possible
                        backup_path = None
                        try:
                            # Try to move to a backup location instead of permanent delete
                            backup_dir = Path("./deletion_backup")
                            backup_dir.mkdir(exist_ok=True)
                            backup_path = backup_dir / f"{deleted_count}_{path.name}"
                            shutil.move(str(path), str(backup_path))
                            
                            with open(self.undo_log_path, 'a') as log:
                                log.write(f"DELETED: {path} -> BACKUP: {backup_path}\n")
                            
                            self.deleted_files.append((str(path), str(backup_path)))
                            
                        except Exception as e:
                            # Fallback to permanent deletion
                            path.unlink()
                            with open(self.undo_log_path, 'a') as log:
                                log.write(f"DELETED: {path} (permanent)\n")
                    
                    deleted_count += 1
                    
                except Exception as e:
                    error_msg = f"Failed to delete {path}: {e}"
                    errors.append(error_msg)
                    print(f"    ERROR: {error_msg}")
                    
                    with open(self.undo_log_path, 'a') as log:
                        log.write(f"ERROR: {path} - {e}\n")
        
        # Summary
        print(f"\n{'='*60}")
        print(f"DELETION SUMMARY")
        print(f"{'='*60}")
        print(f"Files processed: {deleted_count}")
        print(f"Errors: {len(errors)}")
        
        if not self.dry_run:
            total_savings = sum(g.potential_savings for g in groups)
            print(f"Space freed: {self.format_size(total_savings)}")
            print(f"Log file: {self.undo_log_path}")
            
            if self.deleted_files:
                print(f"Backup directory: ./deletion_backup")
                print("Files can be restored from backup if needed.")
        
        if errors:
            print(f"\nErrors encountered:")
            for error in errors:
                print(f"  {error}")
        
        return len(errors) == 0
    
    def run(self) -> None:
        """Main interactive cleaning workflow"""
        print("Interactive Duplicate Cleaner")
        print("=" * 40)
        
        # Load duplicates
        print("Loading duplicates from database...")
        groups = self.load_duplicates()
        
        if not groups:
            print("No duplicates found in database.")
            return
        
        total_groups = len(groups)
        total_wasted = sum(g.wasted_space for g in groups)
        
        print(f"Found {total_groups} duplicate groups")
        print(f"Total wasted space: {self.format_size(total_wasted)}")
        print(f"Mode: {'DRY RUN (safe preview)' if self.dry_run else 'LIVE MODE (will delete files)'}")
        
        # Process each group interactively
        processed = 0
        for i, group in enumerate(groups):
            print(f"\n--- Processing group {i+1} of {total_groups} ---")
            
            if not self.interactive_group_selection(group):
                break  # User quit
            
            processed += 1
        
        if processed == 0:
            print("No groups processed.")
            return
        
        # Preview and execute
        groups_with_selections = [g for g in groups if g.selected_for_deletion]
        
        if not groups_with_selections:
            print("No files selected for deletion.")
            return
        
        self.preview_deletions(groups_with_selections)
        
        print(f"\nReady to {'simulate' if self.dry_run else 'execute'} deletions.")
        proceed = input("Proceed? (y/n): ").lower().strip()
        
        if proceed == 'y':
            self.execute_deletions(groups_with_selections)
        else:
            print("Operation cancelled.")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Interactive Duplicate File Cleaner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--db",
        default="file_scanner.db",
        help="Database path from Ultimate Scanner"
    )
    
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live mode (actually delete files). Default is safe dry-run mode."
    )
    
    parser.add_argument(
        "--max-files",
        type=int,
        default=100,
        help="Maximum files to delete in one operation"
    )
    
    parser.add_argument(
        "--clear-decisions",
        action="store_true",
        help="Clear all saved decisions and start fresh"
    )
    
    parser.add_argument(
        "--list-decisions",
        action="store_true",
        help="List all saved decisions and exit"
    )
    
    parser.add_argument(
        "--auto-apply-saved",
        action="store_true",
        help="Automatically apply saved decisions without prompting"
    )
    
    args = parser.parse_args()
    
    cleaner = InteractiveCleaner(
        db_path=args.db,
        dry_run=not args.live,
        auto_apply_saved=args.auto_apply_saved
    )
    cleaner.max_files_per_operation = args.max_files
    
    # Handle decision management options
    if args.list_decisions:
        cleaner.list_saved_decisions()
        return
    
    if args.clear_decisions:
        print("Clearing all saved decisions...")
        deleted_count = cleaner.clear_saved_decisions()
        print(f"Cleared {deleted_count} saved decisions.")
        if deleted_count > 0:
            print("All duplicate groups will now prompt for decisions again.")
        return
    
    # Safety confirmation for live mode
    if args.live:
        print("⚠️  WARNING: Live mode will actually delete files!")
        print("⚠️  Make sure you have backups of important data!")
        confirm = input("Continue with live mode? (type 'yes'): ")
        if confirm.lower() != 'yes':
            print("Switching to safe dry-run mode.")
            args.live = False
            cleaner.dry_run = True
    
    try:
        cleaner.run()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()