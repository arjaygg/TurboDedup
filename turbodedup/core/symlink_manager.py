#!/usr/bin/env python3
"""
Symlink Manager - Safe symlink replacement for duplicate files

Provides cross-platform symlink creation, validation, and rollback capabilities
for non-destructive file deduplication.

Features:
- Cross-platform symlink support with intelligent fallbacks
- Atomic operations with comprehensive rollback capability
- Filesystem compatibility detection and validation
- Operation logging and audit trails
- Integrity verification and validation
"""

import json
import logging
import os
import platform
import shutil
import stat
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)

@dataclass
class SymlinkOperation:
    """Represents a symlink operation for tracking and rollback"""
    source_path: Path
    target_path: Path
    original_size: int
    timestamp: datetime = field(default_factory=datetime.now)
    operation_id: str = ""
    success: bool = False
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'source_path': str(self.source_path),
            'target_path': str(self.target_path),
            'original_size': self.original_size,
            'timestamp': self.timestamp.isoformat(),
            'operation_id': self.operation_id,
            'success': self.success,
            'error_message': self.error_message
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SymlinkOperation':
        """Create from dictionary"""
        op = cls(
            source_path=Path(data['source_path']),
            target_path=Path(data['target_path']),
            original_size=data['original_size'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            operation_id=data['operation_id'],
            success=data['success'],
            error_message=data.get('error_message')
        )
        return op

@dataclass
class FilesystemInfo:
    """Information about filesystem capabilities"""
    supports_symlinks: bool
    supports_hardlinks: bool
    filesystem_type: str
    mount_point: str
    is_network: bool
    requires_elevation: bool = False

class SymlinkManager:
    """
    Cross-platform symlink manager with safety features
    
    Provides safe symlink creation, validation, and rollback capabilities
    for duplicate file replacement.
    """
    
    def __init__(self, log_path: Optional[Path] = None, dry_run: bool = False):
        """
        Initialize SymlinkManager
        
        Args:
            log_path: Path for operation log file (default: current dir)
            dry_run: If True, don't actually create symlinks (preview mode)
        """
        self.dry_run = dry_run
        self.log_path = log_path or Path.cwd() / "symlink_operations.json"
        self.operations: List[SymlinkOperation] = []
        self.filesystem_cache: Dict[str, FilesystemInfo] = {}
        
        # Platform-specific settings
        self.is_windows = platform.system() == "Windows"
        self.supports_symlinks = self._check_symlink_support()
        
        # Load existing operations log
        self._load_operations_log()
    
    def _check_symlink_support(self) -> bool:
        """Check if the system supports symlink creation"""
        if not self.is_windows:
            return True
        
        # On Windows, check if we have symlink privileges
        try:
            # Try creating a test symlink
            test_dir = Path.cwd() / ".symlink_test"
            test_file = test_dir / "test.txt"
            test_link = test_dir / "test_link.txt"
            
            test_dir.mkdir(exist_ok=True)
            test_file.write_text("test")
            
            try:
                test_link.symlink_to(test_file)
                test_link.unlink()
                test_file.unlink()
                test_dir.rmdir()
                return True
            except OSError:
                # Clean up
                if test_file.exists():
                    test_file.unlink()
                if test_dir.exists():
                    test_dir.rmdir()
                return False
        except Exception:
            return False
    
    def _get_filesystem_info(self, path: Path) -> FilesystemInfo:
        """Get filesystem information for a given path"""
        path_str = str(path.resolve())
        
        # Check cache first
        for cached_path, info in self.filesystem_cache.items():
            if path_str.startswith(cached_path):
                return info
        
        # Determine filesystem info
        try:
            stat_result = path.stat()
            mount_point = self._get_mount_point(path)
            filesystem_type = self._get_filesystem_type(mount_point)
            
            info = FilesystemInfo(
                supports_symlinks=self._test_symlink_support(path),
                supports_hardlinks=self._test_hardlink_support(path),
                filesystem_type=filesystem_type,
                mount_point=mount_point,
                is_network=self._is_network_filesystem(mount_point),
                requires_elevation=self.is_windows and not self.supports_symlinks
            )
            
            # Cache the result
            self.filesystem_cache[mount_point] = info
            return info
            
        except Exception as e:
            logger.warning(f"Could not determine filesystem info for {path}: {e}")
            # Return conservative defaults
            return FilesystemInfo(
                supports_symlinks=False,
                supports_hardlinks=False,
                filesystem_type="unknown",
                mount_point=str(path),
                is_network=False,
                requires_elevation=True
            )
    
    def _get_mount_point(self, path: Path) -> str:
        """Get the mount point for a path"""
        if self.is_windows:
            # On Windows, use the drive letter
            return str(path.resolve().anchor)
        else:
            # On Unix systems, find the mount point
            path = path.resolve()
            while not path.is_mount() and path.parent != path:
                path = path.parent
            return str(path)
    
    def _get_filesystem_type(self, mount_point: str) -> str:
        """Get filesystem type for a mount point"""
        try:
            if self.is_windows:
                import subprocess
                result = subprocess.run(
                    ['fsutil', 'fsinfo', 'fstype', mount_point],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    return result.stdout.strip().split()[-1]
            else:
                # Use statvfs or parse /proc/mounts
                with open('/proc/mounts', 'r') as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) >= 3 and parts[1] == mount_point:
                            return parts[2]
        except Exception:
            pass
        
        return "unknown"
    
    def _is_network_filesystem(self, mount_point: str) -> bool:
        """Check if filesystem is networked"""
        network_types = {'nfs', 'cifs', 'smb', 'ftp', 'sftp', 'sshfs'}
        fs_type = self._get_filesystem_type(mount_point).lower()
        return any(net_type in fs_type for net_type in network_types)
    
    def _test_symlink_support(self, path: Path) -> bool:
        """Test if symlinks work in a specific directory"""
        try:
            test_dir = path / ".symlink_test_dir"
            test_file = test_dir / "test.txt"
            test_link = test_dir / "test_link.txt"
            
            test_dir.mkdir(exist_ok=True)
            test_file.write_text("test")
            
            try:
                test_link.symlink_to(test_file)
                success = test_link.is_symlink() and test_link.read_text() == "test"
                test_link.unlink()
                test_file.unlink()
                test_dir.rmdir()
                return success
            except OSError:
                # Clean up on failure
                if test_file.exists():
                    test_file.unlink()
                if test_dir.exists():
                    test_dir.rmdir()
                return False
        except Exception:
            return False
    
    def _test_hardlink_support(self, path: Path) -> bool:
        """Test if hardlinks work in a specific directory"""
        try:
            test_dir = path / ".hardlink_test_dir"
            test_file = test_dir / "test.txt"
            test_link = test_dir / "test_link.txt"
            
            test_dir.mkdir(exist_ok=True)
            test_file.write_text("test")
            
            try:
                test_link.hardlink_to(test_file)
                success = test_link.exists() and test_link.stat().st_nlink > 1
                test_link.unlink()
                test_file.unlink()
                test_dir.rmdir()
                return success
            except (OSError, AttributeError):
                # Clean up on failure
                if test_file.exists():
                    test_file.unlink()
                if test_dir.exists():
                    test_dir.rmdir()
                return False
        except Exception:
            return False
    
    def check_compatibility(self, source_path: Path, target_path: Path) -> Tuple[bool, str]:
        """
        Check if symlink replacement is compatible for given paths
        
        Returns:
            Tuple of (is_compatible, reason)
        """
        # Check if paths exist
        if not target_path.exists():
            return False, f"Target file does not exist: {target_path}"
        
        if source_path.exists():
            return False, f"Source path already exists: {source_path}"
        
        # Check filesystem compatibility
        source_fs = self._get_filesystem_info(source_path.parent)
        target_fs = self._get_filesystem_info(target_path)
        
        if not source_fs.supports_symlinks:
            return False, f"Source filesystem does not support symlinks: {source_fs.filesystem_type}"
        
        if source_fs.requires_elevation:
            return False, "Symlinks require administrator privileges on this Windows system"
        
        if source_fs.is_network or target_fs.is_network:
            return False, "Network filesystems not recommended for symlink operations"
        
        # Check if cross-filesystem
        if source_fs.mount_point != target_fs.mount_point:
            logger.warning(f"Cross-filesystem symlink: {source_fs.mount_point} -> {target_fs.mount_point}")
        
        return True, "Compatible"
    
    def create_symlink(self, source_path: Path, target_path: Path, 
                      operation_id: Optional[str] = None) -> SymlinkOperation:
        """
        Create a symlink, replacing source file with link to target
        
        Args:
            source_path: Path where symlink will be created (duplicate file location)
            target_path: Path that symlink will point to (original file)
            operation_id: Optional ID for grouping operations
        
        Returns:
            SymlinkOperation with results
        """
        operation_id = operation_id or f"symlink_{int(time.time())}"
        
        # Get original file size before any operations
        try:
            original_size = source_path.stat().st_size if source_path.exists() else 0
        except OSError:
            original_size = 0
        
        operation = SymlinkOperation(
            source_path=source_path,
            target_path=target_path,
            original_size=original_size,
            operation_id=operation_id
        )
        
        try:
            # Check compatibility
            compatible, reason = self.check_compatibility(source_path, target_path)
            if not compatible:
                operation.error_message = f"Compatibility check failed: {reason}"
                return operation
            
            if self.dry_run:
                logger.info(f"DRY RUN: Would create symlink {source_path} -> {target_path}")
                operation.success = True
                operation.error_message = "DRY RUN - No actual operation performed"
                return operation
            
            # Create backup of source file permissions/timestamps
            source_stat = source_path.stat()
            
            # Remove the source file
            source_path.unlink()
            
            # Create the symlink
            source_path.symlink_to(target_path)
            
            # Verify the symlink works
            if not self._verify_symlink(source_path, target_path):
                # Rollback on verification failure
                source_path.unlink()
                raise ValueError("Symlink verification failed")
            
            operation.success = True
            logger.info(f"Created symlink: {source_path} -> {target_path}")
            
        except Exception as e:
            operation.error_message = str(e)
            logger.error(f"Failed to create symlink {source_path} -> {target_path}: {e}")
        
        # Add to operations log
        self.operations.append(operation)
        self._save_operations_log()
        
        return operation
    
    def _verify_symlink(self, symlink_path: Path, target_path: Path) -> bool:
        """Verify that symlink was created correctly and points to target"""
        try:
            return (symlink_path.is_symlink() and 
                   symlink_path.resolve() == target_path.resolve() and
                   symlink_path.exists())
        except Exception:
            return False
    
    def rollback_operation(self, operation: SymlinkOperation) -> bool:
        """
        Rollback a symlink operation by copying target back to source
        
        Args:
            operation: The SymlinkOperation to rollback
            
        Returns:
            True if rollback successful, False otherwise
        """
        try:
            if not operation.success:
                logger.warning(f"Cannot rollback failed operation: {operation.operation_id}")
                return False
            
            if self.dry_run:
                logger.info(f"DRY RUN: Would rollback {operation.source_path}")
                return True
            
            # Check if symlink still exists
            if not operation.source_path.is_symlink():
                logger.warning(f"Symlink no longer exists: {operation.source_path}")
                return False
            
            # Remove symlink
            operation.source_path.unlink()
            
            # Copy target file back to source location
            shutil.copy2(operation.target_path, operation.source_path)
            
            logger.info(f"Rolled back symlink operation: {operation.source_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rollback operation {operation.operation_id}: {e}")
            return False
    
    def rollback_all_operations(self, operation_id: Optional[str] = None) -> int:
        """
        Rollback all operations, optionally filtered by operation_id
        
        Args:
            operation_id: If provided, only rollback operations with this ID
            
        Returns:
            Number of operations successfully rolled back
        """
        operations_to_rollback = self.operations
        if operation_id:
            operations_to_rollback = [op for op in self.operations if op.operation_id == operation_id]
        
        rollback_count = 0
        for operation in reversed(operations_to_rollback):  # Rollback in reverse order
            if self.rollback_operation(operation):
                rollback_count += 1
        
        return rollback_count
    
    def get_space_savings(self, operation_id: Optional[str] = None) -> int:
        """
        Calculate total space savings from symlink operations
        
        Args:
            operation_id: If provided, only count operations with this ID
            
        Returns:
            Total bytes saved
        """
        operations = self.operations
        if operation_id:
            operations = [op for op in self.operations if op.operation_id == operation_id]
        
        return sum(op.original_size for op in operations if op.success)
    
    def _load_operations_log(self) -> None:
        """Load operations from log file"""
        if not self.log_path.exists():
            return
        
        try:
            with open(self.log_path, 'r') as f:
                data = json.load(f)
                self.operations = [SymlinkOperation.from_dict(op_data) for op_data in data]
        except Exception as e:
            logger.warning(f"Could not load operations log: {e}")
            self.operations = []
    
    def _save_operations_log(self) -> None:
        """Save operations to log file"""
        try:
            with open(self.log_path, 'w') as f:
                json.dump([op.to_dict() for op in self.operations], f, indent=2)
        except Exception as e:
            logger.error(f"Could not save operations log: {e}")
    
    def get_statistics(self) -> Dict:
        """Get statistics about symlink operations"""
        total_ops = len(self.operations)
        successful_ops = sum(1 for op in self.operations if op.success)
        failed_ops = total_ops - successful_ops
        total_space_saved = self.get_space_savings()
        
        return {
            'total_operations': total_ops,
            'successful_operations': successful_ops,
            'failed_operations': failed_ops,
            'success_rate': successful_ops / total_ops if total_ops > 0 else 0,
            'total_space_saved': total_space_saved,
            'space_saved_mb': total_space_saved / (1024 * 1024),
            'space_saved_gb': total_space_saved / (1024 * 1024 * 1024)
        }