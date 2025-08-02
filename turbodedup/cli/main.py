#!/usr/bin/env python3
"""
TurboDedup - High-Performance File Deduplication Scanner v4.0
The definitive file deduplication solution with intelligent optimization

Features:
- Three-phase scanning with partial hash optimization (10-100x faster)
- Smart caching system for incremental scanning
- GPU acceleration (CUDA/OpenCL) for massive performance gains
- Similarity detection for near-duplicates (images, audio, documents)
- Auto-detection of optimal settings
- Multiple hash algorithms (MD5, SHA256, xxHash)
- Progress tracking with ETA
- Export to CSV/JSON
- Resume capability
- Comprehensive error handling
"""

import argparse
import csv
import hashlib
import json
import logging
import os
import queue
import re
import shutil
import sqlite3
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

# Optional imports
try:
    import xxhash
    XXHASH_AVAILABLE = True
except ImportError:
    XXHASH_AVAILABLE = False

# ---------------------------
# Version Info
# ---------------------------
__version__ = "4.0.0"
__author__ = "Ultimate Scanner Team"

# ---------------------------
# Logging Configuration
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ---------------------------
# Intelligent Decision System
# ---------------------------

@dataclass
class FileContext:
    """Context information for intelligent decision making"""
    related_files: List[Path] = field(default_factory=list)
    group_size: int = 0
    total_wasted_space: int = 0
    file_types: Set[str] = field(default_factory=set)
    common_directories: Set[str] = field(default_factory=set)
    access_patterns: Dict[str, datetime] = field(default_factory=dict)

@dataclass
class FileScore:
    """Detailed scoring breakdown for a file"""
    total_score: float = 0.0
    path_quality: float = 0.0
    file_integrity: float = 0.0
    usage_context: float = 0.0
    metadata_quality: float = 0.0
    reasoning: List[str] = field(default_factory=list)

class IntelligentFileRanker:
    """Advanced file ranking system using multi-factor analysis"""
    
    def __init__(self):
        # Scoring weights (should sum to 1.0)
        self.weights = {
            'path_quality': 0.40,
            'file_integrity': 0.25, 
            'usage_context': 0.20,
            'metadata_quality': 0.15
        }
        
        # Path quality indicators
        self.path_indicators = {
            # Negative indicators (messy/temporary locations)
            'negative': {
                r'/[Dd]esktop/': -20,
                r'/[Dd]ownloads?/': -30,
                r'/[Tt]emp/': -50,
                r'/[Tt]mp/': -50,
                r'\\[Dd]esktop\\': -20,
                r'\\[Dd]ownloads?\\': -30,
                r'\\[Tt]emp\\': -50,
                r'\\[Tt]mp\\': -50,
                r'/[Rr]ecycle': -40,
                r'\\[Rr]ecycle': -40,
                r'/\.': -25,  # Hidden directories
                r'backup': -15,
                r'old': -10,
            },
            # Positive indicators (organized locations)
            'positive': {
                r'/[Dd]ocuments/': 20,
                r'/[Pp]rojects?/': 30,
                r'/[Ww]ork/': 25,
                r'\\[Dd]ocuments\\': 20,
                r'\\[Pp]rojects?\\': 30,
                r'\\[Ww]ork\\': 25,
                r'/[0-9]{4}/': 15,  # Year folders
                r'\\[0-9]{4}\\': 15,
                r'/src/': 20,
                r'/source/': 20,
                r'\\src\\': 20,
                r'\\source\\': 20,
            }
        }
    
    def calculate_file_score(self, file_path: Path, file_info: Dict, context: FileContext) -> FileScore:
        """Calculate comprehensive score for a file"""
        score = FileScore()
        
        # Calculate individual component scores
        score.path_quality = self._score_path_quality(file_path, score.reasoning)
        score.file_integrity = self._score_file_integrity(file_path, file_info, score.reasoning)
        score.usage_context = self._score_usage_context(file_path, file_info, context, score.reasoning)
        score.metadata_quality = self._score_metadata_quality(file_path, file_info, score.reasoning)
        
        # Calculate weighted total
        score.total_score = (
            score.path_quality * self.weights['path_quality'] +
            score.file_integrity * self.weights['file_integrity'] +
            score.usage_context * self.weights['usage_context'] +
            score.metadata_quality * self.weights['metadata_quality']
        )
        
        return score
    
    def _score_path_quality(self, file_path: Path, reasoning: List[str]) -> float:
        """Score based on path organization and location quality"""
        score = 50.0  # Base score
        path_str = str(file_path).replace('\\', '/')
        
        # Check negative indicators
        for pattern, penalty in self.path_indicators['negative'].items():
            if re.search(pattern, path_str, re.IGNORECASE):
                score += penalty
                reasoning.append(f"Path penalty: {pattern.strip('/')} location ({penalty:+d})")
        
        # Check positive indicators  
        for pattern, bonus in self.path_indicators['positive'].items():
            if re.search(pattern, path_str, re.IGNORECASE):
                score += bonus
                reasoning.append(f"Path bonus: {pattern.strip('/')} location (+{bonus})")
        
        # Directory depth analysis (deeper usually = more organized)
        depth = len(file_path.parts)
        if depth > 3:
            depth_bonus = min((depth - 3) * 3, 15)
            score += depth_bonus
            reasoning.append(f"Organized depth: {depth} levels (+{depth_bonus})")
        elif depth <= 2:
            score -= 10
            reasoning.append(f"Shallow depth: {depth} levels (-10)")
        
        # Filename quality
        filename = file_path.name
        if re.match(r'^[a-zA-Z0-9_\-\s\.]+$', filename):
            score += 5
            reasoning.append("Clean filename format (+5)")
        
        # Path length (very long paths can be problematic)
        if len(str(file_path)) > 200:
            score -= 10
            reasoning.append("Very long path (-10)")
        
        return max(0, min(100, score))
    
    def _score_file_integrity(self, file_path: Path, file_info: Dict, reasoning: List[str]) -> float:
        """Score based on file accessibility and integrity"""
        score = 50.0  # Base score
        
        # File accessibility
        if file_info.get('exists', True):
            score += 20
            reasoning.append("File exists (+20)")
        else:
            score -= 50
            reasoning.append("File missing (-50)")
            return max(0, score)
        
        # File readability
        try:
            if file_path.is_file() and os.access(file_path, os.R_OK):
                score += 15
                reasoning.append("File readable (+15)")
            else:
                score -= 20
                reasoning.append("File access issues (-20)")
        except:
            score -= 15
            reasoning.append("File access error (-15)")
        
        # Size reasonableness (files that are too small might be corrupted)
        size = file_info.get('size', 0)
        if size > 0:
            score += 10
            reasoning.append("Valid file size (+10)")
        else:
            score -= 25
            reasoning.append("Zero/invalid size (-25)")
        
        # Modification time reasonableness
        mtime = file_info.get('mtime')
        if mtime:
            # Files modified in the future are suspicious
            if mtime > time.time():
                score -= 20
                reasoning.append("Future modification time (-20)")
            else:
                score += 5
                reasoning.append("Valid modification time (+5)")
        
        return max(0, min(100, score))
    
    def _score_usage_context(self, file_path: Path, file_info: Dict, context: FileContext, reasoning: List[str]) -> float:
        """Score based on usage patterns and context"""
        score = 50.0  # Base score
        
        # Recent access patterns (if available)
        try:
            access_time = file_path.stat().st_atime
            days_since_access = (time.time() - access_time) / (24 * 3600)
            
            if days_since_access < 7:
                score += 25
                reasoning.append("Recently accessed (+25)")
            elif days_since_access < 30:
                score += 15
                reasoning.append("Accessed this month (+15)")
            elif days_since_access < 90:
                score += 5
                reasoning.append("Accessed recently (+5)")
            else:
                score -= 5
                reasoning.append("Not accessed recently (-5)")
        except:
            pass  # Access time not available
        
        # File relationships (part of a collection)
        if len(context.related_files) > 2:
            score += 15
            reasoning.append(f"Part of {len(context.related_files)} file collection (+15)")
        
        # Cloud sync detection
        path_str = str(file_path).lower()
        cloud_patterns = ['onedrive', 'dropbox', 'google drive', 'icloud']
        for pattern in cloud_patterns:
            if pattern in path_str:
                score += 15
                reasoning.append(f"In cloud sync folder: {pattern} (+15)")
                break
        
        # Backup location detection
        backup_patterns = ['backup', 'bak', 'old', 'archive']
        for pattern in backup_patterns:
            if pattern in path_str:
                score -= 15
                reasoning.append(f"In backup location: {pattern} (-15)")
                break
        
        # Project context (files in same directory as code/project files)
        parent_dir = file_path.parent
        try:
            sibling_files = list(parent_dir.glob('*'))
            project_indicators = ['.git', '.project', 'package.json', 'requirements.txt', '*.sln']
            for indicator in project_indicators:
                if any(f.name == indicator or f.match(indicator) for f in sibling_files):
                    score += 20
                    reasoning.append(f"In project directory (+20)")
                    break
        except:
            pass
        
        return max(0, min(100, score))
    
    def _score_metadata_quality(self, file_path: Path, file_info: Dict, reasoning: List[str]) -> float:
        """Score based on file metadata and naming conventions"""
        score = 50.0  # Base score
        
        # Original file detection (avoid copies)
        filename = file_path.name.lower()
        name_without_ext = file_path.stem.lower()
        
        # Heavy penalties for obvious copies
        copy_patterns = [
            (r'\(\d+\)', -30, "numbered copy"),
            (r'[-_\s]+copy[-_\s]*', -25, "named copy"),
            (r'[-_\s]+\d+$', -15, "numbered suffix"),
            (r'duplicate', -20, "duplicate marker"),
            (r'backup', -15, "backup marker"),
            (r'[-_]bak$', -20, "backup extension"),
        ]
        
        for pattern, penalty, description in copy_patterns:
            if re.search(pattern, name_without_ext):
                score += penalty
                reasoning.append(f"Copy indicator: {description} ({penalty:+d})")
        
        # Bonuses for original-looking names
        if not re.search(r'\d+$', name_without_ext):
            score += 15
            reasoning.append("No trailing numbers (+15)")
        
        # Simple, clean naming
        if re.match(r'^[a-zA-Z0-9_\-\s]+$', name_without_ext):
            score += 10
            reasoning.append("Clean naming convention (+10)")
        
        # Reasonable filename length
        if 5 <= len(file_path.stem) <= 50:
            score += 5
            reasoning.append("Reasonable filename length (+5)")
        elif len(file_path.stem) > 100:
            score -= 10
            reasoning.append("Very long filename (-10)")
        
        # Extension consistency
        extension = file_path.suffix.lower()
        if extension in ['.pdf', '.docx', '.xlsx', '.pptx', '.jpg', '.png', '.mp4', '.mp3']:
            score += 5
            reasoning.append("Standard file format (+5)")
        
        return max(0, min(100, score))

class SmartRecommendationEngine:
    """Generates intelligent recommendations for duplicate file handling"""
    
    def __init__(self):
        self.ranker = IntelligentFileRanker()
    
    def analyze_duplicate_group(self, duplicate_group) -> Dict:
        """Analyze a group of duplicates and generate smart recommendations"""
        
        # Build context
        context = self._build_context(duplicate_group)
        
        # Score each file
        scored_files = []
        for i in range(duplicate_group.count):
            file_path = duplicate_group.paths[i]
            file_info = duplicate_group.get_file_info(i)
            
            score = self.ranker.calculate_file_score(file_path, file_info, context)
            scored_files.append({
                'index': i,
                'path': file_path,
                'info': file_info,
                'score': score
            })
        
        # Sort by total score (highest first)
        scored_files.sort(key=lambda x: x['score'].total_score, reverse=True)
        
        # Calculate confidence
        confidence = self._calculate_confidence(scored_files)
        
        # Generate recommendation
        best_file = scored_files[0]
        
        return {
            'recommended_file': best_file,
            'confidence': confidence,
            'all_scores': scored_files,
            'reasoning': self._generate_reasoning(best_file, scored_files),
            'alternative_suggestions': scored_files[1:3] if len(scored_files) > 1 else []
        }
    
    def _build_context(self, duplicate_group) -> FileContext:
        """Build context information for the duplicate group"""
        context = FileContext()
        
        context.group_size = duplicate_group.count
        context.total_wasted_space = (duplicate_group.count - 1) * duplicate_group.size
        
        # Collect file extensions
        for i in range(duplicate_group.count):
            file_path = duplicate_group.paths[i]
            context.file_types.add(file_path.suffix.lower())
            context.common_directories.add(str(file_path.parent))
        
        # Find related files in same directories
        for directory in context.common_directories:
            try:
                related_files = list(Path(directory).glob('*'))
                context.related_files.extend(related_files[:10])  # Limit to avoid memory issues
            except:
                pass
        
        return context
    
    def _calculate_confidence(self, scored_files: List[Dict]) -> float:
        """Calculate confidence in the recommendation"""
        if len(scored_files) < 2:
            return 50.0
        
        best_score = scored_files[0]['score'].total_score
        second_best_score = scored_files[1]['score'].total_score
        
        # Confidence based on score gap
        score_gap = best_score - second_best_score
        confidence = min(95.0, max(50.0, 50 + score_gap))
        
        return confidence
    
    def _generate_reasoning(self, best_file: Dict, all_files: List[Dict]) -> str:
        """Generate human-readable reasoning for the recommendation"""
        score = best_file['score']
        reasoning_parts = []
        
        # Highlight top reasons
        top_reasons = sorted(score.reasoning, key=lambda x: abs(float(re.search(r'[+-]\d+', x).group())), reverse=True)[:3]
        
        for reason in top_reasons:
            reasoning_parts.append(reason)
        
        total_score = score.total_score
        reasoning_parts.append(f"Overall score: {total_score:.1f}/100")
        
        return " | ".join(reasoning_parts)

class SymlinkSafetyAnalyzer:
    """Intelligent analysis system for symlink vs deletion recommendations"""
    
    def __init__(self):
        # File types that are safer with symlinks (valuable/irreplaceable files)
        self.symlink_preferred_types = {
            # Media files (often irreplaceable)
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp',
            '.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.m4v',
            '.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma',
            '.psd', '.ai', '.svg', '.eps', '.indd',
            
            # Documents (often contain unique content)
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            '.odt', '.ods', '.odp', '.rtf',
            
            # Code and configuration (may have subtle differences)
            '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.cs',
            '.json', '.xml', '.yaml', '.yml', '.toml', '.ini', '.cfg',
            '.sql', '.md', '.txt', '.log',
            
            # Archives (deletion may be risky)
            '.zip', '.rar', '.7z', '.tar', '.gz', '.bz2',
            
            # Executables (may have different versions)
            '.exe', '.msi', '.dmg', '.pkg', '.deb', '.rpm', '.appimage'
        }
        
        # Patterns that indicate safe-to-delete temporary/cache files
        self.deletion_safe_patterns = [
            r'\.tmp$', r'\.temp$', r'\.cache$', r'\.bak$',
            r'\.backup$', r'\.old$', r'\.orig$',
            r'~$', r'\.swp$', r'\.swo$',
            r'_copy\d*\.', r' - Copy\.', r' - Copy \(\d+\)\.',
            r'Thumbs\.db$', r'\.DS_Store$',
            r'desktop\.ini$'
        ]
        
        # Directory patterns that suggest safer symlink use
        self.symlink_safe_directories = [
            r'/[Dd]ocuments?/', r'/[Pp]rojects?/', r'/[Ww]ork/',
            r'/[Pp]hotos?/', r'/[Pp]ictures?/', r'/[Mm]usic/',
            r'/[Vv]ideos?/', r'/[Dd]esktop/',
            r'\\[Dd]ocuments?\\', r'\\[Pp]rojects?\\', r'\\[Ww]ork\\',
            r'\\[Pp]hotos?\\', r'\\[Pp]ictures?\\', r'\\[Mm]usic\\',
            r'\\[Vv]ideos?\\', r'\\[Dd]esktop\\'
        ]
        
        # Directory patterns that suggest deletion is safer
        self.deletion_safe_directories = [
            r'/[Tt]emp/', r'/[Tt]mp/', r'/cache/', r'/logs?/',
            r'\\[Tt]emp\\', r'\\[Tt]mp\\', r'\\cache\\', r'\\logs?\\',
            r'node_modules/', r'\.git/', r'__pycache__/',
            r'build/', r'dist/', r'target/'
        ]
    
    def analyze_group_safety(self, group: 'DuplicateGroup') -> Dict:
        """Analyze a duplicate group and recommend symlink vs deletion strategy"""
        analysis = {
            'recommended_strategy': 'deletion',  # default
            'confidence': 50.0,
            'reasoning': [],
            'symlink_safety_score': 0.0,
            'deletion_safety_score': 0.0,
            'file_value_score': 0.0
        }
        
        symlink_score = 0.0
        deletion_score = 0.0
        reasoning = []
        
        # Analyze each file in the group
        for i, file_path in enumerate(group.paths):
            file_analysis = self._analyze_file_safety(file_path)
            symlink_score += file_analysis['symlink_score']
            deletion_score += file_analysis['deletion_score']
            if file_analysis['reasoning']:
                reasoning.extend(file_analysis['reasoning'])
        
        # Normalize scores by group size
        avg_symlink_score = symlink_score / len(group.paths)
        avg_deletion_score = deletion_score / len(group.paths)
        
        # Calculate file value (larger files are generally more valuable)
        size_gb = group.size / (1024 * 1024 * 1024)
        if size_gb > 1.0:
            file_value_bonus = min(20.0, size_gb * 5)
            avg_symlink_score += file_value_bonus
            reasoning.append(f"Large file value bonus: {size_gb:.1f}GB (+{file_value_bonus:.1f})")
        
        # Make recommendation
        if avg_symlink_score > avg_deletion_score + 10:  # Prefer symlinks with confidence margin
            analysis['recommended_strategy'] = 'symlink'
            analysis['confidence'] = min(95.0, 50 + (avg_symlink_score - avg_deletion_score))
            reasoning.insert(0, f"Symlink recommended (safety: {avg_symlink_score:.1f} vs deletion: {avg_deletion_score:.1f})")
        else:
            analysis['recommended_strategy'] = 'deletion'
            analysis['confidence'] = min(95.0, 50 + (avg_deletion_score - avg_symlink_score))
            reasoning.insert(0, f"Deletion recommended (safety: {avg_deletion_score:.1f} vs symlink: {avg_symlink_score:.1f})")
        
        analysis['symlink_safety_score'] = avg_symlink_score
        analysis['deletion_safety_score'] = avg_deletion_score
        analysis['file_value_score'] = size_gb
        analysis['reasoning'] = reasoning[:5]  # Limit reasoning length
        
        return analysis
    
    def _analyze_file_safety(self, file_path: Path) -> Dict:
        """Analyze individual file for symlink/deletion safety"""
        symlink_score = 0.0
        deletion_score = 0.0
        reasoning = []
        
        path_str = str(file_path).replace('\\', '/')
        filename = file_path.name.lower()
        extension = file_path.suffix.lower()
        
        # Extension-based scoring
        if extension in self.symlink_preferred_types:
            symlink_score += 20.0
            reasoning.append(f"Valuable file type: {extension}")
        else:
            deletion_score += 5.0
        
        # Filename pattern analysis
        for pattern in self.deletion_safe_patterns:
            if re.search(pattern, filename, re.IGNORECASE):
                deletion_score += 25.0
                reasoning.append(f"Temp/backup file pattern: {pattern}")
                break
        
        # Directory-based scoring
        for pattern in self.symlink_safe_directories:
            if re.search(pattern, path_str, re.IGNORECASE):
                symlink_score += 15.0
                reasoning.append(f"User content directory")
                break
        
        for pattern in self.deletion_safe_directories:
            if re.search(pattern, path_str, re.IGNORECASE):
                deletion_score += 20.0
                reasoning.append(f"Temporary/cache directory")
                break
        
        # Naming convention analysis
        if re.search(r'copy|duplicate|backup', filename, re.IGNORECASE):
            deletion_score += 15.0
            reasoning.append("Copy/backup naming pattern")
        
        if re.search(r'original|master|main|final', filename, re.IGNORECASE):
            symlink_score += 10.0
            reasoning.append("Original/master file naming")
        
        # Version number detection
        if re.search(r'v\d+|\d+\.\d+|_\d{4}', filename):
            symlink_score += 8.0
            reasoning.append("Versioned file (may have subtle differences)")
        
        return {
            'symlink_score': symlink_score,
            'deletion_score': deletion_score,
            'reasoning': reasoning
        }

# ---------------------------
# Configuration
# ---------------------------

@dataclass
class Config:
    """Scanner configuration with smart defaults"""
    # Paths
    scan_paths: List[str] = field(default_factory=lambda: ["C:\\"])
    db_path: str = "file_scanner.db"
    
    # Size limits
    min_size: int = 100 * 1024 * 1024  # 100 MB
    max_size: Optional[int] = None
    
    # Performance
    workers: int = field(default_factory=lambda: max(8, (os.cpu_count() or 4) * 4))
    chunk_size: int = 1024 * 1024  # 1 MB
    batch_size: int = 1000
    progress_interval: float = 2.0
    
    # Partial hash settings - THE KEY INNOVATION
    partial_hash_threshold: int = 512 * 1024 * 1024  # 512 MB
    partial_segment_size: int = 8 * 1024 * 1024      # 8 MB
    
    # Algorithm
    hash_algorithm: str = "md5"  # md5, sha1, sha256, xxhash
    
    # Features
    reset_db: bool = False
    force_rerun: bool = False
    verify_duplicates: bool = False
    export_format: Optional[str] = None  # csv, json
    export_path: Optional[str] = None
    
    # Integrated Deletion System
    delete_duplicates: bool = True  # Always process duplicates by default
    delete_strategy: str = "interactive"  # Default to interactive mode
    delete_dry_run: bool = False  # Default to live execution of recommendations
    delete_max_files: int = 1000
    delete_backup: bool = True
    delete_confirm: bool = True
    
    # Symlink Replacement System
    enable_symlinks: bool = False
    prefer_symlinks: bool = False
    symlink_strategy: str = "replace_duplicates"  # replace_duplicates, hybrid, large_files_only
    symlink_dry_run: bool = False
    rollback_symlinks: bool = False
    symlink_min_size: int = 1024 * 1024  # 1MB - minimum size for symlink replacement
    
    # Similarity Detection System
    enable_similarity: bool = False
    similarity_types: Set[str] = field(default_factory=lambda: {"image", "audio", "document"})
    image_similarity: bool = False
    image_sensitivity: str = "medium"  # low, medium, high
    audio_similarity: bool = False
    audio_sensitivity: str = "medium"
    document_similarity: bool = False  
    document_sensitivity: str = "medium"
    
    # Smart Caching System
    enable_cache: bool = True
    cache_path: str = "file_cache.db"
    cache_max_size_mb: int = 1000  # 1GB cache limit
    cache_cleanup_hours: int = 24
    rebuild_cache: bool = False
    clear_cache: bool = False
    cache_stats: bool = False
    
    # GPU Acceleration System
    enable_gpu: bool = False
    gpu_backend: str = "auto"  # cuda, opencl, auto
    gpu_batch_size: int = 32  # Number of files to process in GPU batch
    gpu_chunk_size: int = 4 * 1024 * 1024  # 4MB chunks for GPU processing
    
    # Advanced
    retry_attempts: int = 3
    retry_backoff: float = 0.2
    follow_symlinks: bool = False
    scan_hidden: bool = False
    quiet: bool = False
    verbose: bool = False
    show_scanned: bool = False
    
    # Legacy compatibility (deprecated)
    interactive_clean: bool = False
    
    # Exclusions
    excluded_dirs: Set[str] = field(default_factory=lambda: {
        r"C:\Windows",
        r"C:\Program Files",
        r"C:\Program Files (x86)",
        r"C:\ProgramData",
        r"C:\System Volume Information",
        r"C:\$Recycle.Bin",
        os.path.expandvars(r"%USERPROFILE%\AppData\Local\.pnpm-store"),
        r".git",
        r"node_modules",
        r"__pycache__",
        r".venv",
        r"venv",
    })
    
    excluded_extensions: Set[str] = field(default_factory=lambda: {
        ".sys", ".dll", ".exe", ".log", ".tmp", ".temp", ".cache",
        ".pyc", ".pyo", ".swp", ".swo", ".bak", ".old", ".orig",
    })
    
    # Include patterns
    include_patterns: List[str] = field(default_factory=list)
    
    def validate(self) -> None:
        """Validate configuration"""
        if self.workers < 1:
            raise ValueError("Workers must be >= 1")
        if self.chunk_size < 1024:
            raise ValueError("Chunk size must be >= 1KB")
        if self.min_size < 0:
            raise ValueError("Min size cannot be negative")
        if self.hash_algorithm == "xxhash" and not XXHASH_AVAILABLE:
            logger.warning("xxhash not available, falling back to md5")
            self.hash_algorithm = "md5"

# ---------------------------
# Statistics
# ---------------------------

@dataclass
class ScanStats:
    """Comprehensive scan statistics"""
    # Counts
    files_discovered: int = 0
    files_excluded: int = 0
    files_processed: int = 0
    files_partial_hashed: int = 0
    files_full_hashed: int = 0
    files_verified: int = 0
    
    # Sizes
    bytes_discovered: int = 0
    bytes_processed: int = 0
    bytes_read: int = 0
    
    # Results
    duplicate_sets: int = 0
    duplicate_files: int = 0
    wasted_space: int = 0
    
    # Performance
    start_time: float = field(default_factory=time.time)
    phase_times: Dict[str, float] = field(default_factory=dict)
    
    # Tracking
    largest_file: Optional[Tuple[str, int]] = None
    errors: List[str] = field(default_factory=list)
    error_count: int = 0
    exclusions: Dict[str, int] = field(default_factory=dict)
    
    def add_error(self, msg: str) -> None:
        """Add error message"""
        self.error_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.errors.append(f"[{timestamp}] {msg}")
        if len(self.errors) > 100:  # Keep last 100
            self.errors = self.errors[-100:]
    
    def get_duration(self) -> float:
        """Get elapsed time"""
        return time.time() - self.start_time
    
    def get_rate(self, metric: str = "files_processed") -> float:
        """Calculate rate per second"""
        duration = self.get_duration()
        if duration <= 0:
            return 0.0
        count = getattr(self, metric, 0)
        return count / duration
    
    def start_phase(self, phase: str) -> None:
        """Mark phase start"""
        self.phase_times[f"{phase}_start"] = time.time()
    
    def end_phase(self, phase: str) -> None:
        """Mark phase end"""
        start_key = f"{phase}_start"
        if start_key in self.phase_times:
            self.phase_times[f"{phase}_duration"] = time.time() - self.phase_times[start_key]

# ---------------------------
# Database Manager
# ---------------------------

class DatabaseManager:
    """SQLite database operations with optimization"""
    
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        path TEXT UNIQUE NOT NULL,
        size INTEGER NOT NULL,
        mtime REAL,
        phash TEXT,  -- Partial hash
        hash TEXT,   -- Full hash
        error TEXT,
        scan_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_size ON files(size);
    CREATE INDEX IF NOT EXISTS idx_hash ON files(hash);
    CREATE INDEX IF NOT EXISTS idx_phash ON files(phash);
    CREATE INDEX IF NOT EXISTS idx_size_hash ON files(size, hash);
    CREATE INDEX IF NOT EXISTS idx_size_phash ON files(size, phash);
    
    CREATE TABLE IF NOT EXISTS similarity_hashes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id INTEGER NOT NULL,
        similarity_type TEXT NOT NULL,
        hash_value TEXT NOT NULL,
        confidence REAL DEFAULT 1.0,
        metadata TEXT,  -- JSON metadata
        computed_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE,
        UNIQUE(file_id, similarity_type)
    );
    
    CREATE INDEX IF NOT EXISTS idx_similarity_type ON similarity_hashes(similarity_type);
    CREATE INDEX IF NOT EXISTS idx_similarity_hash ON similarity_hashes(hash_value);
    CREATE INDEX IF NOT EXISTS idx_similarity_type_hash ON similarity_hashes(similarity_type, hash_value);
    
    CREATE TABLE IF NOT EXISTS scan_metadata (
        key TEXT PRIMARY KEY,
        value TEXT,
        updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    def __init__(self, db_path: str, reset: bool = False):
        self.db_path = db_path
        self._init_db(reset)
    
    def _init_db(self, reset: bool) -> None:
        """Initialize database with optimal settings"""
        with sqlite3.connect(self.db_path) as conn:
            # Performance settings
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=MEMORY")
            
            # Create schema
            conn.executescript(self.SCHEMA)
            
            if reset:
                conn.execute("DELETE FROM files")
                conn.execute("DELETE FROM scan_metadata")
                
            # Save scan start
            conn.execute(
                "INSERT OR REPLACE INTO scan_metadata(key, value) VALUES (?, ?)",
                ("scan_start", datetime.now().isoformat())
            )
            conn.commit()
    
    def writer_thread(self, q: queue.Queue, sentinel: object, batch_size: int) -> None:
        """Database writer thread"""
        conn = sqlite3.connect(self.db_path, timeout=60)
        cursor = conn.cursor()
        batch = []
        
        try:
            while True:
                try:
                    item = q.get(timeout=0.2)
                except queue.Empty:
                    if batch:
                        self._insert_batch(cursor, batch)
                        conn.commit()
                        batch.clear()
                    continue
                
                if item is sentinel:
                    q.task_done()
                    break
                
                batch.append(item)
                q.task_done()
                
                if len(batch) >= batch_size:
                    self._insert_batch(cursor, batch)
                    conn.commit()
                    batch.clear()
        
        except Exception as e:
            logger.error(f"Database writer error: {e}")
        finally:
            if batch:
                self._insert_batch(cursor, batch)
                conn.commit()
            conn.close()
    
    def _insert_batch(self, cursor: sqlite3.Cursor, batch: List[Tuple]) -> None:
        """Batch insert records"""
        cursor.executemany(
            "INSERT OR REPLACE INTO files(path, size, mtime, phash, hash, error) VALUES (?,?,?,?,?,?)",
            batch
        )
    
    def get_size_duplicates(self) -> Iterator[Tuple[int, str]]:
        """Get files with duplicate sizes"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT size, path
                FROM files
                WHERE error IS NULL
                  AND size IN (
                    SELECT size FROM files
                    WHERE error IS NULL
                    GROUP BY size
                    HAVING COUNT(*) > 1
                  )
                ORDER BY size DESC
            """)
            yield from cursor
    
    def get_partial_hash_candidates(self, threshold: int) -> Iterator[Tuple[int, str]]:
        """Get large files needing partial hash"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT size, path
                FROM files
                WHERE error IS NULL
                  AND size >= ?
                  AND size IN (
                    SELECT size FROM files
                    WHERE error IS NULL
                    GROUP BY size
                    HAVING COUNT(*) > 1
                  )
                ORDER BY size DESC
            """, (threshold,))
            yield from cursor
    
    def get_full_hash_candidates(self, threshold: int) -> Iterator[str]:
        """Get files needing full hash"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Small files with duplicate sizes
            cursor.execute("""
                SELECT path FROM files
                WHERE error IS NULL
                  AND size < ?
                  AND size IN (
                    SELECT size FROM files
                    WHERE error IS NULL
                    GROUP BY size
                    HAVING COUNT(*) > 1
                  )
            """, (threshold,))
            yield from (row[0] for row in cursor)
            
            # Files with duplicate partial hashes
            cursor.execute("""
                SELECT path FROM files
                WHERE error IS NULL
                  AND phash IN (
                    SELECT phash FROM files
                    WHERE phash IS NOT NULL AND error IS NULL
                    GROUP BY size, phash
                    HAVING COUNT(*) > 1
                  )
            """)
            yield from (row[0] for row in cursor)
    
    def update_hashes(self, updates: List[Tuple], hash_type: str = "hash") -> None:
        """Update hash values in batch"""
        with sqlite3.connect(self.db_path) as conn:
            if hash_type == "phash":
                conn.executemany("UPDATE files SET phash=?, error=? WHERE path=?", updates)
            else:
                conn.executemany("UPDATE files SET hash=?, error=? WHERE path=?", updates)
            conn.commit()
    
    def get_duplicates(self) -> List[Tuple[int, str, int, List[str]]]:
        """Get duplicate files by hash"""
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
            
            duplicates = []
            for size, hash_val, count in cursor.fetchall():
                cursor2 = conn.cursor()
                cursor2.execute(
                    "SELECT path FROM files WHERE hash=? AND error IS NULL",
                    (hash_val,)
                )
                paths = [row[0] for row in cursor2]
                duplicates.append((size, hash_val, count, paths))
            
            return duplicates
    
    def store_similarity_hashes(self, file_path: str, similarity_hashes: Dict[str, str], 
                               metadata: Dict = None) -> None:
        """Store similarity hashes for a file"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get file_id
            cursor.execute("SELECT id FROM files WHERE path = ?", (file_path,))
            result = cursor.fetchone()
            if not result:
                return
            
            file_id = result[0]
            metadata_json = json.dumps(metadata) if metadata else None
            
            # Insert similarity hashes
            for similarity_type, hash_value in similarity_hashes.items():
                cursor.execute("""
                    INSERT OR REPLACE INTO similarity_hashes 
                    (file_id, similarity_type, hash_value, metadata)
                    VALUES (?, ?, ?, ?)
                """, (file_id, similarity_type, hash_value, metadata_json))
            
            conn.commit()
    
    def get_similarity_matches(self, similarity_type: str, 
                              threshold: float = 0.8) -> List[Tuple]:
        """Get potential similarity matches for a given type"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT f1.path, f2.path, s1.hash_value, s2.hash_value, s1.confidence
                FROM similarity_hashes s1
                JOIN similarity_hashes s2 ON s1.similarity_type = s2.similarity_type 
                    AND s1.hash_value = s2.hash_value
                    AND s1.file_id != s2.file_id
                JOIN files f1 ON s1.file_id = f1.id
                JOIN files f2 ON s2.file_id = f2.id
                WHERE s1.similarity_type = ? 
                    AND s1.confidence >= ?
                    AND f1.error IS NULL 
                    AND f2.error IS NULL
                ORDER BY s1.confidence DESC
            """, (similarity_type, threshold))
            
            return cursor.fetchall()
    
    def get_file_similarity_hashes(self, file_path: str) -> Dict[str, str]:
        """Get all similarity hashes for a file"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT s.similarity_type, s.hash_value
                FROM similarity_hashes s
                JOIN files f ON s.file_id = f.id
                WHERE f.path = ?
            """, (file_path,))
            
            return {row[0]: row[1] for row in cursor.fetchall()}

# ---------------------------
# Hash Computer
# ---------------------------

class HashComputer:
    """Compute file hashes with multiple algorithms"""
    
    def __init__(self, algorithm: str, chunk_size: int, retry_attempts: int, retry_backoff: float):
        self.algorithm = algorithm
        self.chunk_size = chunk_size
        self.retry_attempts = retry_attempts
        self.retry_backoff = retry_backoff
    
    def _get_hasher(self):
        """Get hasher for algorithm"""
        if self.algorithm == "md5":
            return hashlib.md5()
        elif self.algorithm == "sha1":
            return hashlib.sha1()
        elif self.algorithm == "sha256":
            return hashlib.sha256()
        elif self.algorithm == "xxhash" and XXHASH_AVAILABLE:
            return xxhash.xxh64()
        else:
            return hashlib.md5()
    
    def compute_full_hash(self, path: Path) -> Tuple[Optional[str], Optional[str], int]:
        """Compute full file hash with retries"""
        bytes_read = 0
        
        for attempt in range(self.retry_attempts):
            try:
                hasher = self._get_hasher()
                
                with path.open("rb") as f:
                    while chunk := f.read(self.chunk_size):
                        hasher.update(chunk)
                        bytes_read += len(chunk)
                
                return hasher.hexdigest(), None, bytes_read
                
            except Exception as e:
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_backoff * (2 ** attempt))
                    continue
                return None, f"Hash error: {e}", bytes_read
    
    def compute_partial_hash(self, path: Path, segment_size: int) -> Tuple[Optional[str], Optional[str], int]:
        """Compute partial hash (head + tail) - THE KEY INNOVATION"""
        bytes_read = 0
        
        for attempt in range(self.retry_attempts):
            try:
                size = path.stat().st_size
                hasher = self._get_hasher()
                
                with path.open("rb") as f:
                    # Read head
                    head = f.read(segment_size)
                    hasher.update(head)
                    bytes_read += len(head)
                    
                    # Read tail
                    if size > segment_size * 2:
                        f.seek(size - segment_size)
                        tail = f.read(segment_size)
                        hasher.update(tail)
                        bytes_read += len(tail)
                
                return hasher.hexdigest(), None, bytes_read
                
            except Exception as e:
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_backoff * (2 ** attempt))
                    continue
                return None, f"Partial hash error: {e}", bytes_read
    
    def verify_identical(self, path1: Path, path2: Path) -> bool:
        """Byte-by-byte comparison"""
        try:
            if path1.stat().st_size != path2.stat().st_size:
                return False
            
            with path1.open("rb") as f1, path2.open("rb") as f2:
                while True:
                    chunk1 = f1.read(self.chunk_size)
                    chunk2 = f2.read(self.chunk_size)
                    
                    if chunk1 != chunk2:
                        return False
                    
                    if not chunk1:  # EOF
                        return True
                        
        except Exception:
            return False

# ---------------------------
# Path Filter
# ---------------------------

class PathFilter:
    """Filter paths based on rules"""
    
    def __init__(self, config: Config):
        self.config = config
        self.excluded_dirs = self._normalize_dirs(config.excluded_dirs)
        self.excluded_extensions = {ext.lower() for ext in config.excluded_extensions}
        self.include_patterns = [p.lower() for p in config.include_patterns]
    
    def _normalize_dirs(self, dirs: Set[str]) -> Set[str]:
        """Normalize directory paths"""
        normalized = set()
        for d in dirs:
            # Handle both absolute and relative paths
            if os.path.isabs(d):
                normalized.add(os.path.normcase(d).rstrip(os.sep))
            else:
                normalized.add(os.path.normcase(d))
        return normalized
    
    def is_excluded_dir(self, path: str) -> bool:
        """Check if directory should be excluded"""
        norm_path = os.path.normcase(path)
        path_parts = norm_path.split(os.sep)
        
        for excluded in self.excluded_dirs:
            if os.path.isabs(excluded):
                # Absolute path comparison
                if norm_path.startswith(excluded):
                    return True
            else:
                # Relative path comparison (e.g., "node_modules")
                if excluded in path_parts:
                    return True
        
        return False
    
    def should_process_file(self, path: Path, size: int) -> Tuple[bool, Optional[str]]:
        """Check if file should be processed"""
        # Size check
        if size < self.config.min_size:
            return False, f"below_min_size:{self.config.min_size}"
        
        if self.config.max_size and size > self.config.max_size:
            return False, f"above_max_size:{self.config.max_size}"
        
        # Extension check
        if path.suffix.lower() in self.excluded_extensions:
            return False, f"excluded_extension:{path.suffix.lower()}"
        
        # Include patterns
        if self.include_patterns:
            path_lower = str(path).lower()
            if not any(pattern in path_lower or path_lower.endswith(pattern) 
                      for pattern in self.include_patterns):
                return False, "not_in_include_patterns"
        
        # Hidden files
        if not self.config.scan_hidden and path.name.startswith('.'):
            return False, "hidden_file"
        
        return True, None

# ---------------------------
# Progress Tracker
# ---------------------------

class ProgressTracker:
    """Track and display progress"""
    
    def __init__(self, config: Config, stats: ScanStats):
        self.config = config
        self.stats = stats
        self.last_update = time.time()
        self.last_count = 0
    
    def update(self, force: bool = False) -> None:
        """Update progress display"""
        now = time.time()
        if not force and now - self.last_update < self.config.progress_interval:
            return
        
        # Calculate rates
        duration = self.stats.get_duration()
        current_count = self.stats.files_processed
        interval_files = current_count - self.last_count
        interval_time = now - self.last_update
        
        current_rate = interval_files / interval_time if interval_time > 0 else 0
        overall_rate = self.stats.get_rate()
        
        # Build progress message
        parts = [
            f"Files: {current_count:,}",
            f"Rate: {current_rate:.1f}/s",
            f"Avg: {overall_rate:.1f}/s",
            f"Time: {timedelta(seconds=int(duration))}",
        ]
        
        if self.stats.bytes_read > 0:
            mb_read = self.stats.bytes_read / (1024 * 1024)
            mb_rate = mb_read / duration if duration > 0 else 0
            parts.append(f"Read: {mb_read:.1f}MB ({mb_rate:.1f}MB/s)")
        
        if not self.config.quiet:
            logger.info(" | ".join(parts))
        
        self.last_update = now
        self.last_count = current_count

# ---------------------------
# Utility Functions
# ---------------------------

def format_size(bytes_val: int) -> str:
    """Format bytes as human readable"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f} PB"

def detect_original_file(paths: List[Path]) -> Optional[int]:
    """
    Detect which file is likely the original among duplicates.
    
    Common patterns detected:
    - Image.png vs Image (1).png, Image (2).png
    - Document.pdf vs Document - Copy.pdf
    - File.txt vs File - Copy (2).txt
    - Photo.jpg vs Photo_copy.jpg
    
    Returns the index of the likely original file, or None if unclear.
    """
    if len(paths) <= 1:
        return None
    
    # Convert paths to strings for pattern matching
    path_strings = [str(p) for p in paths]
    file_scores = []
    
    for i, path_str in enumerate(path_strings):
        score = 0
        filename = Path(path_str).name
        name_without_ext = Path(path_str).stem
        
        # Higher score = more likely to be original
        
        # Check for copy indicators (lower score for copies)
        copy_patterns = [
            r'\(\d+\)$',           # (1), (2), etc. at end of name
            r'[-_\s]+copy[-_\s]*',  # "copy", "Copy", " - Copy", "_copy"
            r'[-_\s]+\d+$',        # ending with numbers like "_2", " 3"
            r'duplicate',          # "duplicate" in name
            r'backup',             # "backup" in name
        ]
        
        filename_lower = filename.lower()
        name_lower = name_without_ext.lower()
        
        for pattern in copy_patterns:
            if re.search(pattern, name_lower):
                score -= 10  # Heavy penalty for copy indicators
        
        # Additional penalties for specific patterns
        if re.search(r'\(\d+\)', filename):  # (1), (2) patterns
            score -= 20
        if 'copy' in filename_lower:
            score -= 15
        if re.search(r'[-_]\d+$', name_without_ext):  # ends with dash/underscore + number
            score -= 10
        
        # Bonus for shorter names (originals tend to be simpler)
        score += max(0, 50 - len(filename))  # Bonus for shorter names
        
        # Bonus for no special characters or numbers at end
        if re.match(r'^[a-zA-Z0-9_\-\s]+\.[a-zA-Z0-9]+$', filename):
            if not re.search(r'\d+$', name_without_ext):
                score += 20
        
        file_scores.append((i, score, filename))
    
    # Sort by score (highest first) and return the index of the best candidate
    file_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Only return if there's a clear winner (significant score difference)
    if len(file_scores) >= 2:
        best_score = file_scores[0][1]
        second_best_score = file_scores[1][1]
        
        # Require at least 10 point difference to be confident
        if best_score - second_best_score >= 10:
            return file_scores[0][0]
    
    return None

# ---------------------------
# Integrated Deletion System  
# ---------------------------

@dataclass
class DuplicateGroup:
    """Represents a group of duplicate files"""
    size: int
    hash_val: str
    paths: List[Path]
    selected_for_deletion: Set[int] = field(default_factory=set)
    selected_for_symlink: Set[int] = field(default_factory=set)
    symlink_target_index: Optional[int] = None
    
    @property
    def count(self) -> int:
        return len(self.paths)
    
    @property
    def wasted_space(self) -> int:
        return self.size * (self.count - 1)
    
    @property
    def potential_savings(self) -> int:
        return self.size * len(self.selected_for_deletion)
    
    @property
    def potential_symlink_savings(self) -> int:
        return self.size * len(self.selected_for_symlink)
    
    @property
    def total_potential_savings(self) -> int:
        return self.size * (len(self.selected_for_deletion) + len(self.selected_for_symlink))
    
    def get_file_info(self, index: int) -> Dict:
        """Get detailed info about a specific file"""
        path = self.paths[index]
        try:
            stat = path.stat()
            return {
                'path': str(path),
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'exists': True,
                'readable': os.access(path, os.R_OK),
                'writable': os.access(path, os.W_OK),
            }
        except (OSError, FileNotFoundError):
            return {
                'path': str(path),
                'exists': False,
                'readable': False,
                'writable': False,
            }

class IntegratedDuplicateManager:
    """Integrated duplicate detection, deletion, and symlink manager"""
    
    def __init__(self, config: Config, stats: ScanStats):
        self.config = config
        self.stats = stats
        self.deleted_files: List[str] = []
        self.backup_dir = Path("./deletion_backup")
        
        if config.delete_backup:
            self.backup_dir.mkdir(exist_ok=True)
        
        # Initialize intelligent analysis systems
        self.file_ranker = IntelligentFileRanker()
        self.symlink_analyzer = SymlinkSafetyAnalyzer()
        
        # Initialize symlink manager if enabled
        self.symlink_manager = None
        if config.enable_symlinks:
            try:
                from ..core.symlink_manager import SymlinkManager
                symlink_log_path = Path(f"symlink_operations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                self.symlink_manager = SymlinkManager(
                    log_path=symlink_log_path, 
                    dry_run=config.symlink_dry_run or config.delete_dry_run
                )
                logger.info("Symlink manager initialized successfully")
            except ImportError:
                logger.warning("Symlink manager not available - symlink features disabled")
                self.config.enable_symlinks = False
            except Exception as e:
                logger.error(f"Failed to initialize symlink manager: {e}")
                self.config.enable_symlinks = False
    
    def apply_deletion_strategy(self, group: DuplicateGroup) -> None:
        """Apply deletion strategy to a group"""
        if group.count <= 1:
            return
        
        strategy = self.config.delete_strategy
        
        if strategy == "keep_newest":
            self._select_keep_newest(group)
        elif strategy == "keep_oldest":
            self._select_keep_oldest(group)
        elif strategy == "keep_first":
            self._select_keep_first(group)
        elif strategy == "keep_priority":
            self._select_keep_priority(group)
        elif strategy == "keep_smart":
            self._select_keep_smart(group)
        elif strategy == "interactive":
            self._interactive_selection(group)
        elif strategy == "skip":
            group.selected_for_deletion.clear()
    
    def _select_keep_newest(self, group: DuplicateGroup) -> None:
        """Keep the newest file, delete or symlink others"""
        file_infos = [(i, group.get_file_info(i)) for i in range(group.count)]
        valid_files = [(i, info) for i, info in file_infos if info['exists']]
        
        if len(valid_files) <= 1:
            return
        
        newest_idx = max(valid_files, key=lambda x: x[1].get('modified', datetime.min))[0]
        
        if self.config.prefer_symlinks and self.config.enable_symlinks:
            # Use symlink replacement
            group.symlink_target_index = newest_idx
            group.selected_for_symlink = {i for i, _ in valid_files if i != newest_idx}
            group.selected_for_deletion.clear()
        else:
            # Use deletion
            group.selected_for_deletion = {i for i, _ in valid_files if i != newest_idx}
    
    def _select_keep_oldest(self, group: DuplicateGroup) -> None:
        """Keep the oldest file, delete others"""
        file_infos = [(i, group.get_file_info(i)) for i in range(group.count)]
        valid_files = [(i, info) for i, info in file_infos if info['exists']]
        
        if len(valid_files) <= 1:
            return
        
        oldest_idx = min(valid_files, key=lambda x: x[1].get('modified', datetime.max))[0]
        group.selected_for_deletion = {i for i, _ in valid_files if i != oldest_idx}
    
    def _select_keep_first(self, group: DuplicateGroup) -> None:
        """Keep the first file alphabetically, delete others"""
        file_infos = [(i, group.get_file_info(i)) for i in range(group.count)]
        valid_files = [(i, info) for i, info in file_infos if info['exists']]
        
        if len(valid_files) <= 1:
            return
        
        sorted_files = sorted(valid_files, key=lambda x: x[1]['path'])
        keep_idx = sorted_files[0][0]
        group.selected_for_deletion = {i for i, _ in valid_files if i != keep_idx}
    
    def _select_keep_priority(self, group: DuplicateGroup) -> None:
        """Keep files in priority directories"""
        priority_patterns = ['/Documents/', '/Desktop/', '/Pictures/', '/Videos/', 
                           'Documents\\\\', 'Desktop\\\\', 'Pictures\\\\', 'Videos\\\\',
                           'home/', 'Users/']
        
        file_infos = [(i, group.get_file_info(i)) for i in range(group.count)]
        valid_files = [(i, info) for i, info in file_infos if info['exists']]
        
        if len(valid_files) <= 1:
            return
        
        priority_files = []
        for i, info in valid_files:
            if any(pattern in info['path'] for pattern in priority_patterns):
                priority_files.append((i, info))
        
        if priority_files:
            # Keep newest priority file
            keep_idx = max(priority_files, key=lambda x: x[1].get('modified', datetime.min))[0]
            group.selected_for_deletion = {i for i, _ in valid_files if i != keep_idx}
        else:
            # Fallback to keep_newest
            self._select_keep_newest(group)
    
    def _select_keep_original(self, group: DuplicateGroup) -> None:
        """Keep the original file based on naming patterns"""
        file_infos = [(i, group.get_file_info(i)) for i in range(group.count)]
        valid_files = [(i, info) for i, info in file_infos if info['exists']]
        
        if len(valid_files) <= 1:
            return
        
        # Use the detect_original_file function
        valid_paths = [group.paths[i] for i, _ in valid_files]
        original_idx_in_valid = detect_original_file(valid_paths)
        
        if original_idx_in_valid is not None:
            # Map back to the original group indices
            keep_idx = valid_files[original_idx_in_valid][0]
            group.selected_for_deletion = {i for i, _ in valid_files if i != keep_idx}
            
            # Show detection result
            original_file = group.paths[keep_idx].name
            copies = [group.paths[i].name for i, _ in valid_files if i != keep_idx]
            print(f"    Detected original: {original_file}")
            print(f"    Will delete {len(copies)} copies: {', '.join(copies[:3])}{'...' if len(copies) > 3 else ''}")
        else:
            # Fallback to keep_newest if detection fails
            print(f"    Could not detect original file, falling back to keep newest")
            self._select_keep_newest(group)
    
    def _select_keep_smart(self, group: DuplicateGroup) -> None:
        """Keep the best file based on intelligent analysis with symlink safety consideration"""
        try:
            # Get symlink safety analysis first
            safety_analysis = self.symlink_analyzer.analyze_group_safety(group)
            
            # Get traditional smart recommendation
            recommendation_engine = SmartRecommendationEngine()
            recommendation = recommendation_engine.analyze_duplicate_group(group)
            
            best_file = recommendation['recommended_file']
            confidence = recommendation['confidence']
            reasoning = recommendation['reasoning']
            keep_idx = best_file['index']
            
            # Determine operation strategy based on safety analysis
            if (safety_analysis['recommended_strategy'] == 'symlink' and 
                self.config.enable_symlinks and 
                safety_analysis['confidence'] > 70.0):
                
                # Use symlink replacement
                group.symlink_target_index = keep_idx
                group.selected_for_symlink = {i for i in range(group.count) if i != keep_idx}
                group.selected_for_deletion.clear()
                
                # Show what was decided
                kept_file = group.paths[keep_idx].name
                symlink_count = len(group.selected_for_symlink)
                
                print(f"    🔗 Smart symlink recommendation: Keep '{kept_file}' as target")
                print(f"    File reasoning: {reasoning}")
                print(f"    Symlink reasoning: {safety_analysis['reasoning'][0] if safety_analysis['reasoning'] else 'Safe for symlinks'}")
                print(f"    Will create {symlink_count} symlinks (Safety: {safety_analysis['confidence']:.0f}%)")
                
            else:
                # Use traditional deletion
                group.selected_for_deletion = {i for i in range(group.count) if i != keep_idx}
                group.selected_for_symlink.clear()
                group.symlink_target_index = None
                
                # Show what was decided
                kept_file = group.paths[keep_idx].name
                deleted_count = len(group.selected_for_deletion)
                
                strategy_reason = "deletion safer" if safety_analysis['recommended_strategy'] == 'deletion' else "symlinks disabled"
                print(f"    🗑️ Smart deletion recommendation: Keep '{kept_file}' ({strategy_reason})")
                print(f"    Reasoning: {reasoning}")
                print(f"    Will delete {deleted_count} copies (Confidence: {confidence:.0f}%)")
            
        except Exception as e:
            logger.warning(f"Smart selection failed: {e}")
            print(f"    Smart selection failed, falling back to keep newest")
            self._select_keep_newest(group)
    
    def _interactive_selection(self, group: DuplicateGroup) -> None:
        """Intelligent interactive selection for a group using AI recommendations"""
        # Get smart recommendation
        try:
            recommendation_engine = SmartRecommendationEngine()
            recommendation = recommendation_engine.analyze_duplicate_group(group)
            
            # Display header with recommendation
            print(f"\n{'='*70}")
            print(f"Duplicate Group - SMART RECOMMENDATION")
            print(f"{'='*70}")
            print(f"Files: {group.count} copies | Size: {format_size(group.size)} each | Total wasted: {format_size(group.wasted_space)}")
            
            # Show recommendation with confidence
            best_file = recommendation['recommended_file']
            confidence = recommendation['confidence']
            reasoning = recommendation['reasoning']
            
            print(f"\n🎯 RECOMMENDED: Keep [{best_file['index']+1}] (Confidence: {confidence:.0f}%)")
            print(f"   Reason: {reasoning}")
            
            # Add symlink safety analysis if enabled
            if self.config.enable_symlinks:
                safety_analysis = self.symlink_analyzer.analyze_group_safety(group)
                strategy_icon = "🔗" if safety_analysis['recommended_strategy'] == 'symlink' else "🗑️"
                strategy_name = "SYMLINK" if safety_analysis['recommended_strategy'] == 'symlink' else "DELETE"
                
                print(f"\n{strategy_icon} SAFETY RECOMMENDATION: {strategy_name} duplicates (Confidence: {safety_analysis['confidence']:.0f}%)")
                if safety_analysis['reasoning']:
                    print(f"   Safety reason: {safety_analysis['reasoning'][0]}")
            
            print()
            
            # Display all files with scores
            for file_data in recommendation['all_scores']:
                i = file_data['index']
                path = file_data['path']
                score = file_data['score']
                info = group.get_file_info(i)
                
                # Highlight recommended file
                marker = "⭐" if i == best_file['index'] else "  "
                status = "MISSING" if not info['exists'] else ""
                
                print(f"{marker} [{i+1}] {path} {status}")
                
                if info['exists']:
                    # Show key metrics
                    try:
                        access_time = path.stat().st_atime
                        days_ago = (time.time() - access_time) / (24 * 3600)
                        if days_ago < 1:
                            access_str = "today"
                        elif days_ago < 7:
                            access_str = f"{int(days_ago)} days ago"
                        elif days_ago < 30:
                            access_str = f"{int(days_ago/7)} weeks ago"
                        else:
                            access_str = "long ago"
                    except:
                        access_str = "unknown"
                    
                    # Determine location type
                    path_str = str(path).lower()
                    if any(x in path_str for x in ['projects', 'work', 'dev']):
                        location_type = "📁 Project folder"
                    elif any(x in path_str for x in ['documents']):
                        location_type = "📁 Documents"
                    elif any(x in path_str for x in ['downloads']):
                        location_type = "📁 Downloads"
                    elif any(x in path_str for x in ['desktop']):
                        location_type = "📁 Desktop"
                    elif any(x in path_str for x in ['onedrive', 'dropbox', 'google drive']):
                        location_type = "☁️ Cloud sync"
                    else:
                        location_type = "📁 Standard location"
                    
                    print(f"      {location_type}  🕒 Last accessed: {access_str}")
                    print(f"      📊 Score: {score.total_score:.0f}/100")
                
                print()
            
        except Exception as e:
            # Fallback to basic display if smart recommendation fails
            logger.warning(f"Smart recommendation failed: {e}")
            print(f"\n{'='*60}")
            print(f"Duplicate Group")
            print(f"{'='*60}")
            print(f"Files: {group.count} copies")
            print(f"Size: {format_size(group.size)} each")
            print(f"Total wasted: {format_size(group.wasted_space)}")
            print()
            
            for i, path in enumerate(group.paths):
                info = group.get_file_info(i)
                status = "MISSING" if not info['exists'] else ""
                print(f"  [{i+1}] {info['path']} {status}")
                if info['exists'] and 'modified' in info:
                    print(f"      Modified: {info['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Interactive menu
        while True:
            print(f"Options:")
            try:
                if 'recommendation' in locals():
                    print(f"  ✅ a) Accept recommendation (keep [{best_file['index']+1}])")
                    if len(recommendation['alternative_suggestions']) > 0:
                        alt = recommendation['alternative_suggestions'][0]
                        print(f"  🔄 r) Reverse (keep [{alt['index']+1}] instead)")
                    print(f"  📊 s) Show detailed scoring breakdown")
            except:
                pass
            
            print(f"  ⚙️  1) Keep [1]")
            print(f"  ⚙️  2) Keep [2]")
            if len(group.paths) > 2:
                print(f"  ⚙️  3) Keep [3]")
            print(f"  ⚙️  p) Keep in priority dirs (auto-select)")
            print(f"  🔗 sl) Force symlinks (keep recommended file as target)")
            print(f"  🗑️  dl) Force deletion (delete duplicates, keep recommended file)")
            print(f"  ✋ m) Manual selection")
            print(f"  ⏭️  k) Skip this group")
            print(f"  🚪 q) Quit")
            
            choice = input("\nChoice: ").lower().strip()
            
            if choice == 'q':
                sys.exit(0)
            elif choice == 'k':
                group.selected_for_deletion.clear()
                break
            elif choice == 'a' and 'recommendation' in locals():
                # Accept smart recommendation
                keep_idx = best_file['index']
                
                # Check if symlinks were recommended and enabled
                if (self.config.enable_symlinks and 
                    'safety_analysis' in locals() and 
                    safety_analysis['recommended_strategy'] == 'symlink' and 
                    safety_analysis['confidence'] > 70.0):
                    
                    # Set up symlink replacement
                    group.symlink_target_index = keep_idx
                    group.selected_for_symlink = {i for i in range(group.count) if i != keep_idx}
                    group.selected_for_deletion.clear()
                    print(f"    🔗 Smart symlink recommendation accepted. Target: [{keep_idx+1}]")
                else:
                    # Set up deletion
                    group.selected_for_deletion = {i for i in range(group.count) if i != keep_idx}
                    group.selected_for_symlink.clear()
                    group.symlink_target_index = None
                    print(f"    🗑️ Smart deletion recommendation accepted. Keeping [{keep_idx+1}]")
                break
            elif choice == 'r' and 'recommendation' in locals() and len(recommendation['alternative_suggestions']) > 0:
                # Use alternative recommendation
                alt = recommendation['alternative_suggestions'][0]
                keep_idx = alt['index']
                
                # Check if symlinks were recommended and enabled
                if (self.config.enable_symlinks and 
                    'safety_analysis' in locals() and 
                    safety_analysis['recommended_strategy'] == 'symlink' and 
                    safety_analysis['confidence'] > 70.0):
                    
                    # Set up symlink replacement
                    group.symlink_target_index = keep_idx
                    group.selected_for_symlink = {i for i in range(group.count) if i != keep_idx}
                    group.selected_for_deletion.clear()
                    print(f"    🔗 Alternative symlink choice accepted. Target: [{keep_idx+1}]")
                else:
                    # Set up deletion
                    group.selected_for_deletion = {i for i in range(group.count) if i != keep_idx}
                    group.selected_for_symlink.clear()
                    group.symlink_target_index = None
                    print(f"    🗑️ Alternative deletion choice accepted. Keeping [{keep_idx+1}]")
                break
            elif choice == 's' and 'recommendation' in locals():
                # Show detailed scoring breakdown
                self._show_detailed_scores(recommendation['all_scores'])
                continue
            elif choice == '1':
                # Keep file [1], use safety recommendation for strategy
                keep_idx = 0
                if ('safety_analysis' in locals() and 
                    safety_analysis['recommended_strategy'] == 'symlink' and 
                    safety_analysis['confidence'] > 70.0):
                    group.symlink_target_index = keep_idx
                    group.selected_for_symlink = set(i for i in range(len(group.paths)) if i != keep_idx)
                    group.selected_for_deletion.clear()
                    print(f"    🔗 Keep file [1] as target, symlink others")
                else:
                    group.selected_for_deletion = set(range(1, len(group.paths)))
                    group.selected_for_symlink.clear()
                    group.symlink_target_index = None
                    print(f"    🗑️ Keep file [1], delete others")
                break
            elif choice == '2':
                # Keep file [2], use safety recommendation for strategy
                keep_idx = 1
                if ('safety_analysis' in locals() and 
                    safety_analysis['recommended_strategy'] == 'symlink' and 
                    safety_analysis['confidence'] > 70.0):
                    group.symlink_target_index = keep_idx
                    group.selected_for_symlink = set(i for i in range(len(group.paths)) if i != keep_idx)
                    group.selected_for_deletion.clear()
                    print(f"    🔗 Keep file [2] as target, symlink others")
                else:
                    group.selected_for_deletion = set(i for i in range(len(group.paths)) if i != 1)
                    group.selected_for_symlink.clear()
                    group.symlink_target_index = None
                    print(f"    🗑️ Keep file [2], delete others")
                break
            elif choice == '3' and len(group.paths) > 2:
                # Keep file [3], use safety recommendation for strategy
                keep_idx = 2
                if ('safety_analysis' in locals() and 
                    safety_analysis['recommended_strategy'] == 'symlink' and 
                    safety_analysis['confidence'] > 70.0):
                    group.symlink_target_index = keep_idx
                    group.selected_for_symlink = set(i for i in range(len(group.paths)) if i != keep_idx)
                    group.selected_for_deletion.clear()
                    print(f"    🔗 Keep file [3] as target, symlink others")
                else:
                    group.selected_for_deletion = set(i for i in range(len(group.paths)) if i != 2)
                    group.selected_for_symlink.clear()
                    group.symlink_target_index = None
                    print(f"    🗑️ Keep file [3], delete others")
                break
            elif choice == 'p':
                self._select_keep_priority(group)
                break
            elif choice == 'sl':
                # Force symlinks with recommended file as target
                if 'recommendation' in locals():
                    keep_idx = best_file['index']
                    group.symlink_target_index = keep_idx
                    group.selected_for_symlink = {i for i in range(len(group.paths)) if i != keep_idx}
                    group.selected_for_deletion.clear()
                    print(f"    🔗 Forced symlinks. Target: [{keep_idx+1}]")
                else:
                    print(f"    Error: No recommendation available for symlink forcing")
                break
            elif choice == 'dl':
                # Force deletion with recommended file kept
                if 'recommendation' in locals():
                    keep_idx = best_file['index']
                    group.selected_for_deletion = {i for i in range(len(group.paths)) if i != keep_idx}
                    group.selected_for_symlink.clear()
                    group.symlink_target_index = None
                    print(f"    🗑️ Forced deletion. Keeping: [{keep_idx+1}]")
                else:
                    print(f"    Error: No recommendation available for deletion forcing")
                break
            elif choice == 'm':
                if self._manual_selection(group):
                    break
            else:
                print("Invalid choice. Please try again.")
    
    def _show_detailed_scores(self, scored_files: List[Dict]) -> None:
        """Show detailed scoring breakdown for each file"""
        print(f"\n{'='*70}")
        print(f"DETAILED SCORING BREAKDOWN")
        print(f"{'='*70}")
        
        for file_data in scored_files:
            path = file_data['path']
            score = file_data['score']
            
            print(f"\n[{file_data['index']+1}] {path.name}")
            print(f"    Path: {path}")
            print(f"    Total Score: {score.total_score:.1f}/100")
            print(f"    ├─ Path Quality: {score.path_quality:.1f}/100 (weight: 40%)")
            print(f"    ├─ File Integrity: {score.file_integrity:.1f}/100 (weight: 25%)")
            print(f"    ├─ Usage Context: {score.usage_context:.1f}/100 (weight: 20%)")
            print(f"    └─ Metadata Quality: {score.metadata_quality:.1f}/100 (weight: 15%)")
            
            if score.reasoning:
                print(f"    Reasoning:")
                for reason in score.reasoning[:5]:  # Show top 5 reasons
                    print(f"      • {reason}")
        
        input("\nPress Enter to continue...")
    
    def _manual_selection(self, group: DuplicateGroup) -> bool:
        """Manual file selection within a group"""
        group.selected_for_deletion.clear()
        
        while True:
            print(f"\nManual Selection:")
            print(f"  Enter file numbers to DELETE (e.g., 1,3,4)")
            print(f"  'all' to select all but first")
            print(f"  'none' to clear selection")
            print(f"  'done' to finish")
            print(f"  'back' to return to auto options")
            
            choice = input("\nSelect files to DELETE: ").lower().strip()
            
            if choice == 'back':
                return False
            elif choice == 'done':
                if not group.selected_for_deletion:
                    print("No files selected.")
                    continue
                if len(group.selected_for_deletion) >= group.count:
                    print("Error: Cannot delete all copies.")
                    continue
                return True
            elif choice == 'all':
                group.selected_for_deletion = set(range(1, group.count))
            elif choice == 'none':
                group.selected_for_deletion.clear()
            else:
                try:
                    indices = []
                    for part in choice.split(','):
                        part = part.strip()
                        if '-' in part:
                            start, end = map(int, part.split('-'))
                            indices.extend(range(start-1, end))
                        else:
                            indices.append(int(part) - 1)
                    
                    valid_indices = {idx for idx in indices if 0 <= idx < group.count}
                    
                    if len(valid_indices) >= group.count:
                        print("Error: Cannot delete all copies.")
                        continue
                    
                    group.selected_for_deletion = valid_indices
                except ValueError:
                    print("Invalid input. Use numbers separated by commas.")
    
    def execute_deletions(self, groups: List[DuplicateGroup], skip_confirmation: bool = False) -> bool:
        """Execute planned deletions"""
        total_files = sum(len(g.selected_for_deletion) for g in groups)
        total_savings = sum(g.potential_savings for g in groups)
        
        if total_files == 0:
            print("\nNo files selected for deletion.")
            return True
        
        if total_files > self.config.delete_max_files:
            print(f"Error: Too many files selected ({total_files}). Max: {self.config.delete_max_files}")
            return False
        
        # Show preview
        print(f"\n{'='*60}")
        print(f"DELETION {'PREVIEW' if self.config.delete_dry_run else 'EXECUTION'}")
        print(f"{'='*60}")
        print(f"Files to delete: {total_files}")
        print(f"Space to free: {format_size(total_savings)}")
        print(f"Mode: {'DRY RUN' if self.config.delete_dry_run else 'LIVE'}")
        
        if not skip_confirmation and self.config.delete_confirm and not self.config.delete_dry_run:
            print(f"\n⚠️  WARNING: About to delete {total_files} files!")
            confirm = input("Type 'DELETE' to confirm: ")
            if confirm != 'DELETE':
                print("Deletion cancelled.")
                return False
        
        # Execute deletions
        deleted_count = 0
        errors = []
        
        for group in groups:
            if not group.selected_for_deletion:
                continue
            
            for file_idx in sorted(group.selected_for_deletion):
                path = group.paths[file_idx]
                
                try:
                    if self.config.delete_dry_run:
                        print(f"  [DRY RUN] Would delete: {path}")
                    else:
                        if self.config.delete_backup:
                            # Move to backup
                            backup_path = self.backup_dir / f"{deleted_count}_{path.name}"
                            shutil.move(str(path), str(backup_path))
                            print(f"  Deleted: {path} -> backup")
                        else:
                            # Permanent deletion
                            path.unlink()
                            print(f"  Deleted: {path}")
                        
                        self.deleted_files.append(str(path))
                    
                    deleted_count += 1
                    
                except Exception as e:
                    error_msg = f"Failed to delete {path}: {e}"
                    errors.append(error_msg)
                    print(f"  ERROR: {error_msg}")
        
        # Summary
        print(f"\n{'='*60}")
        print(f"DELETION SUMMARY")
        print(f"{'='*60}")
        print(f"Files processed: {deleted_count}")
        print(f"Errors: {len(errors)}")
        
        if not self.config.delete_dry_run:
            print(f"Space freed: {format_size(total_savings)}")
            if self.config.delete_backup and self.deleted_files:
                print(f"Backup directory: {self.backup_dir}")
        
        return len(errors) == 0
    
    def execute_mixed_operations(self, deletion_groups: List[DuplicateGroup], symlink_groups: List[DuplicateGroup]) -> bool:
        """Execute both deletion and symlink operations with unified confirmation"""
        total_deletions = sum(len(g.selected_for_deletion) for g in deletion_groups)
        total_symlinks = sum(len(g.selected_for_symlink) for g in symlink_groups)
        
        if total_deletions == 0 and total_symlinks == 0:
            print("\nNo operations selected.")
            return True
        
        # Calculate savings
        deletion_savings = sum(g.potential_savings for g in deletion_groups)
        symlink_savings = sum(g.potential_symlink_savings for g in symlink_groups)
        total_savings = deletion_savings + symlink_savings
        
        # Show unified preview
        print(f"\n{'='*60}")
        if total_deletions > 0 and total_symlinks > 0:
            print(f"MIXED OPERATIONS {'PREVIEW' if self.config.delete_dry_run else 'EXECUTION'}")
        elif total_symlinks > 0:
            print(f"SYMLINK {'PREVIEW' if self.config.delete_dry_run else 'EXECUTION'}")
        else:
            print(f"DELETION {'PREVIEW' if self.config.delete_dry_run else 'EXECUTION'}")
        print(f"{'='*60}")
        
        if total_deletions > 0:
            print(f"Files to delete: {total_deletions}")
        if total_symlinks > 0:
            print(f"Files to symlink: {total_symlinks}")
        print(f"Space to save: {format_size(total_savings)}")
        print(f"Mode: {'DRY RUN' if self.config.delete_dry_run else 'LIVE'}")
        
        # Unified confirmation
        if self.config.delete_confirm and not self.config.delete_dry_run:
            if total_deletions > 0 and total_symlinks > 0:
                print(f"\n🔗 About to create {total_symlinks} symlinks and delete {total_deletions} files!")
                confirm = input("Type 'EXECUTE' to confirm: ")
            elif total_symlinks > 0:
                print(f"\n🔗 About to create {total_symlinks} symlinks!")
                confirm = input("Type 'SYMLINK' to confirm: ")
            else:
                print(f"\n⚠️  WARNING: About to delete {total_deletions} files!")
                confirm = input("Type 'DELETE' to confirm: ")
                
            expected = 'EXECUTE' if (total_deletions > 0 and total_symlinks > 0) else ('SYMLINK' if total_symlinks > 0 else 'DELETE')
            if confirm != expected:
                print("Operation cancelled.")
                return False
        
        # Execute operations
        success = True
        if deletion_groups:
            success &= self.execute_deletions(deletion_groups, skip_confirmation=True)
        if symlink_groups and self.config.enable_symlinks:
            success &= self.execute_symlinks(symlink_groups, skip_confirmation=True)
            
        return success

    def execute_symlinks(self, groups: List[DuplicateGroup], skip_confirmation: bool = False) -> bool:
        """Execute symlink replacement operations"""
        if not self.config.enable_symlinks or not self.symlink_manager:
            print("Error: Symlink operations not enabled or supported.")
            return False
        
        total_symlinks = sum(len(g.selected_for_symlink) for g in groups)
        total_savings = sum(g.potential_symlink_savings for g in groups)
        
        if total_symlinks == 0:
            print("\nNo files selected for symlink replacement.")
            return True
        
        # Show preview
        print(f"\n{'='*60}")
        print(f"SYMLINK {'PREVIEW' if self.config.symlink_dry_run or self.config.delete_dry_run else 'EXECUTION'}")
        print(f"{'='*60}")
        print(f"Files to symlink: {total_symlinks}")
        print(f"Space to save: {format_size(total_savings)}")
        print(f"Mode: {'DRY RUN' if self.config.symlink_dry_run or self.config.delete_dry_run else 'LIVE'}")
        
        if not skip_confirmation and self.config.delete_confirm and not (self.config.symlink_dry_run or self.config.delete_dry_run):
            print(f"\n🔗 About to create {total_symlinks} symlinks!")
            confirm = input("Type 'SYMLINK' to confirm: ")
            if confirm != 'SYMLINK':
                print("Symlink operation cancelled.")
                return False
        
        # Execute symlinks
        operation_id = f"batch_{int(time.time())}"
        symlink_count = 0
        errors = []
        
        for group in groups:
            if not group.selected_for_symlink or group.symlink_target_index is None:
                continue
            
            target_path = group.paths[group.symlink_target_index]
            
            # Verify target still exists
            if not target_path.exists():
                error_msg = f"Target file no longer exists: {target_path}"
                errors.append(error_msg)
                print(f"  ERROR: {error_msg}")
                continue
            
            for file_idx in sorted(group.selected_for_symlink):
                source_path = group.paths[file_idx]
                
                try:
                    if self.config.symlink_dry_run or self.config.delete_dry_run:
                        print(f"  [DRY RUN] Would create symlink: {source_path} -> {target_path}")
                    else:
                        print(f"  Creating symlink: {source_path} -> {target_path}")
                        
                        # Safety check
                        if not source_path.exists():
                            print(f"    Warning: Source file no longer exists")
                            continue
                        
                        # Create the symlink
                        operation = self.symlink_manager.create_symlink(
                            source_path, target_path, operation_id
                        )
                        
                        if not operation.success:
                            error_msg = f"Failed to create symlink {source_path}: {operation.error_message}"
                            errors.append(error_msg)
                            print(f"    ERROR: {error_msg}")
                            continue
                    
                    symlink_count += 1
                    
                except Exception as e:
                    error_msg = f"Failed to create symlink {source_path}: {e}"
                    errors.append(error_msg)
                    print(f"  ERROR: {error_msg}")
        
        # Summary
        print(f"\n{'='*60}")
        print(f"SYMLINK SUMMARY")
        print(f"{'='*60}")
        print(f"Symlinks created: {symlink_count}")
        print(f"Errors: {len(errors)}")
        
        if not (self.config.symlink_dry_run or self.config.delete_dry_run) and self.symlink_manager:
            stats = self.symlink_manager.get_statistics()
            print(f"Space saved: {format_size(stats['total_space_saved'])}")
            print(f"Success rate: {stats['success_rate']:.1%}")
            
            if hasattr(self.symlink_manager, 'log_path'):
                print(f"Operation log: {self.symlink_manager.log_path}")
        
        return len(errors) == 0

# ---------------------------
# Main Scanner
# ---------------------------

class UltimateScanner:
    """The ultimate file scanner with three-phase optimization"""
    
    def __init__(self, config: Config):
        config.validate()
        self.config = config
        self.stats = ScanStats()
        self.db = DatabaseManager(config.db_path, config.reset_db)
        self.hasher = HashComputer(
            config.hash_algorithm,
            config.chunk_size,
            config.retry_attempts,
            config.retry_backoff
        )
        
        # Initialize cache manager if enabled
        self.cache = None
        self.incremental_scanner = None
        if config.enable_cache:
            try:
                from cache_manager import CacheManager, IncrementalScanner
                self.cache = CacheManager(
                    config.cache_path,
                    config.cache_max_size_mb,
                    config.cache_cleanup_hours
                )
                self.incremental_scanner = IncrementalScanner(self.cache)
                logger.info(f"Cache system initialized: {config.cache_path}")
            except ImportError:
                logger.warning("Cache manager not available, proceeding without caching")
            except Exception as e:
                logger.warning(f"Failed to initialize cache: {e}")
        
        # Initialize GPU acceleration if enabled
        self.gpu_manager = None
        if config.enable_gpu:
            try:
                from gpu_accelerator import GPUAccelerationManager
                self.gpu_manager = GPUAccelerationManager(config.gpu_backend)
                
                if self.gpu_manager.is_gpu_available():
                    gpu_info = self.gpu_manager.get_acceleration_info()
                    logger.info(f"GPU acceleration initialized: {gpu_info.get('backend')} - {gpu_info.get('name', 'Unknown device')}")
                else:
                    logger.info("GPU acceleration requested but not available, using CPU")
                    
            except ImportError:
                logger.warning("GPU acceleration not available, proceeding with CPU")
            except Exception as e:
                logger.warning(f"Failed to initialize GPU acceleration: {e}")
        
        self.filter = PathFilter(config)
        self.progress = ProgressTracker(config, self.stats)
        self.db_queue: queue.Queue = queue.Queue(maxsize=config.workers * 10)
        self.stop_sentinel = object()
        
        # Set logging level
        if config.quiet:
            logger.setLevel(logging.WARNING)
        elif config.verbose:
            logger.setLevel(logging.DEBUG)
    
    def scan(self) -> ScanStats:
        """Execute the complete scan"""
        logger.info(f"TurboDedup v{__version__}")
        logger.info(f"Algorithm: {self.config.hash_algorithm} | Workers: {self.config.workers}")
        logger.info(f"Min size: {self.config.min_size / (1024**2):.1f} MB")
        
        # Display enabled features
        features = []
        if self.config.enable_cache:
            features.append("cache")
        if self.config.enable_gpu and self.gpu_manager and self.gpu_manager.is_gpu_available():
            gpu_info = self.gpu_manager.get_acceleration_info()
            features.append(f"gpu-{gpu_info.get('backend', 'unknown')}")
        if self.config.enable_similarity:
            features.append("similarity")
        
        if features:
            logger.info(f"Features: {', '.join(features)}")
        
        try:
            # Phase 1: Discovery
            self.stats.start_phase("discovery")
            self._phase1_discovery()
            self.stats.end_phase("discovery")
            
            # Phase 2: Partial hash
            self.stats.start_phase("partial_hash")
            self._phase2_partial_hash()
            self.stats.end_phase("partial_hash")
            
            # Phase 3: Full hash
            self.stats.start_phase("full_hash")
            self._phase3_full_hash()
            self.stats.end_phase("full_hash")
            
            # Phase 4: Similarity detection (optional)
            if self.config.enable_similarity:
                self.stats.start_phase("similarity")
                self._phase4_similarity_detection()
                self.stats.end_phase("similarity")
            
            # Phase 5: Verify (optional)
            if self.config.verify_duplicates:
                self.stats.start_phase("verify")
                self._phase5_verify()
                self.stats.end_phase("verify")
            
            # Analyze results
            self._analyze_results()
            
            # Export if requested
            if self.config.export_format:
                self._export_results()
                
        except KeyboardInterrupt:
            logger.warning("Scan interrupted by user")
        except Exception as e:
            logger.error(f"Scan failed: {e}", exc_info=True)
            self.stats.add_error(f"Fatal: {e}")
        
        return self.stats
    
    def _phase1_discovery(self) -> None:
        """Discover and catalog files"""
        logger.info("Phase 1: Discovering files...")
        
        # Start database writer
        writer = threading.Thread(
            target=self.db.writer_thread,
            args=(self.db_queue, self.stop_sentinel, self.config.batch_size),
            daemon=True
        )
        writer.start()
        
        try:
            for scan_path in self.config.scan_paths:
                if not os.path.exists(scan_path):
                    logger.warning(f"Path does not exist: {scan_path}")
                    continue
                
                logger.info(f"Scanning: {scan_path}")
                self._scan_directory(scan_path)
                
        finally:
            self.db_queue.put(self.stop_sentinel)
            self.db_queue.join()
            writer.join()
        
        logger.info(
            f"Phase 1 complete: {self.stats.files_processed:,} files cataloged "
            f"({self.stats.files_excluded:,} excluded)"
        )
    
    def _scan_directory(self, start_path: str) -> None:
        """Scan a directory tree"""
        for root, dirs, files in os.walk(start_path, topdown=True, followlinks=self.config.follow_symlinks):
            # Check directory exclusions
            if self.filter.is_excluded_dir(root):
                dirs.clear()  # Don't descend
                continue
            
            # Filter subdirectories
            dirs[:] = [d for d in dirs if not self.filter.is_excluded_dir(os.path.join(root, d))]
            
            # Process files
            for filename in files:
                self.stats.files_discovered += 1
                
                try:
                    path = Path(root) / filename
                    stat = path.stat()
                    size = stat.st_size
                    mtime = stat.st_mtime
                    
                    should_process, exclusion_reason = self.filter.should_process_file(path, size)
                    if not should_process:
                        self.stats.files_excluded += 1
                        if hasattr(self.stats, 'exclusions'):
                            self.stats.exclusions[exclusion_reason] = self.stats.exclusions.get(exclusion_reason, 0) + 1
                        if self.config.show_scanned and exclusion_reason:
                            logger.debug(f"Excluded {path}: {exclusion_reason}")
                        continue
                    
                    # Track statistics
                    self.stats.bytes_discovered += size
                    if not self.stats.largest_file or size > self.stats.largest_file[1]:
                        self.stats.largest_file = (str(path), size)
                    
                    # Queue for database
                    self.db_queue.put((str(path), size, mtime, None, None, None))
                    self.stats.files_processed += 1
                    
                    # Progress update
                    self.progress.update()
                    
                except OSError as e:
                    self.stats.add_error(f"{path}: {e}")
    
    def _phase2_partial_hash(self) -> None:
        """Compute partial hashes for large files"""
        candidates = list(self.db.get_partial_hash_candidates(self.config.partial_hash_threshold))
        
        if not candidates:
            logger.info("Phase 2: No large files need partial hashing")
            return
        
        logger.info(f"Phase 2: Computing partial hashes for {len(candidates):,} large files...")
        
        updates = []
        completed = 0
        
        with ThreadPoolExecutor(max_workers=self.config.workers) as executor:
            futures = {}
            
            for size, path_str in candidates:
                # Check cache first if available
                if self.incremental_scanner:
                    _, cached_partial_hash, _ = self.incremental_scanner.get_cached_hashes(path_str)
                    if cached_partial_hash:
                        # Use cached partial hash
                        updates.append((cached_partial_hash, None, path_str))
                        self.stats.files_partial_hashed += 1
                        completed += 1
                        continue
                
                # Compute partial hash if not cached
                future = executor.submit(
                    self.hasher.compute_partial_hash,
                    Path(path_str),
                    self.config.partial_segment_size
                )
                futures[future] = path_str
            
            for future in as_completed(futures):
                path_str = futures[future]
                phash, error, bytes_read = future.result()
                
                # Store in cache if successful
                if phash and self.incremental_scanner:
                    self.incremental_scanner.store_computed_hashes(path_str, partial_hash=phash)
                
                updates.append((phash, error, path_str))
                self.stats.files_partial_hashed += 1
                self.stats.bytes_read += bytes_read
                completed += 1
                
                if len(updates) >= self.config.batch_size:
                    self.db.update_hashes(updates, "phash")
                    updates.clear()
                
                if completed % 10 == 0:
                    pct = (completed / len(candidates)) * 100
                    logger.info(f"Partial hash progress: {completed}/{len(candidates)} ({pct:.1f}%)")
        
        if updates:
            self.db.update_hashes(updates, "phash")
        
        logger.info(f"Phase 2 complete: {self.stats.files_partial_hashed:,} files partial hashed")
    
    def _phase3_full_hash(self) -> None:
        """Compute full hashes for candidates"""
        candidates = list(self.db.get_full_hash_candidates(self.config.partial_hash_threshold))
        
        if not candidates:
            logger.info("Phase 3: No files need full hashing")
            return
        
        logger.info(f"Phase 3: Computing full hashes for {len(candidates):,} candidates...")
        
        updates = []
        completed = 0
        
        # Use GPU acceleration for batch processing if available
        if self.gpu_manager and self.gpu_manager.is_gpu_available() and len(candidates) >= self.config.gpu_batch_size:
            self._process_full_hash_gpu_batches(candidates, updates, completed)
        else:
            # Fall back to CPU processing
            with ThreadPoolExecutor(max_workers=self.config.workers) as executor:
                futures = {}
                
                for path_str in candidates:
                    # Check cache first if available
                    if self.incremental_scanner:
                        cached_hash, _, _ = self.incremental_scanner.get_cached_hashes(path_str)
                        if cached_hash:
                            # Use cached hash
                            updates.append((cached_hash, None, path_str))
                            self.stats.files_full_hashed += 1
                            completed += 1
                            continue
                    
                    # Compute hash if not cached
                    future = executor.submit(
                        self.hasher.compute_full_hash,
                        Path(path_str)
                    )
                    futures[future] = path_str
            
            for future in as_completed(futures):
                path_str = futures[future]
                hash_val, error, bytes_read = future.result()
                
                # Store in cache if successful
                if hash_val and self.incremental_scanner:
                    self.incremental_scanner.store_computed_hashes(path_str, md5_hash=hash_val)
                
                updates.append((hash_val, error, path_str))
                self.stats.files_full_hashed += 1
                self.stats.bytes_read += bytes_read
                completed += 1
                
                if len(updates) >= self.config.batch_size:
                    self.db.update_hashes(updates, "hash")
                    updates.clear()
                
                if completed % 10 == 0 or completed == len(candidates):
                    pct = (completed / len(candidates)) * 100
                    mb_read = self.stats.bytes_read / (1024 * 1024)
                    logger.info(
                        f"Full hash progress: {completed}/{len(candidates)} ({pct:.1f}%) "
                        f"| {mb_read:.1f} MB read"
                    )
        
        if updates:
            self.db.update_hashes(updates, "hash")
        
        logger.info(f"Phase 3 complete: {self.stats.files_full_hashed:,} files full hashed")
    
    def _process_full_hash_gpu_batches(self, candidates: List[str], updates: List, completed: int) -> None:
        """Process full hash computation using GPU in batches"""
        logger.info("Using GPU acceleration for hash computation")
        
        # Process candidates in batches
        batch_size = self.config.gpu_batch_size
        
        for i in range(0, len(candidates), batch_size):
            batch_candidates = candidates[i:i + batch_size]
            
            # Filter out cached files
            batch_paths = []
            cached_results = []
            
            for path_str in batch_candidates:
                if self.incremental_scanner:
                    cached_hash, _, _ = self.incremental_scanner.get_cached_hashes(path_str)
                    if cached_hash:
                        cached_results.append((cached_hash, None, path_str))
                        continue
                
                batch_paths.append(Path(path_str))
            
            # Add cached results to updates
            updates.extend(cached_results)
            completed += len(cached_results)
            
            # Process remaining files with GPU
            if batch_paths:
                try:
                    gpu_results = self.gpu_manager.compute_hashes_batch(
                        batch_paths, 
                        self.config.gpu_chunk_size
                    )
                    
                    for path_str, hash_val in gpu_results:
                        error = None if hash_val else "GPU computation failed"
                        
                        # Store in cache if successful
                        if hash_val and self.incremental_scanner:
                            self.incremental_scanner.store_computed_hashes(path_str, md5_hash=hash_val)
                        
                        updates.append((hash_val, error, path_str))
                        self.stats.files_full_hashed += 1
                        completed += 1
                    
                    # Update database periodically
                    if len(updates) >= self.config.batch_size:
                        self.db.update_hashes(updates, "hash")
                        updates.clear()
                    
                    # Progress reporting
                    pct = (completed / len(candidates)) * 100
                    logger.info(f"GPU hash progress: {completed}/{len(candidates)} ({pct:.1f}%)")
                    
                except Exception as e:
                    logger.warning(f"GPU batch processing failed, falling back to CPU: {e}")
                    
                    # Fallback to CPU for this batch
                    with ThreadPoolExecutor(max_workers=self.config.workers) as executor:
                        futures = {
                            executor.submit(self.hasher.compute_full_hash, path): str(path)
                            for path in batch_paths
                        }
                        
                        for future in as_completed(futures):
                            path_str = futures[future]
                            hash_val, error, bytes_read = future.result()
                            
                            if hash_val and self.incremental_scanner:
                                self.incremental_scanner.store_computed_hashes(path_str, md5_hash=hash_val)
                            
                            updates.append((hash_val, error, path_str))
                            self.stats.files_full_hashed += 1
                            self.stats.bytes_read += bytes_read
                            completed += 1
    
    def _phase4_similarity_detection(self) -> None:
        """Phase 4: Compute similarity hashes for enabled types"""
        if not self.config.enable_similarity:
            return
        
        # Import similarity detector (optional dependency)
        try:
            from similarity_detector import get_similarity_manager
        except ImportError:
            logger.warning("Similarity detection requested but similarity_detector module not available")
            return
        
        logger.info("Phase 4: Computing similarity hashes...")
        
        # Initialize similarity manager
        similarity_manager = get_similarity_manager()
        
        # Configure enabled similarity types
        if self.config.image_similarity:
            similarity_manager.enable_similarity_type('image')
        if self.config.audio_similarity:
            similarity_manager.enable_similarity_type('audio') 
        if self.config.document_similarity:
            similarity_manager.enable_similarity_type('document')
        
        # Get all files for similarity analysis
        with sqlite3.connect(self.db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT path FROM files 
                WHERE error IS NULL 
                ORDER BY size DESC
            """)
            file_paths = [Path(row[0]) for row in cursor.fetchall()]
        
        if not file_paths:
            logger.info("Phase 4: No files available for similarity detection")
            return
        
        processed = 0
        similarity_hashes_computed = 0
        
        # Compute similarity hashes
        for file_path in file_paths:
            try:
                similarity_hashes = similarity_manager.compute_similarity_hash(file_path)
                
                if similarity_hashes:
                    # Store in database
                    self.db.store_similarity_hashes(str(file_path), similarity_hashes)
                    similarity_hashes_computed += len(similarity_hashes)
                
                processed += 1
                
                # Progress update
                if processed % 100 == 0 or processed == len(file_paths):
                    pct = (processed / len(file_paths)) * 100
                    logger.info(f"Similarity detection: {processed}/{len(file_paths)} ({pct:.1f}%) | {similarity_hashes_computed} hashes computed")
            
            except Exception as e:
                logger.warning(f"Failed to compute similarity hashes for {file_path}: {e}")
        
        logger.info(f"Phase 4 complete: {similarity_hashes_computed} similarity hashes computed for {processed} files")
    
    def _phase5_verify(self) -> None:
        """Verify duplicates with byte comparison"""
        logger.info("Phase 4: Verifying duplicates...")
        
        duplicates = self.db.get_duplicates()
        verified = 0
        mismatches = 0
        
        for size, hash_val, count, paths in duplicates:
            if count < 2:
                continue
                
            base_path = Path(paths[0])
            for other_path_str in paths[1:]:
                if self.hasher.verify_identical(base_path, Path(other_path_str)):
                    verified += 1
                else:
                    mismatches += 1
                    logger.warning(f"Hash collision: {base_path} vs {other_path_str}")
                
                self.stats.files_verified += 1
        
        logger.info(f"Phase 4 complete: {verified} verified, {mismatches} mismatches")
    
    def _analyze_results(self) -> None:
        """Analyze scan results"""
        duplicates = self.db.get_duplicates()
        
        self.stats.duplicate_sets = len(duplicates)
        self.stats.duplicate_files = sum(count for _, _, count, _ in duplicates)
        self.stats.wasted_space = sum(size * (count - 1) for size, _, count, _ in duplicates)
        
        logger.info(
            f"Found {self.stats.duplicate_sets} duplicate sets "
            f"({self.stats.wasted_space / (1024**3):.2f} GB wasted)"
        )
        
        # Display cache statistics if caching is enabled
        if self.cache and self.config.enable_cache:
            cache_stats = self.cache.get_cache_stats()
            if 'error' not in cache_stats:
                hit_rate = cache_stats.get('hit_rate_percent', 0)
                entry_count = cache_stats.get('entry_count', 0)
                cache_size_mb = cache_stats.get('total_size_mb', 0)
                bytes_saved_mb = cache_stats.get('bytes_saved_mb', 0)
                
                logger.info(
                    f"Cache stats: {entry_count:,} entries, {cache_size_mb:.1f} MB cache size, "
                    f"{hit_rate:.1f}% hit rate, {bytes_saved_mb:.1f} MB I/O saved"
                )
    
    def _export_results(self) -> None:
        """Export results to file"""
        duplicates = self.db.get_duplicates()
        
        if self.config.export_format == "csv":
            self._export_csv(duplicates)
        elif self.config.export_format == "json":
            self._export_json(duplicates)
        
        logger.info(f"Results exported to: {self.config.export_path}")
    
    def _export_csv(self, duplicates: List[Tuple]) -> None:
        """Export to CSV"""
        output_path = self.config.export_path or "duplicates.csv"
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Hash", "Size (MB)", "Count", "Wasted (MB)", "Paths"])
            
            for size, hash_val, count, paths in duplicates:
                size_mb = size / (1024 * 1024)
                wasted_mb = size_mb * (count - 1)
                writer.writerow([
                    hash_val,
                    f"{size_mb:.2f}",
                    count,
                    f"{wasted_mb:.2f}",
                    "; ".join(paths)
                ])
    
    def _export_json(self, duplicates: List[Tuple]) -> None:
        """Export to JSON"""
        output_path = self.config.export_path or "duplicates.json"
        
        data = {
            "scan_info": {
                "version": __version__,
                "date": datetime.now().isoformat(),
                "files_scanned": self.stats.files_processed,
                "duplicate_sets": self.stats.duplicate_sets,
                "wasted_space_gb": self.stats.wasted_space / (1024**3),
            },
            "duplicates": [
                {
                    "hash": hash_val,
                    "size": size,
                    "count": count,
                    "paths": paths
                }
                for size, hash_val, count, paths in duplicates
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def _load_duplicate_groups(self) -> List[DuplicateGroup]:
        """Load duplicate groups from database"""
        groups = []
        duplicates = self.db.get_duplicates()
        
        for size, hash_val, count, paths in duplicates:
            if count > 1:
                path_objects = [Path(p) for p in paths]
                groups.append(DuplicateGroup(size, hash_val, path_objects))
        
        return groups

# ---------------------------
# Report Generator
# ---------------------------

def display_scanned_summary(db: DatabaseManager) -> None:
    """Display summary of scanned folders and files"""
    print("\n" + "="*50)
    print("SCANNED FOLDERS SUMMARY")
    print("="*50)
    
    # Get folder summary from database
    with sqlite3.connect(db.db_path) as conn:
        cursor = conn.cursor()
        
        # Windows-style path parsing
        cursor.execute("""
            SELECT 
                CASE 
                    WHEN INSTR(SUBSTR(path, 4), '\') > 0 
                    THEN SUBSTR(path, 1, INSTR(SUBSTR(path, 4), '\') + 2)
                    ELSE path
                END as folder_path,
                COUNT(*) as file_count,
                ROUND(SUM(size)/1024.0/1024.0, 2) as size_mb
            FROM files 
            GROUP BY folder_path 
            ORDER BY file_count DESC
        """)
        
        folders = cursor.fetchall()
        
        if not folders:
            print("No folders found in scan results.")
            return
        
        print(f"\nFound {len(folders)} folders with files:")
        print(f"{'Folder':<50} {'Files':<8} {'Size (MB)':<12}")
        print("-" * 70)
        
        total_files = 0
        total_size = 0
        
        for folder, file_count, size_mb in folders[:20]:  # Show top 20 folders
            print(f"{folder:<50} {file_count:<8} {size_mb:<12}")
            total_files += file_count
            total_size += size_mb or 0
        
        if len(folders) > 20:
            remaining = len(folders) - 20
            print(f"... and {remaining} more folders")
        
        print("-" * 70)
        print(f"{'TOTAL':<50} {total_files:<8} {total_size:<12.2f}")
        
        # Show sample files from largest folder
        if folders:
            largest_folder = folders[0][0]
            print(f"\nSample files from largest folder ({largest_folder}):")
            cursor.execute("""
                SELECT path, ROUND(size/1024.0/1024.0, 2) as size_mb 
                FROM files 
                WHERE path LIKE ? 
                ORDER BY size DESC 
                LIMIT 5
            """, (f"{largest_folder}%",))
            
            sample_files = cursor.fetchall()
            for file_path, size_mb in sample_files:
                filename = file_path.split('\\')[-1] if '\\' in file_path else file_path
                print(f"  {filename:<40} {size_mb} MB")

def generate_report(config: Config, stats: ScanStats, db: DatabaseManager) -> None:
    """Generate comprehensive report"""
    duration = stats.get_duration()
    
    print("\n" + "="*70)
    print("ULTIMATE SCANNER REPORT")
    print("="*70)
    
    # Summary
    print(f"\nScan Summary:")
    print(f"  Duration: {timedelta(seconds=int(duration))}")
    print(f"  Files discovered: {stats.files_discovered:,}")
    print(f"  Files processed: {stats.files_processed:,}")
    print(f"  Files excluded: {stats.files_excluded:,}")
    
    # Exclusion breakdown
    if stats.exclusions:
        print(f"\nExclusion breakdown:")
        for reason, count in sorted(stats.exclusions.items(), key=lambda x: x[1], reverse=True):
            if ":" in reason:
                category, detail = reason.split(":", 1)
                if category == "below_min_size":
                    detail = f"< {format_size(int(detail))}"
                elif category == "above_max_size":
                    detail = f"> {format_size(int(detail))}"
                print(f"  {category.replace('_', ' ').title()}: {count:,} files ({detail})")
            else:
                print(f"  {reason.replace('_', ' ').title()}: {count:,} files")
    
    # Performance
    print(f"\nPerformance:")
    print(f"  Overall rate: {stats.get_rate():.1f} files/sec")
    if stats.bytes_read > 0:
        mb_read = stats.bytes_read / (1024 * 1024)
        mb_rate = mb_read / duration if duration > 0 else 0
        print(f"  Data read: {mb_read:.1f} MB ({mb_rate:.1f} MB/sec)")
    
    # Phase timing
    print(f"\nPhase Timing:")
    for phase in ["discovery", "partial_hash", "full_hash", "similarity", "verify"]:
        key = f"{phase}_duration"
        if key in stats.phase_times:
            phase_time = stats.phase_times[key]
            print(f"  {phase.replace('_', ' ').title()}: {timedelta(seconds=int(phase_time))}")
    
    # Hash statistics
    if stats.files_partial_hashed > 0 or stats.files_full_hashed > 0:
        print(f"\nHash Operations:")
        if stats.files_partial_hashed > 0:
            print(f"  Partial hashes: {stats.files_partial_hashed:,}")
        print(f"  Full hashes: {stats.files_full_hashed:,}")
        
        # Efficiency calculation
        total_possible = stats.files_processed
        total_hashed = stats.files_full_hashed
        if total_possible > 0:
            efficiency = (1 - (total_hashed / total_possible)) * 100
            print(f"  Efficiency gain: {efficiency:.1f}% fewer full hashes needed")
    
    # Results
    if stats.largest_file:
        path, size = stats.largest_file
        print(f"\nLargest file:")
        print(f"  Path: {path}")
        print(f"  Size: {size / (1024**3):.2f} GB")
    
    # Duplicates
    duplicates = db.get_duplicates()
    if duplicates:
        print(f"\nDuplicates found: {stats.duplicate_sets} sets")
        print(f"Total files involved: {stats.duplicate_files}")
        print(f"Wasted space: {stats.wasted_space / (1024**3):.2f} GB")
        
        # Top duplicates
        print("\nTop duplicate sets:")
        for i, (size, hash_val, count, paths) in enumerate(duplicates[:5]):
            size_mb = size / (1024 * 1024)
            wasted = size * (count - 1) / (1024 * 1024)
            
            print(f"\n{i+1}. {count} copies of {size_mb:.1f} MB file")
            print(f"   Hash: {hash_val[:16]}...")
            print(f"   Wasted: {wasted:.1f} MB")
            
            for j, path in enumerate(paths[:3]):
                print(f"   [{j+1}] {path}")
            
            if len(paths) > 3:
                print(f"   ... and {len(paths) - 3} more")
    else:
        print("\nNo duplicates found!")
    
    # Errors
    if stats.errors:
        print(f"\nErrors encountered: {stats.error_count}")
        print("Recent errors:")
        for error in stats.errors[-5:]:
            print(f"  {error}")
    
    # Export info
    if config.export_path:
        print(f"\nResults exported to: {config.export_path}")
    
    print(f"\nDatabase: {config.db_path}")

# ---------------------------
# CLI Interface
# ---------------------------

def parse_size(size_str: str) -> int:
    """Parse human-readable size"""
    size_str = size_str.strip().upper()
    
    # Order matters: longer suffixes first to avoid 'B' matching 'MB'
    multipliers = [
        ('TB', 1024**4),
        ('GB', 1024**3),
        ('MB', 1024**2),
        ('KB', 1024),
        ('B', 1),
    ]
    
    for suffix, multiplier in multipliers:
        if size_str.endswith(suffix):
            number = size_str[:-len(suffix)].strip()
            return int(float(number) * multiplier)
    
    return int(size_str)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="TurboDedup - High-Performance File Deduplication Scanner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Basic options
    parser.add_argument(
        "--path",
        nargs="+",
        default=["C:\\"],
        help="Paths to scan (can specify multiple)"
    )
    parser.add_argument("--db", default="file_scanner.db", help="Database path")
    parser.add_argument("--min-size", type=parse_size, default="100MB", help="Minimum file size")
    parser.add_argument("--max-size", type=parse_size, help="Maximum file size")
    
    # Performance
    parser.add_argument("--workers", type=int, help="Worker threads (default: auto)")
    parser.add_argument("--chunk-size", type=parse_size, default="1MB", help="Hash chunk size")
    parser.add_argument("--batch-size", type=int, default=1000, help="DB batch size")
    
    # Algorithm
    parser.add_argument(
        "--algorithm",
        choices=["md5", "sha1", "sha256", "xxhash"],
        default="md5",
        help="Hash algorithm"
    )
    
    # Partial hash
    parser.add_argument(
        "--partial-threshold",
        type=parse_size,
        default="512MB",
        help="Size threshold for partial hash"
    )
    parser.add_argument(
        "--partial-segment",
        type=parse_size,
        default="8MB",
        help="Segment size for partial hash"
    )
    
    # Features
    parser.add_argument("--reset", action="store_true", help="Reset database")
    parser.add_argument("--force-rerun", action="store_true", help="Force rerun on previously scanned files")
    parser.add_argument("--verify", action="store_true", help="Verify duplicates")
    parser.add_argument(
        "--export",
        choices=["csv", "json"],
        help="Export format"
    )
    parser.add_argument("--export-path", help="Export file path")
    # Integrated Deletion System
    parser.add_argument("--no-delete", action="store_true", help="Disable integrated deletion (scan only)")
    parser.add_argument("--delete-strategy", 
                       choices=["interactive", "keep_newest", "keep_oldest", "keep_first", "keep_priority", "keep_smart", "skip"],
                       default="interactive",
                       help="Deletion strategy: interactive, keep_smart (AI), keep_newest, keep_oldest, keep_priority, skip (default: interactive)")
    parser.add_argument("--dry-run", action="store_true", help="Preview mode only - don't actually execute recommendations")
    parser.add_argument("--delete-max-files", type=int, default=1000, help="Max files to delete in one operation")
    parser.add_argument("--no-delete-backup", action="store_true", help="Skip backup when deleting")
    parser.add_argument("--no-delete-confirm", action="store_true", help="Skip confirmation prompts")
    
    # Symlink Replacement System
    parser.add_argument("--enable-symlinks", action="store_true", help="Enable symlink replacement for duplicates")
    parser.add_argument("--prefer-symlinks", action="store_true", help="Prefer symlinks over deletion for all strategies")
    parser.add_argument("--symlink-strategy", 
                       choices=["replace_duplicates", "hybrid", "large_files_only"],
                       default="replace_duplicates",
                       help="Symlink replacement strategy (default: replace_duplicates)")
    parser.add_argument("--rollback-symlinks", action="store_true", help="Rollback previous symlink operations")
    
    # Similarity Detection System
    parser.add_argument("--enable-similarity", action="store_true", help="Enable similarity detection for near-duplicates")
    parser.add_argument("--image-similarity", action="store_true", help="Enable perceptual image similarity detection") 
    parser.add_argument("--image-sensitivity", choices=["low", "medium", "high"], default="medium",
                       help="Image similarity sensitivity (default: medium)")
    parser.add_argument("--audio-similarity", action="store_true", help="Enable audio fingerprint similarity detection")
    parser.add_argument("--audio-sensitivity", choices=["low", "medium", "high"], default="medium", 
                       help="Audio similarity sensitivity (default: medium)")
    parser.add_argument("--document-similarity", action="store_true", help="Enable document fuzzy text similarity detection")
    parser.add_argument("--document-sensitivity", choices=["low", "medium", "high"], default="medium",
                       help="Document similarity sensitivity (default: medium)")
    
    # Smart Caching System
    parser.add_argument("--no-cache", action="store_true", help="Disable smart caching system")
    parser.add_argument("--cache-path", default="file_cache.db", help="Path to cache database")
    parser.add_argument("--cache-max-size", type=int, default=1000, help="Maximum cache size in MB")
    parser.add_argument("--rebuild-cache", action="store_true", help="Rebuild cache from scratch")
    parser.add_argument("--clear-cache", action="store_true", help="Clear cache and exit")
    parser.add_argument("--cache-stats", action="store_true", help="Show cache statistics and exit")
    
    # GPU Acceleration System
    parser.add_argument("--enable-gpu", action="store_true", help="Enable GPU acceleration for hash computation")
    parser.add_argument("--gpu-backend", choices=["cuda", "opencl", "auto"], default="auto", 
                       help="GPU backend to use (default: auto)")
    parser.add_argument("--gpu-batch-size", type=int, default=32, help="Number of files to process in GPU batch")
    parser.add_argument("--gpu-chunk-size", type=parse_size, default="4MB", help="Chunk size for GPU processing")
    parser.add_argument("--gpu-info", action="store_true", help="Show GPU acceleration info and exit")
    
    # Legacy compatibility
    parser.add_argument("--interactive-clean", action="store_true", help="(DEPRECATED) Launch legacy interactive cleaner")
    
    # Filters
    parser.add_argument("--include", nargs="+", help="Include only matching files")
    parser.add_argument("--exclude-dir", nargs="+", help="Additional dirs to exclude")
    parser.add_argument("--exclude-ext", nargs="+", help="Additional extensions to exclude")
    
    # Options
    parser.add_argument("--follow-symlinks", action="store_true", help="Follow symbolic links")
    parser.add_argument("--scan-hidden", action="store_true", help="Scan hidden files")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--show-scanned", action="store_true", help="Show summary of scanned folders and files")
    
    # Advanced
    parser.add_argument("--retry-attempts", type=int, default=3, help="Retry attempts")
    parser.add_argument("--retry-backoff", type=float, default=0.2, help="Retry backoff")
    
    args = parser.parse_args()
    
    # Build configuration
    config = Config(
        scan_paths=args.path,
        db_path=args.db,
        min_size=args.min_size,
        max_size=args.max_size,
        workers=args.workers or Config().workers,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        hash_algorithm=args.algorithm,
        partial_hash_threshold=args.partial_threshold,
        partial_segment_size=args.partial_segment,
        reset_db=args.reset,
        force_rerun=args.force_rerun,
        verify_duplicates=args.verify,
        export_format=args.export,
        export_path=args.export_path,
        # Integrated deletion system
        delete_duplicates=not args.no_delete,
        delete_strategy=args.delete_strategy,
        delete_dry_run=args.dry_run,
        delete_max_files=args.delete_max_files,
        delete_backup=not args.no_delete_backup,
        delete_confirm=not args.no_delete_confirm,
        
        # Symlink replacement system
        enable_symlinks=args.enable_symlinks,
        prefer_symlinks=args.prefer_symlinks,
        symlink_strategy=args.symlink_strategy if hasattr(args, 'symlink_strategy') else "replace_duplicates",
        symlink_dry_run=args.dry_run,
        rollback_symlinks=args.rollback_symlinks if hasattr(args, 'rollback_symlinks') else False,
        
        # Similarity detection system  
        enable_similarity=args.enable_similarity or args.image_similarity or args.audio_similarity or args.document_similarity,
        image_similarity=args.image_similarity,
        image_sensitivity=args.image_sensitivity,
        audio_similarity=args.audio_similarity,
        audio_sensitivity=args.audio_sensitivity,
        document_similarity=args.document_similarity,
        document_sensitivity=args.document_sensitivity,
        
        # Smart caching system
        enable_cache=not args.no_cache,
        cache_path=args.cache_path,
        cache_max_size_mb=args.cache_max_size,
        rebuild_cache=args.rebuild_cache,
        clear_cache=args.clear_cache,
        cache_stats=args.cache_stats,
        
        # GPU acceleration system
        enable_gpu=args.enable_gpu,
        gpu_backend=args.gpu_backend,
        gpu_batch_size=args.gpu_batch_size,
        gpu_chunk_size=args.gpu_chunk_size,
        
        # Legacy compatibility
        interactive_clean=args.interactive_clean,
        include_patterns=args.include or [],
        follow_symlinks=args.follow_symlinks,
        scan_hidden=args.scan_hidden,
        quiet=args.quiet,
        verbose=args.verbose,
        show_scanned=args.show_scanned,
        retry_attempts=args.retry_attempts,
        retry_backoff=args.retry_backoff,
    )
    
    # Add additional exclusions
    if args.exclude_dir:
        config.excluded_dirs.update(args.exclude_dir)
    if args.exclude_ext:
        config.excluded_extensions.update(args.exclude_ext)
    
    # Handle cache operations before main scan
    if config.clear_cache or config.cache_stats or config.rebuild_cache:
        try:
            from cache_manager import CacheManager
            cache = CacheManager(config.cache_path, config.cache_max_size_mb)
            
            if config.clear_cache:
                cache.clear_cache()
                print("Cache cleared successfully.")
                return
            
            if config.cache_stats:
                stats = cache.get_cache_stats()
                print("Cache Statistics:")
                print("="*50)
                print(f"  Entries: {stats['entry_count']:,}")
                print(f"  Total Size: {stats['total_size_mb']:.1f} MB")
                print(f"  Hit Rate: {stats['hit_rate_percent']:.1f}%")
                print(f"  Bytes Saved: {stats['bytes_saved_mb']:.1f} MB")
                if stats.get('newest_entry'):
                    print(f"  Newest Entry: {stats['newest_entry']}")
                if stats.get('oldest_entry'):
                    print(f"  Oldest Entry: {stats['oldest_entry']}")
                return
            
            if config.rebuild_cache:
                cache.clear_cache()
                print("Cache cleared for rebuild...")
                
        except ImportError:
            print("Cache manager not available")
            if config.clear_cache or config.cache_stats:
                return
    
    # Handle GPU info command
    if args.gpu_info:
        try:
            from gpu_accelerator import get_gpu_accelerator, check_gpu_support
            
            support = check_gpu_support()
            print("GPU Acceleration Support:")
            print("="*50)
            print(f"  CUDA Available: {'✓' if support['cuda'] else '✗'}")
            print(f"  OpenCL Available: {'✓' if support['opencl'] else '✗'}")
            print(f"  Any GPU Available: {'✓' if support['any_gpu'] else '✗'}")
            print()
            
            if support['any_gpu']:
                gpu_manager = get_gpu_accelerator(config.gpu_backend)
                if gpu_manager.is_gpu_available():
                    info = gpu_manager.get_acceleration_info()
                    print(f"Active Backend: {info.get('backend', 'unknown')}")
                    print(f"Device Name: {info.get('name', 'unknown')}")
                    
                    if 'total_memory' in info:
                        print(f"Memory: {info['total_memory'] / (1024**3):.1f} GB")
                    if 'compute_capability' in info:
                        print(f"Compute Capability: {info['compute_capability']}")
                    if 'multiprocessor_count' in info:
                        print(f"Multiprocessors: {info['multiprocessor_count']}")
                
                gpu_manager.cleanup()
            else:
                print("No GPU acceleration available.")
                print("Install dependencies: pip install pycuda pyopencl")
            
            return
                
        except ImportError:
            print("GPU acceleration module not available")
            return
    
    # Print banner
    if not config.quiet:
        print(f"TurboDedup v{__version__}")
        print("="*70)
    
    # Run scan
    try:
        scanner = UltimateScanner(config)
        stats = scanner.scan()
        
        # Generate report
        if not config.quiet:
            generate_report(config, stats, scanner.db)
            
        # Show scanned folders summary if requested
        if config.show_scanned:
            display_scanned_summary(scanner.db)
        
        # Legacy compatibility - deprecated, use integrated system instead
        if config.interactive_clean and stats.duplicate_sets > 0:
            print(f"\nLegacy interactive cleaner is deprecated. Using integrated deletion system instead.")
            config.delete_strategy = "interactive"  # Force interactive mode
        
        # Integrated deletion processing
        if config.delete_duplicates and stats.duplicate_sets > 0:
            print(f"\nProcessing duplicates with integrated deletion system...")
            try:
                duplicate_manager = IntegratedDuplicateManager(config, stats)
                groups = scanner._load_duplicate_groups()
                
                if groups:
                    # Apply deletion strategy to each group
                    for group in groups:
                        duplicate_manager.apply_deletion_strategy(group)
                    
                    # Separate groups by operation type
                    groups_with_deletions = [g for g in groups if g.selected_for_deletion]
                    groups_with_symlinks = [g for g in groups if hasattr(g, 'selected_for_symlink') and g.selected_for_symlink]
                    
                    # Execute operations with unified confirmation
                    if groups_with_deletions or groups_with_symlinks:
                        duplicate_manager.execute_mixed_operations(groups_with_deletions, groups_with_symlinks)
                else:
                    print("No duplicate groups found.")
            except Exception as e:
                print(f"Error in duplicate processing: {e}")
                import traceback
                traceback.print_exc()
            
    except ValueError as e:
        parser.error(str(e))
    except KeyboardInterrupt:
        print("\nScan interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()